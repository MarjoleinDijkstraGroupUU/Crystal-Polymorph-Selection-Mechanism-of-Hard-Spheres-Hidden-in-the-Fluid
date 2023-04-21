# %%
import random
import os
import numpy as np
import math as m
import argparse

import hoomd
import freud
import gsd
import gsd.hoomd

# %%
# options for interface:
# fcc-100 fcc-110 hcp-1010 hcp-1120 hcp-0001

parser = argparse.ArgumentParser()
parser.add_argument('--interface', type=str, required=True)
parser.add_argument('--n_sample_cycles', type=float, required=True)
args = parser.parse_args()
interface = args.interface
n_sample_cycles = int(args.n_sample_cycles)

# interface = 'fcc-100'
# n_sample_cycles = 1e6
d0 = f'./results/coexistence_md/{interface}'
os.makedirs(d0, exist_ok=True)

n_tune_cycles = int(2e3)

# %% [markdown]
# # 1. Create initial configuration

# %%
# find packing fraction for given pressure
# use analytical equations of state

from scipy.optimize import fsolve

def eos_fluid_res(eta, P_target):
    """ Carahan-Starling """
    P = 6*eta/m.pi * (1+eta+eta**2-eta**3)/(1-eta)**3
    return P - P_target

def eos_solid_res(eta, P_target):
    """ Speedy equation of state """
    a, b, c = 0.5921, 0.7072, 0.601
    eta_cp = m.pi * m.sqrt(2) / 6
    y = eta / eta_cp
    P = m.sqrt(2) * y * (3/(1-y) - a*(y-b)/(y-c))

    return P - P_target

pressure = 11.57
eta_fluid = fsolve(eos_fluid_res, x0=0.5, args=(pressure))[0]
eta_solid = fsolve(eos_solid_res, x0=0.5, args=(pressure))[0]
eta_target = (eta_fluid + eta_solid) / 2

print(f'Target packing fraction = {eta_target:.3f}, average of {eta_fluid:.3f} and {eta_solid:.3f}')

# %% [markdown]
# ## 1a. Generate solid

# %%
def scale_box(box, scale, scale_dimension):
    box_ = hoomd.Box.from_box(box)

    if scale_dimension == 0:
        box_.Lx *= scale
    elif scale_dimension == 1:
        box_.Ly *= scale
    elif scale_dimension == 2:
        box_.Lz *= scale
    else:
        raise ValueError()
    return box_

# %%
from scipy.spatial.transform import Rotation
# generate lattice of packing fraction eta_solid

if interface.startswith('fcc'):
    fcc = freud.data.UnitCell.fcc()
    density = eta_solid * 6 / m.pi
    if interface == 'fcc-100':
        box, position = fcc.generate_system((10,10,10), sigma_noise=0.01, scale=(4/density)**(1/3))
        scale_dimension = 2
    elif interface == 'fcc-110':
        box, position = fcc.generate_system((25,25,10), sigma_noise=0, scale=(4/density)**(1/3))
        scale_dimension = 0
        # rotate
        position = Rotation.from_euler('z', 45, degrees=True).apply(position)
        # select smaller box
        box = scale_box(box, 0.4, 0)
        box = scale_box(box, 0.4, 1)
        mask = (np.abs(position)-1e-3 < box.L[None, :]/2).all(axis=-1)
        mask *= (position+0.1 < box.L[None, :]/2)[:, :2].all(axis=-1)
        position = position[mask]
    elif interface == 'fcc-111':
        raise ValueError('not supported')

if interface.startswith('hcp'):
    fractions = np.array([[0, 1/3, 0.5], [0.5, 5/6, 0.5], [0.5, 0.5, 0], [0, 0, 0]])
    hcp = freud.data.UnitCell([1, m.sqrt(3), m.sqrt(8/3)], fractions)
    density = eta_solid * 6 / m.pi
    box, position = hcp.generate_system((14,8,8), sigma_noise=0.01, scale=(m.sqrt(2)/density)**(1/3))

    if interface == 'hcp-1120':
        scale_dimension = 0
    elif interface == 'hcp-1010':
        scale_dimension = 1
    elif interface == 'hcp-0001':
        scale_dimension = 2

# %%
def write_gsd(filename, position, box):
    snapshot = gsd.hoomd.Snapshot()
    snapshot.particles.N = position.shape[0]
    snapshot.particles.position = position

    snapshot.particles.typeid = [0] * snapshot.particles.N
    snapshot.configuration.box = list(box.L) + [0,0,0]

    with gsd.hoomd.open(name=filename, mode='wb') as f:
        f.append(snapshot)

# %%
write_gsd(d0 + '/solid.gsd', position, box)

# %% [markdown]
# ## 1b. Generate fluid

# %%
def get_sim():
    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=20)

    # hard spheres
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.1)
    mc.shape["A"] = dict(diameter=1.0)
    sim.operations.integrator = mc

    return sim

# %%
# generate dilute fluid
print('Generating dilute fluid')
sim = get_sim()
box_fluid = scale_box(box, 3, scale_dimension)
position_fluid = position
position_fluid[:, scale_dimension] *= 3
write_gsd(d0+'/fluid_init.gsd', position_fluid, box_fluid)
sim.create_state_from_gsd(filename=d0+'/fluid_init.gsd')
sim.run(1e3)
hoomd.write.GSD.write(state=sim.state, mode='wb', filename=d0+'/fluid_dilute.gsd')

# %%
# compress fluid
sim = get_sim()
sim.create_state_from_gsd(filename=d0+'/fluid_dilute.gsd')

# compress fluid to target packing fraction
final_box = scale_box(box, eta_solid/eta_fluid, scale_dimension)
compress = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10), target_box=final_box)
sim.operations.updaters.append(compress)

# tune displacement moves
tune_displacement = hoomd.hpmc.tune.MoveSize.scale_solver(moves=['d'], target=0.2, trigger=hoomd.trigger.Periodic(50),)
sim.operations.tuners.append(tune_displacement)

print('Compressing fluid')

for i in range(5):
    sim.run(1000)
    
    eta = sim.state.N_particles / sim.state.box.volume * m.pi / 6
    print(f'Compressed to {eta:.3f}, target {eta_fluid:.3f}')
    if np.abs(eta - eta_fluid) < 5e-3:
        break
    
hoomd.write.GSD.write(state=sim.state, mode='wb', filename=d0+'/fluid_dense.gsd')

# %% [markdown]
# ## 1c. Combine fluid and solid

# %%
solid = gsd.hoomd.open(d0+'/solid.gsd').read_frame(0)
fluid = gsd.hoomd.open(d0+'/fluid_dense.gsd').read_frame(0)

# scale box
solid_box = solid.configuration.box
fluid_box = fluid.configuration.box
box = np.append(hoomd.Box.from_box(solid_box).L, [0, 0, 0])
box[scale_dimension] += fluid_box[scale_dimension] + 2

# place fluid next to solid
fluid.particles.position[:, scale_dimension] += 1 + (fluid_box[scale_dimension] + solid_box[scale_dimension]) / 2
mask = (fluid.particles.position[:, scale_dimension] > box[scale_dimension]/2)
fluid.particles.position[mask, scale_dimension] -= box[scale_dimension]

# set snapshot
snapshot = gsd.hoomd.Snapshot()
snapshot.particles.N = solid.particles.N + fluid.particles.N
snapshot.configuration.box = box
snapshot.particles.position = np.append(solid.particles.position, fluid.particles.position)
snapshot.particles.typeid = np.append(solid.particles.typeid, fluid.particles.typeid)

with gsd.hoomd.open(name=d0+'/coexistence_init_dilute.gsd', mode='wb') as f:
    f.append(snapshot)

# %%
# compress solid and fluid together
sim = get_sim()
sim.create_state_from_gsd(filename=d0+'/coexistence_init_dilute.gsd')
eta = sim.state.N_particles / sim.state.box.volume * m.pi / 6

# compress to target packing fraction
final_box = scale_box(box, eta / eta_target, scale_dimension)
compress = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10), target_box=final_box)
sim.operations.updaters.append(compress)

# tune displacement moves
tune_displacement = hoomd.hpmc.tune.MoveSize.scale_solver(moves=['d'], target=0.2, trigger=hoomd.trigger.Periodic(50),)
sim.operations.tuners.append(tune_displacement)

print('Compressing solid and fluid together')
for i in range(10):
    sim.run(100)
    
    eta = sim.state.N_particles / sim.state.box.volume * m.pi / 6
    print(f'Compressed to {eta:.3f},  target {eta_target:.3f}')
    if np.abs(eta - eta_target) < 1e-3:
        break

hoomd.write.GSD.write(state=sim.state, mode='wb', filename=d0+'/coexistence_init_dense.gsd')


# %%
print('Switch to MD for main simulation')
integrator = hoomd.md.Integrator(dt=0.004)
cell = hoomd.md.nlist.Cell(buffer=0.4)

lj = hoomd.md.pair.LJ(nlist=cell)
lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1/1.097)
lj.r_cut[('A', 'A')] = 2**(1/6) / 1.097
integrator.forces.append(lj)

nvt = hoomd.md.methods.NVT(kT=0.025, filter=hoomd.filter.All(), tau=1.0)
integrator.methods.append(nvt)

cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=20)
sim.create_state_from_gsd(filename=d0+'/coexistence_init_dense.gsd')
sim.operations.integrator = integrator
sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=0.025)

file = '/coexistence.gsd'
gsd_writer = hoomd.write.GSD(filename=d0+file,
                             trigger=hoomd.trigger.Periodic(25000),
                             mode='wb')
sim.operations.writers.append(gsd_writer)

print('Starting main simulation')
sim.run(n_sample_cycles)

# %%
# sim.state.set_snapshot(initial_state)
eta = sim.state.N_particles / sim.state.box.volume * m.pi / 6
print(eta, eta_target)
