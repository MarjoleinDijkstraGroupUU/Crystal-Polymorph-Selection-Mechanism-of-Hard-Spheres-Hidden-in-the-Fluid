#from __future__ import print_function, division
import argparse
from hoomd import *
from hoomd import md
import numpy as np
import linecache

parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=int, required=True)
args = parser.parse_args()
trial = args.trial

sim1 = context.initialize('--mode=cpu')
temp = 0.025
beta = 1./temp
betap = 13.40
d0 = f'results/coli/betap{betap:.2f}/trial{trial}'

# file_name_bp = 'siminfo.dat'
 #float(linecache.getline(file_name_bp, 1))
press = betap*temp

# Get the workspace output dir for storing benchmark metadata
if len(option.get_user()) == 0:
    workspace = '.';
else:
    workspace = option.get_user()[0]

file_name_in = f'{d0}/config.sph'
f = open(file_name_in,'r')
in_data = np.loadtxt(f,skiprows=2, usecols = (1,2,3,4), dtype=float);
f.close()

print(linecache.getline(file_name_in, 1))
n_part = int(linecache.getline(file_name_in, 1))
lx = float(((linecache.getline(file_name_in, 2).split())[0]))
ly = float(((linecache.getline(file_name_in, 2).split())[1]))
lz = float(((linecache.getline(file_name_in, 2).split())[2]))
print(n_part,lx,ly,lz)

f = open(file_name_in,'r')
in_data_ptype = np.loadtxt(f,skiprows=2, usecols = (0), dtype='i1');
f.close()

print(in_data_ptype)
print(in_data[0,0:3])

snap = data.make_snapshot(N=n_part, box=data.boxdim(Lx=lx,Ly=ly,Lz=lz,dimensions=3),particle_types=['A','B']);
print(snap.particles.position)
print(snap.particles.typeid)
print(snap.particles.diameter)

fe_file = open(f'{d0}/printconfig.dat', 'w');
fe_file.write ("%4s  %14s  %14s  %14s  %14s\n" % ("type", "pos_x", "pos_y", "pos_z", "dia"));
for i in range(snap.particles.N):
	snap.particles.typeid[i] = in_data_ptype[i] ;
	snap.particles.position[i] = in_data[i,0:3] ;
	snap.particles.diameter[i] = in_data[i,3] ;
	fe_file.write ("%1i  %16.8f  %16.8f  %16.8f  %16.8f\n" % (snap.particles.typeid[i], snap.particles.position[i][0],snap.particles.position[i][1],snap.particles.position[i][2],snap.particles.diameter[i])) ;

if os.path.isfile('./restart.gsd'):
    system = init.read_gsd('./trajectory.gsd',restart='./restart.gsd')
else:
    system = init.read_snapshot(snap)


all = group.all()
print(system.box)
print(all)

init_snap = system.take_snapshot()
N = len(system.particles)
mass = init_snap.particles.mass[:]     # in case particle masses are not homogeneous 
v_sigma = np.sqrt(temp/mass)
velocities = np.einsum('i,ij->ij', v_sigma, np.random.randn(N, 3))
init_snap.particles.velocity[:] = velocities
system.restore_snapshot(init_snap)

nl = md.nlist.cell()
lj = md.pair.lj(r_cut=2.0**(1.0/6.0), nlist=nl)
lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, r_cut=2.0**(1.0/6.0))
lj.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0, r_cut=2.0**(1.0/6.0))
lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, r_cut=2.0**(1.0/6.0))
lj.set_params(mode="shift")

md.integrate.mode_standard(dt=1e-4)
#md.integrate.nvt(group=group.all(), kT=temp, tau=0.5)
md.integrate.npt(group=group.all(), kT=temp,tau=0.5,P=press,tauP=0.5,x=True, y=True, z=True, xy=False, xz=False, yz=False)

nl.set_params(r_buff=0.6, check_period=10)

analyze.log(f'{d0}/log.dat',['N','potential_energy','temperature','pressure','volume'],10000,overwrite=False)

dumper = dump.gsd(filename=f"{d0}/trajectory.gsd", period=10000, group=group.all(),phase=0,overwrite=False,static=[])
run(1e8)
dump.gsd(filename="restart.gsd", group=group.all(), overwrite=True, period=None, phase=0)


