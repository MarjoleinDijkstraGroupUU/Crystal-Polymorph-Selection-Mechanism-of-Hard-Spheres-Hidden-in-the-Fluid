import sys
import os
# These environment variables prevent numpy multithreading, which is undesirable on computer clusters
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import random
import math
import numpy as np
import hoomd
import hoomd.md
import hoomd.hpmc
import gsd.hoomd
import freud
import hoomd.dem.utils
import networkx as nx


# ========================================== Input =================================================

initial_file = sys.argv[1]
NnFCC_target = int(sys.argv[2])
NnHCP_target = int(sys.argv[3])
pressure = float(sys.argv[4])
coupling_Nn = float(sys.argv[5])
rng_seed = int(sys.argv[6])

# ======================================== Functions ===============================================

# Turn off Freud's multithreading for use on cluster
freud.parallel.set_num_threads(1)

def hoomd_box_to_lattice_vecs(box):
    # https://hoomd-blue.readthedocs.io/en/stable/box.html#definitions-and-formulas-for-the-cell-parameter-matrix
    h = [
     [box.Lx         , 0.0            , 0.0   ],
     [box.xy * box.Ly, box.Ly         , 0.0   ],
     [box.xz * box.Lz, box.yz * box.Lz, box.Lz]
    ]
    a1 = h[0];    a2 = h[1];    a3 = h[2]
    return a1, a2, a3;

solidCalc = freud.order.SolidLiquid(normalize_q=True, l=6, q_threshold=0.7, solid_threshold=7)
nbQueryDict = dict(mode='ball', r_max=1.4)
def orderParameterd6(snapshot):
    nq = freud.locality.AABBQuery.from_system(snapshot)
    nlist = nq.from_system(snapshot).query(snapshot.particles.position, nbQueryDict).toNeighborList()
    # Don't just feed the nbQueryDict to compute, that creates a memory leak (Freud 2.2.0)
    solidCalc.compute(snapshot, neighbors=nlist)
    return solidCalc.largest_cluster_size

w4calc = freud.order.Steinhardt(l=4, average=False, wl=True, wl_normalize=True)
def w4(snapshot):
    nq = freud.locality.AABBQuery.from_system(snapshot)
    nlist = nq.from_system(snapshot).query(snapshot.particles.position, nbQueryDict).toNeighborList()
    # Don't just feed the nbQueryDict to compute, that creates a memory leak (Freud 2.2.0)
    w4calc.compute(snapshot, neighbors=nlist)
    return w4calc.particle_order

def solidw4(snapshot):
    nq = freud.locality.AABBQuery.from_system(snapshot)
    nlist = nq.from_system(snapshot).query(snapshot.particles.position, nbQueryDict).toNeighborList()
    # Don't just feed the nbQueryDict to compute, that creates a memory leak (Freud 2.2.0)
    solidCalc.compute(snapshot, neighbors=nlist)
    w4calc.compute(snapshot, neighbors=nlist)
    # The largest cluster has index 0, so select all particles belonging to that cluster.
    solid_indices = [ p_idx for p_idx, cl_idx in enumerate(solidCalc.cluster_idx) if cl_idx == 0 ]
    return [ w4calc.particle_order[i] for i in solid_indices ]

def nFCCnHCP(snapshot):
    nq = freud.locality.AABBQuery.from_system(snapshot)
    nlist = nq.from_system(snapshot).query(snapshot.particles.position, nbQueryDict).toNeighborList()
    # Don't just feed the nbQueryDict to compute, that creates a memory leak (Freud 2.2.0)
    solidCalc.compute(snapshot, neighbors=nlist)
    w4calc.compute(snapshot, neighbors=nlist)
    # The largest cluster has index 0, so select all particles belonging to that cluster.
    solid_indices = [ p_idx for p_idx, cl_idx in enumerate(solidCalc.cluster_idx) if cl_idx == 0 ]
    fcc_indices = [ p_idx for p_idx in solid_indices if w4calc.particle_order[p_idx] < 0.0 ]
    return ( len(fcc_indices), len(solid_indices) - len(fcc_indices) )

avq6calc = freud.order.Steinhardt(l=6, average=True)
def avq6(snapshot):
    nq = freud.locality.AABBQuery.from_system(snapshot)
    nlist = nq.from_system(snapshot).query(snapshot.particles.position, nbQueryDict).toNeighborList()
    # Don't just feed the nbQueryDict to compute, that creates a memory leak (Freud 2.2.0)
    avq6calc.compute(snapshot, neighbors=nlist)
    return avq6calc.particle_order

def fast_compression(system, target_packfrac, part_volume, scale_factor=0.95, sp=True):
    # Fast compression until we exceed the desired packing fraction
    N = len(hoomd.group.all())
    target_volume = N*part_volume / target_packfrac
    current_volume = system.box.get_volume()
    print("Attempting to rapidly compress to desired packing fraction %0.04f."%target_packfrac)
    while target_volume < current_volume:
        current_volume = max(current_volume*scale_factor, target_volume)
        new_box = system.box.set_volume(current_volume)
        hoomd.update.box_resize(Lx=new_box.Lx, Ly=new_box.Ly, Lz=new_box.Lz, period=None, scale_particles=sp)
        overlaps = mc.count_overlaps()
        current_packfrac = N*part_volume / system.box.get_volume()
        if overlaps > N / 2:
            scale_factor = scale_factor ** 0.5
        if scale_factor > (1 - 5e-5):
            print("Failed to compress to desired packing fraction. Exiting.")
            sys.exit()
        print("phi = %0.07f  scale_factor = %0.06f  overlaps = %d"%(current_packfrac,scale_factor,overlaps), end=' ')
        # run until all overlaps are removed
        tries = 0
        while overlaps > 0:
            hoomd.run(100, quiet=True)
            overlaps = mc.count_overlaps()
            print(overlaps, end=' ')
            sys.stdout.flush()
            if tries > 100:
                current_volume = current_volume/scale_factor
                scale_factor = scale_factor ** 0.5
                break
            if tries < 3 and overlaps == 0:
                scale_factor = scale_factor ** 1.05
            tries += 1
        print()
    print("\nFinished fast compression to packing fraction",N*part_volume / system.box.get_volume() )
    # Expand to the desired packing fraction
    new_box = system.box.set_volume(target_volume)
    hoomd.update.box_resize(Lx=new_box.Lx, Ly=new_box.Ly, Lz=new_box.Lz, period=None)
    print("Expanded to desired packing fraction",N*part_volume / system.box.get_volume() )


# ===================================== Initialization =============================================

hoomd.util.quiet_status() # silence spam
hoomd.context.initialize("--notice-level=1 --nthreads=1") # force single-thread on cluster
# hoomd.context.initialize("--notice-level=1 --mode=cpu") # choose cpu or gpu for running locally

# Read in the initial configuration and its particle shape
system = hoomd.init.read_gsd(initial_file, frame=-1)

# Set up the particle shape / integrator
mc = hoomd.hpmc.integrate.sphere(seed=rng_seed, d=0.3)
diameter = 1.0
mc.shape_param.set('A', diameter=diameter)
mc.shape_param.set('B', diameter=diameter) # Dummy type leftover from initial configuration
part_volume = (math.pi / 6.0) * diameter**3

# Set up a neighbour list
nl = hoomd.md.nlist.tree()

# Set up an NPT ensemble
boxmc = hoomd.hpmc.update.boxmc(mc, betaP=pressure, seed=rng_seed)
boxmc.ln_volume(delta=0.001, weight=1.0)

# ================================== Biasing and sampling ==========================================

# Define snapshot saving, using a custom logger to make chunks for the cluster
d = hoomd.dump.gsd("HSUmbrellaInitialization.gsd", period=int(5e5), phase=-1, group=hoomd.group.all(), overwrite=True)
def logClusterIdx(step):
    # d6 order parameter
    snap = system.take_snapshot()
    Nn = orderParameterd6(snap)
    largest_cl_idx = np.argmax(solidCalc.cluster_sizes)
    cl_indices = solidCalc.cluster_idx
    return np.array(cl_indices, dtype=np.int32)
d.log["particles/dynamic_label"] = logClusterIdx
def logw4(step):
    snap = system.take_snapshot()
    return np.array(w4(snap), dtype=np.float32)
d.log["particles/w4"] = logw4
def logavq6(step):
    snap = system.take_snapshot()
    return np.array(avq6(snap), dtype=np.float32)
d.log["particles/avq6"] = logavq6
d.dump_state(mc)

# A harmonic potential on the order parameter
def biasUNn(Nn, target):
    U = 0.5 * coupling_Nn * (Nn - target) * (Nn - target)
    # print(order,target,U)
    return U

# Define a class that constructs the order parameter histogram we want from umbrella sampling
class sampler:
    def __init__(self, n_bins):
        self.Nn_histogram = [x[:] for x in [[0] * n_bins] * n_bins] # without [:] we get n_bins references to *the same* row!
        self.next_timeseries_step = 0
        with open("Nn_timeseries.txt",'w') as file:
            file.write("")
    def add_sample(self, step, NnFCC, NnHCP):
        self.Nn_histogram[NnFCC][NnHCP] += 1
        if step > self.next_timeseries_step:
            with open("Nn_timeseries.txt",'a') as file:
                file.write("%d %d\n"%(NnFCC,NnHCP))
            self.next_timeseries_step += 1000
    def reset(self):
        n_bins = len(self.Nn_histogram)
        self.Nn_histogram = [x[:] for x in [[0] * n_bins] * n_bins] # without [:] we get n_bins references to *the same* row!
        with open("Nn_timeseries.txt",'w') as file:
            file.write("")
    def get_dG_histogram(self):
        hist = []
        # Calculate norm of count histogram to transform into probability histogram
        norm = 0
        for row in self.Nn_histogram: norm += sum(row)
        if norm == 0: norm = 1.0
        # Calculate the <1/W>_W average needed to unbias the histogram
        one_over_w_av = 0.0
        for NnFCC, row in enumerate(self.Nn_histogram):
            for NnHCP, counts in enumerate(row):
                if math.isnan(counts) or counts == 0: continue
                W = np.exp( -(biasUNn(NnFCC, NnFCC_target) + biasUNn(NnHCP, NnHCP_target)) )
                one_over_w_av += counts / W
        one_over_w_av /= norm
        # Unbias the nucleus size histogram to obtain the free energy barrier dG(Nn)
        for NnFCC, row in enumerate(self.Nn_histogram):
            for NnHCP, counts in enumerate(row):
                if math.isnan(counts) or counts == 0: continue
                W = np.exp( -(biasUNn(NnFCC, NnFCC_target) + biasUNn(NnHCP, NnHCP_target)) )
                # Calculate the unbiased probability histogram P(Nn) = ( P_W(Nn)/W(Nn) ) / <1/W>_W
                P = ((counts / norm) / W) / one_over_w_av
                # Transform to a barrier height dG(Nn) = -ln( P(Nn) )
                # print(NnFCC,NnHCP,counts,W,P)
                dG = -math.log(P)
                hist.append( (NnFCC, NnHCP, dG) )
        return hist
    def save_dG_histogram(self, step):
        with open("dG_hist_nFCC%03d_nHCP%03d"%(NnFCC_target,NnHCP_target) + "_%2.05f"%coupling_Nn + ".txt",'w') as file:
            hist = self.get_dG_histogram()
            samples = self.Nn_histogram
            for NnFCC, NnHCP, dG in hist:
                file.write("%d"%NnFCC + " %d"%NnHCP + " %e"%dG + " %d\n"%samples[NnFCC][NnHCP])
    def save_raw_histogram(self, step):
        with open("rawhist_nFCC%03d_nHCP%03d"%(NnFCC_target,NnHCP_target) + "_%2.05f"%coupling_Nn + ".txt",'w') as file:
            file.write("%d"%NnFCC_target + " %d"%NnHCP_target + " %f\n"%coupling_Nn)
            for NnFCC, row in enumerate(self.Nn_histogram):
                for NnHCP, counts in enumerate(row):
                    if math.isnan(counts) or counts == 0: continue
                    file.write("%d"%NnFCC + " %d"%NnHCP + " %d\n"%counts)
sampler = sampler(n_bins=len(hoomd.group.all()))
sample_save_callback = hoomd.analyze.callback(sampler.save_dG_histogram, period=int(1e4), phase=1)
sample_save_callback.disable()
sample_save_callback2 = hoomd.analyze.callback(sampler.save_raw_histogram, period=int(1e4), phase=1)
sample_save_callback2.disable()

# Define a callback function that adds an additional Metropolis check to bias the sampling
class biaser:
    def __init__(self, system):
        self.system = system
        self.seed = random.seed()
        self.prev_snap = self.system.take_snapshot()
        (self.prev_NnFCC, self.prev_NnHCP) = nFCCnHCP(self.prev_snap)
        self.ratio = 0.0
        self.accepted = 0
        self.samples = 0
        self.restrict_window = False
    def __call__(self, step):
        # Get the required system info
        snap = self.system.take_snapshot()
        (NnFCC, NnHCP) = nFCCnHCP(snap)
        self.samples += 1
        # Check the Metropolis criterion
        dUNnFCC = biasUNn(NnFCC, NnFCC_target) - biasUNn(self.prev_NnFCC, NnFCC_target)
        dUNnHCP = biasUNn(NnHCP, NnHCP_target) - biasUNn(self.prev_NnHCP, NnHCP_target)
        # print(NnFCC, NnHCP)
        # print("uFCC: %.04e"%dUNnFCC + "  uHCP: %.04e"%dUNnHCP)
        if( random.random() < np.exp(-(dUNnFCC+dUNnHCP)) ): # Accepted
            self.prev_snap = snap
            self.prev_NnFCC = NnFCC
            self.prev_NnHCP = NnHCP
            self.accepted += 1
            self.ratio += (1.0 / self.samples) * (1.0 - self.ratio)
        else: # Rejected
            self.system.restore_snapshot(self.prev_snap)
            self.ratio += (1.0 / self.samples) * (0.0 - self.ratio)
        sampler.add_sample(step, self.prev_NnFCC, self.prev_NnHCP)
biaser = biaser(system)
bias_analyzer = hoomd.analyze.callback(biaser, period=1)

# This has a bizzare bug where the biaser disappears
# class biasTuner:
#     def __init__(self, biaser, biaser_analyzer, period):
#         self.period = period
#         self.biaser = biaser
#         self.analyzer = biaser_analyzer
#     def __call__(self, step):
#         print("Bias moves accepted: %d/%d (%0.04f)"%(biaser.accepted,biaser.samples,biaser.ratio)," bias period: %d"%int(self.period))
#         if   self.biaser.ratio < 0.2: self.period *= 0.6
#         elif self.biaser.ratio > 0.4: self.period /= 0.85
#         if self.period < 2:     self.period = 2
#         elif self.period > 100: self.period = 100
#         self.biaser.ratio = 0.0
#         self.biaser.samples = 0
#         self.biaser.accepted = 0
#         self.analyzer.set_period(int(self.period))
# bias_period = 15
# bias_analyzer = hoomd.analyze.callback(biaser, period=bias_period)
# biasTuner = biasTuner(biaser, bias_analyzer, bias_period)
# tunerBias_analyzer = hoomd.analyze.callback(biasTuner, period=int(1e2), phase=-1)

# ============================================ Run =================================================

eq_cycles = int(1e5)
sample_cycles = int(1e6)

# Set up tuners for the MC displacement and NPT volume step size
tunerMC = hoomd.hpmc.util.tune(obj=mc, tunables=['d'], target=0.3, gamma=0.1)
def updateTunerMC(step): tunerMC.update()
tunerMC_callback = hoomd.analyze.callback(updateTunerMC, period=int(1e2))
tunerNPT = hoomd.hpmc.util.tune_npt(obj=boxmc, tunables=['dlnV'], target=0.2, gamma=0.5)
def updateTunerNPT(step): tunerNPT.update()
tunerNPT_callback = hoomd.analyze.callback(updateTunerNPT, period=int(1e2))

# def viewTuners(step):
#     print("d:",tunerMC.tunables['d']['get'](),"dlnV:",tunerNPT.tunables['dlnV']['get']())
# tuner_view_callback = hoomd.analyze.callback(viewTuners, period=1e2)
def printFunction(step):
    print("Nn_FCC: %d (target %d), NnHCP: %d (target %d)"%(biaser.prev_NnFCC,NnFCC_target,biaser.prev_NnHCP,NnHCP_target))
    print("Bias moves accepted: %d/%d (%0.04f)"%(biaser.accepted,biaser.samples,biaser.ratio))
    print("d6Nn:",orderParameterd6(biaser.prev_snap)," nucleus <w4>: %.04f"%np.mean(solidw4(biaser.prev_snap)))
    # print("trial Nn: %d"%orderParameterd6(system.take_snapshot()), "trial nucleus <w4>: %.04f"%np.mean(solidw4(system.take_snapshot())))
    # print("trial NnFCC: %d  trial NnHCP: %d"%nFCCnHCP(system.take_snapshot()))
print_callback = hoomd.analyze.callback(printFunction, period=int(1e4), phase=-1)

# equilibrate first, then sample
print("\n ===== Equilibrating... =====\n")
hoomd.run(eq_cycles)

# Save snapshots from the sampling run separately
d.disable()
d = hoomd.dump.gsd(filename="HSUmbrella_nFCC%03d_nHCP%03d.gsd"%(NnFCC_target,NnHCP_target), period=int(1e5), phase=-1, group=hoomd.group.all(), overwrite=True)
d.log["particles/dynamic_label"] = logClusterIdx
d.log["particles/w4"] = logw4
d.log["particles/avq6"] = logavq6
d.dump_state(mc)

print("\n ===== Sampling... =====\n")
tunerMC_callback.disable()
tunerNPT_callback.disable()
# tuneBias_analyzer.disable()
sampler.reset()
sample_save_callback.enable()
sample_save_callback2.enable()
hoomd.run(sample_cycles)

# Save the histogram
sampler.save_dG_histogram(0)
sampler.save_raw_histogram(0)
