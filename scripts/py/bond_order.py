import subprocess
import numpy as np
import argparse
import pickle
import freud
import ovito
from sann import nlist_sann
from tcc_python_scripts.file_readers import atom

parser = argparse.ArgumentParser(description='BOP analysis')
parser.add_argument('-traj', metavar='traj', type=int, help='trajectory number')
parser.add_argument('-data', metavar='data', type=str, help='which data to analyze')

args = parser.parse_args()
data = args.data


##########################
### ANALYSIS FUNCTIONS ###
##########################

class BOP_Analysis():
    """
    Calculate bond order parameters and save.
    """

    def __init__(self, logdir):
        self.logdir = logdir
        self.logfile = logdir + '/nx.log'
        subprocess.run(f'mkdir -p {self.logdir}/dump-bops'.split())
        with open(self.logfile, "w") as f:
            f.write('frame,Nx\n')

    def _filter(self, box, points):
        # select small cubic region
        cutoff = box[0] / 8
        center = points.mean(axis=-2) #snap.box / 2
        manhattan_dist = (np.abs(points - center)).max(axis=-1)
        particle_mask = (manhattan_dist < cutoff)

        return particle_mask

    def __call__(self, box, points, frame_number, determine_polymorph=False):
        # read
        savedir = self.logdir+f'dump-bops/{frame_number}'
        subprocess.run(f'mkdir -p {savedir}'.split())

        # try:
        #     with open(savedir + '/solid-bonds.pickle', 'rb') as file:
        #         solid_bonds = pickle.load(file)
        # except:
        if True:
            # compute solid bonds
            nlist = nlist_sann(box, points)
            solid_liquid = freud.order.SolidLiquid(normalize_q=True, l=6, q_threshold=0.7, solid_threshold=7)
            solid_liquid.compute((box, points), neighbors=nlist)

            if determine_polymorph:
                # compute w4
                w4 = freud.order.Steinhardt(l=4, wl=True)
                w4 = w4.compute((box, points), neighbors=nlist).particle_order

                nuclei = []
                nucleus_idxs = np.arange(len(solid_liquid.cluster_sizes))
                for nucleus_idx in nucleus_idxs[solid_liquid.cluster_sizes > 5]:
                    nucleus_size = solid_liquid.cluster_sizes[nucleus_idx]
                    w4_cluster = w4[solid_liquid.cluster_idx == nucleus_idx]
                    num_fcc = (w4_cluster < 0).sum()
                    num_hcp = (w4_cluster > 0).sum()
                    nuclei.append((nucleus_size, num_fcc, num_hcp))

            # save with pickle
            solid_bonds  = solid_liquid.num_connections
            with open(savedir + '/solid-bonds.pickle', 'wb') as file:
                pickle.dump(solid_bonds, file)
            with open(savedir + '/nuclei.pickle', 'wb') as file:
                pickle.dump(nuclei, file)

        particle_mask = self._filter(box, points)
        solid_bonds = solid_bonds[particle_mask]
        Nx = (solid_bonds > 6.5).sum()
        
        # write to logfile
        with open(self.logfile, "a") as f:
            f.write(f'{frame_number}, {Nx} \n')


##################
### INPUT DATA ###
##################

if data == 'conversion':
    d0 = f'results/coli/'

    # loop over all trajectories
    for P in [13.4]:
        for i in [2, 3, 6, 9, 11]:
            print(i)
            d_in = d0 + f'betap{P:.2f}/trial{i}/'
            d_out = f'./results/{data}/betap{P:.2f}/trial{i}/'
            logdir = d_out

            bop = BOP_Analysis(logdir)

            trajectory = atom.read(d_in + 'nucleation.dump')
            for i, snap in enumerate(trajectory):
                print(f'Frame {i}')
                box = snap.box[:]
                points = snap.particle_coordinates
                bop(box, points, i)


if data == 'experiments':
    d0 = f'results/experiments/'

    for frame_index in range(1, 2):
        # load
        print(f'{frame_index} / 22')
        filename = f'{d0}/frame_{frame_index}.xyz'
        pipeline = ovito.io.import_file(filename)
        data = pipeline.compute()

        # do BOP analysis
        bop = BOP_Analysis(d0)
        box = np.diag(data.cell[0:3,0:3])
        points = data.particles.positions[:]
        bop(box, points, frame_index, determine_polymorph=True)
