import os
import subprocess
import numpy as np
import argparse
import pickle
import ovito

parser = argparse.ArgumentParser(description='TCC analysis')
parser.add_argument('-traj', metavar='traj', type=int, help='trajectory number')
parser.add_argument('-data', metavar='data', type=str, help='which data to analyze')

args = parser.parse_args()
data = args.data


# setup tcc
from tcc_python_scripts.file_readers import atom
from tcc_python_scripts.tcc import wrapper


full_clusterlist = ["sp3a", "sp3b", "sp3c", "sp4a", "sp4b", "sp4c", "sp5a", "sp5b", "sp5c",
                    "6A", "6Z", "7K", "7T_a", "7T_s", "8A", "8B", "8K", "9A", "9B", "9K", "10A", "10B",
                    "10K", "10W", "11A", "11B", "11C", "11E", "11F", "11W", "12A", "12B", "12D",
                    "12E", "12K", "13A", "13B", "13K", "FCC", "HCP", "BCC_9"]

if data in ['coli', 'conversion', 'coexistence', 'experiments']:
    clusters_to_analyse = {key: 1 for key in full_clusterlist}
else:
    clusters_to_analyse = {'FCC': 1, 'HCP': 1, '8A': 1, 'sp5c': 1,}

tcc = wrapper.TCCWrapper(clusters_to_analyse)
tcc.input_parameters['Output']['Raw'] = True
# tcc.tcc_executable_directory = '/nethome/gispe004/TCC/bin/'
tcc.tcc_executable_directory = '/home/willem/Codes/TCC/build/src/'
tcc.tcc_executable_path = tcc.tcc_executable_directory + 'tcc'
tcc.input_parameters['Simulation']['cell_list'] = True


##########################
### ANALYSIS FUNCTIONS ###
##########################

class TCC_Analysis():
    """
    Do TCC and save.
    """

    def __init__(self, logdir):
        self.logdir = logdir
        subprocess.run(f'mkdir -p {self.logdir}/dump'.split())

    def __call__(self, box, points, frame_number, save_raw=True, save_clusters=True):
        tcc.run(box, points, output_clusters=True, silent=False)
        # save as text
        savedir = self.logdir+f'dump/{frame_number}'
        if save_raw:
            tcc.save(savedir)
        else:
            subprocess.run(f'mkdir -p {self.logdir}/dump/{frame_number}'.split())

        # save with pickle
        cluster_table = tcc.get_cluster_table(amount=True)[0] # get amount of clusters per particle
        clusters = {}
        if save_clusters:
            for cluster_name in clusters_to_analyse:
                clusters[cluster_name] = tcc._parse_cluster_file(cluster_name)
        save = (cluster_table, clusters)
        with open(savedir + '/clusters.pickle', 'wb') as file:
            pickle.dump(save, file)

        solidlike = ((cluster_table['FCC'] + cluster_table['HCP']) > 2).values
        print((cluster_table['BCC_9'][~solidlike]>0).mean())

        tcc.__del__()


##################
### INPUT DATA ###
##################

if data == 'conversion':
    d0 = f'results/coli/'

    # loop over all trajectories
    for P in [13.4]:
        for i in [args.traj]:
            d_in = d0 + f'betap{P:.2f}/trial{i}/'
            logdir = f'./results/conversion/betap{P:.2f}/trial{i}/'
            # logdir = f'./results/conversion/betap{P:.2f}-fluid/trial{i}/'

            r_max = 1.8 * 2**(1/6) / (1+ 0.025**(1/2))**(1/6)
            tcc.input_parameters['Simulation']['rcutAA'] = r_max
            tcc.input_parameters['Simulation']['PBCs'] = True

            tcc_save = TCC_Analysis(logdir)

            trajectory = atom.read(d_in + 'nucleation.dump')
            # trajectory = atom.read(d_in + 'fluid.dump')
            for i, snap in enumerate(trajectory):
                print(f'Frame {i}')
                box = snap.particle_coordinates.max(axis=-2)
                points = snap.particle_coordinates
                tcc_save(box, points, i, save_raw=False)

# conda activate ovito-freud
# python -m scripts.py.tcc -data conversion -traj 2


if data == 'coexistence':
    plane = ['fcc-100', 'fcc-110', 'hcp-0001', 'hcp-1010', 'hcp-1120'][args.traj]
    plane = ['fcc-110', 'hcp-1120'][args.traj]
    d_in = d0 = f'results/coexistence_md/{plane}/'
    logdir = d_in
    # logdir = f'results/coexistence_md/{plane}/'

    # loop over all trajectories
    filename = f'{d_in}/coexistence.gsd' #fluid_dense.gsd' #
    # filename = f'{d_in}/coexistence_init_dense.gsd'
    pipeline = ovito.io.import_file(filename)

    # set TCC parameters
    r_max = 1.8
    tcc.input_parameters['Simulation']['rcutAA'] = r_max
    tcc.input_parameters['Simulation']['PBCs'] = True
    tcc_save = TCC_Analysis(logdir)

    # do TCC analysis
    start = pipeline.source.num_frames // 2
    stop = pipeline.source.num_frames
    step = 20
    for frame_index in range(start, stop, step):
    # for frame_index in range(1):
        data_ = pipeline.compute(frame_index)
        box = np.diag(data_.cell[0:3,0:3])
        points = data_.particles.positions[:]
        tcc_save(box, points, frame_index, save_raw=False, save_clusters=False)


if data == 'experiments':
    d0 = f'results/experiments'

    for frame_index in [20, 21, 22]: #range(2, 23):
        # load
        print(f'{frame_index} / 22')
        filename = f'{d0}/frame_{frame_index}.xyz'
        pipeline = ovito.io.import_file(filename)
        data_ = pipeline.compute()

        # set TCC parameters
        rho = data_.particles.position.shape[0] / data_.cell.volume
        r_max = 1.8 / rho**(1/3)
        tcc.input_parameters['Simulation']['rcutAA'] = r_max
        tcc.input_parameters['Simulation']['PBCs'] = False

        # do TCC analysis
        tcc_save = TCC_Analysis(f'{d0}/')
        box = np.diag(data_.cell[0:3,0:3])
        points = data_.particles.positions[:]
        save_clusters = False if frame_index in [1, 19, 20, 21, 22] else True
        tcc_save(box, points, frame_index, save_raw=False, save_clusters=False)


if data == 'fioru':
    # for traj in [2, 3, 4, 5]:
    # d0 = '/DINS-HPC/users/dijks109/fioru001/NUCLEATION_WCA/RHO_0.78507/'
    d0 = f'results/fioru/RHO_0.77700/RUN_00{args.traj}/'

    d_in = d0
    logdir = d_out = d_in

    tcc.input_parameters['Simulation']['rcutAA'] = 1.6/1.4*1.8
    tcc.input_parameters['Simulation']['PBCs'] = True

    tcc_save = TCC_Analysis(logdir)

    trajectory = atom.read(d_in + 'nucleation.dump')

    for i, snap in enumerate(trajectory):
        print(f'Frame {i}')
        box = snap.particle_coordinates.max(axis=-2)
        points = snap.particle_coordinates
        tcc_save(box, points, i, save_raw=False)


if data == 'hermes':
    d0 = f'results/hermes/rate_1.022_0{args.traj}/'
    d_in = d0
    logdir = d_out = d_in

    sigma = 2.169960
    tcc.input_parameters['Simulation']['rcutAA'] = 1.8 * sigma
    tcc.input_parameters['Simulation']['PBCs'] = False

    tcc_save = TCC_Analysis(logdir)

    trajectory = atom.read(d_in + 'nucleation.dump')

    for i, snap in enumerate(trajectory):
        print(f'Frame {i}')
        box = snap.particle_coordinates.max(axis=-2)
        points = snap.particle_coordinates
        tcc_save(box, points, i, save_raw=False)


if data == 'yukawa':
    d0 = f'results/{args.data}/'
    i = [8769, 8890, 8956][args.traj]
    logdir = d_out = d_in = d0 + f'{i}/'

    tcc.input_parameters['Simulation']['rcutAA'] = 2.0/1.4*1.8
    tcc.input_parameters['Simulation']['PBCs'] = True

    tcc_save = TCC_Analysis(logdir)

    trajectory = atom.read(d_in + 'nucleation.dump')

    for i, snap in enumerate(trajectory):
        print(f'Frame {i}')
        box = snap.particle_coordinates.max(axis=-2)
        points = snap.particle_coordinates
        tcc_save(box, points, i, save_raw=False)

if data == 'lj':
    d0 = f'results/{args.data}/'
    i = [6922, 7867, 8205, 8355][args.traj]
    logdir = d_out = d_in = d0 + f'{i}/'

    tcc.input_parameters['Simulation']['rcutAA'] = 1.8
    tcc.input_parameters['Simulation']['PBCs'] = True

    tcc_save = TCC_Analysis(logdir)

    trajectory = atom.read(d_in + 'nucleation.dump')

    for i, snap in enumerate(trajectory):
        print(f'Frame {i}')
        box = snap.particle_coordinates.max(axis=-2)
        points = snap.particle_coordinates
        tcc_save(box, points, i, save_raw=False)
