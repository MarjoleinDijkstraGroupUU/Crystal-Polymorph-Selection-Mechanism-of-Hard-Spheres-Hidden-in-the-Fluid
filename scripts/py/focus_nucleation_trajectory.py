from multiprocessing import connection
import os
import sys
import pandas
import argparse
import subprocess

import ovito
from ovito.pipeline import Pipeline, PythonScriptSource
from ovito.io import import_file
from ovito.data import Particles, SimulationCell
from ovito.modifiers import ClusterAnalysisModifier, CommonNeighborAnalysisModifier
from ovito.modifiers import ExpressionSelectionModifier, ComputePropertyModifier
from ovito.modifiers import InvertSelectionModifier, DeleteSelectedModifier, ClearSelectionModifier, SliceModifier

import freud
from sann import nlist_sann

sys.path.append('../')
sys.path.append('../../')

parser = argparse.ArgumentParser(description='Center nucleation trajectory with CNA/TenWolde')
parser.add_argument('-data', metavar='data', type=str, help='which data to analyze')
parser.add_argument('-traj', metavar='i', type=int, help='trajectory', required=False)
parser.add_argument('--fine', action=argparse.BooleanOptionalAction, default=False, required=False)


class GetReadXYZ(object):
    def __init__(self, d0):
        self.d0 = d0

    def __call__(self, frame, data):
        """
        Read a single XYZ snapshot from a file.
        """

        file = f'{self.d0}/snapshot_{frame:04}.dat'

        with open(file) as input_file:
            # read number of particles
            line = input_file.readline()
            number_of_particles = int(line)

            # read box dimensions
            if not data.cell:
                data.cell = SimulationCell(pbc=(True, True, True))
            for i in range(3):
                line = (input_file.readline())
                L = float(line.split()[1]) - float(line.split()[0])
                av = (float(line.split()[1]) + float(line.split()[0])) / 2
                data.cell_[i, i] = L
                data.cell_[i, 3] = av

        table = pandas.read_csv(file, sep='\s+', names=('x', 'y', 'z', 'sigma'), nrows=number_of_particles, skiprows=4)

        data.particles = Particles(count=number_of_particles)
        coordinates = data.particles_.create_property('Position')
        coordinates[:,0] = table['x']
        coordinates[:,1] = table['y']
        coordinates[:,2] = table['z']

        ptype = data.particles_.create_property('Particle Type')
        ptype[:] = 1

        radius = data.particles_.create_property('Radius')
        radius[:] = table['sigma'] / 2


def read(**kwargs):
    # read dump
    data = kwargs['data']
    i = kwargs['i']
    step = 10

    if data == 'coli':
        rho, P = 1.0, 13.4
        d0 = '/nethome/coli0001/HardSphereNucleation/five_fold/hooMD/npt/'
        d_in = d0 + f'betap{P:.2f}/trial{i}/run/BOP/frames/'
        d_in = f'results/coli/betap{P:.2f}/trial{i}/'
        d_out = f'./results/{data}/betap{P:.2f}/trial{i}/'
        print(d_in)
        if kwargs['filetype'] == 'xyz':
            pipeline = import_file(d_in+'frames*.xyz')
        if kwargs['filetype'] == 'gsd':
            pipeline = import_file(d_in+'trajectory.gsd')
        elif kwargs['filetype'] == 'dump':
            pipeline = import_file(d_in+'trajectory.dump')
    elif data == 'fioru':
        rho = 0.777
        step = 1
        d0 = 'results/fioru/RHO_0.77700/'
        i_ = os.listdir(d0)[i]
        d_in = d0 + f'{i_}/'
        d_out = f'./results/{data}/RHO_0.77700/{i_}/'
        pipeline = import_file(d_in+'dump_traj.srd.mixture')
    elif data == 'hermes':
        rho = 0.1
        step = 1# 50
        d_in = f"results/hermes/rate_1.022_0{i}/"
        d_out = d_in
        pipeline = import_file(d_in+'preproc-nucleation.dump')
        # pipeline = Pipeline(source=PythonScriptSource(function=GetReadXYZ(d_in)))
    elif data == 'yukawa':
        rho = 0.25
        d0 = f'./results/yukawa/'
        # i_ = os.listdir(d0)[i]
        i_ = [8769, 8890, 8956][args.traj]
        # i_ = i
        d_in = d0 + f'/{i_}/'
        d_out = f'{d0}/{i_}/'
        pipeline = import_file(d_in+'trajectory.dump')
    elif data == 'lj':
        rho = 1.0
        step = 1
        d0 = f'./results/lj/'
        i_ = [6922, 7867, 8205, 8355][args.traj]
        # i_ = os.listdir(d0)[i]
        print(i_)
        d_in = d0 + f'/{i_}/'
        d_out = f'{d0}/{i_}/'
        pipeline = import_file(d_in+'dump.all')

    subprocess.run(f'mkdir -p {d_out}'.split())

    return pipeline, d_out, rho, step


def get_largest_cluster(pipeline, cutoff=1.8, delete=True):
    # Find largest cluster of solidlike particles
    pipeline.modifiers.append(
        ClusterAnalysisModifier(
            cutoff=cutoff, sort_by_size=True,
            only_selected=True, unwrap_particles=False,
            compute_com=True
        )
    )

    # Keep largest cluster only
    if delete:
        pipeline.modifiers.append(ClearSelectionModifier())
        pipeline.modifiers.append(ExpressionSelectionModifier(expression='Cluster == 1'))
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(ClearSelectionModifier())

    return pipeline


def get_largest_cluster_wolde(data):
    # Find largest cluster of solidlike particles
    points = data.particles.positions
    box = data.cell[0:3,0:3]
    box = freud.box.Box.from_box(box)

    nlist = nlist_sann(box, points)
    
    # compute nucleus size with Ten Wolde criterion
    solid_liquid = freud.order.SolidLiquid(normalize_q=True, l=6, q_threshold=0.7, solid_threshold=7)
    solid_liquid.compute((box, points), neighbors=nlist)
    nucleus_size = solid_liquid.largest_cluster_size
    nucleus_points = points[solid_liquid.cluster_idx == 0]
    com = box.center_of_mass(nucleus_points)

    return nucleus_size, com


def get_smaller_box_around_nucleation(**kwargs):
    data = kwargs['data']
    wolde = True
    filetype = 'gsd' if kwargs['i'] > 3 else 'xyz'

    pipeline, d_out, rho, step = read(**kwargs, filetype=filetype)
    if not wolde:
        # find largest cluster with CNA
        pipeline.modifiers.append(ComputePropertyModifier(output_property='Mass', expressions=['0.0']))
        pipeline.modifiers.append(CommonNeighborAnalysisModifier())
        pipeline.modifiers.append(ExpressionSelectionModifier(expression='StructureType > 0'))
        pipeline = get_largest_cluster(pipeline, cutoff=1.5/rho, delete=False)

    # compute center of mass of nucleus
    # if data == 'coli':
    # end_frame = 2000 if data == 'yukawa-spontaneous' else 10000
    end_frame = pipeline.source.num_frames
    stop_N = 500 if data == 'yukawa-spontaneous' else 200
    stop_N = 50 if wolde else stop_N
    stop_N = 100 if data == 'lj' else stop_N

    def find_nucleation(start, end, step):
        for frame in range(start, end+1, step):
            data = pipeline.compute(frame)
            if wolde:
                Nc, com = get_largest_cluster_wolde(data)
            else:
                Nc = data.attributes['ClusterAnalysis.largest_size']
                cluster_table = data.tables['clusters']
                com = cluster_table['Center of Mass'][0]

            print(frame, Nc, com)

            if Nc > stop_N:
                break

        return data, frame, com
            
    for step_ in [50*step, 10*step, step]:
        start_frame = max(end_frame - 20*step_, 0)
        # start_frame = 0
        print(start_frame, end_frame, step_)
        print(' ')
        data, end_frame, com = find_nucleation(start_frame, end_frame, step_)

    # output
    pipeline, _, _, _ = read(**kwargs, filetype=filetype)

    # shift nucleus to center of box
    center_cell = [0, 0, 0]
    for d in range(3):
        center_cell[d] = data.cell[d, 3] + data.cell[d, d] / 2
    mod = ovito.modifiers.AffineTransformationModifier(
        operate_on = {'particles'},
        transformation = [[1, 0, 0, -com[0]+center_cell[0]],
                          [0, 1, 0, -com[1]+center_cell[1]],
                          [0, 0, 1, -com[2]+center_cell[2]]])
    pipeline.modifiers.append(mod)
    pipeline.modifiers.append(ovito.modifiers.WrapPeriodicImagesModifier())

    # select small box around nucleus
    if False:
        for d in range(3):
            for orient in [-1, 1]:
                N = 4000
                L = 0.5 * (N / rho)**(1/3)
                distance = center_cell[d] + orient * L
                normal = [0, 0, 0]
                normal[d] = 1
                inverse = False if orient == 1 else True
                modifier = SliceModifier(distance=distance, normal=normal, inverse=inverse)
                pipeline.modifiers.append(modifier)

    # export to LAMMPS dump
    if args.data == 'coli':
        start_frame = end_frame - 1800 if filetype == 'gsd' else end_frame - 180
        end_frame = start_frame + 2500 if (filetype == 'gsd' and not args.fine) else start_frame + 250
        every_nth_frame = 10 if (filetype == 'gsd' and not args.fine) else 1
    elif args.data == 'fioru':
        start_frame = max(0, end_frame - 100)
        end_frame = end_frame + 50
        every_nth_frame = 10 if not args.fine else 1
    elif args.data == 'hermes':
        start_frame = max(0, end_frame - 100)
        end_frame = end_frame + 100
        every_nth_frame = 1
    elif args.data == 'yukawa':
        start_frame = max(0, end_frame - 800)
        end_frame = end_frame + 800
        every_nth_frame = 8
    elif args.data == 'lj':
        start_frame = max(0, end_frame - 480)
        end_frame = end_frame + 320
        every_nth_frame = 4

    export_filename = 'nucleation.dump' 
    # print(start_frame, )

    if start_frame > -1:
        ovito.io.export_file(pipeline, d_out + export_filename, "lammps/dump",
            multiple_frames=True, columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"],
            start_frame=start_frame, end_frame=end_frame, every_nth_frame=every_nth_frame)


if __name__ == "__main__":
    args = parser.parse_args()
    kwargs = vars(args)
    # for i in range(5): #[2, 3, 6, 9, 11]:
        # print(10*'\n')
        # print(i)
    kwargs['i'] = kwargs['traj']
    get_smaller_box_around_nucleation(**kwargs)
