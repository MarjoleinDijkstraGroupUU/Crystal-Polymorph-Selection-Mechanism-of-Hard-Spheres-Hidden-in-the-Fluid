import os
import subprocess
import numpy as np
import pandas as pd
import argparse
import pickle
from scipy.stats import binned_statistic
import ovito
import freud
import scripts.lammps.iotools as lmpio

parser = argparse.ArgumentParser(description='TCC conversion analysis')
parser.add_argument('-traj', metavar='traj', type=int, help='trajectory number', required=False)
parser.add_argument('-mode', metavar='mode', type=str, help='which level to operate on')
parser.add_argument('-data', metavar='data', type=str, help='which data to analyze', required=False)

args = parser.parse_args()

# setup tcc
from tcc_python_scripts.file_readers import atom
from tcc_python_scripts.tcc import wrapper

tcc = wrapper.TCCWrapper()
tcc.tcc_executable_directory = '~/Codes/TCC/bin/'
tcc.tcc_executable_path = tcc.tcc_executable_directory + 'tcc'

full_clusterlist = [
    "sp3a", "sp3b", "sp3c", "sp4a", "sp4b", "sp4c", "sp5a", "sp5b", "sp5c",
    "6A", "6Z", "7K", "7T_a", "7T_s", "8A", "8B", "8K", "9A", "9B", "9K", "10A", "10B", "10K", "10W", "11A", "11B",
    "11C", "11E", "11F", "11W", "12A", "12B", "12D",
    "12E", "12K", "13A", "13B", "13K", "FCC", "HCP", "BCC_9"
]

excluded_clusterlist = [
    "10W", "11A", "11B",
    "12K", "13A", "13K",
]

included_clusterlist = [clust for clust in full_clusterlist] # if clust not in excluded_clusterlist]
# included_clusterlist = {'FCC': 1, 'HCP': 1, '8A': 1, 'sp5c': 1,}

##########################
### ANALYSIS FUNCTIONS ###
##########################

class TCC_Post_Analysis():
    """
    Basic post-processing of TCC.
    """

    def __init__(self, clusters_to_analyze, logdir, filter=False, div_cutoff=None):
        self.clusters_to_analyze = clusters_to_analyze
        self.logdir = logdir
        self.filter = filter
        self.div_cutoff = div_cutoff

        for c in clusters_to_analyze:
            if self.filter:
                logfile = f'{self.logdir}/tcc-basic-{c}.log'
            else:
                logfile = f'{self.logdir}/tcc-basic-{c}-unfiltered.log'

            # if not os.path.exists(logfile):
            with open(logfile, "w") as f:
                f.write('frame,number_of_particles,')
                f.write('number_of_clusters,clusters_per_particle,particles_in_cluster\n')

    def _load(self, frame_number, file='clusters.pickle'):
        tcc.working_directory = self.logdir+f'/dump/{frame_number}'
        # print(tcc.working_directory)
        subprocess.run(f'mkdir -p {tcc.working_directory}'.split())
        with open(f'{tcc.working_directory}/{file}', 'rb') as file:
            cluster_table, clusters = pickle.load(file)
        return cluster_table, clusters

    def _filter(self, snap, cluster_table, clusters, convert_to_set=True, div_cutoff=8):
        # select small cubic region
        cutoff = snap.box[0] / div_cutoff
        center = snap.particle_coordinates.mean(axis=-2)
        box = freud.box.Box.from_box(snap.box)
        manhattan_dist = (np.abs(snap.particle_coordinates - center)).max(axis=-1)
        particle_mask = (manhattan_dist < cutoff)

        # filter cluster table
        cluster_table = cluster_table[particle_mask].copy()

        # save clusters in region (check if center of mass is in region)
        if clusters is not None:
            for c in self.clusters_to_analyze:
                cluster_com = [box.center_of_mass(snap.particle_coordinates[a]) for a in clusters[c]] if len(clusters[c]) > 0 else np.nan
                # cluster_com = (snap.particle_coordinates[clusters[c]].mean(axis=-2)) if len(clusters[c]) > 0 else np.nan
                cluster_mask = (np.abs(cluster_com - center)).max(axis=-1) < cutoff
                if convert_to_set:
                    clusters[c] = [frozenset(a) for a in clusters[c][cluster_mask]] if cluster_mask.sum() > 0 else []
                else:
                    clusters[c] = [a for a in clusters[c][cluster_mask]] if cluster_mask.sum() > 0 else []

        return cluster_table, clusters

    def _log_populations(self, frame_number, cluster_table, clusters):
        # calculate number of clusters and populations
        for c in self.clusters_to_analyze:
            number_of_particles = len(cluster_table[c])
            number_of_clusters = len(clusters[c])
            clusters_per_particle = cluster_table[c].mean()
            particles_in_cluster = (cluster_table[c] > 0).sum()

            if self.filter:
                logfile = f'{self.logdir}/tcc-basic-{c}.log'
            else:
                logfile = f'{self.logdir}/tcc-basic-{c}-unfiltered.log'

            with open(logfile, "a") as f:
                f.write(f"{frame_number},{number_of_particles},")
                f.write(f'{number_of_clusters},{clusters_per_particle},{particles_in_cluster}\n')

    def __call__(self, snap, frame_number):
        cluster_table, clusters = self._load(frame_number)
        print(frame_number)
        # print(cluster_table.keys())
        if self.filter:
            cluster_table, clusters = self._filter(snap, cluster_table, clusters, div_cutoff=self.div_cutoff)
        self._log_populations(frame_number, cluster_table, clusters)


class TCC_Visualization(TCC_Post_Analysis):
    """
    Put positions and TCC in one file to visualize.
    """
    def __call__(self, N, box, df, frame_number):
        cluster_table, _ = self._load(frame_number)
        dumpfile = f'{self.logdir}/dump/classified-{frame_number}.dump'
        for c in self.clusters_to_analyze:
            df[f'v_{c}'] = cluster_table[c]
        # df.loc[df.index[-1], 'type'] = 'B' # to be able to change types later
        try:
            lmpio.write_dump(dumpfile, [df], [box], write_all_properties=True)
        except:
            box = [box, box, box]
            lmpio.write_dump(dumpfile, [df], [box], write_all_properties=True)



class TCC_Cluster_Conversion(TCC_Post_Analysis):
    """
    Analyze conversion of source_clusters into cluster B.
    Supports analysis of different clusters A at the same time.
    """

    def __init__(self, source_clusters, dest_clust, logdir):
        super(TCC_Cluster_Conversion, self).__init__(source_clusters+[dest_clust], logdir)

        # save cluster names
        self.source_clusters = source_clusters
        self.dest_clust = dest_clust

        # data storage
        self.max_frames = 2 # max number of frames to store
        self.clusters = [] # list of all clusters with their particles
        self.number_of_particles = []

        # open logfile
        subprocess.run(f'mkdir -p {self.logdir}'.split())
        for c in source_clusters:
            logfile = f'{self.logdir}/cluster-conversion-{c}-{self.dest_clust}-max_frames-{self.max_frames}-loose.log'
            with open(logfile, "w") as f:
                f.write('frame,number_of_particles,')
                f.write('number_of_A_clusters,number_of_B_clusters,')
                f.write('conversionsAB,conversionsAB_,conversionsBA,conversionsAA,conversionsBB\n')

    def _filter(self, snap, cluster_table, clusters):
        # first confine to small region
        cluster_table, clusters = super(TCC_Cluster_Conversion, self)._filter(snap, cluster_table, clusters)

        # remove clusters A if they are already part of a cluster B
        clusters_ = {}
        clustersB = clusters_[self.dest_clust] = clusters[self.dest_clust]
        for c in self.source_clusters:
            clustersA = clusters[c]
            if len(clusters[c]) > 0:
                size_clusterA = len(next(iter(clustersA))) # size of one clusterA
                min_overlap =  min(6, size_clusterA)
                clusterA_mask = np.array([not any(len(a.intersection(b)) >= min_overlap for b in clustersB) for a in clustersA])
                clusters_[c] = np.array(clustersA)[clusterA_mask] if clusterA_mask.sum() > 0 else []

        return cluster_table, clusters_

    def __call__(self, snap, frame_number,):
        # get and save clusters of new snap
        cluster_table, clusters = self._load(frame_number)
        cluster_table, clusters = self._filter(snap, cluster_table, clusters)
        self.clusters += [clusters,]
        self.number_of_particles += [len(cluster_table[self.dest_clust])]
        if len(self.number_of_particles) < self.max_frames:
            return

        # conversions
        for c in self.source_clusters:
            conversionsAA = conversionsA_B = conversionsAB_ = conversionsBA = conversionsBB = 0

            # get the set of clusters of last few frames
            clustersA_ = set()
            clustersB_ = set()
            for t in range(1, self.max_frames):
                clustersA_ = clustersA_.union(self.clusters[-t][c])
                clustersB_ = clustersB_.union(self.clusters[-t][self.dest_clust])

            # get clusters of max_frames ago
            number_of_particles = self.number_of_particles[-self.max_frames]
            clustersA = self.clusters[-self.max_frames][c]
            clustersB = self.clusters[-self.max_frames][self.dest_clust]

            # check for cluster conversions
            if len(clustersA) > 0:
                size_clusterA = len(next(iter(clustersA))) # size of one clusterA
                min_overlap =  min(6, size_clusterA)
                conversionsA_B = sum([any(len(a.intersection(b)) >= min_overlap for b in clustersB_) for a in clustersA])
                conversionsAB_ = sum([any(len(a.intersection(b)) >= min_overlap for a in clustersA) for b in clustersB_])
                conversionsBA = sum([any(len(a.intersection(b)) >= min_overlap for a in clustersA_) for b in clustersB])
                conversionsAA = sum([a in clustersA_ for a in clustersA])
                conversionsBB = sum([b in clustersB_ for b in clustersB])

            # write to logfile
            logfile = f'{self.logdir}/cluster-conversion-{c}-{self.dest_clust}-max_frames-{self.max_frames}-loose.log'
            with open(logfile, "a") as f:
                f.write(f"{frame_number+1-self.max_frames},{number_of_particles},")
                f.write(f'{len(clustersA)},{len(clustersB)},')
                f.write(f"{conversionsA_B},{conversionsAB_},{conversionsBA},{conversionsAA},{conversionsBB}\n")


class TCC_Particle_Conversion(TCC_Post_Analysis):
    """
    Analyze conversion on a particle level.
    """

    def __init__(self, source_clusters, dest_clust, logdir, filter=False):
        super(TCC_Particle_Conversion, self).__init__(source_clusters+[dest_clust,], logdir)

        # save cluster names
        self.source_clusters = source_clusters
        self.dest_clust = dest_clust
        if dest_clust == 'FCC':
            self.dest_clust_comp = 'HCP'
        elif dest_clust == 'HCP':
            self.dest_clust_comp = 'FCC'
        else:
            self.dest_clust_comp = None

        # set parameters
        self.cluster_tables = []
        self.num_bins = 13
        self.max_frames = 2
        self.filter = filter

        # open logfile
        for c in source_clusters:
            if filter:
                logfile = f'{self.logdir}/tcc-particle-conversion-{c}-{self.dest_clust}-filtered.log'
            else:
                logfile = f'{self.logdir}/tcc-particle-conversion-{c}-{self.dest_clust}.log'
            with open(logfile, "w") as f:
                f.write('frame,')
                for i in range(self.num_bins):
                    f.write(f'number_of_particles_{i},')
                for i in range(self.num_bins):
                    f.write(f'converted_{i},')
                if self.dest_clust_comp is not None:
                    for i in range(self.num_bins):
                        f.write(f'{dest_clust}_{i},')
                    for i in range(self.num_bins):
                        f.write(f'{self.dest_clust_comp}_{i},')
                f.write('\n')

    def __call__(self, snap, frame_number):
        # get and save clusters of new snap
        cluster_table, _ = self._load(frame_number, file='clusters.pickle')
        if self.filter:
            cluster_table, _ = self._filter(snap, cluster_table, clusters=None)
        self.cluster_tables.append(cluster_table)

        # conversions
        if len(self.cluster_tables) >= self.max_frames:
            # get clusters per particle of last frame and of max_frames ago
            cluster_table = self.cluster_tables[-self.max_frames]
            cluster_table_ = self.cluster_tables[-1]

            # find particles present in both frames
            mask = cluster_table.index.intersection(cluster_table_.index)
            cluster_table = cluster_table.loc[mask]
            cluster_table_ = cluster_table_.loc[mask]

            # find particles that are not yet part of a dest_clust cluster
            if self.max_frames > 1:
                mask = (cluster_table[self.dest_clust] == 0)
                if self.dest_clust_comp is not None:
                    mask = mask & (cluster_table[self.dest_clust_comp] == 0)
            else:
                mask = (cluster_table[self.dest_clust] != np.nan) # trivial mask

            for c in self.source_clusters:
                out = np.zeros(4*self.num_bins) if self.dest_clust_comp is not None else np.zeros(2*self.num_bins)
                for i in range(self.num_bins):
                    # find particles that are part of i different structureA clusters
                    mask_i = mask & (cluster_table[c] == i)
                    out[i] = mask_i.sum()

                    if self.dest_clust_comp is not None:
                        # particles that become part of either dest_clust or dest_clust_comp clusters
                        mask_either = mask_i & ((cluster_table_[self.dest_clust] > 0) | (cluster_table_[self.dest_clust_comp] > 0))
                        out[i+1*self.num_bins] = mask_either.sum()
                        # particles that become part of more dest_clust clusters
                        mask_dest = mask_i & (cluster_table_[self.dest_clust] > cluster_table_[self.dest_clust_comp])
                        out[i+2*self.num_bins] = mask_dest.sum()
                        # particles that become part of more dest_clust_comp clusters
                        mask_comp = mask_i & (cluster_table_[self.dest_clust] < cluster_table_[self.dest_clust_comp])
                        out[i+3*self.num_bins] = mask_comp.sum()
                    else:
                        # particles that become part of dest_clust clusters
                        mask_dest = mask_i & (cluster_table_[self.dest_clust] > 0)
                        out[i+self.num_bins] = mask_dest.sum()
                    
                # write to logfile
                if self.filter:
                    logfile = f'{self.logdir}/tcc-particle-conversion-{c}-{self.dest_clust}-filtered.log'
                else:
                    logfile = f'{self.logdir}/tcc-particle-conversion-{c}-{self.dest_clust}.log'
                with open(logfile, "a") as f:
                    f.write(f'{frame_number},')
                    for c in out[:-1]:
                        f.write(f"{c},")
                    f.write(f"{out[-1]}\n")


class TCC_Particle_Cluster_Competition(TCC_Post_Analysis):
    """
    Analyze competition between clusters on a particle level
    """

    def __init__(self, source_clusters, dest_clusters, logdir, div_cutoff=8.0):
        super(TCC_Particle_Cluster_Competition, self).__init__(source_clusters+dest_clusters, logdir)

        # save cluster names
        self.source_clusters = source_clusters
        self.dest_clusters = dest_clusters
        self.div_cutoff = div_cutoff

        # open logfile
        logfile = f'{self.logdir}/tcc-particle-cluster-competition-{self.source_clusters}-{self.dest_clusters}_.log'
        self.num_bins = 13
        with open(logfile, "w") as f:
            f.write('frame,')
            f.write(f'number_of_particles,particles_in_A,particles_in_B,particles_in_A_or_B,particles_in_A_and_B,')
            for i in range(self.num_bins):
                f.write(f'number_of_particles_{i},')
            for i in range(self.num_bins):
                f.write(f'clusters_per_particle_{i},')
            f.write('\n')

    def __call__(self, snap, frame_number):
        # get and save clusters of new snap
        cluster_table, _ = self._load(frame_number, file='clusters.pickle')
        # cluster_table, _ = self._filter(snap, cluster_table, None, div_cutoff=self.div_cutoff)
        c0 = self.source_clusters[0]
        number_of_particles = len(cluster_table[c0])

        # get particles that are part of at least one source_cluster
        numA = [cluster_table[c] for c in self.source_clusters][0]
        # numA = (functools.reduce(lambda x, y: x.add(y), numsA))
        # get particles that are part of at least one dest_cluster
        numB = [cluster_table[c] for c in self.dest_clusters][0]
        # numB = (functools.reduce(lambda x, y: x.add(y), numsB))

        # compute frequencys of A, B, A or B, A and B
        particles_in_A = (numA>0).sum()
        particles_in_B = (numB>0).sum()
        particles_in_A_or_B = ((numA>0) | (numB>0)).sum()
        particles_in_A_and_B = ((numA*numB)>0).sum()
            
        # compute average number of B clusters for different numbers of A clusters
        counts = np.zeros(self.num_bins)
        averages = []
        for i in range(self.num_bins):
            mask_i = (numA == i)
            counts[i] = mask_i.sum()
            averages.append((numB[mask_i]).mean())

        # log
        logfile = f'{self.logdir}/tcc-particle-cluster-competition-{self.source_clusters}-{self.dest_clusters}_.log'
        with open(logfile, "a") as f:
            f.write(f'{frame_number},')
            f.write(f'{number_of_particles},{particles_in_A},{particles_in_B},{particles_in_A_or_B},{particles_in_A_and_B}')
            for count in counts:
                f.write(f"{count},")
            for av in averages[:-1]:
                f.write(f"{av},")
            f.write(f"{averages[-1]}\n")



class TCC_Particle_Wolde(TCC_Post_Analysis):
    """
    Analyze cluster correlations with Ten Wolde bonds on a particle level.
    """

    def __init__(self, clusters_to_analyze, logdir, mode='particles-in-cluster'):
        self.clusters_to_analyze = clusters_to_analyze
        self.logdir = logdir

        self.cluster_tables = []
        self.num_connections = []
        self.num_bins = 13
        self.mode = mode

        super(TCC_Particle_Wolde, self).__init__(self.clusters_to_analyze, self.logdir)

        for c in self.clusters_to_analyze:
            logfile = self.logdir + f'/dump/tcc-particle-correlation-wolde-{c}-{self.mode}.log'
            with open(logfile, "w") as f:
                f.write('frame,')
                for i in range(self.num_bins):
                    f.write(f'number_of_particles_{i},')
                for i in range(self.num_bins):
                    f.write(f'clusters_per_particle_{i},')
                f.write('\n')

    def __call__(self, frame_number):
        # get and save clusters of new snap
        cluster_table, _ = self._load(frame_number)

        # get number of solidlike bonds
        working_directory = self.logdir+f'/dump-bops/{frame_number}'
        with open(working_directory+'/solid-bonds.pickle', 'rb') as file:
            solid_bonds = pickle.load(file)
        cluster_table['num_connections'] = solid_bonds

        # compute averages for different numbers of solidlike bonds
        counts = np.zeros(self.num_bins)
        averages = []
        for i in range(self.num_bins):
            mask_i = (cluster_table['num_connections'] == i)
            counts[i] = mask_i.sum()
            if self.mode == 'particles-in-cluster':
                averages.append((cluster_table[mask_i]>0).mean())
            elif self.mode == 'clusters-per-particle':
                averages.append((cluster_table[mask_i]).mean())

        # write to logfile
        for c in self.clusters_to_analyze:
            logfile = self.logdir + f'/dump/tcc-particle-correlation-wolde-{c}-{self.mode}.log'
            with open(logfile, "a") as f:
                f.write(f'{frame_number},')
                for count in counts:
                    f.write(f"{count},")
                for av in averages[:-1]:
                    f.write(f"{av[c]},")
                f.write(f"{averages[-1][c]}\n")


class TCC_Subclusters(TCC_Post_Analysis):
    """
    Find clusters with specific subclusters.
    """

    def __init__(self, sub_clust, clust, logdir):
        clusters_to_analyze = [sub_clust, clust, 'FCC', 'HCP']
        super(TCC_Subclusters, self).__init__([sub_clust, clust], logdir)

        # save cluster names
        self.sub_clust = sub_clust
        self.clust = clust

    def _filter(self, snap, cluster_table, clusters):
        # first confine to small region
        if True:
            cluster_table, clusters = super(TCC_Subclusters, self)._filter(snap, cluster_table, clusters)
        else:
            for c in self.clusters_to_analyze:
                clusters[c] = [frozenset(a) for a in clusters[c]]

        clusters_ = {}
        clustersA = clusters_[self.sub_clust] = clusters[self.sub_clust]
        clustersB = clusters[self.clust]

        # only keep clusters that do not have sub_clust as a subcluster
        size_clusterA = len(next(iter(clustersA))) # size of one clusterA
        min_overlap =  min(7, size_clusterA)
        
        clusterB_mask = np.array([any(len(a.intersection(b)) >= min_overlap for a in clustersA) for b in clustersB])
        # clusterB_mask = np.array([sum(len(a.intersection(b)) >= min_overlap for a in clustersA) >= 2 for b in clustersB])
        clusters_[self.clust] = np.array(clustersB)[clusterB_mask] if clusterB_mask.sum() > 0 else []

        # recompute cluster table
        indices = cluster_table.index
        cluster_table.loc[:, self.clust] = np.array([sum(p in c for c in clusters_[self.clust]) for p in indices])
        print(cluster_table[self.clust].mean())

        return cluster_table, clusters_

    def _save(self, frame_number, cluster_table, clusters):
        tcc.working_directory = self.logdir+f'/dump/{frame_number}'
        subprocess.run(f'mkdir -p {tcc.working_directory}'.split())

        save = (cluster_table, clusters)
        with open(tcc.working_directory+'/clusters.pickle', 'wb') as file:
            pickle.dump(save, file)

    def __call__(self, snap, frame_number):
        cluster_table, clusters = self._load(frame_number)
        cluster_table, clusters = self._filter(snap, cluster_table, clusters)
        self._save(frame_number, cluster_table, clusters)



class TCC_Surface_Clusters(TCC_Post_Analysis):
    """
    Find out how much of a cluster is part of FCC or HCP.
    """

    def __init__(self, clust, clust_size, logdir):
        clusters_to_analyze = [clust, 'FCC', 'HCP']
        super(TCC_Surface_Clusters, self).__init__(clusters_to_analyze, logdir)
        self.clust = clust
        self.clust_size = clust_size

        logfile = self.logdir + f'/dump/tcc-surface-clusters-{clust}.log'
        with open(logfile, "w") as f:
            f.write('frame,number_of_particles,')
            for i in range(self.clust_size+1):
                f.write(f'overlap_{i},')
            f.write('\n')

    def __call__(self, snap, frame_number):
        cluster_table, clusters = self._load(frame_number)
        cluster_table_, clusters = self._filter(snap, cluster_table, clusters)
        number_of_particles = len(cluster_table_[self.clust])

        cluster_table = cluster_table['FCC'] + cluster_table['HCP']
        clusters = clusters[self.clust]
        overlap = np.array([(cluster_table[np.array([i for i in c])] > 0).sum() for c in clusters])
        # print(overlap)

        logfile = self.logdir + f'/dump/tcc-surface-clusters-{self.clust}.log'
        with open(logfile, "a") as f:
            f.write(f'{frame_number},{number_of_particles},')
            for i in range(self.clust_size+1):
                f.write(f'{(overlap == i).sum()},')
            f.write('\n')


class TCC_8A_Overlaps(TCC_Post_Analysis):
    """
    Find out which part of an 8A cluster is part of FCC or HCP.
    Distinguish between ring, spindle, shifted.
    """

    def __init__(self, logdir):
        clusters_to_analyze = ['sp5b', '8A', 'FCC', 'HCP']
        super(TCC_8A_Overlaps, self).__init__(clusters_to_analyze, logdir)

        logfile = self.logdir + f'/dump/tcc-8a-overlap.log'
        with open(logfile, "w") as f:
            f.write('frame,number_of_particles,number_rings,number_spindles,number_shifted,solid,solid_rings,solid_spindles,solid_shifted\n')

    def __call__(self, snap, frame_number):
        cluster_table, clusters = self._load(frame_number)
        cluster_table_, clusters = self._filter(snap, cluster_table, clusters, convert_to_set=False)
        number_of_particles = len(cluster_table_['8A'])

        # find 8A clusters that are partly solidlike
        try:
            clustersA = np.array(clusters['8A'])
            solidlike = (cluster_table['FCC'] + cluster_table['HCP']) > 0
            solidlike_ = (cluster_table_['FCC'] + cluster_table_['HCP']) > 0
            num_solid = solidlike_.sum()
            overlap = (solidlike.values[clustersA] > 0).sum(axis=-1)
            clustersA = clustersA[(overlap>0)*(overlap<7)]
        except:
            logfile = self.logdir + f'/dump/tcc-8a-overlap.log'
            with open(logfile, "a") as f:
                f.write(f'{frame_number},{number_of_particles},0,0,0,0,0,0,0\n')

        # find particles that are spindles, rings, or shifted
        spindles_ = set()
        rings_ = set()
        shifted_ = set()
        for i, cA in enumerate(clustersA):

            clustersB = np.array(clusters['sp5b'])
            base_clusters = []
            # find the two underlying sp5b clusters
            for cB in np.array(clustersB):
                if set(cB).issubset(set(cA)):
                    base_clusters.append(cB)
                    if len(base_clusters) == 2:
                        break

            # find the subparts of the 8A cluster
            if len(base_clusters) == 2:
                ring_ID = common_ID = set(base_clusters[0]).intersection(base_clusters[1])
                spindles = [base_clusters[0][-1], base_clusters[1][-1]]
                uncommon_ID = set(base_clusters[0]).symmetric_difference(base_clusters[1])
                fifth = list(set(uncommon_ID).difference(set(spindles)))

            # check correctness
            if (len(base_clusters) != 2 or len(common_ID) != 4 or len(fifth) != 2
                    or len(set(common_ID).intersection(spindles)) != 0
                    or len(set(fifth).intersection(set(spindles))) != 0):
                continue

            # add to sets
            spindles_.update(spindles)
            rings_.update(ring_ID)
            shifted_.update(fifth)
        
        solid_rings = solidlike[list(rings_)].sum()
        solid_spindles = solidlike[list(spindles_)].sum()
        solid_shifted = solidlike[list(shifted_)].sum()

        logfile = self.logdir + f'/dump/tcc-8a-overlap.log'
        with open(logfile, "a") as f:
            f.write(f'{frame_number},{number_of_particles},{len(rings_)},{len(spindles_)},{len(shifted_)},{num_solid},{solid_rings},{solid_spindles},{solid_shifted}\n')


class TCC_Internal_Angles(TCC_Post_Analysis):
    """
    Compute internal angles of the ring of a cluster.
    """

    def __init__(self, clust, num_bins, logdir):
        clusters_to_analyze = [clust, 'FCC', 'HCP']
        if clust == '8A':
            clusters_to_analyze.append('sp5b')
        super(TCC_Internal_Angles, self).__init__(clusters_to_analyze, logdir)
        
        self.interval = 11 if clust == 'sp5c' else 2
        self.clust = clust
        self.num_bins = num_bins
        self.clusters = []
        self.cluster_tables = []
        self.snaps = []

        for label in ['pre', 'post']:
            logfile = self.logdir + f'/dump/tcc-{clust}-ring-angles-{label}_.log'
            with open(logfile, "w") as f:
                f.write('frame,')
                for i in range(self.num_bins):
                    f.write(f'{i},')
                f.write('\n')

    def __call__(self, snap_, frame_number):
        cluster_table_, clusters_ = self._load(frame_number)
        # _, clusters_ = self._filter(snap_, cluster_table_, clusters_, convert_to_set=False, div_cutoff=8)

        # save clusters for next step
        self.clusters.append(clusters_)
        self.cluster_tables.append(cluster_table_)
        self.snaps.append(snap_)

        if len(self.clusters) < self.interval:
            return
        else:
            # load previous clusters
            clusters = self.clusters[-self.interval]
            cluster_table = self.cluster_tables[-self.interval]
            snap = self.snaps[-self.interval]
            box = freud.box.Box.from_box(snap.box)

            if len(self.clusters) > self.interval:    
                # remove unneeded data
                self.clusters.pop(0)
                self.cluster_tables.pop(0)
                self.snaps.pop(0)

        # find clusters that 'convert' to FCC/HCP
        clustersA = clusters[self.clust]
        cluster_table_sum = (cluster_table['FCC'] + cluster_table['HCP']).values
        cluster_table_sum_ = (cluster_table_['FCC'] + cluster_table_['HCP']).values
        overlap = (cluster_table_sum[clustersA] > 0).sum(axis=-1)
        overlap_ = (cluster_table_sum_[clustersA] > 0).sum(axis=-1)
        threshold = 7 if self.clust == '8A' else 6
        cluster_mask = (overlap < threshold) * (overlap_ >= threshold)
        clustersA = np.array(clustersA)[cluster_mask]

        print(tcc.working_directory, len(clustersA))
        # return

        # compute internal angles of 8A
        angles = {}
        angles['pre'] = []
        angles['post'] = []
        if self.clust == '8A':
            clustersB = np.array(clusters['sp5b'])

        for cA in clustersA:
            if self.clust == '8A':
                # find the two underlying sp5b clusters
                base_clusters = []
                for cB in np.array(clustersB):
                    if set(cB).issubset(set(cA)):
                        base_clusters.append(cB)
                        if len(base_clusters) == 2:
                            break

                # find the subparts of the 8A cluster
                if len(base_clusters) == 2:
                    ring_ID = common_ID = set(base_clusters[0]).intersection(base_clusters[1])
                    spindles = [base_clusters[0][-1], base_clusters[1][-1]]
                    uncommon_ID = set(base_clusters[0]).symmetric_difference(base_clusters[1])
                    not_ring_ID = set(cA).difference(common_ID)
                    fifth = list(set(uncommon_ID).difference(set(spindles)))

                # check correctness
                if (len(base_clusters) != 2 or len(common_ID) != 4 or len(fifth) != 2
                        or len(set(common_ID).intersection(spindles)) != 0
                        or len(set(fifth).intersection(set(spindles))) != 0):
                    for i in range(4):
                        angles['pre'].append(np.nan)
                        angles['post'].append(np.nan)
                    continue
            elif self.clust == 'sp5c':
                ring_ID = cA[:5]

            # compute angles before and after conversion
            for label, snapshot in [('pre', snap), ('post', snap_)]:
                # choose either ring or not-ring to compute internal angles of
                points = snapshot.particle_coordinates[list(ring_ID)]
                
                # for each of the ring particles find the maximum internal angle
                bonds = points[:, None, :] - points[None, :, :]
                L = len(points)
                bonds = box.wrap(bonds.reshape(-1, 3)).reshape(L,L,3)

                def calculate_angle(bonds, i, j1, j2):
                    bond_1 = bonds[i][j1]
                    bond_2 = bonds[i][j2]
                    cross = np.linalg.norm(np.cross(bond_1, bond_2))
                    angle_ = np.arctan2(cross, np.dot(bond_1, bond_2))
                    angle_ = np.remainder(angle_, np.pi)
                    return angle_

                for i in range(L):
                    if self.clust == '8A':
                        # find maximum internal angle
                        angle = 0
                        for j1 in range(L):
                            for j2 in range(L):
                                if i == j1 or i == j2 or j1 == j2:
                                    continue
                                angle_ = calculate_angle(bonds, i, j1, j2)
                                if angle_ > angle:
                                    angle = angle_
                    elif self.clust == 'sp5c':
                        # use known ordering
                        j1 = (i+1) % L
                        j2 = (i-1) % L
                        angle = calculate_angle(bonds, i, j1, j2)
                
                    angles[label].append(angle)

        # write histogram per number of overlapping particles
        for label in ['pre', 'post']:
            logfile = self.logdir + f'/dump/tcc-{self.clust}-ring-angles-{label}_.log'
            hist, _ = np.histogram(np.array(angles[label]), bins=self.num_bins, range=(0, np.pi))
            with open(logfile, "a") as f:
                f.write(f'{frame_number},')
                for i in range(self.num_bins):
                    f.write(f'{hist[i]},')
                f.write('\n')


class TCC_Coexistence(TCC_Post_Analysis):

    def __init__(self, clusters_to_analyze, num_bins, logdir, mode='particles-in-cluster'):
        super(TCC_Coexistence, self).__init__(clusters_to_analyze, logdir)
        
        self.num_bins = num_bins
        self.mode = mode
        
        for clust in clusters_to_analyze:
            logfile = self.logdir + f'/dump/tcc-{clust}-spatial-distribution-{mode}.log'
            with open(logfile, "w") as f:
                f.write('frame,')
                for i in range(self.num_bins):
                    f.write(f'{i},')
                f.write('\n')

    def __call__(self, points, box, frame_number):
        # print(frame_number)
        cluster_table, _ = self._load(frame_number)

        # center solid
        d = [0, 0,][args.traj]
        # x = (points[:, d])
        box = freud.box.Box.from_box(box)
        solidlike = ((cluster_table['FCC'] + cluster_table['HCP']) > 2).values
        # a = (solidlike * x).sum(axis=-1) / solidlike.sum(axis=-1)
        a = box.center_of_mass(points[solidlike])
        x = np.abs(box.wrap(points-a))[:, d]

        print((cluster_table['BCC_9'][~solidlike]>0).mean())

        # write histogram as a function of distance
        for c in self.clusters_to_analyze:
            logfile = self.logdir + f'/dump/tcc-{c}-spatial-distribution-{self.mode}.log'

            if self.mode == 'particles-in-cluster':
                c_mean, bins, _ = binned_statistic(x, cluster_table[c]>0, bins=self.num_bins, statistic='mean')
            elif self.mode == 'clusters-per-particle':
                c_mean, bins, _ = binned_statistic(x, cluster_table[c], bins=self.num_bins, statistic='mean')
            # x_ = (bins[1:] + bins[:-1]) / 2

            with open(logfile, "a") as f:
                f.write(f'{frame_number},')
                for i in range(self.num_bins):
                    f.write(f'{c_mean[i]},')
                f.write('\n')


##################
### INPUT DATA ###
##################

# find relevant data
if args.data == 'coli':
    d0 = f'results/coli/'
    P = 13.4
    d_in = d0 + f'betap{P:.2f}/trial{args.traj}/'
    logdir = d_out = f'./results/conversion/betap{P:.2f}/trial{args.traj}/'
    div_cutoff = 8.00
elif args.data == 'fioru':
    d0 = f'results/{args.data}/'
    logdir = d_out = d_in = d0 + f'RHO_0.77700/RUN_00{args.traj}/'
    div_cutoff = 5.32
elif args.data == 'hermes':
    d0 = f'results/{args.data}/'
    logdir = d_out = d_in = d0 + f'rate_1.022_0{args.traj}/'
    div_cutoff = 33.65
elif args.data == 'yukawa':
    d0 = f'results/{args.data}/'
    i = [8769, 8890, 8956][args.traj]
    logdir = d_out = d_in = d0 + f'{i}/'
    div_cutoff = 7.25
elif args.data == 'lj':
    d0 = f'results/{args.data}/'
    i = [6922, 7867, 8205, 8355][args.traj]
    logdir = d_out = d_in = d0 + f'{i}/'
    div_cutoff = 7.25


if args.mode == 'basic':
    # measure the amount of clusters, and how many particles are part of them
    trajectory = atom.read(d_in + 'nucleation.dump')
    tcc = TCC_Post_Analysis(included_clusterlist, d_out, filter=True, div_cutoff=div_cutoff)

    for frame_number, snap in enumerate(trajectory):
        tcc(snap, frame_number)


# output files to visualize
if args.mode == 'visualization':
    if args.data in ['coli', 'fioru', 'hermes', 'yukawa', 'lj']:
        fun = TCC_Visualization(['sp5c', '8A', 'FCC', 'HCP'], d_out)

        dumpfile = d_in + 'nucleation.dump'
        # trajectory = atom.read()
        Ns, L0, L1 = lmpio.box_from_dump(dumpfile)
        Ls = (L1-L0)
        trajectory = lmpio.read_dump(dumpfile)

        for frame_number, snap in enumerate(trajectory):
            print(frame_number)
            N = Ns[frame_number]
            L = Ls[frame_number]

            fun(N, L, snap, frame_number)

    if args.data == 'experiments':
        logdir = f'results/experiments'
        fun = TCC_Visualization(['sp5c', '8A', 'FCC', 'HCP'], logdir)

        for frame_number in range(2, 21):
            print(f'{frame_number} / 22')
            filename = f'{logdir}/frame_{frame_number}.xyz'
            N, box, df = lmpio.read_xyz(filename)
            fun(N, box, df, frame_number)


if args.mode == 'cluster-conversion':
    # measure conversion of clusters by explicitly comparing the particles in them
    P = 13.4
    # (['sp5c', '8A'], 'FCC'), (['sp5c', '8A'], 'HCP')
    for source_clust, dest_clust in [(['sp5c'], '8A')]: #[(['8A'], 'FCC'), (['8A'], 'HCP')]:
        for i in [6, 2, 3, 6, 9, 11]: #
            d0 = f'results/coli/'
            d_in = d0 + f'betap{P:.2f}/trial{i}/'
            logdir = f'./results/conversion/betap{P:.2f}/trial{i}/'
            trajectory = atom.read(d_in + 'nucleation.dump')

            tcc_conversion = TCC_Cluster_Conversion(source_clust, dest_clust, logdir)

            for frame_number, snap in enumerate(trajectory):
                print(frame_number)
                tcc_conversion(snap, frame_number)


if args.mode == 'particle-conversion':
    # measure conversion of particles by measuring 
    # if they become part of a specific destination cluster
    d0 = f'results/coli/'
    P = 13.4
    for i in [2, 3, 6, 9, 11]:
        d_in = d0 + f'betap{P:.2f}/trial{i}/'
        logdir = f'./results/conversion/betap{P:.2f}/trial{i}/'
        trajectory = atom.read(d_in + 'nucleation.dump')
        # logdir = f'./results/conversion/betap{P:.2f}-fluid/trial{i}/'
        tcc_conversion = TCC_Particle_Conversion(included_clusterlist, 'FCC', logdir, filter=False)
        # tcc_conversion = TCC_Particle_Conversion(['6A', 'sp5c', '8A'], 'FCC', logdir, filter=False)
        # tcc_conversion = TCC_Particle_Conversion(['sp5c'], '8A', logdir)
        for frame_number, snap in enumerate(trajectory):
                tcc_conversion(snap, frame_number)


if args.mode == 'particle-cluster-competition':
    # measure if particles are part of two clusters:
    # both at the same time, or only one of them, or none
    trajectory = atom.read(d_in + 'nucleation.dump')
    tcc = TCC_Particle_Cluster_Competition(['sp5c'], ['8A'], logdir, div_cutoff=div_cutoff)
    for frame_number, snap in enumerate(trajectory):
        print(frame_number)
        # if frame_number > 125:
        #     break
        tcc(snap, frame_number)


if args.mode == 'particle-wolde':
    # measure how the average number of clusters per particles varies
    # as a function of the number of solidlike bonds
    if args.data == 'coli':
        d0 = f'results/coli/'
        P = 13.4

        for i in [2, 3, 6, 9, 11]:
            d_in = d0 + f'betap{P:.2f}/trial{i}/'
            logdir = f'./results/conversion/betap{P:.2f}/trial{i}/'
            trajectory = atom.read(d_in + 'nucleation.dump')
            tcc = TCC_Particle_Wolde(included_clusterlist, logdir, mode='clusters-per-particle')

            for frame_number, _ in enumerate(trajectory):
                print(frame_number)
                tcc(frame_number)

    if args.data == 'experiments':
        logdir = f'results/experiments'
        tcc = TCC_Particle_Wolde(included_clusterlist, logdir, mode='clusters-per-particle')

        for frame_number in range(2, 23):
            print(f'{frame_number} / 22')
            filename = f'{logdir}/frame_{frame_number}.xyz'
            pipeline = ovito.io.import_file(filename)
            data = pipeline.compute()
            tcc(frame_number)


if args.mode == 'subclusters':
    P = 13.4

    for i in [2, 3, 6, 9, 11]:
        d0 = f'results/coli/'
        d_in = d0 + f'betap{P:.2f}/trial{i}/'
        logdir = f'./results/conversion/betap{P:.2f}/trial{i}/'
        trajectory = atom.read(d_in + 'nucleation.dump')

        tcc_conversion = TCC_Subclusters('sp5c', '8A', logdir)

        for frame_number, snap in enumerate(trajectory):
            if frame_number > 125:
                break
            tcc_conversion(snap, frame_number)


if args.mode == 'surface-clusters':
    P = 13.4

    for i in [2, 3, 6, 9, 11]:
        d0 = f'results/coli/'
        d_in = d0 + f'betap{P:.2f}/trial{i}/'
        logdir = f'./results/conversion/betap{P:.2f}/trial{i}/'
        trajectory = atom.read(d_in + 'nucleation.dump')

        tcc_analysis = TCC_Surface_Clusters('8A', 8, logdir)

        for frame_number, snap in enumerate(trajectory):
            tcc_analysis(snap, frame_number)


if args.mode == '8a-overlaps':
    P = 13.4

    for i in [2,3,6,9,11]:
        d0 = f'results/coli/'
        d_in = d0 + f'betap{P:.2f}/trial{i}/'
        logdir = f'./results/conversion/betap{P:.2f}/trial{i}/'
        trajectory = atom.read(d_in + 'nucleation.dump')

        tcc_analysis = TCC_8A_Overlaps(logdir)

        for frame_number, snap in enumerate(trajectory):
            print(frame_number)
            tcc_analysis(snap, frame_number)


if args.mode == 'internal-angles':
    P = 13.4
    for i in [2, 3, 6, 9, 11]:
        d0 = f'results/coli/'
        d_in = d0 + f'betap{P:.2f}/trial{i}/'
        logdir = f'./results/conversion/betap{P:.2f}/trial{i}/'
        trajectory = atom.read(d_in + 'nucleation.dump')

        tcc_analysis = TCC_Internal_Angles('sp5c', 200, logdir, )

        for frame_number, snap in enumerate(trajectory):
            if frame_number < 125:
                continue
            # if frame_number > 125:
            #     break
            tcc_analysis(snap, frame_number)


if args.mode == 'coexistence':
    # plane = ['fcc-100', 'fcc-110', 'hcp-0001', 'hcp-1010', 'hcp-1120'][args.traj]
    plane = ['fcc-110', 'hcp-1120'][args.traj]
    logdir = d0 = f'results/coexistence_md/{plane}/'
    filename = f'{logdir}/coexistence.gsd' #fluid_dense.gsd' #
    pipeline = ovito.io.import_file(filename)

    num_bins = 100
    # tcc_analysis = TCC_Coexistence(full_clusterlist, num_bins, logdir, mode='particles-in-cluster')
    tcc_analysis = TCC_Coexistence(full_clusterlist, num_bins, logdir, mode='clusters-per-particle')

    start = pipeline.source.num_frames // 2
    stop = pipeline.source.num_frames
    step = 20
    for frame_index in range(start, stop, step):
        data_ = pipeline.compute(frame_index)
        points = data_.particles.positions[:]
        box = data_.cell[:3, :3]
        tcc_analysis(points, box, frame_index)
