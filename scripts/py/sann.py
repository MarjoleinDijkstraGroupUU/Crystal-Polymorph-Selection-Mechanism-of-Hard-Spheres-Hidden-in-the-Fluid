import numpy as np
import freud


def nlist_sann(box, points, max_num_neighbors=20):
    # use freud to find first max_num_neighbors
    N_particles = points.shape[0]
    nq = freud.locality.AABBQuery(box=box, points=points)
    nlist = nq.query(points, dict(num_neighbors=max_num_neighbors, exclude_ii=True)).toNeighborList()

    # sort neighbor distances per particle
    bond_distances = nlist.distances.reshape((N_particles, max_num_neighbors))
    sorted_bond_distances = np.sort(bond_distances, axis=-1)

    # compute sann cutoffs
    neighbor_index = np.arange(1, max_num_neighbors+1)
    neighbor_index_m2 = np.maximum((neighbor_index-2), 0.01*np.ones_like(neighbor_index))
    sann_cutoffs = np.cumsum(sorted_bond_distances, axis=-1) / neighbor_index_m2[None, :]
    condition = (sann_cutoffs[:, :-1] > sorted_bond_distances[:, 1:])
    num_neighbors = 1 + np.argmin(condition, axis=-1)
    sann_cutoffs = sann_cutoffs[np.arange(N_particles), num_neighbors-1]

    # filter neighbor list
    filt = (bond_distances < sann_cutoffs[:, None])
    filt = filt.reshape((N_particles*max_num_neighbors))
    nlist.filter(filt)

    return nlist
    