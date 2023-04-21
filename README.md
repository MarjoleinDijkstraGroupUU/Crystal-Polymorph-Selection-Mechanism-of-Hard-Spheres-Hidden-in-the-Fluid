# Crystal polymorph selection mechanism of hard spheres hidden in the fluid
by Willem Gispen, Gabriele M. Coli, Robin van Damme, C. Patrick Royall, and Marjolein Dijkstra

This code repository accompagnies our paper published in [ACS Nano](https://doi.org/10.1021/acsnano.3c02182). This code is also available from [github.com/MarjoleinDijkstraGroupUU](github.com/MarjoleinDijkstraGroupUU).

## Abstract

> Nucleation plays a critical role in the birth of crystals and is associated with a vast array of phenomena such as protein crystallization and ice formation in clouds. Despite numerous experimental and theoretical studies, many aspects of the nucleation process, such as the polymorph selection mechanism in the early stages, are far from being understood. Here, we show that the hitherto unexplained excess of particles in a face-centred-cubic (fcc)-like environment, as compared to those in a hexagonal-close-packed (hcp)-like environment, in a crystal nucleus of hard spheres can be explained by the higher order structure in the fluid phase. We show using both simulations and experiments that in the metastable fluid phase  pentagonal bipyramids (PBs) --  clusters with fivefold symmetry known to be inhibitors of crystal nucleation -- transform into a different cluster, Siamese dodecahedra (SDs). These clusters are closely similar to an fcc subunit, which explains the higher propensity to grow fcc than hcp in hard spheres. We show that our crystallization and polymorph selection mechanism is generic for crystal nucleation from a dense, strongly correlated, fluid phase.



## How to use this software

 * Install required python packages using the conda file `conda-ovito-freud.yml`
 * Install the *modified* Topological Cluster Classification algorithm (TCC) [1] python package from TCC.zip or from [github](https://github.com/WillemGispen/TCC/tree/exyz). Please see Malins A, Williams SR, Eggers J & Royall CP "Identification of Structure in Condensed Matter with the Topological Cluster Classification", J. Chem. Phys. (2013). **139** 234506 for more information about the TCC algorithm.

 * Obtain nucleation trajectories (e.g. with `scripts.py.spontaneous_wca_nucleation`)
 * Preprocess the nucleation trajectory using `scripts.py.focus_nucleation_trajectory`. This script centers the nucleus in the simulation cell.
 * Perform a TCC analysis using `scripts.py.tcc`. This computes all TCC clusters defined.
 * Perform a bond order analysis using `scripts.py.bond_order`.
 * Post-process the TCC data using `scripts.py.tcc_conversion`. This script contains various different analyses related to TCC. It also contains a script to output files, with which snapshots can be visualized according to the TCC clusters.
 * Plot the results using `scripts.notebooks.plots`

## License

All source code is made available under the MIT license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

If you use or build on our work, please cite our paper.
