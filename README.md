# InverseSkinning

This is the repository for the code and data used in the paper [*Hyperspectral Inverse Skinning* by Songrun Liu, Jianchao Tan, Zhigang Deng, Yotam Gingold in Computer Graphics Forum 2020](https://cragl.cs.gmu.edu/hyperskinning/).

To generate the weights and transformations used in the paper, run the scripts:

    results_songrun_final_run_sh/*/run.sh

For Table 3:

* `cat-poses`
* `chickenCrossing`
* `elephant-gallop`
* `elephant-poses`
* `face-poses`
* `horse-collapse`
* `horse-gallop`
* `horse-poses`
* `lion-poses`
* `pcow`
* `pdance`
* `pjump`

and compare to the data in `SSD_unconstrained`.

For Table 2:

* `crane`
* `elasticCow`
* `elephant`
* `horse`
* `samba`

and compare to existing output data (see below).

For Table 5:

* `cylinder`
* `cube`
* `cheburashka`
* `wolf`
* `cow`

Various comparison scripts can be found as `compare*.py`.
Various other scripts can be found as `test*.sh`.
Some examples can be found in `COMMANDS.md`.

The error and running time for Le and Deng's [2012] output comes from Table 3 of their paper. All recovered OBJs for our method, and a subset of the computed OBJs for Le and Deng [2012] and Luo et al. [2019] can be directly downloaded from the [project page](https://cragl.cs.gmu.edu/hyperskinning/).

## Dependencies:

- [numpy](http://www.numpy.org/) (e.g. `pip install numpy`)
- [scipy](https://www.scipy.org/) (e.g. `pip install scipy`)
- [MOSEK](https://mosek.com/) (download the software and license, `cd mosek/8/tools/platform/osx64x86/python/3/; python setup.py install`)
- [cvxopt](http://cvxopt.org/) You need an unreleased version with import MOSEK binding fixes: `CVXOPT_BUILD_GLPK=1 pip install git+https://github.com/cvxopt/cvxopt/`
- (viewer only) [libigl](https://github.com/libigl/libigl)(e.g. `git clone https://github.com/libigl/libigl.git --recursive`)
- (viewer only) [CGAL](http://www.cgal.org) (e.g. `brew install cgal`)
- (viewer only) [eigen](http://eigen.tuxfamily.org/) (e.g. `brew install eigen`)
- (viewer only) [GLFW3](http://www.glfw.org/) (e.g. `brew install glfw3`)

## Data

See `models/README.md`.

### Compile the viewer

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

### Running the GUI

View a set of poses with random coloring on each handle's influencing region.

	./viewer ../models/animal/poses-1/animal-1.obj ../models/animal/poses-1/animal-2.obj ../models/animal/poses-1/animal-3.obj ../models/animal-estimated.DMAT
	
### Defunct weight estimation (no longer used)

Assuming a set of poses and corresponding handle(bone)'s transformation matrices(Tmat, #handle*4-by-3) are given,
Estimate each vertex's weights.

	./estimate_weights ../models/animal/poses-*/animal-*.obj ../models/animal/poses-*/animal-*.Tmat --weight ../models/animal/animal.DMAT	

### Test the minimum volume enclosure simplex
On the minimum volume simplex enclosure problem for estimating a linear mixing model
(https://link.springer.com/article/10.1007/s10898-012-9876-5).

	python simplex_hull.py models/cube4/poses-1
