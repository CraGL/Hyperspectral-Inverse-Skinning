# InverseSkinning

## Compiling

This code depends on:

- [libigl](https://github.com/libigl/libigl)
- [CGAL](http://www.cgal.org) (e.g. `brew install cgal`)
- [eigen](http://eigen.tuxfamily.org/) (e.g. `brew install eigen`)
- [GLFW3](http://www.glfw.org/) (e.g. `brew install glfw3`)
- [numpy](http://www.numpy.org/)

### Download libigl and compile the third-party dependencies
    git clone https://github.com/libigl/libigl.git --recursive

### Compile this project
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make

### Running the included GUI
	./viewer ../models/animal/poses-1/animal-1.obj ../models/animal/poses-1/animal-2.obj ../models/animal/poses-1/animal-3.obj ../models/animal-estimated.DMAT
	
### Estimate weights
	./estimate_weights ../models/animal/poses-*/animal-*.obj ../models/animal/poses-*/animal-*.Tmat --weight ../models/animal/animal.DMAT	

### Test the minimum volume enclosure simplex
On the minimum volume simplex enclosure problem for estimating a linear mixing model
(https://link.springer.com/article/10.1007/s10898-012-9876-5)
	python -i ../mves2.py
	test()

### A python implementation of Hyperplane-based Craig-Simplex-Identification algorithm
A fast hyperplane-based minimum-volume enclosing simplex algorithm for blind hyperspectral unmixing
(https://arxiv.org/abs/1510.08917)
	python ../HyperCSI.py ../models/cube4/cube.mat 4