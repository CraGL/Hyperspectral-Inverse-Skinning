### Generate result for a set of poses
The input is a folder containing several pose meshes [mesh.obj], ground truth of per-vertex transformation matrices [mesh-i.DMAT],
and ground truth of per-bone transformation matrices [mesh-i.Tmat].
The output is a "result.txt" file placed in the pose folder. The result contains running time, #iterations, per-bone-per-pose-transformations,
skinning weights. The result has a consistent format as the SSD program's result.

    python convex_hull.py path/to/pose-folder
    e.g.
    python convex_hull.py models/animal/poses-1
    
### Convert our pose settings to SSD input settings
Convert a set of OBJ files of poses in pose folder into a txt file, which is the input file of the SSD program. Place it in the pose folder.

	python serialize_pose.py path/to/pose-folder base-name
	e.g.
	python serialize_pose.py models/cube4/poses-1 cube4-1
	
### Convert the SSD input settings to a folder containing a sequence of pose meshes.
SSD input includes a rest pose OBJ file and a TXT file containing pose information.

	python unserialize_pose.py path/to/SSD_input.obj path/to/SSD_input.txt path/to/folder
	e.g.
	python unserialize_pose.py SSD_Data/cat-poses.obj SSD_Data/cat-poses.txt models/cat
	
### Compare our result with SSD result, both numerically and by generating reconstructed meshes.

	python compare.py path/to/pose-folder path/to/rest_pose path/to/groundtruth_weight path/to/SSD_result.txt
	e.g.
	python compare.py models/cube4/poses-1 models/cube4/cube.obj models/cube4/cube.DMAT SSD_res/cube4-1-output.txt
	
### per-vertex transformation optimization.

	python -m pdb flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.1 --GT models/cube4/poses-1
	python -m pdb flat_intersection.py models/cheburashka/cheburashka.obj models/cheburashka/poses-2/cheb-2.txt --H 11 --recovery 0 --GT models/cheburashka/poses-2
	
	#### Zero energy tests
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0 --strategy gradient --GT models/cube4/poses-1 --energy B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0 --strategy grassmann --GT models/cube4/poses-1 --energy B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0 --strategy gradient --GT models/cube4/poses-1 --energy cayley
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0 --GT models/cube4/poses-1 --energy biquadratic
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0 --GT models/cube4/poses-1 --energy biquadratic --solve-for-rest-pose True
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 20 --GT models/cube4/poses-1 --energy biquadratic
	
	#### Recovery tests
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.01 --strategy grassmann --GT models/cube4/poses-1 --energy B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.01 --strategy gradient --GT models/cube4/poses-1 --energy B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.01 --strategy gradient --GT models/cube4/poses-1 --energy cayley
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.001 --strategy gradient --GT models/cube4/poses-1 --energy cayley
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.1 --strategy grassmann --GT models/cube4/poses-1 --energy B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.1 --strategy gradient --GT models/cube4/poses-1 --energy B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.1 --strategy gradient --GT models/cube4/poses-1 --energy cayley
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.1 --strategy gradient --GT models/cube4/poses-1 --energy cayley+cayley
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.1 --strategy gradient --GT models/cube4/poses-1 --energy B+cayley
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.1 --strategy gradient --GT models/cube4/poses-1 --energy B+B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.1 --GT models/cube4/poses-1 --energy biquadratic --strategy pinv
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.1 --GT models/cube4/poses-1 --energy biquadratic --solve-for-rest-pose True
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 0.1 --GT models/cube4/poses-1 --energy biquadratic
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 1 --GT models/cube4/poses-1 --energy biquadratic
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 1 --GT models/cube4/poses-1 --energy biquadratic --strategy pinv
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 1 --strategy gradient --GT models/cube4/poses-1 --energy cayley
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 1 --strategy gradient --GT models/cube4/poses-1 --energy B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 1 --strategy grassmann --GT models/cube4/poses-1 --energy B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --recovery 100 --GT models/cube4/poses-1 --energy biquadratic
	
	#### Tests from random
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --energy biquadratic
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --strategy grassmann --energy B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 4 --strategy gradient --energy B
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 2 --energy biquadratic
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 1 --energy biquadratic
	
	#### Zero energy tests onto fewer handles than possible
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 3 --recovery 0 --GT models/cube4/poses-1 --energy biquadratic
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 2 --recovery 0 --GT models/cube4/poses-1 --energy biquadratic
	python flat_intersection.py models/cube4/cube.obj models/cube4/poses-1/cube4-1.txt --H 1 --recovery 0 --GT models/cube4/poses-1 --energy biquadratic
