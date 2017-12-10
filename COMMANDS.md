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
	python dataset_maker.py path/to/pose-folder base-name
	e.g.
	python dataset_maker.py models/cube4/poses-1 cube4-1
	
### Compare our result with SSD result, both numerically and by generating reconstructed meshes.
	python compare.py path/to/pose-folder path/to/rest_pose path/to/groundtruth_weight path/to/SSD_result.txt
	e.g.
	python compare.py models/cube4/poses-1 models/cube4/cube.obj models/cube4/cube.DMAT SSD_res/cube4-1-output.txt