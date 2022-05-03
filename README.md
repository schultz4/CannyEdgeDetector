# CannyEdgeDetector
Canny Edge Detector for CUDA implementation using C++, with Otzu's method for 
threshold values

# Build Instructions
This project uses cmake for building. 

## Pre-build setup
Paths to nvcc need to be established prior to building. If running on the UofA 
HPC, you will need to do the following to setup your environment
```
module load cuda11/11.0
```
While this project does make use of the libwb library, it is using 
modifications and was forked into this project

Images used for processing these runs can be found at
```
```
## Compile
Run the following from the top-level directory
```
cd canny/build_dir/
cmake ../labs
make
```
This should generate the following executables 
```
canny/build_dir/CannyImage_Serial
canny/build_dir/CannyImage_Solution
canny/build_dir/CannyImage_Solution_Opt
canny/build_dir/CannyImage_Solution_Best
```

## Experiment Execution
A bash/slurm script has been maintained for executing all of the runs used for 
this experiment. To execute all of the runs on the UofA HPC system, do the 
following from the top-level directory
```
cd canny/build_dir
srun run_canny.slurm
```
If you are running outside this directory, you will need to modify the 
```hw_path``` variable on line 36 of ```run_canny.slurm``` to reference this 
directory

# Project organization
TODO - Fill me out

