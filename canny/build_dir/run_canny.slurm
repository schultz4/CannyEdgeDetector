#!/bin/bash

### Set the job name
#SBATCH --job-name=devicequery

### Specify the group for this job
### List of PI groups available to each user can be found with "va" command

####ece569 students can also replace ece569 with another group id if sponsored by another faculty. 
#SBATCH --account=ece569

### Set the queue for this job as windfall or standard (adjust ### and #)
#SBATCH --partition=standard

#SBATCH -e run.error
#SBATCH -o run.out

### Set the number of nodes, cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=12gb

### Specify "wallclock time", hhh:mm:ss. Required field
#SBATCH --time 00:00:50

### Load required modules/libraries if needed (openmpi example)
### Use "module avail" command to list all available modules
###module load openmpi3
###module load cuda11/11.0

### set directory for job execution, replace the following path with your own path
hw_path=~/ece569_hpc/proj/build_dir
###akoglu/class/ece569/build_dir

cd $hw_path
###
PATHS[0]=$hw_path/CannyImage/Dataset/0;
PATHS[1]=$hw_path/CannyImage/Dataset/1;
PATHS[2]=$hw_path/CannyImage/Dataset/2;
PATHS[3]=$hw_path/CannyImage/Dataset/3;
PATHS[4]=$hw_path/CannyImage/Dataset/4;
PATHS[5]=$hw_path/CannyImage/Dataset/5;
PATHS[6]=$hw_path/CannyImage/Dataset/6;
PATHS[7]=$hw_path/CannyImage/Dataset/7;
PATHS[8]=$hw_path/CannyImage/Dataset/8;
PATHS[9]=$hw_path/CannyImage/Dataset/9;


count=0
for i in ${PATHS[@]}
do
   file=output$((count)).txt
   ./Canny_Solution -e $i/output.pbm -i $i/input.ppm -t image > CannyImage_output/$file
 count=$((count+1))
done


