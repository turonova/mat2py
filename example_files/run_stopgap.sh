#! /usr/bin/env bash
set -e              # Crash on error
set -o nounset      # Crash on unset variables

## run_stopgap.sh
# A script for performing subtomogram averaging using 'stopgap'.
# This script first generates a submission file and then launches 
# 'stopgap_watcher', a MATLAB executable that manages the flow of parallel 
# stopgap averaging jobs.
#
# This specific script was written for clusters at the MPI Biochemistry.
# Use this an example for your specific cluster.
#
# WW 06-2018

### MPI Cryo Cluster Partitions ###
# Raven general


##### RUN OPTIONS #####
scripts_path="/u/$USER/STOPGAP/exec/"		# to export proper path - either 'cryo' or 'local'
run_type='slurm'            # Types supported are 'local', 'sge', and 'slurm', for local, SGE-cluster and slurm-cluster submissions.
nodes=4
n_cores=40              	# Number of cores per node!
mem_limit=180GB         	# Amount of memory per node.
time_limit='23:00:00'            # Maximum run time in seconds (max = 24 hours). Ignored for local jobs.
job_id=$1

special_add="--mail-user=$USER@biophys.mpg.de --mail-type=ALL"

##### DIRECTORIES #####
rootdir="/ptmp/$USER/path/to/the/working/parent/directory/$1/"    # Main subtomogram averaging directory
paramfilename='subtomo_param.star'          # Relative path to stopgap parameter file. 


#### Modules ####
matlab_module="matlab/R2021bU2"
openmpi_module="gcc/10 openmpi/4"


################################################################################################################################################################
##### SUBTOMOGRAM AVERAGING WORKFLOW                                                                                                       ie. the nasty bits...
################################################################################################################################################################

module load ${matlab_module}
export STOPGAPHOME=$scripts_path

echo ${STOPGAPHOME}

# Path to MATLAB executables
watcher="${STOPGAPHOME}/bin/stopgap_watcher.sh"
subtomo="${STOPGAPHOME}/bin/stopgap_mpi_slurm.sh"


# Remove previous submission script
rm -f submit_stopgap

if [ "${run_type}" = "local" ]; then
    echo "Running stopgap locally..."


    # Local submit command
    submit_cmd="mpiexec -np ${n_cores} ${subtomo} ${rootdir} ${paramfilename} ${n_cores}  2> ${rootdir}/error_stopgap 1> ${rootdir}/log_stopgap &"
    # echo ${submit_cmd}

elif [ "${run_type}" = "slurm" ]; then
    echo "Preparing to run stopgap on slurm-cluster..."

    # Write submission script
    echo '#!/bin/bash -l' > submit_stopgap
    echo "#SBATCH -D ${rootdir}" >> submit_stopgap
    echo "#SBATCH -e err_${job_id}" >> submit_stopgap
    echo "#SBATCH -o log_${job_id}" >> submit_stopgap
    echo "#SBATCH --job-name ${job_id}" >> submit_stopgap
    echo "#SBATCH --nodes=${nodes}" >> submit_stopgap
    echo "#SBATCH --ntasks-per-node=${n_cores}" >> submit_stopgap
    echo "#SBATCH --mem=${mem_limit}" >> submit_stopgap
    echo "#SBATCH --time=${time_limit}" >> submit_stopgap

	echo "module purge" >> submit_stopgap

	echo "" >> submit_stopgap

	echo "module load ${openmpi_module}" >> submit_stopgap
	echo "module load ${matlab_module}" >> submit_stopgap
	echo "export STOPGAPHOME=${scripts_path}" >> submit_stopgap
	echo 'echo ${STOPGAPHOME}' >> submit_stopgap
	echo "" >> submit_stopgap
	
	echo 'export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK' >> submit_stopgap 
	echo 'export OMP_PLACES=cores' >> submit_stopgap
	echo "" >> submit_stopgap
	
	printf "%s%s\n" "srun ${subtomo} ${rootdir} ${paramfilename} " '${SLURM_NTASKS}' >> submit_stopgap

    # Make executable
    chmod +x submit_stopgap
    
    # Submission command
    sbatch ${special_add} submit_stopgap

else
    echo 'ACHTUNG!!! Invalid run_type!!!'
    echo 'Only supported run_types are "local", "sge", and "slurm"!!!'
    exit 1
fi


exit





