#!/usr/bin/env bash
# Simple array job sample

# Set SLURM options
#SBATCH --job-name=tfim_EE_analysis                # Job name
#SBATCH --output=tfim_EE_analysis-%A-%a.out        # Standard output and error log
#SBATCH --mail-user=tyou@middlebury.edu     # Where to send mail
#SBATCH --mail-type=NONE                        # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --cpus-per-task=1                       # Run each array job on a single core
#SBATCH --mem=2gb                               # Job memory request
#SBATCH --partition=standard                    # Partition (queue)
#SBATCH --time=00:15:00                         # Time limit hrs:min:sec
#SBATCH --array=0-9                             # Array range: stets number of array jobs

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Starting: "`date +"%D %T"`

# Your calculations here
python tfim_EE_analysis.py 3 3 10 > EE-${SLURM_ARRAY_TASK_ID}.dat
cat EE-* > EE.dat
rm EE-*

# End of job info
