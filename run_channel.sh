#!/bin/bash -l
### Job Name
#PBS -N run_channel
### Project Code Allocation
#PBS -A UMCP0036
### Resources
#PBS -l select=1:ncpus=1:ngpus=1:mem=8GB
### Run Time
#PBS -l walltime=20:00:00
### Type of GPU
#PBS -l gpu_type=v100
### To the Casper queue
#PBS -q casper
### Log file
#PBS -o Logs/run_channel.log
### Join output and error streams into single file
#PBS -j oe
### Email
#PBS -M zhihua@umd.edu
### Send email on abort, begin and end
#PBS -m abe

### Clear and load all the modules needed
module --force purge
module load ncarenv
module load cuda/12.2.1

export TMPDIR=/glade/derecho/scratch/$USER/temp
mkdir -p $TMPDIR

### Run spinup simulation
proj_dir=/glade/u/home/zhihuaz/Projects/TRACE-SEAS/TracerInversion
#--project=<...> activates julia environment
julia --project=$proj_dir Simulations/channel.jl

### Overwrite previous log file
LOG=$proj_dir/Logs/channel.log
if [ -f "$LOG" ]; then
    rm -f $LOG
fi
mv $proj_dir/Logs/run_channel.log $LOG

qstat -f $PBS_JOBID >> Logs/channel.log
