#!/bin/bash -f
#$ -cwd
#$ -o $JOB_ID.log
#$ -e $JOB_ID.err
#$ -pe dc* 32
#$ -l h_data=4G,h_vmem=16G,h_rt=24:00:00

source /u/local/Modules/default/init/modules.sh
module add intel/17.0.7
export PYTHONPATH=/u/home/y/yaniguan/miniconda3/envs/ase/:$PYTHONPATH
export PATH=/u/home/y/yaniguan/miniconda3/envs/ase/bin:$PATH

export VASP_PP_PATH=$HOME/vasp/mypps

export OMP_NUM_THREAD=1
export I_MPI_COMPATIBILITY=4

export VASP_COMMAND='mpirun -np ${NSLOTS} ~/vasp_std_vtst_sol'
python ase-opt.py
echo “run complete on `hostname`: `date` `pwd`” >> ~/job.log