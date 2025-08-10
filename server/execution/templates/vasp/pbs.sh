#!/bin/bash -f
#$ -cwd
#$ -o ${job_name}.log
#$ -e ${job_name}.err
#$ -pe dc* ${ntasks}
#$ -l h_data=4G,h_vmem=16G,h_rt=${walltime}

source /u/local/Modules/default/init/modules.sh
module add intel/17.0.7

# ----- 你的软件环境 -----
export PYTHONPATH=/u/home/y/yaniguan/miniconda3/envs/ase/:$PYTHONPATH
export PATH=/u/home/y/yaniguan/miniconda3/envs/ase/bin:$PATH
export VASP_PP_PATH=$HOME/vasp/mypps

export OMP_NUM_THREAD=1
export I_MPI_COMPATIBILITY=4

export VASP_COMMAND="mpirun -np ${ntasks} /u/home/y/yaniguan/vasp_std_vtst_sol"

python ase_opt.py

echo "run complete on $(hostname): $(date) $(pwd)" >> ~/job.log
