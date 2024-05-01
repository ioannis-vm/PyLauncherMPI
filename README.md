# PyLauncherMPI
[![Flake8 Lint](https://github.com/ioannis-vm/PyLauncherMPI/actions/workflows/flake8.yml/badge.svg)](https://github.com/ioannis-vm/PyLauncherMPI/actions/workflows/flake8.yml/badge.svg)

A simple MPI-based task scheduler for dynamically distributing commands across MPI processes.

## How to use

Assuming there is access to HPC resources with MPI capabilities,
`PyLauncherMPI` can be used to dynamically allocate tasks to
processes. Tasks are defined as shell commands in a file, one line per task, like so:

`taskfile`
```
./my_program arg1 arg2 arg3
./my_program arg4 arg5 arg6
...
```

Two environment variables need to be speified, `LAUNCHER_WORKDIR` (the
absolute path from which all tasks will be executed), and
`LAUNCHER_JOB_FILE` (the path to the task file relative to
`LAUNCHER_WORKDIR`).

The following is an example SLURM script using `PyLauncherMPI`.
```
#!/bin/bash
#SBATCH -J test_job        # Job name
#SBATCH -o test_job.o%j    # Name of stdout file
#SBATCH -e test_job.e%j    # Name of stderr error file
#SBATCH -p xyz             # Queue (partition) name
#SBATCH -N 3               # Total # of nodes
#SBATCH -n 144             # Total # of mpi tasks
#SBATCH -t 00:02:30        # Run time (hh:mm:ss)
##SBATCH --mail-user=username@address.xyz
##SBATCH --mail-type=all
#SBATCH -A allocation_name # Allocation name (req'd if you have more than 1)

source $HOME/.bashrc

micromamba activate rid_prj
# (Or use any other way to activate the environment where PyLauncherMPI is installed.)

export OMP_NUM_THREADS=1
export PYTHONPATH=$PYTHONPATH:$(pwd):$HOME/bin
export LAUNCHER_WORKDIR=/path/to/project/root/directory
export LAUNCHER_JOB_FILE=relative/path/name_of_taskfile

ibrun pylaunchermpi
```
