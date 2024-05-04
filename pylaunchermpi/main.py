from mpi4py import MPI
import os
import subprocess
from datetime import datetime
from time import perf_counter
from time import sleep


def message(text):
    """
    Prints a message to stdout including the process ID and a
    timestamp.

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    current_time = datetime.now()
    time_string = current_time.strftime("%H:%M:%S")
    message = f'{time_string} | Process {rank}: ' + text
    print(message, flush=True)


def main():

    t_start = perf_counter()

    # Initialize the MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # The master process (rank 0) will read the commands and distribute them
    if rank == 0:

        # Get environment variables
        work_dir = os.environ.get('LAUNCHER_WORKDIR')
        job_file = os.environ.get('LAUNCHER_JOB_FILE')

        if not work_dir:
            raise ValueError('Did not find LAUNCHER_WORKDIR.')
        if not job_file:
            raise ValueError('Did not find LAUNCHER_JOB_FILE.')

        message(f'`LAUNCHER_WORKDIR={work_dir}`.')
        message(f'`LAUNCHER_JOB_FILE={job_file}`.')

        # Load commands from a file
        job_file_path = f'{work_dir}/{job_file}'
        exists = os.path.isfile(job_file_path)
        if not exists:
            raise ValueError(f'Job file does not exist: `{job_file_path}`.')
        with open(job_file_path, 'r') as file:
            commands = file.readlines()
        # Remove newline characters
        commands = [command.strip() for command in commands]

        message(f'Parsed {len(commands)} tasks.')

    else:
        commands = None

    # Scatter commands to all processes, assuming the number of commands
    # is at least the number of processes
    if rank == 0:
        # Allocate commands to processes
        allocated_commands = [[] for _ in range(size)]
        for i, command in enumerate(commands):
            allocated_commands[i % size].append(command)
    else:
        allocated_commands = None

    # Distribute the commands
    commands_for_process = comm.scatter(allocated_commands, root=0)

    # Wait a bit for other processes to perform their IO operations
    sleep(rank / 1000.00)  # i.e., process 1000 will start after 1 sec

    # Each process runs its allocated commands
    for command in commands_for_process:
        message(f"Executing command: `{command}`")
        out = subprocess.run(command, capture_output=True, shell=True)
        if out.returncode == 0:
            message(
                f'Command `{command}` '
                f'finished successfully. '
                f'stderr: `{out.stderr}`. '
                f'stdout: `{out.stdout}`.'
            )
        else:
            message(
                f'There was an error with command `{command}`. '
                f'stderr: `{out.stderr}`. '
                f'stdout: `{out.stdout}`.'
            )

    t_end = perf_counter()

    message(f'Done with all tasks. ' f'Elapsed time: {t_end - t_start:.2f} s.')


if __name__ == '__main__':
    main()
