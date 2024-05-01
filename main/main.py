from mpi4py import MPI
import os
import subprocess
from time import perf_counter
import hashlib


def generate_hash(s):
    """
    Uses SHA-256 to hash a given string and then encode the result to
    a 6-character string.

    Parameters
    ----------
    s: string
        The string to hash

    Returns
    -------
    str
        6-charachter hash

    """

    hash_object = hashlib.sha256(s.encode())
    hex_dig = hash_object.hexdigest()  # Convert to hexadecimal
    return hex_dig[:6]  # Return the first 6 characters


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

        print(f'Process {rank}: `LAUNCHER_WORKDIR={work_dir}`.')
        print(f'Process {rank}: `LAUNCHER_JOB_FILE={job_file}`.')

        # Load commands from a file
        job_file_path = f'{work_dir}/{job_file}'
        exists = os.path.isfile(job_file_path)
        if not exists:
            raise ValueError(f'Job file does not exist: `{job_file_path}`.')
        with open(job_file_path, 'r') as file:
            commands = file.readlines()
        # Remove newline characters
        commands = [command.strip() for command in commands]

        print(f'Process {rank}: Parsed {len(commands)} tasks.')

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

    # Each process runs its allocated commands
    for command in commands_for_process:
        command_hash = generate_hash(command)
        print(f"Process {rank}: Executing command {command_hash}: `{command}`")
        out = subprocess.run(command, capture_output=True, shell=True)
        if out.returncode == 0:
            print(
                f'Process {rank}: Command {command_hash} '
                f'finished successfully. '
                f'stderr: `{out.stderr}`. '
                f'stdout: `{out.stdout}`.'
            )
        else:
            print(
                f'Process {rank}: There was an error with {command_hash}. '
                f'stderr: `{out.stderr}`. '
                f'stdout: `{out.stdout}`.'
            )

    t_end = perf_counter()

    print(
        f'Process {rank}: Done with all tasks. '
        f'Elapsed time: {t_end - t_start:.2f} s.'
    )


if __name__ == '__main__':
    main()
