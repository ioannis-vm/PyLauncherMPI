from mpi4py import MPI
import os
import subprocess
from datetime import datetime
from time import perf_counter


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

    if rank == 0:
        # The master process (rank 0) will read the commands and
        # manage task distribution

        # Get environment variables
        work_dir = os.environ.get('LAUNCHER_WORKDIR')
        job_file = os.environ.get('LAUNCHER_JOB_FILE')

        if not work_dir:
            raise ValueError('Did not find LAUNCHER_WORKDIR.')
        if not job_file:
            raise ValueError('Did not find LAUNCHER_JOB_FILE.')

        message(f'`LAUNCHER_WORKDIR={work_dir}`.')
        message(f'`LAUNCHER_JOB_FILE={job_file}`.')
        message(f'The size is {size}.')

        # Load commands from a file
        job_file_path = f'{work_dir}/{job_file}'
        exists = os.path.isfile(job_file_path)
        if not exists:
            raise ValueError(f'Job file does not exist: `{job_file_path}`.')
        with open(job_file_path, 'r') as file:
            commands = [command.strip() for command in file.readlines()]

        message(f'Parsed {len(commands)} tasks.')
        for i, command in enumerate(commands):
            message(f'  Task ID {i}: `{command}`')

        # Dispatch tasks dynamically
        task_id = 0
        num_tasks = len(commands)
        active_requests = size - 1

        while active_requests > 0:
            status = MPI.Status()
            if task_id < num_tasks:
                # Receive any signal
                comm.recv(
                    source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
                )
                comm.send(
                    (
                        task_id,
                        commands[task_id],
                    ),
                    dest=status.source,
                    tag=1,
                )
                task_id += 1
                message(f'Sending task {task_id} to process {status.source}')
            else:
                # No more tasks, receive final signals and send termination tag
                comm.recv(
                    source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
                )
                comm.send((None, None), dest=status.source, tag=0)
                active_requests -= 1

    else:

        # Worker processes requesting tasks and executing them
        while True:

            comm.send(None, dest=0, tag=1)  # Signal readiness to receive task
            task_id, command = comm.recv(
                source=0, tag=MPI.ANY_TAG, status=MPI.Status()
            )
            if command is None:
                # No more tasks, break out of loop
                break

            message(f"Executing task {task_id}.")

            out = subprocess.run(command, capture_output=True, shell=True)
            if out.returncode == 0:
                message(
                    f'Task {task_id} finished successfully. '
                    f'stderr: `{out.stderr}`. stdout: `{out.stdout}`.'
                )
            else:
                message(
                    f'There was an error with task {task_id}. '
                    f'stderr: `{out.stderr}`. stdout: `{out.stdout}`.'
                )

    t_end = perf_counter()
    message(f'Done with all tasks. Elapsed time: {t_end - t_start:.2f} s.')


if __name__ == '__main__':
    main()
