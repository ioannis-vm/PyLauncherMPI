"""

pylaunchermpi

A simple MPI-based task scheduler for dynamically distributing
commands across MPI processes.

"""

import os
import subprocess
from datetime import datetime
from time import perf_counter
from mpi4py import MPI


def message(text):
    """
    Prints a message to stdout including the process ID and a
    timestamp.

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    current_time = datetime.now()
    time_string = current_time.strftime("%H:%M:%S")
    message_contents = f'{time_string} | Process {rank}: ' + text
    print(message_contents, flush=True)


def main():
    """
    Main function.

    """

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
        with open(job_file_path, 'r', encoding='utf-8') as file:
            commands = [command.strip() for command in file.readlines()]

        message(
            f'Parsed {len(commands)} tasks. Tasks: '
            + str({i: commands[i] for i in range(len(commands))})
        )

        # Dispatch tasks dynamically
        task_id = 0
        num_tasks = len(commands)
        active_requests = size - 1

        while active_requests > 0:
            if task_id < num_tasks:
                # Receive any signal
                # get data
                status = MPI.Status()
                comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                # get sender
                sender = status.Get_source()
                message(f'Sending task {task_id} to process {sender}')
                comm.send(
                    (
                        task_id,
                        commands[task_id],
                    ),
                    dest=sender,
                    tag=2,
                )
                task_id += 1
            else:
                # No more tasks, receive final signals and send termination tag
                status = MPI.Status()
                comm.recv(
                    source=MPI.ANY_SOURCE,
                    tag=MPI.ANY_TAG,
                    status=status,
                )
                sender = status.Get_tag()
                comm.send((None, None), dest=sender, tag=2)
                active_requests -= 1

    else:

        # Worker processes requesting tasks and executing them
        while True:

            # Signal readiness to receive task
            comm.send(None, dest=0, tag=rank)
            task_id, command = comm.recv(source=0, tag=2)
            if command is None:
                # No more tasks, break out of loop
                break

            message(f"Executing task {task_id}.")

            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                message(
                    f'Task {task_id} finished successfully. '
                    f'stderr: `{stderr}`. stdout: `{stdout}`.'
                )
            else:
                message(
                    f'There was an error with task {task_id}. '
                    f'stderr: `{stderr}`. stdout: `{stdout}`.'
                )

    t_end = perf_counter()
    message(
        f'Done with all tasks. Elapsed time: '
        f'{t_end - t_start:.2f} s. Waiting at barrier.'
    )

    comm.Barrier()
    MPI.Finalize()


if __name__ == '__main__':
    main()
