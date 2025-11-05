from mpi4py import MPI
import os
import sys
import time
from collections import defaultdict


def analyse_log_file(filepath):
    counts = defaultdict(int)
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if '[INFO]' in line:
                    counts['INFO'] += 1
                elif '[WARN]' in line or '[WARNING]' in line:
                    counts['WARN'] += 1
                elif '[ERROR]' in line:
                    counts['ERROR'] += 1
                elif '[DEBUG]' in line:
                    counts['DEBUG'] += 1
    except Exception as e:
        print(f"Error reading {filepath}: {e}", flush=True)
    return counts


def merge_counts(total_counts, new_counts):
    for level, count in new_counts.items():
        total_counts[level] += count


def divide_work(log_files, num_procs):
    """Return a list of lists: one sub-list of files per process."""
    chunks = [[] for _ in range(num_procs)]
    for i, file in enumerate(log_files):
        chunks[i % num_procs].append(file)
    return chunks
# Main Program----

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) < 2:
        if rank == 0:
            print("Usage: mpirun -np <n> python3 parallel_log_analyser.py <log_directory>")
        sys.exit(1)

    log_dir = sys.argv[1]


    # STEP 1: Master gathers list of .log files

    if rank == 0:
        if not os.path.isdir(log_dir):
            print(f"Error: {log_dir} is not a valid directory")
            sys.exit(1)

        log_files = [os.path.join(log_dir, f)
                     for f in os.listdir(log_dir) if f.endswith(".log")]

        if not log_files:
            print(f"No .log files found in {log_dir}")
            sys.exit(1)

        print(f"Found {len(log_files)} log file(s) to analyse")
        print("Starting parallel analysis...\n")

        start_time = time.time()

        # Divide work among all processes (including master)
        chunks = divide_work(log_files, size)
    else:
        log_files = None
        chunks = None
        start_time = None


    # STEP 2: Scatter file lists

    # Each process receives its sub-list of files
    my_files = comm.scatter(chunks, root=0)


    # STEP 3: Each process analyses its assigned files

    local_counts = defaultdict(int)
    for log_file in my_files:
        counts = analyse_log_file(log_file)
        merge_counts(local_counts, counts)


    # STEP 4: Gather all results at master

    all_counts = comm.gather(local_counts, root=0)


    # STEP 5: Master merges results and prints summary

    if rank == 0:
        total_counts = defaultdict(int)
        for c in all_counts:
            merge_counts(total_counts, c)

        end_time = time.time()

        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        for level in sorted(total_counts.keys()):
            print(f"{level}: {total_counts[level]}")
        print("="*50)
        total_time = end_time - start_time
        print(f"Total time: {total_time:.2f}s")

        # Optional: compare with sequential baseline if known
        print("="*50)

    comm.Barrier()  # ensure clean exit for all ranks
    MPI.Finalize()


if __name__ == "__main__":
    main()
