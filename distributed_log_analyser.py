from mpi4py import MPI
import os, sys, time, json
from collections import defaultdict
from datetime import datetime

# MPI Message Tags
WORK_REQUEST = 1
WORK_ASSIGN  = 2
WORK_RESULT  = 3
NO_MORE_WORK = 4
HEARTBEAT    = 5

# Config
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.json")
CHECKPOINT_INTERVAL = 5          # after every 5 processed files
HEARTBEAT_TIMEOUT  = 4.0         # declare failure after this many seconds


# Helper Functions 
def analyse_log_file(filepath, hb_callback=None):
    counts = defaultdict(int)
    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f, 1):
                if "[INFO]" in line:
                    counts["INFO"] += 1
                elif "[WARN]" in line or "[WARNING]" in line:
                    counts["WARN"] += 1
                elif "[ERROR]" in line:
                    counts["ERROR"] += 1
                elif "[DEBUG]" in line:
                    counts["DEBUG"] += 1

                if hb_callback and (i % 1000 == 0):
                    hb_callback()
    except:
        pass
    return counts


def merge_counts(total, new):
    for k, v in new.items():
        total[k] += v


def save_checkpoint(counts, processed, pending, failed):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "counts": dict(counts),
        "processed_files": list(processed),
        "pending_files": list(pending),
        "failed_workers": list(failed),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=4)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return defaultdict(int), set(), [], set(), False
    with open(CHECKPOINT_FILE, "r") as f:
        data = json.load(f)
    print("Loading checkpoint from previous run...")
    processed = set(data.get("processed_files", []))
    pending = data.get("pending_files", [])
    counts = defaultdict(int, data.get("counts", {}))
    failed = set(data.get("failed_workers", []))
    print(f"Resuming from checkpoint: {len(processed)} files already processed")
    return counts, processed, pending, failed, True


# MASTER PROCESS 
def master_process(comm, size, log_dir):
    all_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".log")]

    counts, processed, pending, failed_workers, resumed = load_checkpoint()
    if not pending:
        pending = [f for f in all_files if f not in processed]

    print(f"Found {len(all_files)} log file(s) to analyse ({len(pending)} remaining)")
    print("Starting distributed analysis...\n")

    start_time = time.time()
    assigned = {}
    last_heartbeat = {r: time.monotonic() for r in range(1, size)}
    processed_since_ckpt = 0
    checkpoints_saved = 0

    while pending or assigned:
        while comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
            status = MPI.Status()
            msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            src, tag = status.Get_source(), status.Get_tag()

            if tag == WORK_REQUEST:
                if pending:
                    file = pending.pop(0)
                    assigned[src] = file
                    last_heartbeat[src] = time.monotonic()
                    comm.send(file, dest=src, tag=WORK_ASSIGN)
                    print(f"[Master] Assigning {os.path.basename(file)} to worker {src}")
                else:
                    # if pending empty but worker still had file, delay NO_MORE_WORK
                    if src in assigned:
                        continue
                    comm.send(None, dest=src, tag=NO_MORE_WORK)

            elif tag == WORK_RESULT:
                last_heartbeat[src] = time.monotonic()
                merge_counts(counts, msg["counts"])
                processed.add(msg["file"])
                assigned.pop(src, None)
                processed_since_ckpt += 1
                if processed_since_ckpt >= CHECKPOINT_INTERVAL:
                    save_checkpoint(counts, processed, pending, failed_workers)
                    checkpoints_saved += 1
                    processed_since_ckpt = 0

            elif tag == HEARTBEAT:
                last_heartbeat[src] = time.monotonic()

        # timeout check
        now = time.monotonic()
        for w, last in list(last_heartbeat.items()):
            if (now - last) > HEARTBEAT_TIMEOUT and w in assigned:
                print(f"[Master] Worker {w} timeout detected - marking as failed")
                failed_workers.add(w)
                pending.insert(0, assigned[w])
                print(f"[Master] Reassigning {os.path.basename(assigned[w])} to another worker")
                assigned.pop(w, None)
                last_heartbeat[w] = now

        time.sleep(0.05)

    # Drain phase
    # Handle any results that arrived right after loop exit
    while comm.Iprobe(source=MPI.ANY_SOURCE, tag=WORK_RESULT):
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=WORK_RESULT)
        merge_counts(counts, msg["counts"])
        processed.add(msg["file"])


    # tell all workers to stop
    for w in range(1, size):
        comm.send(None, dest=w, tag=NO_MORE_WORK)

    end_time = time.time()
    save_checkpoint(counts, processed, pending, failed_workers)
    checkpoints_saved += 1

    # FINAL OUTPUT
    print("\n" + "=" * 49)
    print("ANALYSIS RESULTS")
    print("=" * 49)
    for lvl in sorted(counts.keys()):
        print(f"{lvl}: {counts[lvl]}")
    print("=" * 49)
    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"Files processed: {len(processed)}")
    print(f"Failed workers: {len(failed_workers)}")
    print(f"Checkpoints saved: {checkpoints_saved}")
    print("=" * 49)


# WORKER PROCESS 
def worker_process(comm, rank):
    master = 0

    def heartbeat():
        comm.send(None, dest=master, tag=HEARTBEAT)

    comm.send(None, dest=master, tag=WORK_REQUEST)
    while True:
        status = MPI.Status()
        comm.probe(source=master, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == WORK_ASSIGN:
            file = comm.recv(source=master, tag=WORK_ASSIGN)
            if file is None:
                break
            # <-- RECOMMENDED: send an immediate heartbeat before starting processing
            # this helps the master know the worker is alive even for very small files
            try:
                comm.send(None, dest=master, tag=HEARTBEAT)
            except:
                pass

            print(f"[Worker {rank}] Processing {os.path.basename(file)}")
            counts = analyse_log_file(file, hb_callback=heartbeat)
            # send result and then request next work
            comm.send({"file": file, "counts": counts}, dest=master, tag=WORK_RESULT)
            comm.send(None, dest=master, tag=WORK_REQUEST)

        elif tag == NO_MORE_WORK:
            comm.recv(source=master, tag=NO_MORE_WORK)
            break
        else:
            comm.recv(source=master, tag=tag)
    return


#  MAIN
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) < 2:
        if rank == 0:
            print("Usage: mpirun -np <n> python3 distributed_log_analyser.py <log_directory>")
        sys.exit(1)

    log_dir = sys.argv[1]
    if rank == 0:
        if not os.path.isdir(log_dir):
            print(f"Error: {log_dir} is not a valid directory")
            sys.exit(1)
        master_process(comm, size, log_dir)
    else:
        worker_process(comm, rank)

    comm.Barrier()
    MPI.Finalize()


if __name__ == "__main__":
    main()
