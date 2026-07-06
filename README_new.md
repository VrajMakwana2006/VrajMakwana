# Distributed Log Analyser

A log analysis system built in three stages — sequential, parallel, and fully distributed with fault tolerance — to explore how real distributed systems handle scale, failure, and recovery.

The task itself (counting log-level occurrences across files) is intentionally simple. The point of the project is everything needed to do that reliably across multiple workers: **dynamic work distribution, heartbeat-based failure detection, automatic work reassignment, and checkpoint/recovery**, implemented with MPI (`mpi4py`).

## Why this project

Most "distributed systems" class projects stop at parallelizing a for-loop. This one goes a step further and asks: what happens when a worker dies mid-task? How do you resume from a crash without reprocessing everything, or double-counting? The three stages below build up to answering that.

## Architecture

**Stage 0 — Sequential baseline** (`base_log_analyzer.py`)
Reads every `.log` file in a directory one at a time and counts `INFO` / `WARN` / `ERROR` / `DEBUG` lines. This is the correctness and performance baseline everything else is measured against.

**Stage 1 — Parallel** (`parallel_log_analyser.py`)
Static round-robin distribution of files across MPI processes (`chunks[i % num_procs]`), with each process analyzing its share independently and the master merging partial counts.

**Stage 2 — Distributed with fault tolerance** (`distributed_log_analyser.py`)
A master-worker architecture with:
- **Dynamic (pull-based) work assignment** — workers request files as they finish, instead of a static upfront split, so faster workers naturally pick up more work
- **Heartbeat failure detection** — each worker pings the master every 1,000 lines; if a worker goes silent for more than 4 seconds, the master marks it failed
- **Automatic work reassignment** — a failed worker's in-progress file is pushed back onto the pending queue and picked up by the next available worker
- **Checkpointing** — every 5 files processed (or on clean shutdown), the master snapshots counts, processed files, pending files, and failed workers to `checkpoint.json`, so a restarted run resumes instead of starting over
- **Eventual consistency** — workers push partial results as soon as they finish a file rather than waiting to synchronize with others, trading strict ordering for lower latency and higher throughput

## Results

Sequential vs. parallel (Stage 1), same machine, same run:

| Processes | Time    | Speedup | Efficiency |
|-----------|---------|---------|------------|
| 1         | 0.20s   | 1.27x   | 1.27       |
| 2         | 0.12s   | 2.17x   | 1.09       |
| 4         | 0.11s   | 2.93x   | 0.73       |
| 8         | 0.10s   | 6.75x   | 0.84       |

(Sequential time was re-measured on every run via subprocess rather than once upfront, since a single machine's runtime for the baseline was inconsistent enough to skew the comparison.)

Stage 2 fault-tolerance testing (10,000 log files, ~304K lines analyzed):

| Scenario           | Total time | Files processed | Failed workers | Checkpoints |
|--------------------|-----------|------------------|-----------------|-------------|
| Normal run         | 641.2s    | 10,000           | 0               | 2,001       |
| Resume from crash  | 649.8s    | 10,000 (415 restored from checkpoint) | 0 | 2,001 |
| Worker killed mid-run | 523.5s | 10,000           | 1 (auto-reassigned, no data loss) | 2,001 |

Final counts were identical across all three runs (`DEBUG: 89447, ERROR: 18404, INFO: 152485, WARN: 43916`), confirming that a crashed worker's work was fully recovered with no duplication or loss.

## Known limitations

- **No master failover.** If the master process itself dies, there's no re-election — the run just stops. This is a real architectural gap I'd tackle next (a Raft-style leader election, or a passive standby master, would fix it).
- Failure detection resolution is bounded by the 4-second heartbeat timeout, so very short-lived worker failures near the end of a file might not be caught before the work naturally completes.

## Running it

Requires `mpi4py` (`pip install mpi4py`) and Open MPI or MPICH installed locally.

```bash
# Sequential
python3 base_log_analyzer.py <path-to-log-directory>

# Parallel (n = number of processes)
mpiexec -n 4 python3 parallel_log_analyser.py <path-to-log-directory>

# Distributed, with fault tolerance
mpiexec -n 4 python3 distributed_log_analyser.py <path-to-log-directory>
```

`logs_generator.py` can generate a synthetic set of test log files if you don't have your own.

## Repo contents

```
base_log_analyzer.py         # Stage 0: sequential baseline
parallel_log_analyser.py     # Stage 1: static MPI parallelization
distributed_log_analyser.py  # Stage 2: dynamic distribution + fault tolerance
logs_generator.py            # Synthetic log file generator for testing
report_stage1.txt            # Stage 1 performance analysis
report_stage2.txt            # Stage 2 fault-tolerance test results & design notes
analysis.md                  # Analysis of a foundational distributed systems paper
```

## Background

This project started as a distributed systems induction assignment (see `implementation.md` / `paper_reading.md` for the original task spec), paired with a reading/analysis component on a foundational distributed systems paper (`analysis.md`).
