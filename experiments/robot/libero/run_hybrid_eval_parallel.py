"""Independent-process episode sharding. No DDP, shared model or vectorized env."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal


def terminate(signum, frame):
    raise SystemExit(128 + signum)


def wait_workers(processes):
    """Detect any failure promptly and reap all peers before returning/raising."""
    try:
        while any(p.poll() is None for p in processes):
            failed = [(i, p.returncode) for i, p in enumerate(processes) if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Evaluation worker failed: {failed}; partials retained")
            time.sleep(.2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError(f"Evaluation workers failed: {[p.returncode for p in processes]}")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def worker_environment(threads, egl_device):
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK"):
        env.pop(key, None)
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(egl_device))
    # CUDA_VISIBLE_DEVICES is deliberately preserved: --devices uses its logical indices.
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--num-workers", type=int, choices=(1, 4), default=4)
    parser.add_argument("--devices", type=int, nargs="+")
    parser.add_argument("--egl-devices", type=int, nargs="+", required=True,
                        help="Explicit EGL enumeration indices; verify server mapping before use")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-id", type=int, help=argparse.SUPPRESS)
    args, forwarded = parser.parse_known_args(argv)
    devices = args.devices if args.devices is not None else list(range(args.num_workers))
    if (len(devices) != args.num_workers or len(set(devices)) != len(devices)
            or len(args.egl_devices) != args.num_workers or min(devices + args.egl_devices) < 0 or args.threads < 1):
        parser.error("Need distinct logical CUDA devices, one explicit EGL index per worker, and positive threads")
    if any(s.split("=")[0] in ("--device", "--partial", "--output", "--reference") for s in forwarded):
        parser.error("Device/output/partial are controlled by launcher; reference mode is serial-only")
    paths = [args.run_dir / f"worker-{i}.partial.json" for i in range(args.num_workers)]
    if args.worker_id is not None:
        if not 0 <= args.worker_id < args.num_workers:
            parser.error("Invalid worker ID")
        signal.signal(signal.SIGTERM, terminate)
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        import torch
        torch.set_num_threads(args.threads)
        torch.cuda.set_device(devices[args.worker_id])
        print(f"worker={args.worker_id} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
              f"torch=cuda:{devices[args.worker_id]} EGL={os.environ.get('MUJOCO_EGL_DEVICE_ID')} TF_GPU=[]",
              file=sys.stderr, flush=True)
        from experiments.robot.libero.run_hybrid_eval import main as serial_main
        serial_main(["--checkpoint", str(args.checkpoint), "--device", f"cuda:{devices[args.worker_id]}",
                     "--partial", str(paths[args.worker_id]), *( ["--resume"] if args.resume else []), *forwarded],
                    worker_id=args.worker_id, num_workers=args.num_workers)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.resume and args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError("Use an empty run directory, or --resume")
    if args.output.resolve() in {p.resolve() for p in paths}:
        raise ValueError("Final output cannot be a worker partial")
    protected = {args.checkpoint.resolve(), (args.run_dir / "launcher.lock").resolve()}
    if args.output.resolve() in protected or args.output.resolve() in {p.with_name(p.name + ".lock").resolve() for p in paths}:
        raise ValueError("Final output conflicts with input/lock paths")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Parent lock prevents two launchers from writing the same worker journals.
    lock = args.run_dir / "launcher.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    processes = []
    started = time.monotonic()
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        for index in range(args.num_workers):
            command = [sys.executable, "-m", "experiments.robot.libero.run_hybrid_eval_parallel",
                "--worker-id", str(index), "--num-workers", str(args.num_workers),
                "--devices", *map(str, devices), "--egl-devices", *map(str, args.egl_devices),
                "--threads", str(args.threads), "--run-dir", str(args.run_dir), "--output", str(args.output),
                "--checkpoint", str(args.checkpoint), *(["--resume"] if args.resume else []), *forwarded]
            processes.append(subprocess.Popen(command, env=worker_environment(args.threads, args.egl_devices[index])))
        wait_workers(processes)
        from experiments.robot.libero.hybrid_eval_results import atomic_json, merge_partials
        result = merge_partials(paths, checkpoint=args.checkpoint)
        atomic_json(args.output, result)
        print(json.dumps(result, indent=2), flush=True)
        elapsed = time.monotonic() - started
        print(f"parallel_wall_sec={elapsed:.3f} total_episodes={result['total_trials']} "
              "(resume totals include previously completed episodes)", file=sys.stderr, flush=True)
    finally:
        # Also handles failure during process creation, before wait_workers was entered.
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        lock.unlink()
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
