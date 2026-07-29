# Copyright 2026 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Distributed (SPMD / ``torch.distributed``) driver for the CTM contraction
benchmarks.

This is the multi-rank sibling of :mod:`bench_ctm`. It leaves ``bench_ctm.py``
untouched and drives the same contraction models over the SPMD dispatch path in
``yastn.tensor._oe_blocksparse_dist`` (reached via
``contract_with_unroll(..., distributed=True)``). Every rank runs the identical
program, owns a cost-balanced slice of the unrolled combos, and ``all_reduce``s
the assembled output.

Launch with ``torchrun`` (one process per GPU); each rank builds a numerically
identical copy of the inputs on its own ``cuda:{LOCAL_RANK}`` (or ``cpu`` for a
gloo run) and writes its own results file (``..._mode=dist{N}_rank{r}.out``)::

    NGPU=$(nvidia-smi -L | wc -l)
    PYTHONPATH=/path/to/yastn-dev \
    torchrun --nproc_per_node=$NGPU --nnodes=1 bench_ctm_dist.py \
        -backend torch -device cuda -model CtmBenchMeasureNconFermionic \
        -params "dims=(2,2),unroll={('v',1,1,'k'):5}" -fname <input> ...

``distributed=True`` requires a **dict** ``unroll`` (supply it in ``-params``)
and a torch backend. Run without ``torchrun`` (no ``WORLD_SIZE`` in the
environment) and it safely falls back to a serial single-process run.

Notes
-----
* ``-devices`` / ``-mp_workers_per_device`` from :mod:`bench_ctm` are absent
  here: they are ignored under ``distributed=True`` (the SPMD path replaces the
  node-local multiprocess pool).
* Only the contraction-family models (which subclass
  ``CtmBenchContractionParent`` and therefore accept the ``distributed`` param)
  are selectable.
"""
import argparse
import ast
import contextlib
import gc
import glob
import logging
import os
from pathlib import Path
import sys
import timeit
import tracemalloc

import block_stats
# Importing bench_ctm also installs block_stats; reuse its readable_size helper.
from bench_ctm import readable_size
block_stats.install()  # idempotent


# Contraction-family models that support the ``distributed`` param
# (all subclass CtmBenchContractionParent).
DIST_MODELS = (
    "CtmBenchMeasureNconFermionic",
    "CtmBenchContraction1x1",
    "CtmBenchContraction2x2",
    "CtmBenchContraction2x2Measure",
    "CtmBenchContraction2x3",
    "CtmBenchContractionLxLy",
)

HERE = os.path.dirname(os.path.abspath(__file__))


def init_distributed(base_device):
    """Initialise the ``torch.distributed`` process group for an SPMD run.

    Mirrors the launch contract in yastn-dev's
    ``tests/tensor/test_oe_blocksparse_dist.py`` (``_run_spmd``): pick
    ``nccl``/CUDA when each rank can own a distinct GPU, else ``gloo``/CPU.
    Works for single- and multi-node ``torchrun`` launches — the CUDA decision
    compares each node's local GPU count against its *local* world size
    (``LOCAL_WORLD_SIZE``), not the global world, so N nodes × G GPUs runs on
    NCCL as long as every node has G GPUs for its G local ranks.

    Returns ``(rank, world_size, device, initialised)``. If not launched via
    ``torchrun`` (no ``WORLD_SIZE`` in env) this is a no-op returning
    ``(0, 1, base_device, False)`` — a safe serial fallback.
    """
    if "WORLD_SIZE" not in os.environ:
        print("[bench_ctm_dist] WORLD_SIZE not set (not launched via torchrun); "
              "running serial single-process.", flush=True)
        return 0, 1, base_device, False

    import torch
    import torch.distributed as dist

    world = int(os.environ["WORLD_SIZE"])
    # Per-node worker count (torchrun sets LOCAL_WORLD_SIZE). The local-GPU-count
    # decision must compare against this
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    want_cpu = (str(base_device) == "cpu"
                or os.environ.get("YASTN_DIST_TEST_CPU", "0") == "1")
    use_cuda = (not want_cpu and torch.cuda.is_available()
                and torch.cuda.device_count() >= local_world)
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    if use_cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    if rank == 0:
        print(f"[bench_ctm_dist] distributed init: backend={backend}; "
              f"world={world}; local_world={local_world}; device(rank0)={device}",
              flush=True)
    return rank, world, device, True


def teardown_distributed(initialised):
    if not initialised:
        return
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def configure_logging(rank, world, level_name):
    """Route yastn's diagnostic logging (e.g. the per-rank combo/LPT-load line
    from ``yastn.tensor._oe_blocksparse_dist``) to **stdout**.

    Those records are emitted at INFO/DEBUG on the ``yastn.tensor`` package
    logger; with no handler and the default WARNING root level they are dropped.
    We attach a ``StreamHandler(sys.stdout)`` (the default is ``sys.stderr`` — it
    would otherwise land in ``stderr.log`` under ``torchrun -r 3``), so the
    diagnostics show up in each rank's ``stdout.log``. A ``[r{rank}]`` prefix
    keeps them attributable if streams are ever merged.

    ``level_name='NONE'`` (or ``'OFF'``) disables the diagnostics.
    """
    logger = logging.getLogger("yastn.tensor")
    # Remove any handler we added on a previous call (idempotent across fnames).
    for h in list(logger.handlers):
        if getattr(h, "_bench_ctm_dist", False):
            logger.removeHandler(h)
    if str(level_name).upper() in ("NONE", "OFF"):
        return
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(
        f"[r{rank}/{world}] %(name)s %(levelname)s: %(message)s"))
    handler._bench_ctm_dist = True  # tag so we can find/replace it later
    logger.addHandler(handler)
    logger.setLevel(level)
    # Don't propagate to the root logger to avoid duplicate lines if something
    # (e.g. torch) has also configured the root handler.
    logger.propagate = False


def fname_output_dist(bench, fname, args, rank, world):
    """Per-rank output path.

    Same directory/param layout as ``bench_ctm.fname_output``, but every rank
    writes its own file via the ``_mode=dist{world}_rank{rank}`` stem suffix
    (no collisions; per-rank LPT load balance is inspectable). ``distributed``
    is dropped from the path params (the mode tag already conveys it).
    """
    if args.to_file is False:
        return None
    device = args.device.replace(":", "-")
    ss = f"{HERE}/results_ctm/{type(bench).__name__}/"
    _skip_path_keys = {'f_out', 'unroll', 'sites', 'devices', 'distributed'}
    path_params = {k: v for k, v in bench.params.items()
                   if k not in _skip_path_keys and v is not None and v is not False and v != 0}
    if path_params:
        ss += '_'.join(f"{k}={v}" for k, v in sorted(path_params.items())) + '/'
    ss += (f"{args.dtype}/num_threads={args.num_threads}/policy={args.tensordot_policy}"
           f"/lru_cache={args.lru_cache}/{args.backend}/{device}")
    path = Path(ss)
    path.mkdir(parents=True, exist_ok=True)
    stem = f"{fname.stem}_mode=dist{world}_rank{rank}"
    return path / f"{stem}.out"


def run_bench_dist(model, args, fname, rank, world, device):
    """Run a single benchmark on this rank over the distributed path.

    Mirrors ``bench_ctm.run_bench`` but builds tensors on this rank's ``device``
    and forces ``distributed=True`` into the model kwargs.
    """
    config = {"backend": args.backend, "default_device": device, "default_dtype": args.dtype,
              "lru_cache": args.lru_cache, "tensordot_policy": args.tensordot_policy}
    if args.fermionic is not None:
        config["fermionic"] = ast.literal_eval(args.fermionic)
    #
    expr = ast.parse(f"dict({args.params}\n)", mode="eval")
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in expr.body.keywords}
    kwargs["distributed"] = True
    #
    bench = model(fname, config, **kwargs)
    #
    fname_out = fname_output_dist(bench, fname, args, rank, world)

    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(fname_out, 'w')) if fname_out else sys.stdout
        bench.params.update({'f_out': f})

        bench.print_header(file=f)

        tasks = bench.bench_pipeline
        if len(args.pipeline) > 0 and not ("all" in args.pipeline):
            assert all([(t in bench.bench_pipeline) for t in args.pipeline]), \
                f"Some provided pipeline tasks are not in the model's benchmark pipeline. {tasks}"
            tasks = [t for t in bench.bench_pipeline if (t in args.pipeline)]
        print(f"Model = {type(bench).__name__}; fname = {fname.name}", file=f, flush=True)
        print(f"backend = {args.backend}; device = {device}; dtype = {args.dtype}", file=f, flush=True)
        print(f"num_threads = {args.num_threads}; tensordot_policy = {args.tensordot_policy}; lru_cache = {args.lru_cache}", file=f, flush=True)
        if args.fermionic is not None:
            print(f"fermionic = {args.fermionic}", file=f, flush=True)
        print(f"dispatch = dist{world}; rank = {rank}/{world}", file=f, flush=True)
        print(f"Selected pipeline tasks to run: {tasks}", file=f, flush=True)
        for task in tasks:
            print(task + "; times [seconds]", file=f, flush=True)
            times = []
            results = []
            max_blocks_per_run = []
            for r in range(args.repeat):
                gc.collect()
                if 'torch' in args.backend and 'cuda' in device:
                    import torch
                    torch.cuda.empty_cache()
                block_stats.reset()
                t = timeit.timeit(stmt=f'bench.{task}()', number=1, globals=locals())
                mb, where = block_stats.report()
                max_blocks_per_run.append(mb)
                times.append(t)
                result_val = None
                if hasattr(bench, 'tensors') and 'result' in bench.tensors:
                    result_val = float(bench.tensors['result']._data[0])
                    del bench.tensors['result']
                results.append(result_val)
                print(f"  run {r+1}/{args.repeat}: {t:.4f}  max_blocks={mb} ({where})", file=f, flush=True)
            print(*(f"{t:.4f}" for t in times), file=f, flush=True)
            if max_blocks_per_run:
                overall_max = max(max_blocks_per_run)
                print(f"max_blocks per run: {max_blocks_per_run}; overall_max={overall_max}",
                      file=f, flush=True)
            if any(r is not None for r in results):
                print("results:", *(f"{r}" for r in results), file=f, flush=True)
            if args.memory_profile:
                tracemalloc.start()
                current, peak = tracemalloc.get_traced_memory()
                print(f"memory: {readable_size(current)}, {readable_size(peak)}", file=f, flush=True)
                tracemalloc.stop()

        bench.print_properties(file=f)
        f.flush()
        bench.final_cleanup()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Distributed (SPMD/torchrun) driver for CTM contraction benchmarks.")
    parser.add_argument("-backend", type=str, default='torch',
                        help="torch backend variant (distributed requires a torch backend), "
                             "e.g. torch, torch_cutensor, torch_cpp.")
    parser.add_argument("-dtype", type=str, default='float64',
                        choices=['float32', 'float64', 'complex64', 'complex128'])
    parser.add_argument("-device", type=str, default='cuda',
                        help="Base device: 'cuda' (each rank builds on cuda:LOCAL_RANK via NCCL) "
                             "or 'cpu' (gloo). The per-rank device is derived from LOCAL_RANK.")
    parser.add_argument("-tensordot_policy", type=str, default='no_fusion',
                        choices=['fuse_to_matrix', 'fuse_contracted', 'no_fusion'])
    parser.add_argument("-fermionic", type=str, default=None,
                        help="Optional Python literal passed to yastn.make_config as fermionic, "
                             "e.g. 'True' or '(False, False, True)'.")
    parser.add_argument("-no_lru_cache", dest='lru_cache', action='store_false',
                        help="Switch off yastn's lru_cache backing the symmetry algebra.")
    parser.add_argument("-stdout", dest='to_file', action='store_false',
                        help="Print to stdout (all ranks, interleaved) instead of per-rank files.")
    parser.add_argument("-memory_profile", dest='memory_profile', action='store_true',
                        help="Profile memory usage with tracemalloc. High overhead.")
    parser.add_argument("-repeat", type=int, default=4, help='Number of repeated runs; passed to timeit')
    parser.add_argument("-fname", type=str, default='Heisenberg_U1_d=2_D=4_chi=30',
                        help="Use glob to match basenames of json files in ./input_shapes")
    parser.add_argument("-model", type=str, default='CtmBenchMeasureNconFermionic',
                        choices=list(DIST_MODELS),
                        help="Contraction model to run (must support the distributed param).")
    parser.add_argument("-params", type=str, default='',
                        help="Model-specific parameters; MUST include a dict 'unroll=...' for the "
                             "distributed path to engage, e.g. \"dims=(2,2),unroll={('v',1,1,'k'):5}\".")
    parser.add_argument(
        "-pipeline",
        nargs="*",
        choices=["all", "contract", "precompute_A_mat", "enlarged_corner",
                 "fuse_enlarged_corner", "svd_enlarged_corner", "ctmrg_update"],
        default=["all"],
        help="Pipeline steps to run; provide multiple values separated by space. Default: all.",
    )
    parser.add_argument("-num_threads", type=str, default='none',
                        help="Set number of threads for CPU backends; 'none' keeps default settings.")
    parser.add_argument("-log_level", type=str, default='INFO',
                        help="Level for yastn's diagnostic logging routed to stdout (per-rank "
                             "combo/LPT-load line from _oe_blocksparse_dist). One of DEBUG, INFO, "
                             "WARNING, ...; 'NONE' disables it. Default INFO.")
    return parser


def main():
    args = build_parser().parse_args()

    if args.num_threads.lower() != 'none':
        os.environ["OMP_NUM_THREADS"] = args.num_threads
        os.environ["OPENBLAS_NUM_THREADS"] = args.num_threads
        os.environ["MKL_NUM_THREADS"] = args.num_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = args.num_threads
        os.environ["NUMEXPR_NUM_THREADS"] = args.num_threads

    assert 'torch' in args.backend, \
        f"distributed path requires a torch backend, got backend={args.backend!r}"

    # Initialise the process group (and pick this rank's device) before the
    # models import builds any tensors.
    rank, world, device, initialised = init_distributed(args.device)

    # Surface yastn's diagnostic logging (e.g. the dist per-rank combo/LPT-load
    # line) on stdout so it is captured in each rank's stdout.log (torchrun -r 3).
    configure_logging(rank, world, args.log_level)

    # import models here to set num_threads before importing backends
    import models
    model_cls = getattr(models, args.model)

    fnames = glob.glob(os.path.join(HERE, "input_shapes/", args.fname + '.json'))
    fnames = [Path(fname) for fname in sorted(fnames)]
    if len(fnames) == 0:
        if rank == 0:
            print(f"No input files found for pattern {args.fname} in "
                  f"{os.path.join(HERE, 'input_shapes/')}")
        teardown_distributed(initialised)
        sys.exit(1)

    try:
        for fname in fnames:
            run_bench_dist(model_cls, args, fname, rank, world, device)
    finally:
        teardown_distributed(initialised)


if __name__ == "__main__":
    main()
