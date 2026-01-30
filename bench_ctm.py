# Copyright 2024 The YASTN Authors. All Rights Reserved.
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
import argparse
import ast
import contextlib
import glob
import os
from pathlib import Path
import sys
import timeit
import tracemalloc

def readable_size(size):
    units = ('KB', 'MB', 'GB', 'TB')
    size_list = [f'{int(size):,} B'] + [f'{int(size) / 1024 ** (i + 1):,.2f} {u}' for i, u in enumerate(units)]
    return [size for size in size_list if not size.startswith('0.')][-1]

def fname_output(bench, fname, args):
    if args.to_file is False:
        return None
    fpath = os.path.dirname(__file__)
    device = args.device.replace(":", "-")
    ss = f"{fpath}/results_ctm/{type(bench).__name__}/"
    if bench.params:
        ss += '_'.join(f"{k}={v}" for k, v in sorted(bench.params.items())) + '/'
    ss += f"{args.dtype}/num_threads={args.num_threads}/policy={args.tensordot_policy}/lru_cache={args.lru_cache}/{args.backend}/{device}"
    path = Path(ss)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{fname.stem}.out"

def run_bench(model, args):
    """
    Run a single benchmark and output results to file or to stdout
    """
    config = {"backend": args.backend, "default_device": args.device, "default_dtype": args.dtype,
              "lru_cache": args.lru_cache, "tensordot_policy": args.tensordot_policy}
    #
    expr = ast.parse(f"dict({args.params}\n)", mode="eval")
    kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in expr.body.keywords}
    #
    bench = model(fname, config, **kwargs)
    #
    fname_out = fname_output(bench, fname, args)

    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(fname_out, 'w')) if fname_out else sys.stdout
        bench.params.update({'f_out': f})

        bench.print_header(file=f)

        tasks = bench.bench_pipeline
        if len(args.pipeline) > 0 and not ("all" in args.pipeline):
            assert all([(t in bench.bench_pipeline) for t in args.pipeline]), \
                f"Some provided pipeline tasks are not in the model's benchmark pipeline. {tasks}"
            tasks = [t for t in bench.bench_pipeline if (t in args.pipeline)]
        print(f"Model = {type(bench).__name__}; fname = {fname.name}")
        print(f"Selected pipeline tasks to run: {tasks}")
        for task in tasks:
            try:
                times = timeit.repeat(stmt=f'bench.{task}()', repeat=args.repeat, number=1, globals=locals())
            except AssertionError:
                print("Model too large to execute (check conditions in /models/model_parent.py)", file=f)
                return None
            print(task + "; times [seconds]", file=f)
            print(*(f"{t:.4f}" for t in times), file=f)
            if args.memory_profile:
                tracemalloc.start()
                current, peak =  tracemalloc.get_traced_memory()
                print(f"memory: {readable_size(current)}, {readable_size(peak)}", file=f)
                tracemalloc.stop()

        bench.print_properties(file=f)
        bench.final_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-backend", type=str, default='np', choices=['np', 'torch', 'torch_cpp'])
    parser.add_argument("-dtype", type=str, default='float64', choices=['float32', 'float64', 'complex64', 'complex128'])
    parser.add_argument("-device", type=str, default='cpu', help="cpu, cuda, cuda:<device_id>, etc.")
    parser.add_argument("-tensordot_policy", type=str, default='no_fusion', choices=['fuse_to_matrix', 'fuse_contracted', 'no_fusion'])
    parser.add_argument("-no_lru_cache", dest='lru_cache', action='store_false', help="Yastn is using lru_cache to back up algebra of symmetries. Use this option to switch it off.")
    parser.add_argument("-stdout", dest='to_file', action='store_false', help="By default, write results to files in /results; Use this option to print to stdout.")
    parser.add_argument("-memory_profile", dest='memory_profile', action='store_true', help="Profile memory usage with tracemalloc. High overhead.")
    parser.add_argument("-repeat", type=int, default=4, help='Number of repeated runs; passed to timeit')
    parser.add_argument("-fname", type=str, default='Heisenberg_U1_d=2_D=4_chi=30', help="Use glob to match basenames of json files in ./input_shapes")
    parser.add_argument("-model", type=str, default='Ctm', help="Use 'args.model in model_class_name' to select models")
    parser.add_argument("-params", type=str, default='', help="Model-specific parameters, e.g. 'dims=(2,2)' for BenchCtmUpdate")
    parser.add_argument(
        "-pipeline",
        nargs="*",
        choices=["all", "precompute_A_mat", "enlarged_corner", "fuse_enlarged_corner", "svd_enlarged_corner", "ctmrg_update"],
        default=["all"],
        help="Pipeline steps to run (any combination of the choices); provide multiple values separated by space.",
    )
    parser.add_argument("-num_threads", type=str, default='none', choices=['none'] + [str(n) for n in range(1, 33)])
    args = parser.parse_args()

    if args.num_threads != 'none':
        os.environ["OMP_NUM_THREADS"] = args.num_threads
        os.environ["OPENBLAS_NUM_THREADS"] = args.num_threads
        os.environ["MKL_NUM_THREADS"] = args.num_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = args.num_threads
        os.environ["NUMEXPR_NUM_THREADS"] = args.num_threads

    # import models here to set num_threads before importing backends
    from models import CtmBenchYastnBasic, CtmBenchYastnDoublePepsTensor, CtmBenchYastnDoublePepsTensorFuseLayers, CtmBenchUpdate, \
        CtmBenchYastnBasicFused, CtmBenchUpdateMP
    models = {"CtmBenchYastnBasic": CtmBenchYastnBasic,
              "CtmBenchYastnBasicFused": CtmBenchYastnBasicFused,
              "CtmBenchYastnDoublePepsTensor": CtmBenchYastnDoublePepsTensor,
              "CtmBenchYastnDoublePepsTensorFuseLayers": CtmBenchYastnDoublePepsTensorFuseLayers,
              "CtmBenchUpdate": CtmBenchUpdate,
              "CtmBenchUpdateMP": CtmBenchUpdateMP}

    # identify models and input files to run
    use_models = [model for model in models if args.model in model]
    fnames = glob.glob(os.path.join(os.path.dirname(__file__), "input_shapes/", args.fname + '.json'))
    fnames = [Path(fname) for fname in sorted(fnames)]

    if len(fnames) == 0:
        print(f"No input files found for pattern {args.fname} in {os.path.join(os.path.dirname(__file__), 'input_shapes/')}")
        sys.exit(1)

    # execute benchmarks
    for fname in fnames:
        for model in use_models:
            run_bench(models[model], args)
