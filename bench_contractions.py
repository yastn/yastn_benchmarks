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
import contextlib
import glob
import os
from pathlib import Path
import re
import sys
import timeit


def fname_output(model, fname, config):
    fpath = os.path.dirname(__file__)
    path = Path(f"{fpath}/results/{model}/lru_cache={config['lru_cache']}/{config['backend']}/{config['device']}")
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{fname.stem}.out"


def run_bench(model, fname, config, repeat, to_file):
    """
    Run a single benchmark and output results to file or to stdout
    """
    bench = models[model](fname, config)

    target = fname_output(model, fname, config) if to_file else None

    with contextlib.ExitStack() as stack:
        f = stack.enter_context(open(target, 'w')) if target else sys.stdout

        bench.print_header(file=f)

        for task in bench.bench_pipeline:
            try:
                times = timeit.repeat(stmt='bench.' + task + '()', repeat=repeat, number=1, globals=locals())
            except AssertionError:
                print("Model too large to execute (check conditions in /models/model_parent.py)", file=f)
                return None
            print(task + "; times [seconds]", file=f)
            print(*(f"{t:.4f}" for t in times), file=f)

        bench.print_properties(file=f)
        bench.final_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-backend", type=str, default='np', choices=['np', 'torch'])
    parser.add_argument("-device", type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument("-no_lru_cache", dest='lru_cache', action='store_false', help="Yastn is using lru_cache to back up algebra of symmetries. Use this option to switch it off.")
    parser.add_argument("-stdout", dest='to_file', action='store_false', help="By default, write results to files in /results; Use this option to print to stdout.")
    parser.add_argument("-repeat", type=int, default=4)
    parser.add_argument("-fname", type=str, default='*', help="Use glob to match basenames of json files in ./input_shapes")
    parser.add_argument("-model", type=str, default='Ctm', help="Use 'args.model in model_class_name' to select models")
    parser.add_argument("-num_threads", type=str, default='none', choices=['none'] + [str(n) for n in range(1, 33)])
    args = parser.parse_args()

    if args.num_threads != 'none':
        os.environ["OMP_NUM_THREADS"] = args.num_threads
        os.environ["OPENBLAS_NUM_THREADS"] = args.num_threads
        os.environ["MKL_NUM_THREADS"] = args.num_threads
        os.environ["VECLIB_MAXIMUM_THREADS"] = args.num_threads
        os.environ["NUMEXPR_NUM_THREADS"] = args.num_threads

    # import models here to set num_threads before importing backends
    from models import CtmBenchYastnBasic, CtmBenchYastnDL, CtmBenchYastnfpeps, CtmBenchYastnPrecompute
    models = {"CtmBenchYastnDL": CtmBenchYastnDL,
              "CtmBenchYastnBasic": CtmBenchYastnBasic,
              "CtmBenchYastnfpeps": CtmBenchYastnfpeps,
              "CtmBenchYastnPrecompute": CtmBenchYastnPrecompute}

    # identify models and input files to run
    use_models = [model for model in models if args.model in model]
    fnames = glob.glob(os.path.join(os.path.dirname(__file__), "input_shapes/", args.fname + '.json'))
    fnames = [Path(fname) for fname in fnames]

    config = {"backend": args.backend, "device": args.device, "lru_cache": args.lru_cache}

    # execute benchmarks
    for fname in fnames:
        for model in use_models:
            run_bench(model, fname, config, args.repeat, args.to_file)
