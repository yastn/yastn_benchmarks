
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
""" Parent class specifying contractions for benchmarks. """
import abc
import json
import yastn

double_layer_D_limit = {"dense": 16, "Z2": 18, "U1": 20, "U1xU1": 25, "U1xU1xZ2": 25}


def nvtx(func):
    def wrapper(self, *args, **kwargs):
        if self.use_nvtx:
            self.config.backend.cuda.nvtx.range_push(f"{type(self).__name__} {func.__name__}")
        res = func(self, *args, **kwargs)
        if self.use_nvtx:
            self.config.backend.cuda.nvtx.range_pop()
        return res
    return wrapper

class CtmBenchParent(metaclass=abc.ABCMeta):

    def __init__(self, fname, config, **kwargs):
        """
        Read tensors legs and other information into dictionary self.input
        """

        self.bench_pipeline = []
        self.params = {}

        with open(fname, "r", encoding='utf-8') as f:
            self.input = json.load(f)

        if not config["lru_cache"]:
            yastn.set_cache_maxsize(maxsize=0)
        self.config = yastn.make_config(sym=self.input["symmetry"], **config)
        self.config.backend.random_seed(seed=0)  # makes outputs of different models comparable

        self.use_nvtx = ("torch" in self.config.backend.BACKEND_ID) and self.config.backend.cuda_is_available()

        Ds = [sum(self.input[dirn]["dimensions"]) for dirn in ["a_leg_t", "a_leg_l", "a_leg_b", "a_leg_r"]]
        D4 = Ds[0] * Ds[1] * Ds[2] * Ds[3]
        self.allow_explicit_double_layer = (D4 <= double_layer_D_limit[self.input["symmetry"]] ** 4)

    def print_header(self, file=None):
        print(" No benchmark ", file=file)

    def print_properties(self, file=None):
        print(" Fill-in contractions. ", file=file)

    def final_cleanup(self):
        r""" For operations done after executing benchmarks """
        yastn.clear_cache()  # yastn is using lru_cache to store contraction logic
