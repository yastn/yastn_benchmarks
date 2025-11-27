
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

double_layer_D_limit = {"dense": 16, "Z2": 18, "U1": 20, "U1xU1": 25, "U1xU1xZ2": 25}

class CtmBenchParent(metaclass=abc.ABCMeta):

    def __init__(self, fname, *args):
        """
        Read tensors legs and other information into dictionary self.input
        """

        self.bench_pipeline = ["enlarged_corner_ctm",
                               "enlarged_corner",
                               "fuse_enlarged_corner",
                               "svd_enlarged_corner"]

        with open(fname, "r", encoding='utf-8') as f:
            self.input = json.load(f)

        self.tensors = {}

        Ds = [sum(self.input[dirn]["dimensions"]) for dirn in ["a_leg_t", "a_leg_l", "a_leg_b", "a_leg_r"]]
        D4 = Ds[0] * Ds[1] * Ds[2] * Ds[3]
        self.allow_explicit_double_layer = (D4 <= double_layer_D_limit[self.input["symmetry"]] ** 4)

    def print_header(self, file=None):
        print(" No benchmark ", file=file)

    def print_properties(self, file=None):
        print(" Fill-in contractions. ", file=file)

    def enlarged_corner(self):
        r"""
        Contract the network

        -----Tt---Ctr
            / |    |
           |  |    |
        ---a--|----Tr
           |\ |   /|
        ---|--a*-/ |
           |  |    |
        """
        self.tensors["C2x2tr"] = None

    def enlarged_corner_ctm(self):
        r"""
        Contract the network

        -----Tt---Ctr
            / |    |
           |  |    |
        ---a--|----Tr
           |\ |   /|
        ---|--a*-/ |
           |  |    |
        """
        self.tensors["C2x2tr"] = None

    def fuse_enlarged_corner(self):
        r"""
        From block-sparse tensor to block-sparse matrix

        (0----C2x2  -> 0--C2x2
         1----|   |       |
         2)---|___|       1
              | | |
             (4 5 3)

        """
        self.tensors["C2x2mat"] = None

    def svd_enlarged_corner(self):
        r"""
        Perform svd of block-sparse matrix
        """
        self.tensors["U"], self.tensors["S"], self.tensors["V"] = None, None, None

    def final_cleanup(self):
        r""" For operations done after executing benchmarks """
        pass

