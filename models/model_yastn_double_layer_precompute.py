
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
""" Contractions for benchmarks: yastn with ctm tensors with no legs fused. """
from __future__ import annotations
from .model_yastn_basic import CtmBenchYastnBasic
import yastn


class CtmBenchYastnDLPrecompute(CtmBenchYastnBasic):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bench_pipeline = ["precompute_A_mat",
                               "enlarged_corner",
                               "fuse_enlarged_corner",
                               "svd_enlarged_corner"]

    def print_header(self, file=None):
        print("Form double-layer A tensor on the fly; No fusion in building tensors.", file=file)

    def precompute_A_mat(self):
        assert self.allow_explicit_double_layer
        a = self.tensors["a"]
        tmp = yastn.einsum('Gabcd,Gefgh->abcdefgh', a, a.conj(), order='G')
        tmp = yastn.fuse_legs(tmp, axes=((0, 4, 3, 7), (1, 5, 2, 6)))
        self.tensors["Amat"] = tmp

    def enlarged_corner(self):
        Amat, Tt, Tr, Ctr = [self.tensors[k] for k in ["Amat", "Tt", "Tr", "Ctr"]]
        tmp = yastn.einsum('abcA,AB,Bdef->abcdef', Tt, Ctr, Tr, order="AB")
        tmp = tmp.fuse_legs(axes=((0, 5), (1, 2, 3, 4)))
        tmp = tmp @ Amat
        tmp = tmp.unfuse_legs(axes=(0, 1))
        self.tensors["C2x2tr"] = tmp

    def fuse_enlarged_corner(self):
        self.tensors["C2x2mat"] = self.tensors["C2x2tr"].fuse_legs(axes=((0, 2, 3), (1, 4, 5)))
