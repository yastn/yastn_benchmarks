
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
from .model_parent import CtmBenchParent, nvtx
import yastn


class CtmBenchYastnBasic(CtmBenchParent):

    def __init__(self, fname, config, fuse=False, **kwargs):
        """ Initialize tensors for contraction. """
        super().__init__(fname, config)

        self.bench_pipeline = ["enlarged_corner",
                               "fuse_enlarged_corner",
                               "svd_enlarged_corner"]

        legs = {k: yastn.Leg(self.config, s=v['signature'], t=v['charges'], D=v['dimensions'])
                for k, v in self.input.items() if "leg" in k}

        legs_a = ["a_leg_s", "a_leg_a", "a_leg_t", "a_leg_l", "a_leg_b", "a_leg_r"]
        legs_a = [legs[k] for k in legs_a if k in legs]
        a = yastn.rand(self.config, legs=legs_a)
        if a.ndim == 6:  # ancilla leg is present
            a = a.fuse_legs(axes=((0, 1), 2, 3, 4, 5))  # system and ancilla legs are fused

        legs_Tt = [legs["Tt_leg_l"], legs["a_leg_t"].conj(), legs["a_leg_t"], legs["Tt_leg_r"]]
        Tt = yastn.rand(self.config, legs=legs_Tt)

        legs_Tr = [legs["Tr_leg_t"], legs["a_leg_r"].conj(), legs["a_leg_r"], legs["Tr_leg_b"]]
        Tr = yastn.rand(self.config, legs=legs_Tr)

        legs_Ctr = [legs["Tt_leg_r"].conj(), legs["Tr_leg_t"].conj()]
        Ctr = yastn.rand(self.config, legs=legs_Ctr)

        self.tensors = {"a": a, "Tt": Tt, "Tr": Tr, "Ctr": Ctr}

    def print_header(self, file=None):
        print("Attach a and a* sequentially; No fusion in building tensors.", file=file)

    def print_properties(self, file=None):
        print("", file=file)
        print("Config:", file=file)
        print("backend:", self.config.backend, file=file)
        print("sym:", self.config.sym, file=file)
        print("default_fusion:", self.config.default_fusion, file=file)
        print("", file=file)
        print("Cache info", file=file)  # auxiliary information from lru_cache
        for rec in yastn.get_cache_info().items():
            print(*rec, file=file)

        if self.config.backend.BACKEND_ID in ["torch_cpp",] and self.config.backend.cuda_is_available():
            print("", file=file)
            print("cutensor cache stats: "+str(list(yastn.backend.backend_torch_cpp.cutensor_cache_stats().values())), file=file)

        print("", file=file)
        for k, v in self.tensors.items():
            print(f"{k} tensor properties:", file=file)
            v.print_properties(file=file)

    @nvtx
    def enlarged_corner(self):
        r"""
        Contract the network

        a(0-)--Tt--(3+)A(0-)--Ctr
              / |              |
             (1+)(2-)         (1+)
             C   E             B
             |   |            (0-)
        b----a---|-----D(1+)---Tr
             | G |            / |
        c----|---a*----F(2-)-/  |
             |   |            (3+)
             e   f              d
        """
        a, Tt, Tr, Ctr = [self.tensors[k] for k in ["a", "Tt", "Tr", "Ctr"]]
        self.tensors["C2x2tr"] = yastn.einsum('aCEA,AB,BDFd,GCbeD,GEcfF->abcdef',
                                            Tt, Ctr, Tr, a, a.conj(),
                                            order='ABCDEFG')

    @nvtx
    def fuse_enlarged_corner(self):
        self.tensors["C2x2mat"] = self.tensors["C2x2tr"].fuse_legs(axes=((0, 1, 2), (3, 4, 5)))

    @nvtx
    def svd_enlarged_corner(self):
        self.tensors["U"], self.tensors["S"], self.tensors["V"] = self.tensors["C2x2mat"].svd()
