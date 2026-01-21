
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
from .model_parent import nvtx
import yastn


class CtmBenchYastnBasicFused(CtmBenchYastnBasic):

    def print_header(self, file=None):
        print("Attach a and a* sequentially; Extra fusions when building enlarged corners.", file=file)

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
        tmp = yastn.einsum('aCEA,AB,BDFd->aCEDFd', Tt, Ctr, Tr, order='AB')
        tmp = yastn.fuse_legs(tmp, axes=(1, 2, (0, 5), 3, 4))
        tmp = yastn.einsum('CExDF,CbeDG,EcfFG->xbcef',
                           tmp, a, a.conj(),
                           order='CDEFG')
        tmp = yastn.unfuse_legs(tmp, axes=0)
        self.tensors["C2x2tr"] = tmp.transpose(axes=(0, 2, 3, 1, 4, 5))
