
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
from .model_yastn_ctmrg import CtmBenchUpdate, nvtx
import yastn
import yastn.tn.fpeps as peps


class CtmBenchUpdateParallel(CtmBenchUpdate):

    def __init__(self, fname, config, dims=(2, 2), **kwargs):
        """ Initialize tensors for contraction. """
        super().__init__(fname, config, dims=dims)
        #


    def print_header(self, file=None):
        print(f"Perform parallel ctmrg update in {self.params['dims']} SquareLattice", file=file)

    @nvtx
    def ctmrg_update(self):
        r""" update """
        v = self.input['Tt_leg_l']
        leg = yastn.Leg(self.config, s=v['signature'], t=v['charges'], D=v['dimensions'])
        opts_svd = {'D_block': leg.tD, 'tol': 1e-12}
        self.env.update_(opts_svd=opts_svd)
