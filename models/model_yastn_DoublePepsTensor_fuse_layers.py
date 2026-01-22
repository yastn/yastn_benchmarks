
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
from .model_yastn_DoublePepsTensor import CtmBenchYastnDoublePepsTensor
from .model_parent import nvtx
import yastn


class CtmBenchYastnDoublePepsTensorFuseLayers(CtmBenchYastnDoublePepsTensor):

    def __init__(self, *args, **kwargs):
        """ Initialize tensors for contraction. """
        super().__init__(*args, **kwargs)

        self.bench_pipeline = ["precompute_A_mat",
                               "enlarged_corner",
                               "fuse_enlarged_corner",
                               "svd_enlarged_corner"]

    def print_header(self, file=None):
        print("Attach a and a* sequentially; Fusion of some legs in the input and intermediate tensors; Used in yastn.tn.fpeps.", file=file)

    @nvtx
    def precompute_A_mat(self):
        assert self.allow_explicit_double_layer
        self.tensors["Af"] = self.tensors["A"].fuse_layers()

    @nvtx
    def enlarged_corner(self):
        r"""
        Contract the network

        (0)-----Tt---Ctr
                /|    |
        (1)=----a|---=Tr
            \---|a*-/ |
                \|    |
                (3)  (2)
        """
        A, Tt, Tr, Ctr = [self.tensors[k] for k in ["Af", "Tt", "Tr", "Ctr"]]
        self.tensors["C2x2tr"] = yastn.tensordot(Tt @ (Ctr @ Tr), A, axes=((1, 2), (0, 3)))
