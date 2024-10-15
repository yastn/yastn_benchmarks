
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
from .model_yastn_basic import CtmBenchYastnBasic
import yastn.tn.fpeps as fpeps


class CtmBenchYastnfpeps(CtmBenchYastnBasic):

    def __init__(self, *args):
        """ Initialize tensors for contraction. """
        super().__init__(*args)
        self.tensors["a"] = self.tensors["a"].fuse_legs(axes=((1, 2), (3, 4), 0))
        self.tensors["Tt"] = self.tensors["Tt"].fuse_legs(axes=(0, (1, 2), 3))
        self.tensors["Tr"] = self.tensors["Tr"].fuse_legs(axes=(0, (1, 2), 3))
        self.tensors["A"] = fpeps.DoublePepsTensor(self.tensors["a"], self.tensors["a"])

    def print_header(self, file=None):
        print("Attach a and a* sequentially; Fusion of some legs in input and intermidiate tensors; Used in yastn.tn.fpeps.", file=file)

    def enlarged_corner(self):
        """
        Contract the network

        (0)-----Tt---Ctr
                /|    |
        (1)=----a|---=Tr
            \---|a*-/ |
                \|    |
                (3)  (2)
        """
        A, Tt, Tr, Ctr = [self.tensors[k] for k in ["A", "Tt", "Tr", "Ctr"]]
        self.tensors["C2x2tr"] = A._attach_30(Tt @ (Ctr @ Tr))

    def fuse_enlarged_corner(self):
        self.tensors["C2x2mat"] = self.tensors["C2x2tr"].fuse_legs(axes=((0, 1), (2, 3)))
