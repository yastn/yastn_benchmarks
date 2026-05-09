# Copyright 2026 The YASTN Authors. All Rights Reserved.
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
from __future__ import annotations
from .model_parent import nvtx
from .model_yastn_contraction_parent import CtmBenchContractionParent
import yastn


class CtmBenchContraction2x3(CtmBenchContractionParent):

    def __init__(self, fname, config, **kwargs):
        r"""
        # C1------(1)1 1(0)----T1----(3)44 44(0)----T1_x----(3)39 39(0)---T1_2x---(3)24 24(0)--C2_2x
        # 0(0)               (1,2)                 (1,2)                  (1,2)                25(1)
        # 0(0)           100  2  5             102 40 42              104 26 28                25(0)
        # |                 \ 2  5               \ |  |                  \ |  |                 |
        # T4-------(2)3 3-----a--|------45 45----a_x--6(1)----41  41-------a_2x-------27 27(1)--T2_2x
        # |                   |  |                 |  |                    |  |                 |
        # |        (3)6 6-------a*------46 46--------a*_x-----43  43----------a*_2x---29 29(2)  |
        # 15(1)               16 17 \101          47 48 \103               37 38 \105          36(3)
        # 15(0)          106  16 17           108 47 48                    37 38               36(0)
        # |                 \ |   |              \ |  |               110\ |  |                 |
        # T4_y--(2)9 9--------a_y-------20 20-----a_xy--------49 49(1)----a_2xy-------33 33(1)--T2_2xy
        # |                   |   |                |  |                    |  |                 |
        # |     (3)12 12---------a*_y---22 22------- a*_xy----50 50(2)--------a*_2xy--35 35(2)  |
        # |                   10 13 \107           21 23 \109             32 34 \111            |
        # 8(1)                10 13                21 23                  32 34                 31(3)
        # 8(0)                (0,1)                (0,1)                  (0,1)                 31(0)
        # C4_y---(1)7 7(2)-----T3_y--(3)19 19(2)----T3_xy---(3)51 51(2)---T3_2xy--(3)30 30(1)---C3_2xy
        """
        # This network contains 3 unique legs D, X, p
        # The signatures are adjusted at creation as neccessary to match.
        #
        # Every label can appear at most twice (contracted leg pair) otherwise, the label is corresponsing to an open leg.  
        super().__init__(fname, config, **kwargs)
        self.params['open_idx']= kwargs.get('open_idx', [])

        self._gen_network_spec = lambda I,I_out : (
            "C1",[0,1],"T1",[1,2,5,44],"T4",[0,15,3,6],"a",[I[0],2,3,16,45],"a.conj()",[I[1],5,6,17,46],\
            "T4_y",[15,8,9,12],"C4_y",[8,7],"T3_y",[10,13,7,19],"a_y",[I[6],16,9,10,20],"a_y.conj()",[I[7],17,12,13,22],\
            "T3_xy",[21,23,19,51],"a_xy",[I[8],47,20,21,49],"a_xy.conj()",[I[9],48,22,23,50],\
            "T1_2x",[39,26,28,24],"C2_2x",[24,25],"T2_2x",[25,27,29,36],"a_2x",[I[4],26,41,37,27],"a_2x.conj()",[I[5],28,43,38,29],\
            "T2_2xy",[36,33,35,31],"C3_2xy",[31,30],"T3_2xy",[32,34,51,30],"a_2xy",[I[10],37,49,32,33],"a_2xy.conj()",[I[11],38,50,34,35],\
            "T1_x",[44,40,42,39],"a_x",[I[2],40,45,47,41],"a_x.conj()",[I[3],42,46,48,43],I_out)

        # self.input.items() # contains user supplied legs
        tensor_ids, inputs, output = self.build_network(open_idx=self.params['open_idx'])

        self.tensors= self.make_tensors_peps_torch_tn_convention(tensor_ids, inputs, self.legs)
        self.tn= sum(zip([self.tensors[t_id] for t_id in tensor_ids], inputs) , ()) + (tuple(output) if len(output)>0 else ((),))
        
        self.path, self.path_info= self.compute_contraction_path(*self.tn, names=tuple(tensor_ids), optimizer="default") # dynamic-programming


    def build_network(self, open_idx=[]):
        """
        """
        # optionally allows to keep some physical indices open 
        assert set(open_idx) <= set([0,1,2,3,4,5]), "open_idx should be a subset of [0,...,5]"
        I= sum([[100+2*x,100+2*x+1] if x in open_idx else [100+2*x]*2 for x in [0,1,2,3,4,5]],[])
        I_out= [100+2*x for x in open_idx]+[100+2*x+1 for x in open_idx]

        tn= self._gen_network_spec(I,I_out)

        tensor_ids = [tn[i] for i in range(0,len(tn)-1,2)]
        inputs = [tuple(tn[i]) for i in range(1,len(tn)-1,2)]
        output = tuple(tn[-1])

        return tensor_ids, inputs, output
    

    def print_header(self, file=None):
        print("Contract 2x3 patch in CTM environment.", file=file)
