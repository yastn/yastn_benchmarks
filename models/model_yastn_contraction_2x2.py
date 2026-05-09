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
from .model_yastn_contraction_parent import CtmBenchContractionParent
import yastn


class CtmBenchContraction2x2(CtmBenchContractionParent):

    def __init__(self, fname, config, fuse=False, **kwargs):
        r"""
        Contract the network::

            C1------(1)1 1(0)----T1----(3)36 36(0)----T1_x----(3)18 18(0)----C2_x
            0(0)               (1,2)                 (1,2)                   19(1)
            0(0)           100  2  5             102 20 23                   19(0)
            |                 \ 2  5               \ |  |                    |
            T4-------(2)3 3-----a--|------37 37----a_x--6(1)----21 21(1)-----T2_x
            |                   |  |                 |  |                    |
            |        (3)6 6-------a*------38 38--------a*_x-----24 24(2)     |
            15(1)               16 17 \101          34 35 \103               33(3)
            15(0)          104  16 17           106 34 35                    33(0)
            |                 \ |   |              \ |  |                    |
            T4_y--(2)9 9--------a_y-------39 39-----a_xy--------28 28(1)-----T2_xy
            |                   |   |                |  |                    |
            |     (3)12 12---------a*_y---40 40------- a*_xy----31 31(2)     |
            |                   10 13 \105           29 32 \107              |
            8(1)                10 13                29 32                  26(3)
            8(0)                (0,1)                (0,1)                  26(0)
            C4_y---(1)7 7(2)-----T3_y--(3)41 41(2)----T3_xy---(3)27 27(1)----C3_xy

        """
        # This network contains 3 unique legs D, X, p
        # The signatures are adjusted at creation as neccessary to match.
        #
        # Every label can appear at most twice (contracted leg pair) otherwise, the label is corresponsing to an open leg.  
        super().__init__(fname, config, **kwargs)
        self.params['open_idx']= kwargs.get('open_idx', [])

        self._gen_network_spec = lambda I,I_out : ("C1",[0,1],"T1",[1,2,5,36],"T4",[0,15,3,6],
            "a",[I[0],2,3,16,37],"a.conj()",[I[1],5,6,17,38],"T4_y",[15,8,9,12],"C4_y",[8,7],"T3_y",[10,13,7,41],
            "a_y",[I[4],16,9,10,39],"a_y.conj()",[I[5],17,12,13,40],"T1_x",[36,20,23,18],"C2_x",[18,19],"T2_x",[19,21,24,33],
            "a_x",[I[2],20,37,34,21],"a_x.conj()",[I[3],23,38,35,24],"T2_xy",[33,28,31,26],"C3_xy",[26,27],"T3_xy",[29,32,41,27],
            "a_xy",[I[6],34,39,29,28],"a_xy.conj()",[I[7],35,40,32,31],I_out)

        tensor_ids, inputs, output = self.build_network(open_idx=self.params['open_idx'])
        
        self.tensors= self.make_tensors_peps_torch_tn_convention(tensor_ids, inputs, self.legs)
        self.tn= sum(zip([self.tensors[t_id] for t_id in tensor_ids], inputs) , ()) + ((tuple(output),) if len(output)>0 else ((),))
        
        self.path, self.path_info= self.compute_contraction_path(*self.tn, names=tuple(tensor_ids), optimizer="default") # dynamic-programming


    def build_network(self, open_idx=[]):
        """
        """
        # optionally allows to keep some physical indices open 
        assert set(open_idx) <= set([0,1,2,3]), "open_idx should be a subset of [0,1,2,3]"
        I= sum([[100+2*x,100+2*x+1] if x in open_idx else [100+2*x]*2 for x in [0,1,2,3]],[])
        I_out= [100+2*x for x in open_idx]+[100+2*x+1 for x in open_idx]

        tn= self._gen_network_spec(I,I_out)

        tensor_ids = [tn[i] for i in range(0,len(tn)-1,2)]
        inputs = [tuple(tn[i]) for i in range(1,len(tn)-1,2)]
        output = tuple(tn[-1])

        return tensor_ids, inputs, output
    

    def print_header(self, file=None):
        print("Contract 2x2 patch in CTM environment.", file=file)
