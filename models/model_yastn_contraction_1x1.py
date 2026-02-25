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


class CtmBenchContraction1x1(CtmBenchContractionParent):

    def __init__(self, fname, config, fuse=False, **kwargs):
        r"""
        Contract the network::

            C1--(1)1 1(0)----T1--(3)9 9(0)----C2
            0(0)            (1,2)             8(1)
            0(0)        100  2  5             8(0)
            |              \ 2  5              |
            T4--(2)3 3-------a--|---10 10(1)---T2
            |                |  |              |
            |   (3)6 6----------a*--11 11(2)   |
            14(1)           15 16 \101        17(3)
            14(0)           (0,1)             17(0)
            C4--(1)12 12(2)--T3--(3)13 13(1)--C3

        """
        # This network contains 3 unique legs D, X, p
        # The signatures are adjusted at creation as neccessary to match.
        #
        # Every label can appear at most twice (contracted leg pair) otherwise, the label is corresponsing to an open leg.  
        super().__init__(fname, config, **kwargs)

        self.idx_labels= { 
            "a_leg_t": [2,3,16,10, 5,6,15,11],
            "Tt_leg_l": [0,14, 12,13, 17,8, 9,1],
            "a_leg_s": [100,101] 
        }
        self._gen_network_spec = lambda I,I_out : ("C1",[0,1],"T1",[1,2,5,9],"T4",[0,14,3,6], \
            "C2",[9,8],"T2",[8,10,11,17],"C3",[17,13],"T3",[15,16,12,13],"C4",[14,12], \
            "a",[I[0],2,3,15,10],"a.conj()",[I[1],5,6,16,11],I_out)

        tensor_ids, inputs, output, legs_dict = self.build_network(self.legs)
        
        self.tensors= self.make_tensors(tensor_ids, inputs, legs_dict)
        self.tn= sum(zip([self.tensors[t_id] for t_id in tensor_ids], inputs) , ()) + (tuple(output) if len(output)>0 else ((),))
        
        self.path, self.path_info= self.compute_contraction_path(*self.tn, names=tuple(tensor_ids), optimizer="default") # dynamic-programming


    def build_network(self, legs, open_idx=[]):
        """
        """
        # optionally allows to keep some physical indices open 
        assert set(open_idx) <= set([0,]), "open_idx should be a subset of [0,]"
        I= sum([[100+2*x,100+2*x+1] if x in open_idx else [100+2*x]*2 for x in [0,]],[])
        I_out= [100+2*x for x in open_idx]+[100+2*x+1 for x in open_idx]

        assert set(self.idx_labels.keys()) <= set(legs.keys()), \
            f"Legs have to provided for every unique leg in the network. Expected {set(self.idx_labels.keys())}, got {set(legs.keys())}"
        tn= self._gen_network_spec(I,I_out)

        tensor_ids = [tn[i] for i in range(0,len(tn)-1,2)]
        inputs = [tuple(tn[i]) for i in range(1,len(tn)-1,2)]
        output = tuple(tn[-1])
        legs_dict = { idx: legs[k] for k in self.idx_labels for idx in self.idx_labels[k] }

        return tensor_ids, inputs, output, legs_dict
    

    def print_header(self, file=None):
        print("Contract 1x1 patch in CTM environment.", file=file)
