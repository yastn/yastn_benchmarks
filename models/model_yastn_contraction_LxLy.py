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


class CtmBenchContractionLxLy(CtmBenchContractionParent):

    def __init__(self, fname, config, **kwargs):
        r"""
        #                 
        #                   o               o+Ly                    o+(Lx-1)*Ly                
        #                   |               |                       |           
        #             0-----a--------1 1----a_x------2  ... Lx-1----a_(Lx-1)--0
        #                   |               |                       |                 
        #                   o+1             o+Ly+1                  o+(Lx-1)*Ly+1
        #                   |               |                       |        
        #           Lx+0----a_y----Lx+1 ----a_xy--Lx+2 ... 2*Lx-1 --a_2xy-----Lx+0
        #                   |               |                       |                   
        #                   o+2             o+Ly+2                  o+(Lx-1)*Ly+2
        #                   ...             ...                     ...
        #                   o+(Ly-1)        o+2*Ly-1                o+Lx*Ly-1
        #                   |                                       | 
        #    Lx*(Ly-1)+0----a*_Ly-1--Lx*(Ly-1)+0 ...      Lx*Ly-1 --a*_Lx-1,Ly-1--Lx*(Ly-1)+0
        #                   |                                       |
        #                   o                                       o+(Lx-1)*Ly
        # 
        """
        # This network contains 3 unique legs D, X, p
        # The signatures are adjusted at creation as neccessary to match.
        #
        # Every label can appear at most twice (contracted leg pair) otherwise, the label is corresponsing to an open leg.  
        super().__init__(fname, config, **kwargs)
        self.params.update({'L': 2, 'Lx': None, 'Ly': None, 'bc': 'pbc'})
        for k in ['L', 'Lx', 'Ly', 'bc']:
            if k in kwargs:
                self.params[k] = kwargs[k]

        if self.params['L']:
            self.params['Lx']= self.params['L']
            self.params['Ly']= self.params['L']
        else:
            assert self.params['Lx'] and self.params['Ly'], "Either L or both Lx (cols) and Ly (rows) have to be provided."

        # self.input.items() # contains user supplied legs
        tensor_ids, inputs, output, legs_dict = self.build_network(self.legs)

        self.tensors= self.make_tensors(tensor_ids, inputs, legs_dict)
        self.tn= sum(zip([self.tensors[t_id] for t_id in tensor_ids], inputs) , ()) + (tuple(output) if len(output)>0 else ((),))
        
        self.path, self.path_info= self.compute_contraction_path(*self.tn, names=tuple(tensor_ids), 
                                        optimizer=self.params['optimizer'])


    def build_network(self, legs, **kwargs):
        """
        """
        Lx= self.params['Lx']
        Ly= self.params['Ly']
        offset_virtual= Lx*Ly
        _gen_row_left= lambda row,col: Lx*row + col % Lx
        _gen_row_right= lambda row,col: Lx*row + (col+1) % Lx
        _gen_row_up= lambda row,col: offset_virtual + Ly*col + row % Ly
        _gen_row_down= lambda row,col: offset_virtual + Ly*col + (row+1) % Ly

        def _gen_network_spec():
            network_spec= []
            for row in range(Ly):
                for col in range(Lx):
                    network_spec+= [
                        f"A_{row}_{col}", [_gen_row_up(row,col), _gen_row_left(row,col), _gen_row_down(row,col), _gen_row_right(row,col)]
                    ]
            return network_spec + [[],]
        self._gen_network_spec = _gen_network_spec

        self.tn= _gen_network_spec()

        self.idx_labels= { 
            "a_leg_t": list(set(sum(self.tn[1::2],[]))),
        }

        assert set(self.idx_labels.keys()) <= set(legs.keys()), \
            f"Legs have to provided for every unique leg in the network. Expected {set(self.idx_labels.keys())}, got {set(legs.keys())}"

        tensor_ids = [self.tn[i] for i in range(0,len(self.tn)-1,2)]
        inputs = [tuple(self.tn[i]) for i in range(1,len(self.tn)-1,2)]
        output = tuple(self.tn[-1])
        legs_dict = { idx: legs[k] for k in self.idx_labels for idx in self.idx_labels[k] }

        return tensor_ids, inputs, output, legs_dict
    

    def print_header(self, file=None):
        print(f"Contract Lx x Ly patch with BC {self.params['bc']}", file=file)
        
        print("", file=file)
        print(self._gen_network_spec(), file=file)
        print("", file=file)
        print(self.path_info, file=file)
        print("", file=file)


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

        if self.config.backend.cuda_is_available():
            if hasattr(self.config.backend, "cutensor_cache_stats"):
                print("", file=file)
                print("cutensor cache stats: "+str(list(self.config.backend.cutensor_cache_stats().values())), file=file)

        print("", file=file)
        for k, v in self.tensors.items():
            print(f"{k} tensor properties:", file=file)
            v.print_properties(file=file)
            break # only print one tensor as all are identical in this model


    @nvtx
    def contract(self):
        self.tensors["result"] = yastn.tensor.oe_blocksparse.contract_with_unroll(
                *self.tn, optimize=self.path, who=self.__class__.__name__
            )