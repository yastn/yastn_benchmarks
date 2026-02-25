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
from .model_parent import CtmBenchParent, nvtx
import yastn

class CtmBenchContractionParent(CtmBenchParent):

    def __init__(self, fname, config, **kwargs):
        r"""
        Contract a network::

            ... network specification to be added ...

        """
        super().__init__(fname, config)
        self.bench_pipeline = ["contract",]
        self.params = {'seed': 0,
                       'dense': False }  # default params
        for k in self.params:
            if k in kwargs:
                self.params[k] = kwargs[k]

        # self.input.items() # contains user supplied legs
        self.legs = {k: yastn.Leg(self.config, s=v['signature'], t=v['charges'], D=v['dimensions'])
                for k, v in self.input.items() if "leg" in k}
        if self.params['dense']: 
            self.config= self.config._replace(sym=yastn.sym.sym_none)
            for k in self.legs:
                self.legs[k]= yastn.Leg(s=self.legs[k].s, t=(), D=(sum(self.legs[k].D),))
    

    def make_tensors(self, tensor_ids, inputs, legs_dict, **kwargs):
        """
        Create tensors according to the network specification and supplied legs.
        The signature adjustment for contracted leg pairs is done on the fly.
        """
        tensors= {}
        label_counts={}
        for t_id, leg_labels in zip(tensor_ids, inputs):
            if t_id in tensors:
                # TODO treat .conj()
                continue # tensor already created
            t_legs = []
            for l in leg_labels:
                if l not in label_counts:
                    label_counts[l]=1
                else:
                    label_counts[l]+=1
                if label_counts[l]==2: # contracted leg pair
                    t_legs.append(legs_dict[l].conj()) # adjust signature for the second leg in the pair
                elif label_counts[l]==1:
                    t_legs.append(legs_dict[l])
                else:
                    raise ValueError(f"Label {l} appears more than twice in the network specification, which is not allowed.")
            # TODO other than uniform random tensors
            tensors[t_id] = yastn.rand(self.config, legs=t_legs)
        return tensors


    def compute_contraction_path(self, *tn, names=None, **kwargs):
        path, path_info = yastn.tensor.oe_blocksparse.get_contraction_path(*tn, names=names, 
                                                         who=self.__class__.__name__, **kwargs)
        return path, path_info


    def print_header(self, file=None):
        print("Contract 1x1 patch in CTM environment.", file=file)


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
        print(self.path_info, file=file)

        print("", file=file)
        for k, v in self.tensors.items():
            print(f"{k} tensor properties:", file=file)
            v.print_properties(file=file)

    @nvtx
    def contract(self):
        self.tensors["result"] = yastn.tensor.oe_blocksparse.contract_with_unroll(
                *self.tn, optimize=self.path, who=self.__class__.__name__
            )
