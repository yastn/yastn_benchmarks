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
                       'dense': False,
                       'checkpoint_loop': False,
                       'unroll': None,
                       'optimizer': "default",
                       'devices': None}  # default params
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


    def make_tensors_simple(self, tensor_ids, inputs, legs_dict, **kwargs):
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


    def make_tensors_peps_torch_tn_convention(self, tensor_ids, inputs, legs_dict, **kwargs):
        """
        Create tensors according to the network specification and supplied legs.
        The signature adjustment for contracted leg pairs is done on the fly.
        """
        tensors= {}
        leg_dir={"1": "t", "2": "r", "3": "b", "4": "l"}

        # loop through the tensors in the network
        # For boundary tensors, signature of env indices follows either clock (-)->(+) or anti-clockwise (+)->(-) order
        for t_id in tensor_ids:
            t_legs = []
            if t_id[:2] in ["T1","T2","T3","T4"]: # edge tensor
                t_dir= t_id[1] # edge tensor
                aux_leg_ind= "a_leg_"+ leg_dir[f"{(int(t_dir)+1)%4+1}"]
                aux_legs= [ legs_dict[aux_leg_ind], legs_dict[aux_leg_ind].conj() ]
                if t_id[1]=="1": # T1 edge tensor
                    t_legs= [ legs_dict["Tt_leg_l"], *aux_legs, legs_dict["Tt_leg_r"] ]
                if t_id[1]=="2": # T2 edge tensor
                    t_legs= [ legs_dict["Tr_leg_t"], *aux_legs, legs_dict["Tr_leg_b"] ]
                if t_id[1]=="3": # T3 edge tensor
                    t_legs= [ *aux_legs, legs_dict["Tt_leg_l"].conj(), legs_dict["Tt_leg_r"].conj() ]
                if t_id[1]=="4": # T4 edge tensor
                    t_legs= [ legs_dict["Tr_leg_t"].conj(),  legs_dict["Tr_leg_b"].conj(), *aux_legs ]
            elif t_id[:2] in ["C1","C2","C3","C4"]: # corner tensor
                if t_id[1]=="1":
                    t_legs= [ legs_dict["Tr_leg_t"], legs_dict["Tt_leg_l"].conj() ]
                if t_id[1]=="2":
                    t_legs= [ legs_dict["Tt_leg_r"].conj(), legs_dict["Tr_leg_t"].conj() ]
                if t_id[1]=="3":
                    t_legs= [ legs_dict["Tr_leg_b"].conj(), legs_dict["Tt_leg_r"] ]
                if t_id[1]=="4":
                    t_legs= [ legs_dict["Tr_leg_b"], legs_dict["Tt_leg_l"] ]
            # otherwise a site tensor
            else:
                t_legs= [ legs_dict["a_leg_s"],] + ( legs_dict["a_leg_a"] if "a_leg_a" in legs_dict else [] ) \
                    + [legs_dict["a_leg_"+leg_dir[t_dir]] for t_dir in ["1","4","3","2"]]
            T = yastn.rand(self.config, legs=t_legs) 
            tensors[t_id]= T.conj() if ("*" in t_id) or ("conj()" in t_id) else T
        
        import pdb; pdb.set_trace()  # for debugging
        
        return tensors


    def compute_contraction_path(self, *tn, names=None, **kwargs):
        path, path_info = yastn.tensor.oe_blocksparse.get_contraction_path(*tn,
                            unroll=self.params['unroll'], names=names,
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
                *self.tn, optimize=self.path, unroll=self.params['unroll'],
                checkpoint_loop=self.params['checkpoint_loop'],
                devices=self.params['devices'],
                who=self.__class__.__name__
            )
        result= float(self.tensors["result"]._data[0]) # force synchronization
