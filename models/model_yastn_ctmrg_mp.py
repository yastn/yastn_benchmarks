
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
from yastn.tn.fpeps.envs._env_ctm_dist_mp import iterate_D_

class CtmBenchUpdateMP(CtmBenchUpdate):
    """ 
    Benchmarking class for CTMRG update with single-node multiprocessing parallelization 
    over different sites in the unit cell.

    Parameters
    ----------  

    Can be passed via -params 'param_name=value, ...' command line argument.

    seed: int
        Random seed for tensor initialization. Default is 0.
    max_sweeps : int
        Number of CTMRG update sweeps to perform. Default is 5.
    devices : list of str
        List of devices to use for multiprocessing.
    """
    def __init__(self, fname, config, **kwargs):
        """ Initialize tensors for contraction. """
        super().__init__(fname, config, **kwargs)
        #
        self.bench_pipeline = ["ctmrg_update_mp"]
        self.params.update({'devices': ['cpu','cpu'],
                            'max_sweeps': 5})  # default params
        #
        for k in self.params:
            if k in kwargs:
                self.params[k] = kwargs[k]
        

    def print_header(self, file=None):
        print(f"Perform ctmrg update mp in {self.params['dims']} SquareLattice", file=file)

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

        if self.config.backend.BACKEND_ID in ["torch_cpp",] and self.config.backend.cuda_is_available():
            print("", file=file)
            print("cutensor cache stats: "+str(list(yastn.backend.backend_torch_cpp.cutensor_cache_stats().values())), file=file)

    @nvtx
    def ctmrg_update_mp(self):
        r"""  """
        v = self.input['Tt_leg_l']
        leg = yastn.Leg(self.config, s=v['signature'], t=v['charges'], D=v['dimensions'])
        opts_svd = {'D_total': sum(leg.tD.values()), 'D_block': leg.tD, 'tol': 1e-12, 'policy': self.params['policy']}
        
        # corner_tol=-1 always performs max_sweeps
        iterate_D_(self.env, devices=self.params['devices'],
            opts_svd=opts_svd, method=self.params['method'],
            moves='hv', max_sweeps=self.params['max_sweeps'], 
            iterator=False, corner_tol=-1, truncation_f = None)
        
        X= self.env.calculate_corner_svd()
        import pdb; pdb.set_trace()

    def final_cleanup(self):
        yastn.clear_cache()  # yastn is using lru_cache to store contraction logic
