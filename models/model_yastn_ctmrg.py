
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
from .model_parent import CtmBenchParent, nvtx
import yastn
import yastn.tn.fpeps as peps


class CtmBenchUpdate(CtmBenchParent):

    def __init__(self, fname, config, **kwargs):
        """ Initialize tensors for contraction. """
        super().__init__(fname, config)
        #
        self.bench_pipeline = ["ctmrg_update"]
        self.params = {'seed': 0,
                       'dims': (2, 2),
                       'method': '2x2',
                       'policy': 'fullrank'}  # default params
        #
        for k in self.params:
            if k in kwargs:
                self.params[k] = kwargs[k]
        
        #
        legs = {k: yastn.Leg(self.config, s=v['signature'], t=v['charges'], D=v['dimensions'])
                for k, v in self.input.items() if "leg" in k}
        legs_a = ["a_leg_t", "a_leg_l", "a_leg_b", "a_leg_r", "a_leg_s", "a_leg_a"]
        legs_a = [legs[k] for k in legs_a if k in legs]
        if len(legs_a) == 6:  # if ancilla leg is present, system and ancilla legs are fused
            legs_a = legs_a[:4] + [yastn.leg_product(legs_a[4], legs_a[5])]
        
        #
        if self.params['dims'][0] % 2 == 1 or self.params['dims'][1] % 2 == 1:
            try:
                self.env = self.init_any_unitcell(legs_a, legs)
            except AssertionError as e:
                print('unit-cell size should be even in both directions for consistency of leg dimensions')
                raise e
        else:
            self.env = self.init_even_unitcell(legs_a, legs)

    def init_onsite_t(self, legs, seed=None):
        # always build via NumPy to get identical initial tensors for all backends and devices
        config_np= yastn.make_config(**{**self.config._asdict(), **{"backend": "np", "default_device":"cpu"}})
        config_np.backend.random_seed(seed=seed if seed else self.params['seed'])
        
        tmp= yastn.rand(config_np, legs=legs)
        res= yastn.zeros(self.config, legs=legs)
        res._data= self.config.backend.to_tensor( tmp._data, device=self.config.default_device, dtype=self.config.default_dtype )
        return res        

    def init_even_unitcell(self, legs_a, legs):
        ls_a = {0: [legs_a[0], legs_a[1], legs_a[2], legs_a[3], legs_a[4]],
                1: [legs_a[2].conj(), legs_a[3].conj(), legs_a[0].conj(), legs_a[1].conj(), legs_a[4]]}
        
        geometry = peps.SquareLattice(dims=self.params['dims'])
        psi = peps.Peps(geometry)

        for site in psi.sites():
            psi[site] = self.init_onsite_t(legs=ls_a[sum(site) % 2],seed=None)

        #
        legs_t = {0: [legs["Tt_leg_l"], legs_a[0].conj(), legs_a[0], legs["Tt_leg_r"]],
                  1: [legs["Tt_leg_r"].conj(), legs_a[2], legs_a[2].conj(), legs["Tt_leg_l"].conj()]}
        legs_r = {0: [legs["Tr_leg_t"], legs_a[3].conj(), legs_a[3], legs["Tr_leg_b"]],
                  1: [legs["Tr_leg_b"].conj(), legs_a[1], legs_a[1].conj(), legs["Tr_leg_t"].conj()]}
        legs_b = {0: [legs["Tt_leg_l"], legs_a[2].conj(), legs_a[2], legs["Tt_leg_r"]],
                  1: [legs["Tt_leg_r"].conj(), legs_a[0], legs_a[0].conj(), legs["Tt_leg_l"].conj()]}
        legs_l = {0: [legs["Tr_leg_t"], legs_a[1].conj(), legs_a[1], legs["Tr_leg_b"]],
                  1: [legs["Tr_leg_b"].conj(), legs_a[3], legs_a[3].conj(), legs["Tr_leg_t"].conj()]}
        legs_tr = {0: [legs["Tt_leg_r"].conj(), legs["Tr_leg_t"].conj()],
                   1: [legs["Tt_leg_l"], legs["Tr_leg_b"]]}
        legs_br = {0: [legs["Tr_leg_b"].conj(), legs["Tt_leg_l"].conj()],
                   1: [legs["Tr_leg_t"], legs["Tt_leg_r"]]}
        legs_bl = {0: [legs["Tt_leg_r"].conj(), legs["Tr_leg_t"].conj()],
                   1: [legs["Tt_leg_l"], legs["Tr_leg_b"]]}
        legs_tl = {0: [legs["Tr_leg_b"].conj(), legs["Tt_leg_l"].conj()],
                   1: [legs["Tr_leg_t"], legs["Tt_leg_r"]]}

        #
        env = peps.EnvCTM(psi, init=None)
        for site in env.sites():
            s2 = sum(site) % 2
            env[site].t = yastn.rand(self.config, legs=legs_t[s2]).fuse_legs(axes=(0, (1, 2), 3))
            env[site].r = yastn.rand(self.config, legs=legs_r[s2]).fuse_legs(axes=(0, (1, 2), 3))
            env[site].b = yastn.rand(self.config, legs=legs_b[s2]).fuse_legs(axes=(0, (1, 2), 3))
            env[site].l = yastn.rand(self.config, legs=legs_l[s2]).fuse_legs(axes=(0, (1, 2), 3))
            env[site].tr = yastn.rand(self.config, legs=legs_tr[s2])
            env[site].br = yastn.rand(self.config, legs=legs_br[s2])
            env[site].bl = yastn.rand(self.config, legs=legs_bl[s2])
            env[site].tl = yastn.rand(self.config, legs=legs_tl[s2])
        #
        assert env.is_consistent()
        return env

    def init_any_unitcell(self, legs_a, legs):
        #
        ls_a = [legs_a[0], legs_a[1], legs_a[2], legs_a[3], legs_a[4]]

        geometry = peps.SquareLattice(dims=self.params['dims'])
        psi = peps.Peps(geometry)

        # iPEPS initialization: 
        # always build via NumPy to get identical initial tensors for all backends and devices
        for site in psi.sites():
            psi[site] = self.init_onsite_t(legs=ls_a,seed=None)

        # Env initialization
        legs_t = [legs["Tt_leg_l"], legs_a[0].conj(), legs_a[0], legs["Tt_leg_r"]]
        legs_r = [legs["Tr_leg_t"], legs_a[3].conj(), legs_a[3], legs["Tr_leg_b"]]
        legs_b = [legs["Tt_leg_l"], legs_a[2].conj(), legs_a[2], legs["Tt_leg_r"]]
        legs_l = [legs["Tr_leg_t"], legs_a[1].conj(), legs_a[1], legs["Tr_leg_b"]]
        legs_tr = [legs["Tt_leg_r"].conj(), legs["Tr_leg_t"].conj()]
        legs_br = [legs["Tr_leg_b"].conj(), legs["Tt_leg_l"].conj()]
        legs_bl = [legs["Tt_leg_r"].conj(), legs["Tr_leg_t"].conj()]
        legs_tl = [legs["Tr_leg_b"].conj(), legs["Tt_leg_l"].conj()]

        #
        env = peps.EnvCTM(psi, init=None)
        for site in env.sites():
            env[site].t = yastn.rand(self.config, legs=legs_t).fuse_legs(axes=(0, (1, 2), 3))
            env[site].r = yastn.rand(self.config, legs=legs_r).fuse_legs(axes=(0, (1, 2), 3))
            env[site].b = yastn.rand(self.config, legs=legs_b).fuse_legs(axes=(0, (1, 2), 3))
            env[site].l = yastn.rand(self.config, legs=legs_l).fuse_legs(axes=(0, (1, 2), 3))
            env[site].tr = yastn.rand(self.config, legs=legs_tr)
            env[site].br = yastn.rand(self.config, legs=legs_br)
            env[site].bl = yastn.rand(self.config, legs=legs_bl)
            env[site].tl = yastn.rand(self.config, legs=legs_tl)
        #
        # assert env.is_consistent()
        return env


    def print_header(self, file=None):
        print(f"Perform ctmrg update in {self.params['dims']} SquareLattice", file=file)

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
    def ctmrg_update(self):
        r""" update """
        v = self.input['Tt_leg_l']
        leg = yastn.Leg(self.config, s=v['signature'], t=v['charges'], D=v['dimensions'])
        opts_svd = {'D_block': leg.tD, 'tol': 1e-12, 'policy': self.params['policy']}
        self.env.update_(opts_svd=opts_svd, method=self.params['method'])

    def final_cleanup(self):
        yastn.clear_cache()  # yastn is using lru_cache to store contraction logic
