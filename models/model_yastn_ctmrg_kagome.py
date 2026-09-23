
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
""" CTMRG benchmark on optimized dense iPEPS of J1-JD model on Kagome lattice from j1jD_kagome_ipeps_states dataset. """
from __future__ import annotations
from pathlib import Path
import re
import sys
from .model_yastn_ctmrg import CtmBenchUpdate
import yastn
import yastn.tn.fpeps as peps


class CtmBenchUpdateKagome(CtmBenchUpdate):

    input_dir = "j1jD_kagome_ipeps_states"

    def __init__(self, fname, config, **kwargs):
        """ Initialize translationally-invariant iPEPS from single-site state and its 'dl' environment. """
        kwargs.setdefault('dims', (1, 1))  # the state is single-site; 1x1 unit cell is the default
        super().__init__(fname, config, **kwargs)
        #
        self.bench_pipeline = ["ctmrg_full"]
        if self.params['chi'] is None:  # default to environment dimension used in the optimization of the state
            m = re.search(r"_chi_opt(\d+)", Path(fname).name)
            if m is None:
                raise ValueError(f"Cannot infer chi from {Path(fname).name}. Provide it via -params 'chi=...'")
            self.params['chi'] = int(m.group(1))

    def read_input(self, fname):
        r"""
        Read dense on-site tensor a[s,u,l,d,r], where the physical index s of dimension 8 enumerates
        states of the three spin-1/2's on an up-pointing triangle of the Kagome lattice.
        Legs are stored in the format of input_shapes, with the signature of the tensor
        built in :meth:`init_onsite_t`.
        """
        # ipeps_io parses sys.argv at import time; hide arguments of bench_ctm.py
        argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            from j1jD_kagome_ipeps_states.ipeps_io import load_peps_from_json_dense
        finally:
            sys.argv = argv

        a = load_peps_from_json_dense(fname)

        res = {"symmetry": "dense", "A": a}
        for k, D, s in zip(["a_leg_s", "a_leg_t", "a_leg_l", "a_leg_b", "a_leg_r"], a.shape, (-1, -1, -1, 1, 1)):
            res[k] = {"signature": s, "charges": [], "dimensions": [D]}
        return res

    def init_onsite_t(self, legs=None, seed=None):
        a = yastn.Tensor(config=self.config, s=(-1, -1, -1, 1, 1))  # [s,u,l,d,r]
        a.set_block(ts=(), Ds=self.input['A'].shape, val=self.input['A'])
        return a.to(device=self.config.default_device, dtype=self.config.default_dtype)

    def init_even_unitcell(self, legs_a, legs):
        r""" The state is translationally invariant; tile any unit cell with the same tensor. """
        return self.init_any_unitcell(legs_a, legs)

    def init_any_unitcell(self, legs_a, legs):
        r""" Tile square lattice with a single non-symmetric (dense) on-site tensor. """
        # [s,u,l,d,r] -> [t,l,b,r,s]
        a = self.init_onsite_t().transpose(axes=(1, 2, 3, 4, 0))
        assert a.get_legs() == tuple(legs_a)

        geometry = peps.SquareLattice(dims=self.params['dims'])
        psi = peps.Peps(geometry)
        for site in psi.sites():
            psi[site] = a

        #
        env = peps.EnvCTM(psi, init=None)
        env.reset_(init='dl')
        assert env.is_consistent()
        return env

    def print_properties(self, file=None):
        print("CtmBenchUpdateKagome properties", file=file)
        super().print_properties(file=file)
