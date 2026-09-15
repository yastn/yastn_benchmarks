
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
""" CTMRG benchmark on optimized U(1)-symmetric C4v-A1 iPEPS of J1-J2 model from j1j2_ipeps_states dataset. """
from __future__ import annotations
from pathlib import Path
import re
import sys
from .model_yastn_ctmrg import CtmBenchUpdate
import yastn
import yastn.tn.fpeps as peps


class CtmBenchUpdateJ1J2(CtmBenchUpdate):

    input_dir = "j1j2_ipeps_states"

    def __init__(self, fname, config, **kwargs):
        """ Initialize bipartite iPEPS from single-site C4v-symmetric state and its 'dl' environment. """
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
        Read block-sparse on-site tensor a[s,u,l,d,r] with signature (1, 1, 1, 1, 1) and total charge n.
        Legs are stored in the format of input_shapes, with signature and charges of the A-sublattice tensor,
        see :meth:`init_even_unitcell`.
        """
        # ipeps_io parses sys.argv at import time; hide arguments of bench_ctm.py
        argv, sys.argv = sys.argv, sys.argv[:1]
        try:
            from j1j2_ipeps_states.ipeps_io import load_from_pepstorch_json_blocksparse
        finally:
            sys.argv = argv

        blocks = load_from_pepstorch_json_blocksparse(fname)
        n = sum(next(iter(blocks)))
        tDs = [{} for _ in range(5)]
        for ts, b in blocks.items():
            for tD, t, D in zip(tDs, ts, b.shape):
                tD[t] = D

        res = {"symmetry": "U1", "blocks": blocks, "n": n}
        for k, tD, s in zip(["a_leg_s", "a_leg_t", "a_leg_l", "a_leg_b", "a_leg_r"], tDs, (-1, -1, -1, 1, 1)):
            res[k] = {"signature": s, "charges": [[s * t] for t in sorted(tD)], "dimensions": [tD[t] for t in sorted(tD)]}
        return res

    def init_onsite_t(self, legs=None, seed=None):
        a = yastn.Tensor(config=self.config, s=(1, 1, 1, 1, 1), n=self.input['n'])
        for ts, b in self.input['blocks'].items():
            a.set_block(ts=ts, Ds=b.shape, val=b)
        return a

    def init_even_unitcell(self, legs_a, legs):
        r"""
        Tile square lattice with [[A, B], [B, A]] pattern, following get_bipartite_state of IPEPS_ABELIAN_C4V in peps-torch.
        B-sublattice tensor is rotated by -i\sigma^y on physical index.
        """
        a = self.init_onsite_t()
        # signature [s,u,l,d,r]: [1,1,1,1,1] -> [-1,-1,-1,1,1]
        a0 = a.flip_charges(axes=(0, 1, 2))

        phase_op = yastn.Tensor(config=self.config, s=(-1, 1))
        phase_op.set_block(ts=(1, 1), Ds=(1, 1), val=[[-1.]])
        phase_op.set_block(ts=(-1, -1), Ds=(1, 1), val=[[1.]])

        a1 = a0.flip_signature().switch_signature(axes='all')
        a1 = phase_op @ a1

        # [s,u,l,d,r] -> [t,l,b,r,s]
        ts = {0: a0.transpose(axes=(1, 2, 3, 4, 0)), 1: a1.transpose(axes=(1, 2, 3, 4, 0))}
        assert ts[0].get_legs() == tuple(legs_a)

        geometry = peps.SquareLattice(dims=self.params['dims'])
        psi = peps.Peps(geometry)
        for site in psi.sites():
            psi[site] = ts[sum(site) % 2].to_nonsymmetric() if self.params['dense'] else ts[sum(site) % 2]

        #
        env = peps.EnvCTM(psi, init=None)
        env.reset_(init='dl')
        assert env.is_consistent()
        return env

    def init_any_unitcell(self, legs_a, legs):
        raise AssertionError(f"{type(self).__name__} uses bipartite tiling; unit-cell size should be even in both directions.")