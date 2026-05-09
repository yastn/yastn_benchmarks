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
from .model_parent import nvtx
import yastn


class CtmBenchContraction2x2Measure(CtmBenchContractionParent):

    def __init__(self, fname, config, **kwargs):
        r"""
        Same 2×2 CTM network as CtmBenchContraction2x2 but using the
        explicit contraction order from ``measure_2x2`` in
        ``_env_ctm_measure.py`` instead of an opt_einsum-optimised path.

        Network layout::

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

        Contraction order (follows ``measure_2x2``):

        For each of the four corners the three CTM environment tensors (corner C
        and two adjacent edge tensors T) are contracted in the order
        ``T_outer @ (C @ T_inner)``, then the result is contracted with the
        on-site ket (``a``) and bra (``a*``) double-PEPS tensors sequentially.
        The four resulting rank-2 corner matrices are combined as::

            tensordot(cor_tl @ cor_tr,
                      tensordot(cor_bl, cor_br, axes=(0, 1)),
                      axes=((0, 1), (0, 1)))

        Note: only ``open_idx=[]`` (fully contracted network) is supported.
        """
        super().__init__(fname, config, **kwargs)
        self.params['open_idx'] = kwargs.get('open_idx', [])
        assert self.params['open_idx'] == [], \
            "CtmBenchContraction2x2Measure only supports open_idx=[] (fully contracted network)."

        self.idx_labels = {
            "a_leg_t": [2,5,3,6, 16,17,37,38, 20,23,34,35, 21,24, 9,12,10,13, 39,40, 29,32,28,31],
            "Tt_leg_l": [0,15, 8,7, 41,27, 26,33, 19,18, 36,1],
            "a_leg_s": [100,101,102,103,104,105,106,107]
        }
        # open_idx=[] → I[k] = 100+2*(k//2) for all k (physical legs are traced)
        I = [100, 100, 102, 102, 104, 104, 106, 106]
        self._gen_network_spec = lambda _I, _I_out: (
            "C1",    [0, 1],
            "T1",    [1, 2, 5, 36],
            "T4",    [0, 15, 3, 6],
            "a",     [_I[0], 2, 3, 16, 37],
            "a.conj()", [_I[1], 5, 6, 17, 38],
            "T4_y",  [15, 8, 9, 12],
            "C4_y",  [8, 7],
            "T3_y",  [10, 13, 7, 41],
            "a_y",   [_I[4], 16, 9, 10, 39],
            "a_y.conj()", [_I[5], 17, 12, 13, 40],
            "T1_x",  [36, 20, 23, 18],
            "C2_x",  [18, 19],
            "T2_x",  [19, 21, 24, 33],
            "a_x",   [_I[2], 20, 37, 34, 21],
            "a_x.conj()", [_I[3], 23, 38, 35, 24],
            "T2_xy", [33, 28, 31, 26],
            "C3_xy", [26, 27],
            "T3_xy", [29, 32, 41, 27],
            "a_xy",  [_I[6], 34, 39, 29, 28],
            "a_xy.conj()", [_I[7], 35, 40, 32, 31],
            _I_out)

        tensor_ids, inputs, output, legs_dict = self.build_network(self.legs)
        self.tensors = self.make_tensors(tensor_ids, inputs, legs_dict)
        # No contraction path is computed: the order is fixed (measure_2x2).

    def build_network(self, legs, open_idx=[]):
        """Build network specification for the fully-closed 2×2 patch."""
        I = [100, 100, 102, 102, 104, 104, 106, 106]
        I_out = []

        assert set(self.idx_labels.keys()) <= set(legs.keys()), \
            f"Legs must be provided for all unique legs. Expected {set(self.idx_labels.keys())}, got {set(legs.keys())}"
        tn = self._gen_network_spec(I, I_out)

        tensor_ids = [tn[i] for i in range(0, len(tn) - 1, 2)]
        inputs     = [tuple(tn[i]) for i in range(1, len(tn) - 1, 2)]
        output     = tuple(tn[-1])
        legs_dict  = {idx: legs[k] for k in self.idx_labels for idx in self.idx_labels[k]}

        return tensor_ids, inputs, output, legs_dict

    def print_header(self, file=None):
        print("Contract 2x2 patch in CTM environment (measure_2x2 contraction order).", file=file)

    def print_properties(self, file=None):
        print("", file=file)
        print("Config:", file=file)
        print("backend:", self.config.backend, file=file)
        print("sym:", self.config.sym, file=file)
        print("default_fusion:", self.config.default_fusion, file=file)
        print("", file=file)
        print("Cache info", file=file)
        for rec in yastn.get_cache_info().items():
            print(*rec, file=file)
        if self.config.backend.cuda_is_available():
            if hasattr(self.config.backend, "cutensor_cache_stats"):
                print("", file=file)
                print("cutensor cache stats: " +
                      str(list(self.config.backend.cutensor_cache_stats().values())), file=file)
        print("", file=file)
        for k, v in self.tensors.items():
            if k != "result":
                print(f"{k} tensor properties:", file=file)
                v.print_properties(file=file)

    @nvtx
    def contract(self):
        r"""
        Contract the 2×2 network using the explicit order from ``measure_2x2``.

        Each corner is built in three stages:

        1. Contract the two CTM environment tensors that do NOT touch the
           on-site double-PEPS tensor first (``C @ T_inner``), then contract
           that result with the third environment tensor (``T_outer``).

        2. Contract the on-site ket tensor ``a`` (contracting its top/left
           virtual legs with the environment vector).

        3. Contract the on-site bra tensor ``a*`` (contracting its top/left
           virtual legs and the shared physical leg with the running result).

        4. Fuse the four remaining free legs into two groups to produce a
           rank-2 corner matrix.

        The four corner matrices are then assembled as::

            result = tensordot(cor_tl @ cor_tr,
                               tensordot(cor_bl, cor_br, axes=(0, 1)),
                               axes=((0, 1), (0, 1)))

        Index-label trace for each corner (positions in square brackets):

        Top-left (tl) – tensors C1=[0,1], T1=[1,2,5,36], T4=[0,15,3,6],
                                a=[100,2,3,16,37], a*=[100,5,6,17,38]
            step 1  C1[1] × T1[0]             → legs [0,2,5,36]
            step 2  T4[0] × step1[0]          → legs [15,3,6,2,5,36]
            step 3  step2[1,3] × a[2,1]       → legs [15,6,5,36,100,16,37]
            step 4  step3[4,2,1] × a*[0,1,2]  → legs [15,36,16,37,17,38]
            fuse    (0,2,4) and (1,3,5)        → cor_tl: [(15,16,17),(36,37,38)]

        Top-right (tr) – tensors C2_x=[18,19], T2_x=[19,21,24,33],
                                 T1_x=[36,20,23,18], a_x=[102,20,37,34,21],
                                 a_x*=[102,23,38,35,24]
            step 1  C2_x[1] × T2_x[0]              → legs [18,21,24,33]
            step 2  T1_x[3] × step1[0]              → legs [36,20,23,21,24,33]
            step 3  step2[1,3] × a_x[1,4]           → legs [36,23,24,33,102,37,34]
            step 4  step3[4,1,2] × a_x*[0,1,4]      → legs [36,33,37,34,38,35]
            fuse    (0,2,4) and (1,3,5)              → cor_tr: [(36,37,38),(33,34,35)]

        Bottom-right (br) – tensors C3_xy=[26,27], T3_xy=[29,32,41,27],
                                    T2_xy=[33,28,31,26], a_xy=[106,34,39,29,28],
                                    a_xy*=[106,35,40,32,31]
            step 1  C3_xy[1] × T3_xy[3]              → legs [26,29,32,41]
            step 2  T2_xy[3] × step1[0]              → legs [33,28,31,29,32,41]
            step 3  step2[3,1] × a_xy[3,4]           → legs [33,31,32,41,106,34,39]
            step 4  step3[4,2,1] × a_xy*[0,3,4]      → legs [33,41,34,39,35,40]
            fuse    (0,2,4) and (1,3,5)              → cor_br: [(33,34,35),(41,39,40)]

        Bottom-left (bl) – tensors C4_y=[8,7], T4_y=[15,8,9,12],
                                   T3_y=[10,13,7,41], a_y=[104,16,9,10,39],
                                   a_y*=[104,17,12,13,40]
            step 1  C4_y[0] × T4_y[1]               → legs [7,15,9,12]
            step 2  T3_y[2] × step1[0]               → legs [10,13,41,15,9,12]
            step 3  step2[0,4] × a_y[3,2]            → legs [13,41,15,12,104,16,39]
            step 4  step3[4,0,3] × a_y*[0,3,2]       → legs [41,15,16,39,17,40]
            fuse    (0,3,5) and (1,2,4)              → cor_bl: [(41,39,40),(15,16,17)]
        """
        T = self.tensors

        # ── Top-left corner ──────────────────────────────────────────────────
        tmp = yastn.tensordot(T["C1"], T["T1"], axes=(1, 0))
        # legs: [0, 2, 5, 36]
        tmp = yastn.tensordot(T["T4"], tmp, axes=(0, 0))
        # legs: [15, 3, 6, 2, 5, 36]
        tmp = yastn.tensordot(tmp, T["a"], axes=((1, 3), (2, 1)))
        # legs: [15, 6, 5, 36, 100, 16, 37]
        tmp = yastn.tensordot(tmp, T["a.conj()"], axes=((4, 2, 1), (0, 1, 2)))
        # legs: [15, 36, 16, 37, 17, 38]
        cor_tl = tmp.fuse_legs(axes=((0, 2, 4), (1, 3, 5)))
        # cor_tl: [(15,16,17), (36,37,38)]

        # ── Top-right corner ─────────────────────────────────────────────────
        tmp = yastn.tensordot(T["C2_x"], T["T2_x"], axes=(1, 0))
        # legs: [18, 21, 24, 33]
        tmp = yastn.tensordot(T["T1_x"], tmp, axes=(3, 0))
        # legs: [36, 20, 23, 21, 24, 33]
        tmp = yastn.tensordot(tmp, T["a_x"], axes=((1, 3), (1, 4)))
        # legs: [36, 23, 24, 33, 102, 37, 34]
        tmp = yastn.tensordot(tmp, T["a_x.conj()"], axes=((4, 1, 2), (0, 1, 4)))
        # legs: [36, 33, 37, 34, 38, 35]
        cor_tr = tmp.fuse_legs(axes=((0, 2, 4), (1, 3, 5)))
        # cor_tr: [(36,37,38), (33,34,35)]

        # ── Bottom-right corner ───────────────────────────────────────────────
        tmp = yastn.tensordot(T["C3_xy"], T["T3_xy"], axes=(1, 3))
        # legs: [26, 29, 32, 41]
        tmp = yastn.tensordot(T["T2_xy"], tmp, axes=(3, 0))
        # legs: [33, 28, 31, 29, 32, 41]
        tmp = yastn.tensordot(tmp, T["a_xy"], axes=((3, 1), (3, 4)))
        # legs: [33, 31, 32, 41, 106, 34, 39]
        tmp = yastn.tensordot(tmp, T["a_xy.conj()"], axes=((4, 2, 1), (0, 3, 4)))
        # legs: [33, 41, 34, 39, 35, 40]
        cor_br = tmp.fuse_legs(axes=((0, 2, 4), (1, 3, 5)))
        # cor_br: [(33,34,35), (41,39,40)]

        # ── Bottom-left corner ────────────────────────────────────────────────
        tmp = yastn.tensordot(T["C4_y"], T["T4_y"], axes=(0, 1))
        # legs: [7, 15, 9, 12]
        tmp = yastn.tensordot(T["T3_y"], tmp, axes=(2, 0))
        # legs: [10, 13, 41, 15, 9, 12]
        tmp = yastn.tensordot(tmp, T["a_y"], axes=((0, 4), (3, 2)))
        # legs: [13, 41, 15, 12, 104, 16, 39]
        tmp = yastn.tensordot(tmp, T["a_y.conj()"], axes=((4, 0, 3), (0, 3, 2)))
        # legs: [41, 15, 16, 39, 17, 40]
        cor_bl = tmp.fuse_legs(axes=((0, 3, 5), (1, 2, 4)))
        # cor_bl: [(41,39,40), (15,16,17)]

        # ── Final assembly ────────────────────────────────────────────────────
        # cor_tl @ cor_tr: [(15,16,17), (33,34,35)]
        top = cor_tl @ cor_tr

        # tensordot(cor_bl, cor_br, axes=(0,1)):
        #   cor_bl.leg0=(41,39,40) × cor_br.leg1=(41,39,40)
        #   result: [(15,16,17), (33,34,35)]
        bot = yastn.tensordot(cor_bl, cor_br, axes=(0, 1))

        # Full contraction → scalar
        self.tensors["result"] = yastn.tensordot(top, bot, axes=((0, 1), (0, 1)))
        float(self.tensors["result"]._data[0])  # force synchronisation
