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
r"""
Benchmark modelled after ``yastn.tn.fpeps.envs.CtmEnv.measure_nsite_exact_oe``
(see ``yastn/tn/fpeps/envs/_env_ctm_measure.py``).

        j=-1             j=0               j=1          j=Ny-1    j=Ny
        :               :                 :             :         :
i=-1   TL --h,-1,-1-- T[0] --h,-1,0-- T[1] -- ... -- h,-1,Ny-1 -- TR
        |               |               |               |         |
        v,0,-1             v,0,0          v,0,1          v,0,Ny-1   v,0,Ny
        |               |               |               |         |
i=0    L[0]-h,0,-1------*---h,0,0-------*--- ... --h,0,Ny-1------R[0]
        |               |               |               |         |
        v,1,-1             v,1,0          v,1,1          v,1,Ny-1  v,1,Ny
        |               |               |               |         |
i=1    L[1]-h,1,-1------*---h,1,0-------*--- ... --h,1,Ny-1------R[1]
        :               :                :              :         :
        |               |                |              |         |
        v,Nx,-1             v,Nx,0          v,Nx,1       v,Nx,Ny-1  v,Nx,Ny
        |               |                |              |         |
i=Nx   BL --h,Nx,-1-- B[0] --h,Nx,0-- B[1] -- ... -- h,Nx,Ny-1 -- BR

"""
from __future__ import annotations

from .model_parent import nvtx
from .model_yastn_contraction_parent import CtmBenchContractionParent
import yastn
from yastn.tn.fpeps import DoublePepsTensor
from yastn.tensor.oe_blocksparse import contract_with_unroll


def _build_interleaved_unfused(corners, edges, tens, Nx, Ny):
    args = []
    swap_pairs = []

    site_tensors = {}
    for i in range(Nx):
        for j in range(Ny):
            dpt = tens[(i, j)]
            Ab, Ak = dpt.Ab_Ak_with_charge_swap()

            if dpt.op is not None:
                Ak = yastn.tensordot(Ak, dpt.op, axes=(4, 1))

            Ab_c = Ab.conj()
            tt = yastn.tensordot(Ak, Ab_c, axes=(4, 4))
            tt = tt.swap_gate(axes=((1, 5), 4, (2, 6), 7))
            tt = tt.transpose(axes=(0, 4, 1, 5, 2, 6, 3, 7))

            trans8 = []
            for k in dpt.trans:
                trans8.extend([2 * k, 2 * k + 1])
            site_tensors[(i, j)] = tt.transpose(axes=tuple(trans8)).drop_leg_history()

    args += [corners["tl"], [('v', 0, -1), ('h', -1, -1)]]
    args += [corners["bl"], [('h', Nx, -1), ('v', Nx, -1)]]
    args += [corners["tr"], [('h', -1, Ny - 1), ('v', 0, Ny)]]
    args += [corners["br"], [('v', Nx, Ny), ('h', Nx, Ny - 1)]]

    for i in range(Nx):
        args += [edges["l"][i],
                 [('v', i + 1, -1), ('h', i, -1, 'k'), ('h', i, -1, 'b'), ('v', i, -1)]]

    for i in range(Nx):
        args += [edges["r"][i],
                 [('v', i, Ny), ('h', i, Ny - 1, 'k'), ('h', i, Ny - 1, 'b'), ('v', i + 1, Ny)]]

    for j in range(Ny):
        args += [edges["t"][j],
                 [('h', -1, j - 1), ('v', 0, j, 'k'), ('v', 0, j, 'b'), ('h', -1, j)]]

    for j in range(Ny):
        args += [edges["b"][j],
                 [('h', Nx, j), ('v', Nx, j, 'k'), ('v', Nx, j, 'b'), ('h', Nx, j - 1)]]

    def _bond_labels(i, j):
        return [('v', i, j), ('h', i, j - 1), ('v', i + 1, j), ('h', i, j)]

    for i in range(Nx):
        for j in range(Ny):
            lbls = _bond_labels(i, j)
            args += [site_tensors[(i, j)],
                     [lbls[0] + ('k',), lbls[0] + ('b',),
                      lbls[1] + ('k',), lbls[1] + ('b',),
                      lbls[2] + ('k',), lbls[2] + ('b',),
                      lbls[3] + ('k',), lbls[3] + ('b',)]]

    args.append(())
    return tuple(args), swap_pairs


def _build_separate_unfused(corners, edges, tens, Nx, Ny):
    args = []
    swap_pairs = []

    args += [corners["tl"], [('v', 0, -1), ('h', -1, -1)]]
    args += [corners["bl"], [('h', Nx, -1), ('v', Nx, -1)]]
    args += [corners["tr"], [('h', -1, Ny - 1), ('v', 0, Ny)]]
    args += [corners["br"], [('v', Nx, Ny), ('h', Nx, Ny - 1)]]

    for i in range(Nx):
        args += [edges["l"][i],
                 [('v', i + 1, -1), ('h', i, -1, 'k'), ('h', i, -1, 'b'), ('v', i, -1)]]

    for i in range(Nx):
        args += [edges["r"][i],
                 [('v', i, Ny), ('h', i, Ny - 1, 'k'), ('h', i, Ny - 1, 'b'), ('v', i + 1, Ny)]]

    for j in range(Ny):
        args += [edges["t"][j],
                 [('h', -1, j - 1), ('v', 0, j, 'k'), ('v', 0, j, 'b'), ('h', -1, j)]]

    for j in range(Ny):
        args += [edges["b"][j],
                 [('h', Nx, j), ('v', Nx, j, 'k'), ('v', Nx, j, 'b'), ('h', Nx, j - 1)]]

    def _bond_labels(i, j):
        return [('v', i, j), ('h', i, j - 1), ('v', i + 1, j), ('h', i, j)]

    for i in range(Nx):
        for j in range(Ny):
            dpt = tens[(i, j)]
            Ab, Ak = dpt.Ab_Ak_with_charge_swap()

            if dpt.op is not None:
                Ak = yastn.tensordot(Ak, dpt.op, axes=(4, 1))

            Ab_c = Ab.conj().swap_gate(axes=(1, 0, 2, 3))
            Ak = Ak.transpose(axes=dpt.trans + (4,)).drop_leg_history()
            Ab_c = Ab_c.transpose(axes=dpt.trans + (4,)).drop_leg_history()

            lbls = _bond_labels(i, j)
            args += [Ak,
                     [lbls[0] + ('k',), lbls[1] + ('k',),
                      lbls[2] + ('k',), lbls[3] + ('k',), ('p', i, j)]]
            args += [Ab_c,
                     [lbls[0] + ('b',), lbls[1] + ('b',),
                      lbls[2] + ('b',), lbls[3] + ('b',), ('p', i, j)]]

            inv = [dpt.trans.index(d) for d in range(4)]
            swap_pairs.append((lbls[inv[1]] + ('k',), lbls[inv[0]] + ('b',)))
            swap_pairs.append((lbls[inv[2]] + ('k',), lbls[inv[3]] + ('b',)))

    args.append(())
    return tuple(args), swap_pairs


class CtmBenchMeasureNconFermionic(CtmBenchContractionParent):

    def __init__(self, fname, config, **kwargs):
        super().__init__(fname, config, **kwargs)
        self.params.update({
            # (Nx, Ny) shape of the synthetic CTM patch tiled with copies of the input site tensor.
            'dims': (2, 2),
            # Pair of (i, j) site coordinates where the two operators are inserted.
            # None defaults to the diagonal corners ((0, 0), (Nx-1, Ny-1)).
            'sites': None,
            # If True, feed Ak and Ab_c into the network as separate 5-leg tensors
            # (ket/bra layers contracted by einsum). If False, pre-contract them into
            # an 8-leg double-layer site tensor before handing the network to ncon.
            'separate_layers': False,
            # If True, attach the operators on the chosen sites and add the fermionic
            # charge-swap string between them. If False, time the bare network only.
            'insert_operator': True,
            # If True, treat the JSON-supplied site tensor as the (0,0) entry of a
            # 2x2 checkerboard iPEPS and build the sublattice-B tensor at sites
            # with (i+j) % 2 == 1 by sublattice-rotating its physical/ancilla legs.
            'checkerboard': False,
        })
        for k in ('dims', 'sites', 'separate_layers', 'insert_operator',
                  'checkerboard'):
            if k in kwargs:
                self.params[k] = kwargs[k]

        self.swap_pairs = None
        self.result = None
        self.tensors = {}

        self._init_from_input()
        self.build_ncon_call()

    def _normalized_sites(self):
        if self.params['sites'] is None:
            Nx, Ny = self.params['dims']
            sites = ((0, 0), (Nx - 1, Ny - 1))
        else:
            sites = tuple(tuple(site) for site in self.params['sites'])

        if len(sites) != 2:
            raise ValueError("Fermionic measurement benchmark expects exactly two sites.")
        if sites[0] == sites[1]:
            raise ValueError("Fermionic measurement benchmark needs two distinct sites.")
        return sites

    def _init_from_input(self):
        self.config.backend.random_seed(seed=self.params['seed'])

        legs_a = ["a_leg_t", "a_leg_l", "a_leg_b", "a_leg_r", "a_leg_s", "a_leg_a"]
        legs_a = [self.legs[k] for k in legs_a if k in self.legs]
        self.site_ket = yastn.rand(self.config, legs=legs_a)
        if self.site_ket.ndim == 6:
            self.site_ket = self.site_ket.fuse_legs(axes=(0, 1, 2, 3, (4, 5)))

        self.phys_leg = self.site_ket.get_legs(axes=self.site_ket.ndim - 1)

        # Sublattice B for a 2x2 checkerboard. The exported tensor is sublattice A;
        # B is its partner so that A only ever tiles against B (never A-against-A).
        # B is A rotated 180 degrees (t<->b, l<->r) with every leg conjugated:
        #   B.top = conj(A.bottom), B.bottom = conj(A.top),
        #   B.left = conj(A.right), B.right = conj(A.left).
        # Then each shared bond closes by construction, e.g. A.bottom == a_leg_b
        # meets B.top == conj(a_leg_b) regardless of whether a_leg_b == conj(a_leg_t).
        # (legs are [t, l, b, r, phys]; transpose (2,3,0,1,4) performs the rotation.)
        self.site_ket_B = None
        if self.params['checkerboard']:
            self.site_ket_B = self.site_ket.conj().transpose(axes=(2, 3, 0, 1, 4))

        # Use one chi leg per boundary direction across the whole synthetic patch.
        # Reusing the left/right or top/bottom chi legs from a single exported local
        # tensor would make repeated boundary edges inconsistent when tiled.
        self.chi_h_leg = self.legs["Tt_leg_l"]
        self.chi_v_leg = self.legs["Tr_leg_t"]

        # Edge (T) tensors. An edge facing a site whose boundary leg is X carries
        # legs [chi, X.conj(), X, chi.conj()] so both ket/bra layers contract.
        def _edge(X, chi):
            return yastn.rand(self.config, legs=[chi, X.conj(), X, chi.conj()])

        lt, ll, lb, lr = (self.legs["a_leg_t"], self.legs["a_leg_l"],
                          self.legs["a_leg_b"], self.legs["a_leg_r"])
        # A-facing edges (boundary leg is A's own t/l/b/r leg).
        self.edge_t = _edge(lt, self.chi_h_leg)
        self.edge_b = _edge(lb, self.chi_h_leg)
        self.edge_l = _edge(ll, self.chi_v_leg)
        self.edge_r = _edge(lr, self.chi_v_leg)
        # B-facing edges: B's boundary legs are the rotated-conjugated A legs.
        self.edge_t_B = self.edge_b_B = self.edge_l_B = self.edge_r_B = None
        if self.params['checkerboard']:
            self.edge_t_B = _edge(lb.conj(), self.chi_h_leg)   # B.top    = conj(a_leg_b)
            self.edge_b_B = _edge(lt.conj(), self.chi_h_leg)   # B.bottom = conj(a_leg_t)
            self.edge_l_B = _edge(lr.conj(), self.chi_v_leg)   # B.left   = conj(a_leg_r)
            self.edge_r_B = _edge(ll.conj(), self.chi_v_leg)   # B.right  = conj(a_leg_l)

        # Corner (C) tensors are rank-2 chi x chi environment blocks; they only
        # touch chi bonds (never site a-legs), so there is exactly one per patch
        # corner and nothing sublattice-dependent to match.
        self.corners = {
            "tl": yastn.rand(self.config, legs=[self.chi_v_leg, self.chi_h_leg.conj()]),
            "tr": yastn.rand(self.config, legs=[self.chi_h_leg, self.chi_v_leg.conj()]),
            "bl": yastn.rand(self.config, legs=[self.chi_h_leg, self.chi_v_leg.conj()]),
            "br": yastn.rand(self.config, legs=[self.chi_v_leg, self.chi_h_leg.conj()]),
        }

    def _is_B(self, i, j):
        return self.params['checkerboard'] and (i + j) % 2 == 1

    def _site_ket_for(self, i, j):
        return self.site_ket_B if self._is_B(i, j) else self.site_ket

    def _make_patch(self):
        Nx, Ny = self.params['dims']
        tens = {(i, j): DoublePepsTensor(self._site_ket_for(i, j).copy(),
                                         self._site_ket_for(i, j).copy())
                for i in range(Nx) for j in range(Ny)}
        # Each boundary edge must match the sublattice of the site it borders:
        # top col j -> (0, j); bottom col j -> (Nx-1, j); left row i -> (i, 0);
        # right row i -> (i, Ny-1).
        edges = {
            "t": {j: (self.edge_t_B if self._is_B(0, j) else self.edge_t).copy() for j in range(Ny)},
            "b": {j: (self.edge_b_B if self._is_B(Nx - 1, j) else self.edge_b).copy() for j in range(Ny)},
            "l": {i: (self.edge_l_B if self._is_B(i, 0) else self.edge_l).copy() for i in range(Nx)},
            "r": {i: (self.edge_r_B if self._is_B(i, Ny - 1) else self.edge_r).copy() for i in range(Nx)},
        }
        corners = {k: v.copy() for k, v in self.corners.items()}
        return corners, edges, tens

    def _make_operator_pair(self):
        s_A = self.legs["a_leg_s"]
        op_A = yastn.eye(self.config, legs=[s_A, s_A.conj()], isdiag=False)
        if self.params['checkerboard']:
            # B's physical space is the conjugate of A's (rotated-conjugated site),
            # so its operator lives on the conjugated system leg.
            s_B = s_A.conj()
            op_B = yastn.eye(self.config, legs=[s_B, s_B.conj()], isdiag=False)
        else:
            op_B = op_A
        return op_A, op_B

    def _op_for_site(self, op_A, op_B, site):
        return op_B if self._is_B(*site) else op_A

    def _insert_operators(self, tens):
        op_A, op_B = self._make_operator_pair()
        site_l, site_r = self._normalized_sites()
        site_ops = {site_l: self._op_for_site(op_A, op_B, site_l),
                    site_r: self._op_for_site(op_A, op_B, site_r)}
        Nx, Ny = self.params['dims']
        axes_string_x = ['b3', 'k4', 'k1']
        axes_string_y = ['k2', 'k4', 'b0']

        for j in range(Ny):
            for i in range(Nx):
                site = (i, j)
                if site not in site_ops:
                    continue
                tens[site].set_operator_(site_ops[site])
                if i > 0:
                    tens[site].add_charge_swaps_(site_ops[site].n, axes='k1')
                    for i1 in range(i - 1, 0, -1):
                        tens[(i1, j)].add_charge_swaps_(site_ops[site].n, axes=axes_string_x)
                    tens[(0, j)].add_charge_swaps_(site_ops[site].n, axes=['b3', 'k4'])
                if j > 0:
                    tens[(0, j)].add_charge_swaps_(site_ops[site].n, axes='b0')
                    for j1 in range(j - 1, 0, -1):
                        tens[(0, j1)].add_charge_swaps_(site_ops[site].n, axes=axes_string_y)
                    tens[(0, 0)].add_charge_swaps_(site_ops[site].n, axes=['k2', 'k4'])

    def _clear_operators(self, tens):
        Nx, Ny = self.params['dims']
        for i in range(Nx):
            for j in range(Ny):
                tens[(i, j)].del_operator_()
                tens[(i, j)].del_charge_swaps_()

    @nvtx
    def build_ncon_call(self):
        corners, edges, tens = self._make_patch()
        if self.params['insert_operator']:
            self._insert_operators(tens)
        build_fn = _build_separate_unfused if self.params['separate_layers'] else _build_interleaved_unfused
        self.tn, self.swap_pairs = build_fn(corners, edges, tens, *self.params['dims'])
        self.tensors.update({
            "corner_tl": corners["tl"],
            "corner_bl": corners["bl"],
            "corner_tr": corners["tr"],
            "corner_br": corners["br"],
        })
        for i, edge in edges["l"].items():
            self.tensors[f"edge_l_{i}"] = edge
        for i, edge in edges["r"].items():
            self.tensors[f"edge_r_{i}"] = edge
        for j, edge in edges["t"].items():
            self.tensors[f"edge_t_{j}"] = edge
        for j, edge in edges["b"].items():
            self.tensors[f"edge_b_{j}"] = edge
        if self.params['insert_operator']:
            self._clear_operators(tens)

    @nvtx
    def contract(self):
        kwargs = dict(
            optimizer=self.params['optimizer'], 
            optimizer_kwargs=self.params['optimizer_kwargs'],
            unroll=self.params['unroll'],
            checkpoint_loop=self.params['checkpoint_loop'],
            devices=self.params['devices'],
            mp_workers_per_device=self.params['mp_workers_per_device'],
            per_combo_path=self.params['per_combo_path'],
            combo_path_kwargs=self.params['combo_path_kwargs'],
            distributed=self.params['distributed'],
            swap=self.swap_pairs,
            who=self.__class__.__name__,
        )
        self.tensors["result"] = contract_with_unroll(*self.tn, **kwargs)
        self.result = self.tensors["result"].to_number()

    def print_header(self, file=None):
        print("Generate post-double-layer ncon call from input-derived CTM tensors.", file=file)

    def print_properties(self, file=None):
        print("Benchmark params:", file=file)
        for k, v in self.params.items():
            if k == 'f_out':
                continue
            print(f"{k}: {v}", file=file)
        print("", file=file)
        if self.tn is not None:
            print("Generated network:", file=file)
            print(f"num_tensors: {len(self.tn[0::2])}", file=file)
            print(f"num_connects: {len(self.tn[1::2])}", file=file)
            print(f"num_swap_pairs: {len(self.swap_pairs)}", file=file)
            print("", file=file)
        super().print_properties(file=file)
        if self.result is not None:
            print(f"result: {self.result}", file=file)
