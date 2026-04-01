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

Simplifications relative to the real measurement function:

* Edge tensors are created already unfused (4-leg), skipping the
  ``_uf_middle_padded`` unfuse-and-pad step.
* Only the operator contraction is timed; the norm contraction and
  ``sign * val_op / val_no`` normalisation are omitted.
* All site / edge / corner tensors are random with uniform structure
  (no position-dependent charge sectors).
"""
from __future__ import annotations

from .model_parent import nvtx
from .model_yastn_contraction_parent import CtmBenchContractionParent
import yastn
from yastn.tn.fpeps import DoublePepsTensor
from yastn.tensor.oe_blocksparse import contract_with_unroll, contract_with_unroll_compute_constants


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
            'dims': (2, 2),
            'sites': None,
            'separate_layers': False,
            'insert_operator': True,
            'precontract_constants': False,
        })
        for k in ('dims', 'sites', 'separate_layers', 'insert_operator', 'precontract_constants'):
            if k in kwargs:
                self.params[k] = kwargs[k]

        self.swap_pairs = None
        self.path = None
        self.path_info = None
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
        # Use one chi leg per boundary direction across the whole synthetic patch.
        # Reusing the left/right or top/bottom chi legs from a single exported local
        # tensor would make repeated boundary edges inconsistent when tiled.
        self.chi_h_leg = self.legs["Tt_leg_l"]
        self.chi_v_leg = self.legs["Tr_leg_t"]

        self.edge_t = yastn.rand(
            self.config,
            legs=[self.chi_h_leg, self.legs["a_leg_t"].conj(), self.legs["a_leg_t"], self.chi_h_leg.conj()],
        )
        self.edge_b = yastn.rand(
            self.config,
            legs=[self.chi_h_leg, self.legs["a_leg_b"].conj(), self.legs["a_leg_b"], self.chi_h_leg.conj()],
        )
        self.edge_l = yastn.rand(
            self.config,
            legs=[self.chi_v_leg, self.legs["a_leg_l"].conj(), self.legs["a_leg_l"], self.chi_v_leg.conj()],
        )
        self.edge_r = yastn.rand(
            self.config,
            legs=[self.chi_v_leg, self.legs["a_leg_r"].conj(), self.legs["a_leg_r"], self.chi_v_leg.conj()],
        )

        self.corners = {
            "tl": yastn.rand(self.config, legs=[self.chi_v_leg, self.chi_h_leg.conj()]),
            "tr": yastn.rand(self.config, legs=[self.chi_h_leg, self.chi_v_leg.conj()]),
            "bl": yastn.rand(self.config, legs=[self.chi_h_leg, self.chi_v_leg.conj()]),
            "br": yastn.rand(self.config, legs=[self.chi_v_leg, self.chi_h_leg.conj()]),
        }

    def _make_patch(self):
        Nx, Ny = self.params['dims']
        tens = {(i, j): DoublePepsTensor(self.site_ket.copy(), self.site_ket.copy())
                for i in range(Nx) for j in range(Ny)}
        edges = {
            "t": {j: self.edge_t.copy() for j in range(Ny)},
            "b": {j: self.edge_b.copy() for j in range(Ny)},
            "l": {i: self.edge_l.copy() for i in range(Nx)},
            "r": {i: self.edge_r.copy() for i in range(Nx)},
        }
        corners = {k: v.copy() for k, v in self.corners.items()}
        return corners, edges, tens

    def _make_operator_pair(self):
        op = yastn.eye(
            self.config,
            legs=[self.legs["a_leg_s"], self.legs["a_leg_s"].conj()],
            isdiag=False,
        )
        return op, op

    def _insert_operators(self, tens):
        cp_op, c_op = self._make_operator_pair()
        site_l, site_r = self._normalized_sites()
        site_ops = {site_l: cp_op, site_r: c_op}
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
        self.path, self.path_info = self.compute_contraction_path(*self.tn, optimizer=self.params['optimizer'])
        if self.params['insert_operator']:
            self._clear_operators(tens)

    @nvtx
    def contract(self):
        _contract = contract_with_unroll_compute_constants if self.params['precontract_constants'] else contract_with_unroll
        self.tensors["result"] = _contract(
            *self.tn,
            optimize=self.path,
            unroll=self.params['unroll'],
            checkpoint_loop=self.params['checkpoint_loop'],
            devices=self.params['devices'],
            swap=self.swap_pairs,
            who=self.__class__.__name__,
        )
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
