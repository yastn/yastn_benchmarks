"""Tracks block statistics seen during benchmark contractions.

Two independent sets of numbers are kept:

* tensor level -- the largest block count (`struct` of the tensors handed to /
  returned by `yastn.tensordot`) and the largest block within those tensors.
* fused-matrix level (`f2m`) -- with `tensordot_policy='fuse_to_matrix'` the
  tensors are merged into matrices before the GEMM, so the largest buffer
  actually materialized is a block of the intermediate matrix, which never
  appears in any tensor `struct`. Those come from the `meta_dot` built by
  `_meta_tensordot_f2m`.

Both counters are process-local: with `mp_workers_per_device > 0` the workers
each have their own counter and the parent only sees what it ran itself, plus
whatever the workers ship back over `res_q`.

Note: `yastn.set_cache_maxsize()` rebinds `_contractions._meta_tensordot_f2m`
from `__wrapped__`, which drops the f2m hook. Nothing in the benchmarks calls
it; if that changes, re-run `install()` afterwards.
"""

from collections import namedtuple
from functools import lru_cache
import threading

import yastn
from yastn.tensor import _contractions, _einsum
from yastn.tensor._auxiliary import get_blocks


Stats = namedtuple("Stats", "max_blocks max_block where_blocks where_block f2m_blocks f2m_block")

_lock = threading.Lock()
_max_blocks = 0
_max_block = 0
_max_blocks_context = ""  # short tag for which slot held the max (a / b / out)
_max_block_context = ""  # short tag for which slot held the max block size (a / b / out)
_max_f2m_blocks = 0  # blocks of the fused result matrix in _meta_tensordot_f2m
_max_f2m_block = 0  # largest block of that matrix, in elements
_installed = False


def _consider(tensor, tag):
    global _max_blocks, _max_blocks_context, _max_block, _max_block_context
    struct= getattr(tensor, "struct", None)
    if struct is None:
        return
    st_full = get_blocks(tensor.config.sym, struct)
    # int(): slc is a numpy array, and a np.int64 would leak its repr into the report
    max_block= int(max(st_full.slc[:, 1] - st_full.slc[:, 0])) if len(st_full.slc) else 0
    nb = st_full.nblocks
    if nb <= _max_blocks and max_block <= _max_block:
        return
    with _lock:
        if nb > _max_blocks:
            _max_blocks = nb
            _max_blocks_context = tag
        if max_block > _max_block:
            _max_block = max_block
            _max_block_context = tag


def _consider_f2m(nb, mb):
    """Track the fused-matrix block count / largest block of one f2m contraction."""
    global _max_f2m_blocks, _max_f2m_block
    if nb <= _max_f2m_blocks and mb <= _max_f2m_block:
        return
    with _lock:
        if nb > _max_f2m_blocks:
            _max_f2m_blocks = nb
        if mb > _max_f2m_block:
            _max_f2m_block = mb


_orig_tensordot = _contractions.tensordot
_orig_meta_f2m = _contractions._meta_tensordot_f2m  # the lru-cached callable


def _instrumented_tensordot(a, b, axes, conj=(0, 0)):
    _consider(a, "in")
    _consider(b, "in")
    c = _orig_tensordot(a, b, axes, conj=conj)
    _consider(c, "out")
    return c


@lru_cache(maxsize=4096)
def _f2m_stats_for(sym, struct_a, struct_b):
    """Block count and largest block of the fused result matrix, memoized.

    `meta` entries are `(slc, Dc, sla, Da, slb, Db)` with `Dc` the `(M, N)`
    shape of one output matrix block; there is exactly one such block per
    contracted charge sector, so `len(meta)` is the matrix' block count.

    Keyed on the same `(sym, struct_a, struct_b)` triple as yastn's own
    `lru_cache`, so the O(nblocks) scan runs once per distinct struct pair --
    the hook has to sit outside that cache to see every call, and a scan on
    every call would show up in the benchmark timings.
    """
    meta, _size_cm, _struct_cm = _orig_meta_f2m(sym, struct_a, struct_b)  # cache hit
    return len(meta), max((Dc[0] * Dc[1] for _, Dc, *_ in meta), default=0)


def _instrumented_meta_tensordot_f2m(sym, struct_a, struct_b):
    out = _orig_meta_f2m(sym, struct_a, struct_b)
    _consider_f2m(*_f2m_stats_for(sym, struct_a, struct_b))
    return out


_instrumented_meta_tensordot_f2m._block_stats_hook = True
# yastn's _control_lru reaches through the module attribute for these:
# get_cache_info() calls .cache_info(), clear_cache() calls .cache_clear(), and
# set_cache_maxsize() rebuilds from .__wrapped__ (which drops the hook -- see
# the module docstring).
_instrumented_meta_tensordot_f2m.cache_info = _orig_meta_f2m.cache_info
_instrumented_meta_tensordot_f2m.cache_clear = _orig_meta_f2m.cache_clear
_instrumented_meta_tensordot_f2m.__wrapped__ = _orig_meta_f2m.__wrapped__


def accumulate(nb, mb, f2m_nb=0, f2m_mb=0, where="ext"):
    """Bump the counters from an external source (e.g. an MP worker)."""
    global _max_blocks, _max_block, _max_blocks_context, _max_block_context
    if nb <= _max_blocks and mb <= _max_block:
        _consider_f2m(f2m_nb, f2m_mb)
        return
    with _lock:
        if nb > _max_blocks:
            _max_blocks = nb
            _max_blocks_context = where
        if mb > _max_block:
            _max_block = mb
            _max_block_context = where
    _consider_f2m(f2m_nb, f2m_mb)


def install():
    """Patch every module that holds a reference to `tensordot`. Idempotent.

    Also patches `_contractions._meta_tensordot_f2m` -- around its `lru_cache`,
    so the hook fires on cache hits too (after the first repeat run every call
    is a hit).

    In MP mode this also monkey-patches ``_oe_blocksparse_mp`` so that worker
    processes (a) install block_stats in-process (patching their own
    tensordot) and (b) ship their per-forward max back via the existing
    ``res_q`` by extending the ``forward_done`` tuple with trailing fields.
    The parent strips those fields on receive and feeds them into
    ``accumulate``. No yastn source edits required.
    """
    global _installed
    # Guarded separately from `_installed` so a re-install recovers the hook if
    # something (e.g. yastn.set_cache_maxsize) rebound the attribute.
    if not getattr(_contractions._meta_tensordot_f2m, "_block_stats_hook", False):
        _contractions._meta_tensordot_f2m = _instrumented_meta_tensordot_f2m
    if _installed:
        return
    _contractions.tensordot = _instrumented_tensordot
    _einsum.tensordot = _instrumented_tensordot  # ncon does `from ._contractions import tensordot`
    yastn.tensordot = _instrumented_tensordot
    try:
        from yastn.tensor import _oe_blocksparse_mp
        _install_mp_patches(_oe_blocksparse_mp)
    except ImportError:
        pass
    # Make this module importable by spawned MP workers (spawn does not
    # inherit the parent's sys.path[0]; PYTHONPATH does propagate).
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    if here not in parts:
        parts.insert(0, here)
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    _installed = True


def _install_mp_patches(mp_mod):
    """Wire block_stats into ``_oe_blocksparse_mp`` purely via monkey-patches.

    Idempotent: stashes the real ``_worker_main`` under a side name on the
    first call and skips on subsequent calls. Safe to run in parent and
    worker (each process has its own module copy).
    """
    if getattr(mp_mod, "_block_stats_patched", False):
        return
    mp_mod._worker_main_original = mp_mod._worker_main
    mp_mod._worker_main = _worker_main_with_stats

    _orig_pool_init = mp_mod._PersistentWorkerPool.__init__

    def _pool_init_with_stats(self, *args, **kwargs):
        _orig_pool_init(self, *args, **kwargs)
        _orig_get = self.res_q.get

        def _get_with_stats(*a, **kw):
            msg = _orig_get(*a, **kw)
            # the real message is ('forward_done', rank, txn_id, out); the
            # worker appends (nb, mb, f2m_nb, f2m_mb) -- see _put_with_stats.
            if isinstance(msg, tuple) and len(msg) >= 8 and msg[0] == 'forward_done':
                try:
                    accumulate(int(msg[4]), int(msg[5]), int(msg[6]), int(msg[7]))
                except (TypeError, ValueError, IndexError):
                    pass
                msg = msg[:4]
            return msg

        # Instance-level shadow of the bound Queue.get. multiprocessing.Queue
        # has a __dict__ so this is allowed; if a future version flips to
        # __slots__ this will raise AttributeError and we'd need a Queue
        # subclass instead.
        self.res_q.get = _get_with_stats

    mp_mod._PersistentWorkerPool.__init__ = _pool_init_with_stats
    mp_mod._block_stats_patched = True


def _worker_main_with_stats(rank, gpu_dev, config_desc, cmd_q, res_q, *extra):
    """Replacement for ``_oe_blocksparse_mp._worker_main`` set up via
    ``_install_mp_patches``. Runs in each spawned worker; installs
    block_stats locally so the worker's own ``tensordot`` is hooked, and
    intercepts ``res_q.put`` to attach the per-forward max + reset the
    counter before delegating to the stashed original ``_worker_main``.

    ``*extra`` forwards any trailing args the real ``_worker_main`` gained
    (e.g. ``log_queue, log_level, logger_levels`` from yastn's "logging for
    mp" change) so this wrapper stays signature-proof across yastn updates.
    """
    install()  # patches tensordot in this worker; also stashes the original
    from yastn.tensor import _oe_blocksparse_mp as _mp_mod
    _orig_worker_main = _mp_mod._worker_main_original

    _orig_put = res_q.put

    def _put_with_stats(msg, *a, **kw):
        if isinstance(msg, tuple) and msg and msg[0] == 'forward_done':
            try:
                st = report()
            except Exception:
                st = Stats(0, 0, "", "", 0, 0)
            msg = msg + (int(st.max_blocks), int(st.max_block),
                         int(st.f2m_blocks), int(st.f2m_block))
            reset()
        return _orig_put(msg, *a, **kw)

    res_q.put = _put_with_stats

    _orig_worker_main(rank, gpu_dev, config_desc, cmd_q, res_q, *extra)


def reset():
    global _max_blocks, _max_blocks_context, _max_block, _max_block_context
    global _max_f2m_blocks, _max_f2m_block
    with _lock:
        _max_blocks = 0
        _max_block = 0
        _max_blocks_context = ""
        _max_block_context = ""
        _max_f2m_blocks = 0
        _max_f2m_block = 0


def report():
    return Stats(_max_blocks, _max_block, _max_blocks_context, _max_block_context,
                 _max_f2m_blocks, _max_f2m_block)
