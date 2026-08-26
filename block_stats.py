"""Tracks the largest block count (`len(tensor.struct.t)`) seen during
benchmark contractions.

Hooks `yastn.tensordot` so every input and every result is inspected. The
counter is process-local: with `mp_workers_per_device > 0` the workers each
have their own counter and the parent only sees what it ran itself.
"""

import threading

import yastn
from yastn.tensor import _contractions, _einsum
from yastn.tensor._auxiliary import get_blocks


_lock = threading.Lock()
_max_blocks = 0
_max_block = 0 
_max_blocks_context = ""  # short tag for which slot held the max (a / b / out)
_max_block_context = ""  # short tag for which slot held the max block size (a / b / out)
_installed = False


def _consider(tensor, tag):
    global _max_blocks, _max_blocks_context, _max_block, _max_block_context
    struct= getattr(tensor, "struct", None)
    if struct is None:
        return
    st_full = get_blocks(tensor.config.sym, struct)
    max_block= max(st_full.slc[:, 1] - st_full.slc[:, 0]) if st_full.slc is not None else 0
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


_orig_tensordot = _contractions.tensordot


def _instrumented_tensordot(a, b, axes, conj=(0, 0)):
    _consider(a, "in")
    _consider(b, "in")
    c = _orig_tensordot(a, b, axes, conj=conj)
    _consider(c, "out")
    return c


def accumulate(nb, mb, where="ext"):
    """Bump the counter from an external source (e.g. an MP worker)."""
    global _max_blocks, _max_block, _max_blocks_context, _max_block_context
    if nb <= _max_blocks:
        return
    with _lock:
        if nb > _max_blocks:
            _max_blocks = nb
            _max_blocks_context = where
        if mb > _max_block:
            _max_block = mb
            _max_block_context = where


def install():
    """Patch every module that holds a reference to `tensordot`. Idempotent.

    In MP mode this also monkey-patches ``_oe_blocksparse_mp`` so that worker
    processes (a) install block_stats in-process (patching their own
    tensordot) and (b) ship their per-forward max back via the existing
    ``res_q`` by extending the ``forward_done`` tuple with a trailing field.
    The parent strips that field on receive and feeds it into ``accumulate``.
    No yastn source edits required.
    """
    global _installed
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
            if isinstance(msg, tuple) and len(msg) >= 5 and msg[0] == 'forward_done':
                try:
                    accumulate(int(msg[4]))
                    accumulate(int(msg[5]))
                except Exception:
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
                nb, mb, _, _ = report()
            except Exception:
                nb = 0
                mb = 0
            msg = msg + (int(nb), int(mb))
            reset()
        return _orig_put(msg, *a, **kw)

    res_q.put = _put_with_stats

    _orig_worker_main(rank, gpu_dev, config_desc, cmd_q, res_q, *extra)


def reset():
    global _max_blocks, _max_blocks_context, _max_block, _max_block_context
    with _lock:
        _max_blocks = 0
        _max_block = 0
        _max_blocks_context = ""
        _max_block_context = ""


def report():
    return _max_blocks, _max_block, _max_blocks_context, _max_block_context
