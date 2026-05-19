"""Tracks the largest block count (`len(tensor.struct.t)`) seen during
benchmark contractions.

Hooks `yastn.tensordot` so every input and every result is inspected. The
counter is process-local: with `mp_workers_per_device > 0` the workers each
have their own counter and the parent only sees what it ran itself.
"""

import threading

import yastn
from yastn.tensor import _contractions, _einsum


_lock = threading.Lock()
_max_blocks = 0
_max_context = ""  # short tag for which slot held the max (a / b / out)
_installed = False


def _consider(tensor, tag):
    global _max_blocks, _max_context
    nb = len(getattr(getattr(tensor, "struct", None), "t", ()) or ())
    if nb <= _max_blocks:
        return
    with _lock:
        if nb > _max_blocks:
            _max_blocks = nb
            _max_context = tag


_orig_tensordot = _contractions.tensordot


def _instrumented_tensordot(a, b, axes, conj=(0, 0)):
    _consider(a, "in")
    _consider(b, "in")
    c = _orig_tensordot(a, b, axes, conj=conj)
    _consider(c, "out")
    return c


def accumulate(nb, where="ext"):
    """Bump the counter from an external source (e.g. an MP worker)."""
    global _max_blocks, _max_context
    if nb <= _max_blocks:
        return
    with _lock:
        if nb > _max_blocks:
            _max_blocks = nb
            _max_context = where


def install():
    """Patch every module that holds a reference to `tensordot`. Idempotent.
    Also wires the MP worker-result sink so worker maxes feed the same
    counter when ``mp_workers_per_device > 0``.
    """
    global _installed
    if _installed:
        return
    _contractions.tensordot = _instrumented_tensordot
    _einsum.tensordot = _instrumented_tensordot  # ncon does `from ._contractions import tensordot`
    yastn.tensordot = _instrumented_tensordot
    try:
        from yastn.tensor import _oe_blocksparse_mp
        _oe_blocksparse_mp._block_stats_sink = accumulate
    except ImportError:
        pass
    _installed = True


def reset():
    global _max_blocks, _max_context
    with _lock:
        _max_blocks = 0
        _max_context = ""


def report():
    return _max_blocks, _max_context
