# SPDX-License-Identifier: Apache-2.0
"""Persistent farm runtime.

Sequence contexts that outlive a single ``model.forward()``.

The farm scheduler (``scheduler.get_continuous_scheduler``) is already a
process-local singleton keyed by a caller-supplied ``ctx_id``. This module adds
the *runner* half that was still per-forward:

* owned ``hidden`` / ``residual`` / ``positions`` tensors,
* an owned child ``ForwardBatch`` snapshot (no parent views),
* the pending ``FarmHop``,
* the ready / deferred result bookkeeping.

With ``SGLANG_AFD_FARM_PERSISTENT=0`` (the default) none of this is used and the
farm keeps its original one-shot intra-forward behavior.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)


# ``TboForwardBatchPreparer.filter_batch`` slices these out of the parent
# ``ForwardBatch``. Every one of them is therefore a *view* that a later
# forward will overwrite, so a context that outlives the forward must clone it.
_TOKEN_INDEXED_FIELDS = (
    "input_ids",
    "positions",
    "out_cache_loc",
    "mrope_positions",
)

_SEQ_INDEXED_FIELDS = (
    "req_pool_indices",
    "seq_lens",
    "seq_lens_cpu",
    "orig_seq_lens",
    "extend_seq_lens",
    "extend_prefix_lens",
    "extend_start_loc",
    "extend_prefix_lens_cpu",
    "extend_seq_lens_cpu",
    "extend_logprob_start_lens_cpu",
    "lora_ids",
    "rids",
    "top_logprobs_nums",
    "token_ids_logprobs",
    "next_token_logits_buffer",
)

_SCALAR_TENSOR_FIELDS = (
    "num_token_non_padded",
    "global_dp_buffer_len",
)

# ``SamplingBatchInfo`` fields sliced by ``slice_sampling_info``. Those slices
# are views into the parent row range, so they need the same treatment.
_SAMPLING_ROW_FIELDS = (
    "temperatures",
    "top_ps",
    "top_ks",
    "min_ps",
    "sampling_seed",
    "acc_additive_penalties",
    "acc_scaling_penalties",
    "logit_bias",
    "rids_int",
    "bootstrap_room_ids_int",
    "grammar_mask",
)


def _clone_field(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        # Row lists carry per-row objects whose tensors are not mutated by the
        # farm; a shallow copy is enough to stop the parent list from being
        # rebound underneath us.
        return list(value)
    return value


def snapshot_sampling_info(sampling_info: Any) -> Any:
    """Detach per-row sampling tensors from the parent view."""
    if sampling_info is None:
        return None
    for name in _SAMPLING_ROW_FIELDS:
        if not hasattr(sampling_info, name):
            continue
        value = getattr(sampling_info, name)
        cloned = _clone_field(value)
        if cloned is not value:
            try:
                setattr(sampling_info, name, cloned)
            except Exception:  # pragma: no cover - defensive
                logger.debug("farm could not snapshot sampling field %s", name)
    return sampling_info


def snapshot_forward_batch(forward_batch: Any) -> Any:
    """Detach a child ``ForwardBatch`` from its parent tensor views.

    Called once, at context adoption. After this the context owns every tensor
    the farm later reads, so overwriting the parent batch next forward cannot
    corrupt it.
    """
    if forward_batch is None:
        return None
    for name in (
        *_TOKEN_INDEXED_FIELDS,
        *_SEQ_INDEXED_FIELDS,
        *_SCALAR_TENSOR_FIELDS,
    ):
        if not hasattr(forward_batch, name):
            continue
        value = getattr(forward_batch, name)
        cloned = _clone_field(value)
        if cloned is value:
            continue
        try:
            setattr(forward_batch, name, cloned)
        except Exception:  # pragma: no cover - defensive
            logger.debug("farm could not snapshot forward-batch field %s", name)
    snapshot_sampling_info(getattr(forward_batch, "sampling_info", None))
    return forward_batch


@dataclass
class FarmContext:
    """One *group* of sequences at the same layer, spanning forwards.

    ``req_pool_idx`` is the stable identity of the group's first member. Never
    key on a batch row: rows are reused between steps and are not guaranteed to
    survive a deferral.

    The group, not the row, is the unit that advances: every member is at the
    same layer, so one A2F hop carries all of them. That keeps the remote FFN
    call fat (``n_rows`` tokens), which is what the FFN's per-call host dispatch
    cost demands. ``G=1`` degenerates to a single fat chain; one row per context
    is the other extreme.
    """

    req_pool_idx: int
    # Every member of the group, primary first. ``req_pool_idx`` is
    # ``req_pool_idxs[0]``; the rest is kept for single-row callers/tests.
    req_pool_idxs: List[int] = field(default_factory=list)
    ctx_id: int = 0
    # Owned tensors (copies, not parent views).
    hidden: Optional[torch.Tensor] = None
    residual: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    child_fb: Any = None
    # Dependency state.
    current_layer: int = 0
    prev_topk: Optional[torch.Tensor] = None
    pending_hop: Any = None
    output_ready: bool = False
    # Bookkeeping.
    age: int = 0
    sampled_token_ids: Optional[torch.Tensor] = None
    sampled_logits: Any = None

    @property
    def live(self) -> bool:
        """True while the context still owns work or an unsampled token."""
        return self.pending_hop is not None or not self.output_ready

    @property
    def n_tokens(self) -> int:
        if self.hidden is None:
            return 0
        return int(self.hidden.shape[0])

    @property
    def members(self) -> List[int]:
        """Every ``req_pool_idx`` in the group, primary first."""
        return list(self.req_pool_idxs) or [int(self.req_pool_idx)]

    def note_sample(self, token_ids: torch.Tensor, logits: Any) -> None:
        self.sampled_token_ids = token_ids
        self.sampled_logits = logits
        self.output_ready = True
        self.pending_hop = None


class PersistentFarmRuntime:
    """Process-local store of live :class:`FarmContext` objects."""

    def __init__(self) -> None:
        self._contexts: Dict[int, FarmContext] = {}
        self._order: List[int] = []
        self._by_ctx: Dict[int, FarmContext] = {}
        # member req_pool_idx -> group primary key. Every member is deferred
        # while the group is in flight, so this is the mask authority.
        self._members: Dict[int, int] = {}
        self.forward_id = 0
        self._queue_base = 0

    # -- lifecycle ---------------------------------------------------------
    def note_forward(self) -> int:
        self.forward_id += 1
        return self.forward_id

    def reset(self) -> None:
        self._contexts.clear()
        self._order.clear()
        self._by_ctx.clear()
        self._members.clear()
        self._queue_base = 0

    # -- lookup ------------------------------------------------------------
    def get(self, req_pool_idx: int) -> Optional[FarmContext]:
        key = int(req_pool_idx)
        primary = self._members.get(key)
        if primary is None:
            return None
        return self._contexts.get(primary)

    def get_by_ctx(self, ctx_id: int) -> Optional[FarmContext]:
        """Return the context a scheduler ticket belongs to."""
        return self._by_ctx.get(int(ctx_id))

    def live_contexts(self) -> List[FarmContext]:
        return [self._contexts[k] for k in self._order if k in self._contexts]

    def deferred_req_pool_indices(self) -> List[int]:
        """Every member of every live group, in admission order."""
        out: List[int] = []
        for key in self._order:
            ctx = self._contexts.get(key)
            if ctx is not None:
                out.extend(ctx.members)
        return out

    def pending_hops(self) -> List[Any]:
        return [
            ctx.pending_hop
            for ctx in self.live_contexts()
            if ctx.pending_hop is not None
        ]

    def __contains__(self, req_pool_idx: int) -> bool:
        return int(req_pool_idx) in self._members

    def __len__(self) -> int:
        return len(self._contexts)

    # -- mutation ----------------------------------------------------------
    def adopt(
        self,
        *,
        req_pool_idx: int,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        child_fb: Any,
        req_pool_idxs: Optional[Sequence[int]] = None,
    ) -> FarmContext:
        """Create a context for a group of newly sampled tokens.

        Tensors are cloned so the context survives the parent forward.
        """
        key = int(req_pool_idx)
        if key in self._members:
            raise RuntimeError(
                f"farm context for req_pool_idx={key} already exists; "
                "a sequence must be sampled before its next token is adopted"
            )
        members = [int(x) for x in (req_pool_idxs or [key])]
        if members[0] != key:
            members.insert(0, key)
        ctx = FarmContext(
            req_pool_idx=key,
            req_pool_idxs=members,
            hidden=hidden.detach().clone(),
            residual=None if residual is None else residual.detach().clone(),
            positions=positions.detach().clone(),
            child_fb=snapshot_forward_batch(child_fb),
        )
        self._contexts[key] = ctx
        self._order.append(key)
        for member in members:
            self._members[member] = key
        return ctx

    def drop(self, req_pool_idx: int) -> Optional[FarmContext]:
        key = int(req_pool_idx)
        primary = self._members.get(key, key)
        ctx = self._contexts.pop(primary, None)
        if ctx is not None:
            if primary in self._order:
                self._order.remove(primary)
            if ctx.ctx_id in self._by_ctx:
                del self._by_ctx[ctx.ctx_id]
            for member in ctx.members:
                if self._members.get(member) == primary:
                    del self._members[member]
            ctx.pending_hop = None
        return ctx

    def bind_ctx_id(self, req_pool_idx: int, ctx_id: int) -> None:
        key = int(req_pool_idx)
        ctx = self._contexts.get(self._members.get(key, key))
        if ctx is not None:
            ctx.ctx_id = int(ctx_id)
            self._by_ctx[int(ctx_id)] = ctx

    def alloc_queue_idxs(self, n: int) -> List[int]:
        """Reserve ``n`` consecutive scheduler indices.

        The farm queue identifies a run by consecutive indices
        (``take_contiguous_run``), and keys a context by ``ctx_id``. Using
        ``req_pool_idx`` here would break the run: pool indices are sparse, so a
        multi-row group would be split into several tickets and a single
        context's rows would then advance independently. A dense, monotonically
        growing block keeps a group contiguous and unique across contexts.
        """
        count = max(1, int(n))
        base = self._queue_base
        self._queue_base += count
        return list(range(base, base + count))

    def take_deferred(self) -> List[int]:
        """Return and keep the current deferred keys (order preserved)."""
        return self.deferred_req_pool_indices()

    def aged_out(self, max_age: int) -> List[int]:
        if max_age <= 0:
            return []
        return [
            ctx.req_pool_idx
            for ctx in self.live_contexts()
            if ctx.age >= max_age
        ]


_RUNTIME: Optional[PersistentFarmRuntime] = None


def get_persistent_runtime() -> PersistentFarmRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = PersistentFarmRuntime()
    return _RUNTIME


def reset_persistent_runtime() -> None:
    """Test hook: drop all live contexts."""
    runtime = get_persistent_runtime()
    runtime.reset()


# ---------------------------------------------------------------------------
# ForwardBatch attribute hand-off
# ---------------------------------------------------------------------------
# ``EagerRunner.load_batch`` builds the model's ForwardBatch with
# ``dataclasses.replace``, so it is a *different object* from the one the
# scheduler worker holds. Any ``_afd_farm_*`` attribute therefore has to be
# carried across explicitly, in both directions:
#
#   caller -> executed : ``_afd_farm_sample_fn`` must reach the model, or the
#                        persistent farm can never sample a token.
#   executed -> caller : the ready/deferred result must reach ``tp_worker``.
#
# Gated on the persistent flag so the one-shot farm path is unchanged.
_FARM_CARRIED_ATTRS = (
    "_afd_farm_sample_fn",
    "_afd_farm_context_sampler",
    "_afd_farm_logits_output",
    "_afd_farm_next_token_ids",
    "_afd_farm_handled",
    "_afd_farm_ready_req_pool_indices",
    "_afd_farm_deferred_req_pool_indices",
)


def copy_farm_attrs(src: Any, dst: Any) -> None:
    """Copy every farm attribute present on ``src`` onto ``dst``."""
    if src is None or dst is None or src is dst:
        return
    for name in _FARM_CARRIED_ATTRS:
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name))


def sync_farm_before_forward(caller_batch: Any, executed_batch: Any) -> None:
    """Carry farm inputs (notably the sampler) into the executed batch."""
    from sglang.srt.afd.farm.env import farm_persistent_enabled

    if not farm_persistent_enabled():
        return
    copy_farm_attrs(caller_batch, executed_batch)


def sync_farm_after_forward(executed_batch: Any, caller_batch: Any) -> None:
    """Carry the farm result out of the executed batch back to the caller."""
    from sglang.srt.afd.farm.env import farm_persistent_enabled

    if not farm_persistent_enabled():
        return
    copy_farm_attrs(executed_batch, caller_batch)


def farm_ready_row_mask(req_pool_indices: Any) -> Optional[torch.Tensor]:
    """Per-row bool mask: ``True`` where the row may advance this forward.

    ``None`` means "not in persistent farm mode, advance everything", which
    keeps the non-farm decode path byte-for-byte unchanged.

    The live context set is the *authority*, not a flag carried on the batch:
    ``ScheduleBatch.copy`` builds a fresh object for the overlap result queue,
    and the running batch is merged/filtered between forwards, so anything
    stored on a batch object would be lost or stale. The runtime is a
    process-level singleton keyed by the stable ``req_pool_idx``, which is
    exactly the identity the deferral is defined on.

    A row whose request still owns a context is mid-flight: it must not
    allocate a KV slot, must not advance ``seq_len``, and must not be treated
    as having produced a token.
    """
    from sglang.srt.afd.farm.env import farm_persistent_enabled

    if not farm_persistent_enabled():
        return None
    if not isinstance(req_pool_indices, torch.Tensor):
        return None
    deferred = set(get_persistent_runtime().deferred_req_pool_indices())
    if not deferred:
        return None
    rows = req_pool_indices.to("cpu").tolist()
    if not any(int(v) in deferred for v in rows):
        return None
    return torch.tensor([int(v) not in deferred for v in rows], dtype=torch.bool)
