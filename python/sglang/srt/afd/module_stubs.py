# SPDX-License-Identifier: Apache-2.0
"""Skip constructing unused modules for AFD roles (P5/P6).

When ``SGLANG_AFD_MODULE_STUBS`` is true (default when AFD mode != null):

* **FFN worker** — do not build ``self_attn`` (MLA / KV dominate memory).
* **Attn worker (scheme A)** — do not build MoE ``experts`` / ``shared_experts``
  or dense MLP; keep ``gate`` + ``topk`` for routing.
* **Attn worker (scheme B)** — stub the entire ``mlp`` (routing runs on FFN).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch.nn as nn

from sglang.srt.afd.mode import AfdMode, get_afd_mode
from sglang.srt.afd.routing_scheme import is_scheme_b
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def afd_module_stubs_enabled() -> bool:
    mode = get_afd_mode()
    if mode == AfdMode.NULL:
        return False
    return bool(envs.SGLANG_AFD_MODULE_STUBS.get())


def afd_skip_self_attn(layer_id: Optional[int] = None) -> bool:
    """Skip building unused self_attn for AFD roles.

    * **FFN**: stub self_attn except merge-group interior layers (Attn runs there).
    * **Attn**: stub self_attn on merge interiors (entire layer runs on FFN).
    """
    if not afd_module_stubs_enabled():
        return False
    mode = get_afd_mode()
    from sglang.srt.afd.remote_policy import afd_is_merge_interior, layer_merge_k

    if mode == AfdMode.ATTN:
        if layer_id is not None and afd_is_merge_interior(int(layer_id)):
            return True
        return False

    if mode != AfdMode.FFN:
        return False
    if layer_id is not None:
        if afd_is_merge_interior(int(layer_id)):
            return False
        return True
    # No layer id: if merge is on, do not globally stub (layer ctor passes id).
    if layer_merge_k() > 1:
        return False
    return True


def afd_skip_experts(layer_id: Optional[int] = None) -> bool:
    """Skip MoE experts / shared_experts on Attn (scheme A keeps gate+topk).

    When ``layer_id`` is given and that layer keeps local FFN, do not stub.
    """
    if not (
        afd_module_stubs_enabled()
        and get_afd_mode() == AfdMode.ATTN
        and not is_scheme_b()
    ):
        return False
    if layer_id is not None:
        from sglang.srt.afd.remote_policy import afd_attn_keeps_local_ffn

        if afd_attn_keeps_local_ffn(int(layer_id), is_moe=True):
            return False
    return True


def afd_skip_dense_mlp(layer_id: Optional[int] = None) -> bool:
    """Skip dense MLP body on Attn (remote FFN runs it).

    Disabled for layers that keep local FFN (``REMOTE_MOE_ONLY`` dense).
    """
    if not (afd_module_stubs_enabled() and get_afd_mode() == AfdMode.ATTN):
        return False
    if layer_id is not None:
        from sglang.srt.afd.remote_policy import afd_attn_keeps_local_ffn

        if afd_attn_keeps_local_ffn(int(layer_id), is_moe=False):
            return False
    # Without layer_id: if MoE-only remote, dense is never stubbed globally.
    from sglang.srt.afd.remote_policy import remote_moe_only

    if remote_moe_only():
        return False
    return True


def afd_skip_entire_moe() -> bool:
    """Scheme B Attn: no local MoE module (gate+experts both remote)."""
    return (
        afd_module_stubs_enabled()
        and get_afd_mode() == AfdMode.ATTN
        and is_scheme_b()
    )


class AfdMissingModule(nn.Module):
    """Placeholder that must never run in the hot path."""

    def __init__(self, name: str):
        super().__init__()
        self._afd_stub_name = name

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            f"AFD stub module {self._afd_stub_name!r} was called; "
            f"check SGLANG_AFD_MODE / MODULE_STUBS wiring."
        )

    def prepare_qkv_latent(self, *args, **kwargs):
        raise RuntimeError(
            f"AFD stub self_attn.prepare_qkv_latent called ({self._afd_stub_name})"
        )

    def __repr__(self) -> str:
        return f"AfdMissingModule({self._afd_stub_name!r})"


class AfdExpertsStub(nn.Module):
    """Stand-in for FusedMoE on Attn so TopK can read fuse flags."""

    should_fuse_routed_scaling_factor_in_topk = False

    def __init__(self):
        super().__init__()
        self.moe_runner_config = None

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "AFD experts stub invoked on Attn worker; experts run remotely on FFN."
        )

    def forward_impl(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
