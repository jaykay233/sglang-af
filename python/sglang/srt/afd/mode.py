# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from enum import Enum

from sglang.srt.environ import envs


class AfdMode(str, Enum):
    """Process role for Attention–FFN Disaggregation.

    Distinct from PD ``DisaggregationMode`` (KV transfer).
    """

    NULL = "null"
    ATTN = "attn"
    FFN = "ffn"

    @classmethod
    def from_str(cls, value: str | None) -> "AfdMode":
        if value is None or value == "":
            return cls.NULL
        v = value.strip().lower()
        for m in cls:
            if m.value == v:
                return m
        raise ValueError(
            f"Unknown AFD mode {value!r}; expected one of "
            f"{[m.value for m in cls]}"
        )


def get_afd_mode() -> AfdMode:
    return AfdMode.from_str(envs.SGLANG_AFD_MODE.get())


def get_afd_transport_name() -> str:
    name = (envs.SGLANG_AFD_TRANSPORT.get() or "fake").strip().lower()
    if name in ("nvlink", "cuda-ipc", "p2p"):
        name = "cuda_ipc"
    if name not in ("fake", "stepmesh", "cuda_ipc"):
        raise ValueError(
            f"Unknown AFD transport {name!r}; expected 'fake', 'stepmesh', "
            f"or 'cuda_ipc' (alias nvlink)"
        )
    return name
