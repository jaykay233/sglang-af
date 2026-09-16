# SPDX-License-Identifier: Apache-2.0
"""AFD MoE routing placement: scheme A (Attn) vs B (FFN)."""

from __future__ import annotations

from enum import Enum

from sglang.srt.environ import envs


class AfdRoutingScheme(str, Enum):
    """Where gate+topk runs.

    A — Attn computes routing; A2F carries topk_ids/weights (default, P1).
    B — FFN computes gate+topk+experts; A2F is hidden (+ meta) only.
    """

    A = "a"
    B = "b"

    @classmethod
    def from_str(cls, value: str | None) -> "AfdRoutingScheme":
        if value is None or value == "":
            return cls.A
        v = value.strip().lower()
        if v in ("a", "scheme_a", "attn"):
            return cls.A
        if v in ("b", "scheme_b", "ffn"):
            return cls.B
        raise ValueError(f"Unknown AFD routing scheme {value!r}; expected a|b")


def get_routing_scheme() -> AfdRoutingScheme:
    return AfdRoutingScheme.from_str(envs.SGLANG_AFD_ROUTING_SCHEME.get())


def is_scheme_a() -> bool:
    return get_routing_scheme() == AfdRoutingScheme.A


def is_scheme_b() -> bool:
    return get_routing_scheme() == AfdRoutingScheme.B
