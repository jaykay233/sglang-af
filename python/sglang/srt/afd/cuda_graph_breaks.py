# SPDX-License-Identifier: Apache-2.0
"""Helpers for breakable CUDA Graph + StepMesh wait_flag (P2).

P0.5 wires ``@eager_on_graph`` on ``attn_bridge._afd_remote_ffn_graph_break`` so
remote FFN is a CUDA Graph break under ``BreakableCudaGraphBackend``.

P2 will replace the host ``push_pull``+``wait`` inside that break with the
GPU ``write_flag`` / ``wait_flag`` pattern from StepMesh
``tests/fserver/test_kernel_wait.py`` (see ``AfdGraphSyncFlags``).
"""

from __future__ import annotations

from typing import Optional

import torch


def afd_should_use_breakable_cuda_graph(afd_enabled: bool) -> bool:
    """AFD decode must not use full CUDA Graph."""
    return afd_enabled


class AfdGraphSyncFlags:
    """GPU-visible flags for StepMesh wait_kernel pattern.

    Capture sequence (P2)::

        seq_add_one(seq)
        write_flag(signal_dev, seq)
        # CPU thread: push_pull + wait → ack_host = seq
        wait_flag(ack_dev, seq)

    True-overlap (deferred wait) uses one flag set **per microbatch** so two
    outstanding issues do not share a single sequence counter.
    """

    def __init__(self, device: str = "cuda"):
        self.sequence = torch.zeros(1, dtype=torch.int64, device=device)
        self.signal_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self.ack_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self.signal_dev: Optional[torch.Tensor] = None
        self.ack_dev: Optional[torch.Tensor] = None
        self._device = device

    def map_with_stepmesh(self, fserver_lib, gpu: int) -> None:
        self.signal_dev = fserver_lib.map_pinned_tensor(self.signal_host, gpu)
        self.ack_dev = fserver_lib.map_pinned_tensor(self.ack_host, gpu)

    def reset(self) -> None:
        self.sequence.zero_()
        self.signal_host.zero_()
        self.ack_host.zero_()
