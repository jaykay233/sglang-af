# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AFD FakeTransport, bootstrap policy, and graph-break bridge."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.afd.attn_bridge import maybe_remote_ffn
from sglang.srt.afd.bootstrap import apply_afd_cuda_graph_policy
from sglang.srt.afd.buffers import AfdBufferPool, AfdBufferPoolConfig
from sglang.srt.afd.mode import AfdMode
from sglang.srt.afd.protocol import AfdServerBatch, gen_pull_key, gen_push_key
from sglang.srt.afd.runtime import (
    get_afd_runtime,
    init_afd_runtime,
    shutdown_afd_runtime,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase


class TestAfdKeys(unittest.TestCase):
    def test_push_pull_bits(self):
        push = gen_push_key(3, microbatch=1, worker_rank=2)
        pull = gen_pull_key(3, microbatch=1, worker_rank=2)
        self.assertEqual(push & ((1 << 24) - 1), pull & ((1 << 24) - 1))
        self.assertEqual(pull >> 24, 1)
        self.assertEqual(push >> 24, 0)


class TestAfdBufferPool(unittest.TestCase):
    def test_fill_and_pad(self):
        device = "cpu"
        pool = AfdBufferPool(
            AfdBufferPoolConfig(
                num_mb=1,
                max_num_token=8,
                hidden_size=4,
                dtype=torch.float32,
                device=device,
            )
        )
        x = torch.randn(3, 4)
        payload = pool.fill_a2f(0, hidden=x, layer_id=7)
        self.assertEqual(int(payload.num_tokens.item()), 3)
        self.assertEqual(int(payload.layer_id.item()), 7)
        self.assertTrue(torch.allclose(payload.hidden[:3], x))
        self.assertTrue(torch.all(payload.hidden[3:] == 0))

    def test_fill_rejects_tokens_over_capacity(self):
        pool = AfdBufferPool(
            AfdBufferPoolConfig(
                num_mb=1,
                max_num_token=4,
                hidden_size=2,
                dtype=torch.float32,
                device="cpu",
            )
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"tokens=5 capacity=4.*SGLANG_AFD_MAX_NUM_TOKEN",
        ):
            pool.fill_a2f(0, hidden=torch.randn(5, 2), layer_id=0)


class TestFakeAfdRoundtrip(unittest.TestCase):
    def tearDown(self):
        shutdown_afd_runtime()
        envs.SGLANG_AFD_MODE.set("null")
        envs.SGLANG_AFD_TRANSPORT.set("fake")

    def test_identity_ffn_remote(self):
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_TRANSPORT.set("fake")
        envs.SGLANG_AFD_NUM_MB.set(1)
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(16)

        def identity_ffn(batch: AfdServerBatch):
            t = batch.num_tokens
            out = batch.hidden.clone()
            out[:t] = batch.hidden[:t] * 2
            return [out]

        rt = init_afd_runtime(
            hidden_size=8,
            mode=AfdMode.ATTN,
            transport_name="fake",
            device="cpu",
            dtype=torch.float32,
            ffn_compute=identity_ffn,
        )
        assert rt is not None
        x = torch.randn(5, 8)
        y = rt.remote_ffn(layer_id=0, hidden=x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, x * 2))

    def test_remote_ffn_splits_oversized_token_batch(self):
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_TRANSPORT.set("fake")
        envs.SGLANG_AFD_NUM_MB.set(1)
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(2)

        seen_chunks = []

        def chunked_ffn(batch: AfdServerBatch):
            t = batch.num_tokens
            seen_chunks.append(t)
            out = batch.hidden.clone()
            out[:t] = batch.hidden[:t] + 1
            return [out]

        rt = init_afd_runtime(
            hidden_size=4,
            mode=AfdMode.ATTN,
            transport_name="fake",
            device="cpu",
            dtype=torch.float32,
            ffn_compute=chunked_ffn,
        )
        assert rt is not None
        x = torch.arange(20, dtype=torch.float32).view(5, 4)
        y = rt.remote_ffn(layer_id=0, hidden=x)
        self.assertEqual(seen_chunks, [2, 2, 1])
        self.assertEqual(tuple(y.shape), (5, 4))
        self.assertTrue(torch.allclose(y, x + 1))

    def test_maybe_remote_ffn_bridge(self):
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(16)

        def mul2(batch: AfdServerBatch):
            t = batch.num_tokens
            out = batch.hidden.clone()
            out[:t] = batch.hidden[:t] * 2
            return [out]

        init_afd_runtime(
            hidden_size=4,
            mode=AfdMode.ATTN,
            transport_name="fake",
            device="cpu",
            dtype=torch.float32,
            ffn_compute=mul2,
        )
        x = torch.ones(2, 4)
        y = maybe_remote_ffn(layer_id=1, hidden_states=x)
        self.assertIsNotNone(y)
        self.assertTrue(torch.allclose(y, x * 2))

    def test_maybe_remote_ffn_null_mode(self):
        envs.SGLANG_AFD_MODE.set("null")
        self.assertIsNone(
            maybe_remote_ffn(layer_id=0, hidden_states=torch.zeros(1, 4))
        )


class TestAfdCudaIpcTransportName(unittest.TestCase):
    def tearDown(self):
        envs.SGLANG_AFD_TRANSPORT.set("fake")

    def test_alias_nvlink(self):
        from sglang.srt.afd.mode import get_afd_transport_name
        from sglang.srt.afd.transport import create_transport

        envs.SGLANG_AFD_TRANSPORT.set("nvlink")
        self.assertEqual(get_afd_transport_name(), "cuda_ipc")
        envs.SGLANG_AFD_TRANSPORT.set("cuda_ipc")
        self.assertEqual(get_afd_transport_name(), "cuda_ipc")
        t = create_transport("cuda_ipc")
        self.assertEqual(type(t).__name__, "CudaIpcAfdTransport")


class TestAfdCudaGraphPolicy(unittest.TestCase):
    def tearDown(self):
        envs.SGLANG_AFD_MODE.set("null")
        envs.SGLANG_AFD_LAYER_PIPELINE.set(False)
        envs.SGLANG_AFD_IN_GRAPH_WAIT.set(False)
        envs.SGLANG_AFD_STEPMESH_STAGES.set(0)

    def test_force_breakable(self):
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_LAYER_PIPELINE.set(False)
        decode = SimpleNamespace(backend=Backend.FULL)
        cfg = SimpleNamespace(decode=decode)
        args = SimpleNamespace(
            cuda_graph_config=cfg,
            _cuda_graph_config_locked=set(),
        )
        apply_afd_cuda_graph_policy(args)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.BREAKABLE)
        self.assertIn((Phase.DECODE, "backend"), args._cuda_graph_config_locked)

    def test_layer_pipeline_keeps_breakable_cg(self):
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_LAYER_PIPELINE.set(True)
        envs.SGLANG_AFD_IN_GRAPH_WAIT.set(False)
        envs.SGLANG_AFD_STEPMESH_STAGES.set(3)
        decode = SimpleNamespace(backend=Backend.FULL)
        cfg = SimpleNamespace(decode=decode)
        args = SimpleNamespace(
            cuda_graph_config=cfg,
            _cuda_graph_config_locked=set(),
        )
        apply_afd_cuda_graph_policy(args)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.BREAKABLE)
        envs.SGLANG_AFD_LAYER_PIPELINE.set(False)
        envs.SGLANG_AFD_STEPMESH_STAGES.set(0)

    def test_in_graph_wait_forces_full(self):
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_IN_GRAPH_WAIT.set(True)
        envs.SGLANG_AFD_LAYER_PIPELINE.set(False)
        decode = SimpleNamespace(backend=Backend.BREAKABLE)
        cfg = SimpleNamespace(decode=decode)
        args = SimpleNamespace(
            cuda_graph_config=cfg,
            _cuda_graph_config_locked=set(),
        )
        apply_afd_cuda_graph_policy(args)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
        envs.SGLANG_AFD_IN_GRAPH_WAIT.set(False)

    def test_ffn_forces_disabled(self):
        envs.SGLANG_AFD_MODE.set("ffn")
        decode = SimpleNamespace(backend=Backend.FULL)
        cfg = SimpleNamespace(decode=decode)
        args = SimpleNamespace(
            cuda_graph_config=cfg,
            _cuda_graph_config_locked=set(),
        )
        apply_afd_cuda_graph_policy(args)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertIn((Phase.DECODE, "backend"), args._cuda_graph_config_locked)

    def test_null_mode_noop(self):
        envs.SGLANG_AFD_MODE.set("null")
        decode = SimpleNamespace(backend=Backend.FULL)
        cfg = SimpleNamespace(decode=decode)
        args = SimpleNamespace(cuda_graph_config=cfg, _cuda_graph_config_locked=set())
        apply_afd_cuda_graph_policy(args)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)


class TestAfdFfnCudaGraphBuckets(unittest.TestCase):
    def test_pick_bucket(self):
        from sglang.srt.afd.ffn_cuda_graph import (
            default_ffn_cg_buckets,
            pick_ffn_cg_bucket,
        )

        buckets = default_ffn_cg_buckets(16)
        self.assertEqual(buckets[-1], 16)
        self.assertEqual(pick_ffn_cg_bucket(1, buckets), 1)
        self.assertEqual(pick_ffn_cg_bucket(3, buckets), 4)
        self.assertEqual(pick_ffn_cg_bucket(16, buckets), 16)


class TestAfdBootstrap(unittest.TestCase):
    def tearDown(self):
        shutdown_afd_runtime()
        envs.SGLANG_AFD_MODE.set("null")
        envs.SGLANG_AFD_TRANSPORT.set("fake")
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(256)

    def test_maybe_init_from_model_runner(self):
        from sglang.srt.afd.bootstrap import maybe_init_afd_from_model_runner

        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_TRANSPORT.set("fake")
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(8)

        runner = MagicMock()
        runner.model_config.hidden_size = 16
        runner.device = "cpu"
        runner.gpu_id = 0
        runner.dtype = torch.float32
        runner.tp_rank = 0
        runner.server_args.cuda_graph_config.decode.max_bs = 8

        maybe_init_afd_from_model_runner(runner)
        rt = get_afd_runtime()
        self.assertIsNotNone(rt)
        self.assertEqual(rt.mode, AfdMode.ATTN)
        self.assertGreaterEqual(rt.pool.cfg.max_num_token, 8)
        y = rt.remote_ffn(layer_id=0, hidden=torch.randn(3, 16))
        self.assertEqual(tuple(y.shape), (3, 16))


class TestAfdMoEProtocol(unittest.TestCase):
    def tearDown(self):
        shutdown_afd_runtime()
        envs.SGLANG_AFD_MODE.set("null")

    def test_detect_moe_topk(self):
        from sglang.srt.afd.moe_bridge import detect_moe_topk_from_config

        cfg = SimpleNamespace(
            n_routed_experts=256,
            num_experts_per_tok=8,
            n_shared_experts=1,
        )
        # Non-fused shared experts are not part of A2F topk buffers.
        self.assertEqual(detect_moe_topk_from_config(cfg), 8)
        cfg_v2_lite = SimpleNamespace(
            n_routed_experts=64,
            num_experts_per_tok=6,
            n_shared_experts=2,
            routed_scaling_factor=1.0,
        )
        self.assertEqual(detect_moe_topk_from_config(cfg_v2_lite), 8)
        self.assertEqual(detect_moe_topk_from_config(SimpleNamespace()), 0)

    def test_a2f_with_topk_roundtrip(self):
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(8)

        def check_topk(batch: AfdServerBatch):
            t = batch.num_tokens
            self.assertIsNotNone(batch.topk_ids)
            self.assertIsNotNone(batch.topk_weights)
            self.assertEqual(tuple(batch.topk_ids[:t].shape), (t, 4))
            # Weight-sum as "FFN" so Attn can verify routing arrived.
            w = batch.topk_weights[:t].sum(dim=-1, keepdim=True)
            out = batch.hidden.clone()
            out[:t] = batch.hidden[:t] * w
            return [out]

        rt = init_afd_runtime(
            hidden_size=4,
            mode=AfdMode.ATTN,
            transport_name="fake",
            device="cpu",
            dtype=torch.float32,
            ffn_compute=check_topk,
            moe_topk=4,
        )
        assert rt is not None
        x = torch.ones(3, 4)
        ids = torch.arange(12, dtype=torch.int32).view(3, 4)
        weights = torch.full((3, 4), 0.25, dtype=torch.float32)
        y = rt.remote_ffn(
            layer_id=2, hidden=x, topk_ids=ids, topk_weights=weights
        )
        # sum of weights per token = 1.0 → y == x
        self.assertTrue(torch.allclose(y, x))


class TestFusedSharedExpertWeightSplit(unittest.TestCase):
    def test_split_weights_match_wide_shared_mlp(self):
        from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
            DeepseekV2WeightLoaderMixin,
        )

        hidden_size = 3
        intermediate_size = 4
        output_size = 2
        num_fused = 2
        torch.manual_seed(0)

        wide_gate = torch.randn(intermediate_size, hidden_size)
        wide_up = torch.randn(intermediate_size, hidden_size)
        wide_down = torch.randn(output_size, intermediate_size)
        weights = [
            ("model.layers.0.mlp.shared_experts.gate_proj.weight", wide_gate),
            ("model.layers.0.mlp.shared_experts.up_proj.weight", wide_up),
            ("model.layers.0.mlp.shared_experts.down_proj.weight", wide_down),
            ("model.layers.0.self_attn.q_proj.weight", torch.randn(2, 2)),
        ]
        loader = SimpleNamespace(
            num_fused_shared_experts=num_fused,
            config=SimpleNamespace(n_routed_experts=64),
        )
        split_weights = dict(
            DeepseekV2WeightLoaderMixin._maybe_split_fused_shared_expert_weights(
                loader, weights
            )
        )

        self.assertIn("model.layers.0.self_attn.q_proj.weight", split_weights)
        for local_id in range(num_fused):
            prefix = f"model.layers.0.mlp.experts.{64 + local_id}"
            self.assertEqual(
                split_weights[f"{prefix}.gate_proj.weight"].shape,
                (intermediate_size // num_fused, hidden_size),
            )
            self.assertEqual(
                split_weights[f"{prefix}.up_proj.weight"].shape,
                (intermediate_size // num_fused, hidden_size),
            )
            self.assertEqual(
                split_weights[f"{prefix}.down_proj.weight"].shape,
                (output_size, intermediate_size // num_fused),
            )

        x = torch.randn(hidden_size)
        wide_output = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(x, wide_gate))
            * torch.nn.functional.linear(x, wide_up),
            wide_down,
        )
        split_output = torch.zeros_like(wide_output)
        for local_id in range(num_fused):
            prefix = f"model.layers.0.mlp.experts.{64 + local_id}"
            gate = split_weights[f"{prefix}.gate_proj.weight"]
            up = split_weights[f"{prefix}.up_proj.weight"]
            down = split_weights[f"{prefix}.down_proj.weight"]
            expert_output = torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(x, gate))
                * torch.nn.functional.linear(x, up),
                down,
            )
            split_output.add_(expert_output)

        self.assertTrue(torch.allclose(split_output, wide_output, atol=1e-6))


class TestAfdPipeline(unittest.TestCase):
    def tearDown(self):
        shutdown_afd_runtime()
        envs.SGLANG_AFD_MODE.set("null")
        envs.SGLANG_AFD_PIPELINE.set(False)
        envs.SGLANG_AFD_NUM_MB.set(1)
        envs.SGLANG_AFD_FAKE_ASYNC_FFN.set(False)
        envs.SGLANG_AFD_FAKE_TRANSFER_MS.set(0.0)

    def test_split_token_ranges(self):
        from sglang.srt.afd.pipeline import split_token_ranges

        self.assertEqual(split_token_ranges(5, 1), [(0, 5)])
        self.assertEqual(split_token_ranges(5, 3), [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(split_token_ranges(2, 3), [(0, 1), (1, 2)])
        self.assertEqual(split_token_ranges(0, 3), [])

    def test_pipelined_roundtrip(self):
        from sglang.srt.afd.pipeline import remote_ffn_pipelined

        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_NUM_MB.set(3)
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(32)
        envs.SGLANG_AFD_PIPELINE.set(True)
        envs.SGLANG_AFD_FAKE_ASYNC_FFN.set(False)
        envs.SGLANG_AFD_FAKE_TRANSFER_MS.set(0.0)

        seen_mbs = []

        def ffn(batch: AfdServerBatch):
            # layer_id unused; prove each mb sees a slice
            t = batch.num_tokens
            seen_mbs.append(t)
            out = batch.hidden.clone()
            out[:t] = batch.hidden[:t] + 1
            return [out]

        rt = init_afd_runtime(
            hidden_size=4,
            mode=AfdMode.ATTN,
            transport_name="fake",
            device="cpu",
            dtype=torch.float32,
            ffn_compute=ffn,
            moe_topk=0,
        )
        assert rt is not None
        x = torch.arange(20, dtype=torch.float32).view(5, 4)
        y = remote_ffn_pipelined(layer_id=0, hidden=x, runtime=rt)
        self.assertEqual(tuple(y.shape), (5, 4))
        self.assertTrue(torch.allclose(y, x + 1))
        # 5 tokens → 3 mb sizes 2,2,1
        self.assertEqual(sorted(seen_mbs), [1, 2, 2])

    def test_async_in_flight_guard(self):
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_NUM_MB.set(2)
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(16)
        # Need async Fake so push_pull returns before FFN finishes (in-flight slots).
        envs.SGLANG_AFD_FAKE_ASYNC_FFN.set(True)
        envs.SGLANG_AFD_FAKE_TRANSFER_MS.set(0.0)

        def slow_ffn(batch: AfdServerBatch):
            import time

            time.sleep(0.05)
            t = batch.num_tokens
            out = batch.hidden.clone()
            out[:t] = batch.hidden[:t]
            return [out]

        rt = init_afd_runtime(
            hidden_size=2,
            mode=AfdMode.ATTN,
            transport_name="fake",
            device="cpu",
            dtype=torch.float32,
            ffn_compute=slow_ffn,
        )
        assert rt is not None
        p0 = rt.remote_ffn_async(layer_id=0, hidden=torch.ones(1, 2), mb_id=0)
        p1 = rt.remote_ffn_async(layer_id=0, hidden=torch.ones(1, 2) * 2, mb_id=1)
        with self.assertRaises(RuntimeError):
            rt.remote_ffn_async(layer_id=0, hidden=torch.ones(1, 2), mb_id=0)
        y0 = rt.wait_remote_ffn(p0)
        y1 = rt.wait_remote_ffn(p1)
        self.assertTrue(torch.allclose(y0, torch.ones(1, 2)))
        self.assertTrue(torch.allclose(y1, torch.ones(1, 2) * 2))
        envs.SGLANG_AFD_FAKE_ASYNC_FFN.set(False)

    def test_pipeline_overlap_speedup(self):
        """Fake async + transfer delay: pipelined wall < sequential (Step-3 style)."""
        from sglang.srt.afd.bench_pipeline import _run_once

        seq_ms, _ = _run_once(
            pipelined=False,
            num_mb=3,
            num_tokens=24,
            hidden=8,
            ffn_ms=20.0,
            xfer_ms=20.0,
            rounds=3,
        )
        pipe_ms, _ = _run_once(
            pipelined=True,
            num_mb=3,
            num_tokens=24,
            hidden=8,
            ffn_ms=20.0,
            xfer_ms=20.0,
            rounds=3,
        )
        speedup = seq_ms / pipe_ms
        self.assertGreaterEqual(
            speedup,
            1.2,
            f"expected pipeline overlap seq={seq_ms:.1f} pipe={pipe_ms:.1f}",
        )


class TestAfdWeightFilter(unittest.TestCase):
    def tearDown(self):
        envs.SGLANG_AFD_MODE.set("null")
        envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(0)
        envs.SGLANG_AFD_REMOTE_MOE_ONLY.set(False)

    def test_classification(self):
        from sglang.srt.afd.weight_filter import (
            is_ffn_exclusive_weight,
            should_load_weight,
        )

        self.assertTrue(
            is_ffn_exclusive_weight("model.layers.0.mlp.experts.0.weight")
        )
        self.assertTrue(
            is_ffn_exclusive_weight("model.layers.0.mlp.shared_experts.down_proj.weight")
        )
        self.assertTrue(
            is_ffn_exclusive_weight("model.layers.0.mlp.gate_up_proj.weight")
        )
        self.assertTrue(
            is_ffn_exclusive_weight("model.layers.0.mlp.gate_proj.weight")
        )
        self.assertTrue(
            is_ffn_exclusive_weight("model.layers.0.mlp.up_proj.weight")
        )
        self.assertFalse(is_ffn_exclusive_weight("model.layers.0.mlp.gate.weight"))
        self.assertFalse(is_ffn_exclusive_weight("model.layers.0.self_attn.q_proj.weight"))

        self.assertTrue(
            should_load_weight(
                "model.layers.0.mlp.experts.0.weight", AfdMode.FFN
            )
        )
        self.assertFalse(
            should_load_weight(
                "model.layers.0.self_attn.q_proj.weight", AfdMode.FFN
            )
        )
        self.assertTrue(
            should_load_weight(
                "model.layers.0.mlp.gate.weight", AfdMode.ATTN
            )
        )
        self.assertFalse(
            should_load_weight(
                "model.layers.0.mlp.experts.0.weight", AfdMode.ATTN
            )
        )
        envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(9)
        self.assertTrue(
            should_load_weight(
                "model.layers.3.mlp.experts.0.weight", AfdMode.ATTN
            )
        )
        self.assertFalse(
            should_load_weight(
                "model.layers.9.mlp.experts.0.weight", AfdMode.ATTN
            )
        )
        envs.SGLANG_AFD_REMOTE_MOE_ONLY.set(True)
        self.assertTrue(
            should_load_weight(
                "model.layers.0.mlp.gate_up_proj.weight", AfdMode.ATTN
            )
        )
        envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(0)
        envs.SGLANG_AFD_REMOTE_MOE_ONLY.set(False)


class TestAfdRemotePolicy(unittest.TestCase):
    def tearDown(self):
        envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(0)
        envs.SGLANG_AFD_REMOTE_MOE_ONLY.set(False)

    def test_remote_from_layer_and_moe_only(self):
        from sglang.srt.afd.remote_policy import afd_should_remote_ffn

        envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(0)
        envs.SGLANG_AFD_REMOTE_MOE_ONLY.set(True)
        self.assertFalse(afd_should_remote_ffn(0, is_moe=False))
        self.assertTrue(afd_should_remote_ffn(1, is_moe=True))

        envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(9)
        self.assertFalse(afd_should_remote_ffn(8, is_moe=True))
        self.assertTrue(afd_should_remote_ffn(9, is_moe=True))

    def test_filter_iterator(self):
        from sglang.srt.afd.weight_filter import filter_weights_for_afd

        envs.SGLANG_AFD_MODE.set("ffn")
        weights = [
            ("model.layers.0.self_attn.q_proj.weight", torch.ones(2)),
            ("model.layers.0.mlp.experts.0.weight", torch.ones(3)),
            ("model.layers.0.mlp.gate.weight", torch.ones(1)),
        ]
        kept = list(filter_weights_for_afd(weights, AfdMode.FFN))
        self.assertEqual([n for n, _ in kept], ["model.layers.0.mlp.experts.0.weight"])


class TestAfdPdPolicy(unittest.TestCase):
    def tearDown(self):
        envs.SGLANG_AFD_MODE.set("null")
        envs.SGLANG_AFD_ALLOW_PREFILL_ATTN.set(False)

    def test_ffn_rejects_pd(self):
        from sglang.srt.afd.pd_policy import apply_afd_pd_policy

        envs.SGLANG_AFD_MODE.set("ffn")
        args = SimpleNamespace(disaggregation_mode="decode")
        with self.assertRaises(ValueError):
            apply_afd_pd_policy(args)

    def test_attn_decode_ok(self):
        from sglang.srt.afd.pd_policy import apply_afd_pd_policy

        envs.SGLANG_AFD_MODE.set("attn")
        args = SimpleNamespace(disaggregation_mode="decode")
        apply_afd_pd_policy(args)  # no raise

    def test_attn_prefill_refused(self):
        from sglang.srt.afd.pd_policy import apply_afd_pd_policy

        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_ALLOW_PREFILL_ATTN.set(False)
        args = SimpleNamespace(disaggregation_mode="prefill")
        with self.assertRaises(ValueError):
            apply_afd_pd_policy(args)


class TestAfdModuleStubs(unittest.TestCase):
    def tearDown(self):
        envs.SGLANG_AFD_MODE.set("null")
        envs.SGLANG_AFD_MODULE_STUBS.set(True)
        envs.SGLANG_AFD_ROUTING_SCHEME.set("a")
        envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(0)
        envs.SGLANG_AFD_REMOTE_MOE_ONLY.set(False)

    def test_skip_flags(self):
        from sglang.srt.afd.module_stubs import (
            afd_module_stubs_enabled,
            afd_skip_dense_mlp,
            afd_skip_entire_moe,
            afd_skip_experts,
            afd_skip_self_attn,
        )

        envs.SGLANG_AFD_MODE.set("null")
        self.assertFalse(afd_module_stubs_enabled())

        envs.SGLANG_AFD_MODE.set("ffn")
        envs.SGLANG_AFD_MODULE_STUBS.set(True)
        self.assertTrue(afd_skip_self_attn())
        self.assertFalse(afd_skip_experts())
        self.assertFalse(afd_skip_dense_mlp())

        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_ROUTING_SCHEME.set("a")
        self.assertFalse(afd_skip_self_attn())
        self.assertTrue(afd_skip_experts())
        self.assertTrue(afd_skip_dense_mlp())
        self.assertFalse(afd_skip_entire_moe())

        envs.SGLANG_AFD_ROUTING_SCHEME.set("b")
        self.assertFalse(afd_skip_experts())
        self.assertTrue(afd_skip_entire_moe())

        envs.SGLANG_AFD_MODULE_STUBS.set(False)
        self.assertFalse(afd_skip_entire_moe())

        # Local-FFN layers: do not stub experts / dense for layer_id < K.
        envs.SGLANG_AFD_MODULE_STUBS.set(True)
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_ROUTING_SCHEME.set("a")
        envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(9)
        envs.SGLANG_AFD_REMOTE_MOE_ONLY.set(True)
        self.assertFalse(afd_skip_experts(3))
        self.assertTrue(afd_skip_experts(9))
        self.assertFalse(afd_skip_dense_mlp(0))
        envs.SGLANG_AFD_REMOTE_FROM_LAYER.set(0)
        envs.SGLANG_AFD_REMOTE_MOE_ONLY.set(False)

    def test_stub_forward_raises(self):
        from sglang.srt.afd.module_stubs import AfdExpertsStub, AfdMissingModule

        m = AfdMissingModule("self_attn")
        with self.assertRaises(RuntimeError):
            m(torch.zeros(1))
        e = AfdExpertsStub()
        with self.assertRaises(RuntimeError):
            e(torch.zeros(1), None)
        self.assertFalse(e.should_fuse_routed_scaling_factor_in_topk)


class TestAfdParity(unittest.TestCase):
    def test_linear_local_vs_fake(self):
        from sglang.srt.afd.parity import compare_local_vs_afd_fake

        local, afd = compare_local_vs_afd_fake(
            hidden_size=8, num_tokens=5, seed=42, device="cpu"
        )
        self.assertEqual(tuple(local.shape), tuple(afd.shape))
        self.assertTrue(torch.allclose(local, afd, atol=1e-5))


class TestAfdSchemeBAndQuant(unittest.TestCase):
    def tearDown(self):
        envs.SGLANG_AFD_MODE.set("null")
        envs.SGLANG_AFD_ROUTING_SCHEME.set("a")
        envs.SGLANG_AFD_A2F_DTYPE.set("auto")

    def test_scheme_b_gate_weights(self):
        from sglang.srt.afd.weight_filter import should_load_weight

        envs.SGLANG_AFD_ROUTING_SCHEME.set("b")
        gate = "model.layers.0.mlp.gate.weight"
        experts = "model.layers.0.mlp.experts.0.weight"
        attn = "model.layers.0.self_attn.q_proj.weight"
        self.assertTrue(should_load_weight(gate, AfdMode.FFN))
        self.assertTrue(should_load_weight(experts, AfdMode.FFN))
        self.assertFalse(should_load_weight(attn, AfdMode.FFN))
        self.assertFalse(should_load_weight(gate, AfdMode.ATTN))
        self.assertFalse(should_load_weight(experts, AfdMode.ATTN))
        self.assertTrue(should_load_weight(attn, AfdMode.ATTN))

    def test_a2f_quant_roundtrip_bf16(self):
        from sglang.srt.afd.a2f_quant import (
            dequantize_hidden_from_a2f,
            quantize_hidden_for_a2f,
        )

        x = torch.randn(4, 8)
        q, scale = quantize_hidden_for_a2f(x, torch.bfloat16)
        self.assertIsNone(scale)
        y = dequantize_hidden_from_a2f(q, scale, torch.float32)
        self.assertTrue(torch.allclose(x.float(), y.float(), atol=1e-2))

    def test_a2f_fp8_roundtrip_if_available(self):
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("no float8")
        from sglang.srt.afd.a2f_quant import (
            dequantize_hidden_from_a2f,
            quantize_hidden_for_a2f,
        )

        x = torch.randn(4, 8)
        q, scale = quantize_hidden_for_a2f(x, torch.float8_e4m3fn)
        self.assertIsNotNone(scale)
        y = dequantize_hidden_from_a2f(q, scale, torch.float32)
        self.assertTrue(torch.allclose(x.float(), y, rtol=0.08, atol=0.08))

    def test_buffer_fp8_fill_if_available(self):
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("no float8")
        envs.SGLANG_AFD_A2F_DTYPE.set("fp8")
        pool = AfdBufferPool(
            AfdBufferPoolConfig(
                num_mb=1,
                max_num_token=8,
                hidden_size=4,
                dtype=torch.float32,
                device="cpu",
                a2f_wire_dtype=torch.float8_e4m3fn,
            )
        )
        x = torch.randn(3, 4)
        payload = pool.fill_a2f(0, hidden=x, layer_id=1)
        self.assertIsNotNone(payload.hidden_scale)
        tensors = payload.as_tensor_list()
        self.assertEqual(tensors[-1].numel(), 1)


class TestAfdLayerPipeline(unittest.TestCase):
    def tearDown(self):
        shutdown_afd_runtime()
        envs.SGLANG_AFD_MODE.set("null")
        envs.SGLANG_AFD_TRANSPORT.set("fake")
        envs.SGLANG_AFD_NUM_MB.set(1)
        envs.SGLANG_AFD_PIPELINE.set(False)
        envs.SGLANG_AFD_LAYER_PIPELINE.set(False)
        envs.SGLANG_AFD_TRUE_OVERLAP.set(False)
        envs.SGLANG_AFD_FAKE_ASYNC_FFN.set(False)
        envs.SGLANG_AFD_FAKE_TRANSFER_MS.set(0.0)
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(256)
        envs.SGLANG_AFD_STEPMESH_STAGES.set(0)

    def _init_async_rt(self, *, transfer_ms: float = 40.0, num_mb: int = 2):
        envs.SGLANG_AFD_MODE.set("attn")
        envs.SGLANG_AFD_TRANSPORT.set("fake")
        envs.SGLANG_AFD_NUM_MB.set(num_mb)
        envs.SGLANG_AFD_MAX_NUM_TOKEN.set(32)
        envs.SGLANG_AFD_FAKE_ASYNC_FFN.set(True)
        envs.SGLANG_AFD_FAKE_TRANSFER_MS.set(transfer_ms)
        envs.SGLANG_AFD_LAYER_PIPELINE.set(True)
        envs.SGLANG_AFD_PIPELINE.set(False)

        def mul2(batch: AfdServerBatch):
            t = batch.num_tokens
            out = batch.hidden.clone()
            out[:t] = batch.hidden[:t] * 2
            return [out]

        return init_afd_runtime(
            hidden_size=4,
            mode=AfdMode.ATTN,
            transport_name="fake",
            device="cpu",
            dtype=torch.float32,
            ffn_compute=mul2,
        )

    def test_enabled_gate(self):
        from sglang.srt.afd.layer_pipeline import afd_layer_pipeline_enabled

        envs.SGLANG_AFD_LAYER_PIPELINE.set(False)
        self.assertFalse(afd_layer_pipeline_enabled())

        self._init_async_rt(transfer_ms=0.0)
        self.assertTrue(afd_layer_pipeline_enabled())

        envs.SGLANG_AFD_PIPELINE.set(True)
        self.assertFalse(afd_layer_pipeline_enabled())

    def test_true_overlap_env_wires_layer_pipe(self):
        from sglang.srt.afd.layer_pipeline import (
            apply_true_overlap_env,
            afd_layer_pipeline_enabled,
        )

        envs.SGLANG_AFD_TRUE_OVERLAP.set(True)
        envs.SGLANG_AFD_LAYER_PIPELINE.set(False)
        envs.SGLANG_AFD_IN_GRAPH_WAIT.set(True)
        envs.SGLANG_AFD_NUM_MB.set(1)
        envs.SGLANG_AFD_USE_WAIT_FLAG.set(False)
        self.assertTrue(apply_true_overlap_env())
        self.assertTrue(envs.SGLANG_AFD_LAYER_PIPELINE.get())
        self.assertFalse(envs.SGLANG_AFD_IN_GRAPH_WAIT.get())
        self.assertGreaterEqual(int(envs.SGLANG_AFD_NUM_MB.get()), 2)
        self._init_async_rt(transfer_ms=0.0)
        self.assertTrue(afd_layer_pipeline_enabled())

    def test_issue_wait_parity(self):
        from sglang.srt.afd.attn_bridge import issue_remote_ffn, wait_remote_ffn

        self._init_async_rt(transfer_ms=5.0)
        x = torch.randn(3, 4)
        y_sync = get_afd_runtime().remote_ffn(layer_id=0, hidden=x, mb_id=0)
        # mb_id=1 for async so we don't collide with in-flight bookkeeping after sync.
        pending = issue_remote_ffn(
            layer_id=1, hidden_states=x, mb_id=1
        )
        y_async = wait_remote_ffn(pending)
        self.assertTrue(torch.allclose(y_sync, x * 2))
        self.assertTrue(torch.allclose(y_async, x * 2))

    def test_stagger_faster_than_sequential(self):
        """P7 schedule should hide FFN latency of mb0 behind Attn of mb1."""
        import time

        from sglang.srt.afd.attn_bridge import issue_remote_ffn, wait_remote_ffn

        transfer_ms = 50.0
        self._init_async_rt(transfer_ms=transfer_ms)
        attn_ms = 10.0
        n_layers = 3
        xs = [
            [torch.randn(2, 4), torch.randn(2, 4)] for _ in range(n_layers)
        ]

        def run_sequential():
            outs = []
            for li in range(n_layers):
                layer_outs = []
                for mb in range(2):
                    time.sleep(attn_ms / 1000.0)
                    y = get_afd_runtime().remote_ffn(
                        layer_id=li, hidden=xs[li][mb], mb_id=mb
                    )
                    layer_outs.append(y)
                outs.append(layer_outs)
            return outs

        def run_stagger():
            outs = [[None, None] for _ in range(n_layers)]
            pending = [None, None]
            for li in range(n_layers):
                for mb in range(2):
                    if pending[mb] is not None:
                        outs[li - 1][mb] = wait_remote_ffn(pending[mb])
                        pending[mb] = None
                    time.sleep(attn_ms / 1000.0)
                    pending[mb] = issue_remote_ffn(
                        layer_id=li, hidden_states=xs[li][mb], mb_id=mb
                    )
            for mb in range(2):
                outs[n_layers - 1][mb] = wait_remote_ffn(pending[mb])
            return outs

        t0 = time.perf_counter()
        seq = run_sequential()
        t_seq = time.perf_counter() - t0

        t0 = time.perf_counter()
        pipe = run_stagger()
        t_pipe = time.perf_counter() - t0

        for li in range(n_layers):
            for mb in range(2):
                self.assertTrue(torch.allclose(seq[li][mb], xs[li][mb] * 2))
                self.assertTrue(torch.allclose(pipe[li][mb], xs[li][mb] * 2))

        self.assertLess(
            t_pipe,
            t_seq * 0.85,
            f"expected layer pipeline wall time win: pipe={t_pipe:.3f}s seq={t_seq:.3f}s",
        )

    def test_balanced_seq_ranges_mb3(self):
        from sglang.srt.afd.layer_pipeline import _balanced_seq_ranges

        ranges = _balanced_seq_ranges(n_seq=9, num_mb=3, token_num_per_seq=1)
        self.assertEqual(len(ranges), 3)
        self.assertEqual(ranges[0], (0, 3, 0, 3))
        self.assertEqual(ranges[1], (3, 6, 3, 6))
        self.assertEqual(ranges[2], (6, 9, 6, 9))

    def test_run_layers_pipelined_parity(self):
        from sglang.srt.afd import layer_pipeline as lp
        from sglang.srt.afd.layer_pipeline import AfdMbSlice, run_layers_pipelined

        self._init_async_rt(transfer_ms=5.0)

        class _MockLayer:
            def __init__(self, layer_id: int):
                self.layer_id = layer_id

            def forward_pre_ffn(
                self,
                positions,
                hidden_states,
                forward_batch,
                residual,
                zero_allocator,
                gemm_output_zero_allocator=None,
                llama_4_scaling=None,
                prev_topk_indices=None,
                captured_last_layer_outputs=None,
            ):
                meta = {
                    "forward_batch": forward_batch,
                    "should_allreduce_fusion": False,
                    "use_reduce_scatter": False,
                    "gemm_output_zero_allocator": None,
                    "topk_indices": prev_topk_indices,
                    "hidden_states_orig": hidden_states,
                    "hidden_for_mlp": hidden_states,
                }
                return hidden_states, residual, None, None, meta

            def forward_post_ffn(self, mlp_out, residual, meta):
                return mlp_out, residual, meta.get("topk_indices")

            def __call__(self, positions, hidden_states, forward_batch, residual, *args, **kwargs):
                h, r, _, _, meta = self.forward_pre_ffn(
                    positions, hidden_states, forward_batch, residual, None
                )
                from sglang.srt.afd.attn_bridge import maybe_remote_ffn

                out = maybe_remote_ffn(layer_id=self.layer_id, hidden_states=h)
                return self.forward_post_ffn(out, r, meta)

        H, T = 4, 4
        hidden = torch.arange(T * H, dtype=torch.float32).reshape(T, H)
        residual = hidden.clone()
        positions = torch.arange(T)
        fb = SimpleNamespace()  # unused by mock layers

        slices = [
            AfdMbSlice(
                mb_id=0,
                hidden_states=hidden[:2].clone(),
                residual=residual[:2].clone(),
                positions=positions[:2],
                forward_batch=fb,
                token_lo=0,
                token_hi=2,
            ),
            AfdMbSlice(
                mb_id=1,
                hidden_states=hidden[2:].clone(),
                residual=residual[2:].clone(),
                positions=positions[2:],
                forward_batch=fb,
                token_lo=2,
                token_hi=4,
            ),
        ]

        orig_split = lp.split_decode_mbs
        lp.split_decode_mbs = lambda **kwargs: slices
        try:
            layers = [_MockLayer(0), _MockLayer(1)]
            y_pipe, r_pipe, _ = run_layers_pipelined(
                layers,
                positions=positions,
                hidden_states=hidden.clone(),
                forward_batch=fb,
                residual=residual.clone(),
                zero_allocator=None,
            )
            # Each layer multiplies by 2 → 2 layers → *4
            self.assertTrue(torch.allclose(y_pipe, hidden * 4))
            self.assertEqual(y_pipe.shape, hidden.shape)
            self.assertIsNotNone(r_pipe)
        finally:
            lp.split_decode_mbs = orig_split


if __name__ == "__main__":
    unittest.main()
