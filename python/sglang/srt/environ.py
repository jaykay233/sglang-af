import functools
import os
import subprocess
import warnings
from contextlib import ExitStack, contextmanager
from enum import IntEnum
from typing import Any, Optional


@functools.lru_cache(maxsize=1)
def _default_hip() -> bool:
    """Lazy ROCm/HIP detection for platform-conditional env defaults.

    Avoids importing torch at environ import time (this module is intentionally
    stdlib-only and loaded very early). Resolved on first EnvField.get() that uses
    it as a default, by which point torch is already imported in any real run;
    falls back to False if torch is unavailable.
    """
    try:
        import torch

        return torch.version.hip is not None
    except Exception:
        return False


@contextmanager
def temp_set_env(*, allow_sglang: bool = False, **env_vars: Any):
    """Temporarily set environment variables, restoring originals on exit.

    By default, SGLANG_*/SGL_* keys are rejected — use ``Envs`` descriptors
    for those.  Pass ``allow_sglang=True`` only for special env vars that
    intentionally bypass ``environ.py``.
    """
    if not allow_sglang:
        for key in env_vars:
            if key.startswith("SGLANG_") or key.startswith("SGL_"):
                raise ValueError("temp_set_env should not be used for sglang env vars")

    backup = {key: os.environ.get(key) for key in env_vars}
    try:
        for key, value in env_vars.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class EnvField:
    _allow_set_name = True

    def __init__(self, default: Any):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        assert EnvField._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def _resolve_default(self) -> Any:
        # Support a callable default for lazily/platform-computed defaults
        # (e.g. EnvBool(_default_hip)); evaluated only when the env is unset.
        return self.default() if callable(self.default) else self.default

    def get(self) -> Any:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self._resolve_default()

        try:
            return self.parse(value)
        except ValueError as e:
            default = self._resolve_default()
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{default}"'
            )
            return default

    def is_set(self):
        return self.name in os.environ

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvTuple(EnvField):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class _DeprecatedEnvFallback:
    """Mixin for EnvField subclasses: if the canonical env var is not set,
    check *deprecated_name* and emit DeprecationWarning before reading it.

    Usage:
        SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, deprecated_name="SGLANG_NSA_FUSE_TOPK")
    """

    def __init__(self, default: Any, deprecated_name: str):
        super().__init__(default)
        self.deprecated_name = deprecated_name

    def get(self) -> Any:
        if os.getenv(self.name) is None:
            fallback = os.getenv(self.deprecated_name)
            if fallback is not None:
                warnings.warn(
                    f"Environment variable '{self.deprecated_name}' is deprecated; "
                    f"use '{self.name}' instead. "
                    "The alias will be removed in a future release.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                os.environ[self.name] = fallback
        return super().get()


class EnvBoolWithAlias(_DeprecatedEnvFallback, EnvBool):
    pass


class EnvIntWithAlias(_DeprecatedEnvFallback, EnvInt):
    pass


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class ToolStrictLevel(IntEnum):
    """
    Defines the strictness levels for tool call parsing and validation.

    OFF: No strict validation
    FUNCTION: Enables structural tag constraints for all tools
    PARAMETER: Enforces strict parameter validation for all tools
    """

    OFF = 0
    FUNCTION = 1
    PARAMETER = 2


class Envs:
    # fmt: off

    # Model & File Download
    SGLANG_USE_MODELSCOPE = EnvBool(False)
    # Controls weight-file ordering for load-time I/O optimization.
    #   -1 : no sorting, no staggering; preserves original file order.
    #    0 : sort files only; maximizes ordering but may reduce cross-rank I/O concurrency.
    #   k>0: sort files and stagger per-rank order with factor k.
    #        Files are processed in groups of (tp_size * k), and rank r starts each
    #        group at offset (r * k), improving multi-rank I/O concurrency while
    #        keeping access relatively ordered.
    SGLANG_SORT_WEIGHT_FILES = EnvInt(0)
    SGLANG_DISABLED_MODEL_ARCHS = EnvTuple(tuple())
    SGLANG_PREFETCH_BLOCK_SIZE_MB = EnvInt(16)
    SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION = EnvBool(False)

    # Logging Options
    SGLANG_LOG_GC = EnvBool(False)
    SGLANG_LOG_FORWARD_ITERS = EnvBool(False)
    SGLANG_LOG_MS = EnvBool(False)
    SGLANG_LOG_REQUEST_EXCEEDED_MS = EnvInt(-1)
    SGLANG_LOG_REQUEST_HEADERS = EnvTuple(tuple())
    SGLANG_LOG_SCHEDULER_STATUS_TARGET = EnvStr("")
    SGLANG_LOG_SCHEDULER_STATUS_INTERVAL = EnvFloat(60.0)

    # SGLang CI
    SGLANG_IS_IN_CI = EnvBool(False)
    SGLANG_IS_IN_CI_AMD = EnvBool(False)
    SGLANG_CUDA_COREDUMP = EnvBool(False)
    # None = unset, letting get_dump_dir() resolve the base (RUNNER_TEMP in CI,
    # else /tmp); see debug_utils/cuda_coredump.py.
    SGLANG_CUDA_COREDUMP_DIR = EnvStr(None)
    SGLANG_TEST_MAX_RETRY = EnvInt(None)

    # Constrained Decoding (Grammar)
    SGLANG_GRAMMAR_POLL_INTERVAL = EnvFloat(0.005)
    SGLANG_GRAMMAR_MAX_POLL_ITERATIONS = EnvInt(10000)
    SGLANG_DISABLE_OUTLINES_DISK_CACHE = EnvBool(False)

    # Test & Debug
    SGLANG_DETECT_SLOW_RANK = EnvBool(False)
    SGLANG_TEST_STUCK_DETOKENIZER = EnvFloat(0)
    SGLANG_TEST_STUCK_DP_CONTROLLER = EnvFloat(0)
    SGLANG_TEST_STUCK_SCHEDULER_INIT = EnvFloat(0)
    SGLANG_TEST_STUCK_TOKENIZER = EnvFloat(0)
    SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS = EnvInt(0)
    IS_H200 = EnvBool(False)
    SGLANG_SET_CPU_AFFINITY = EnvBool(False)
    SGLANG_ENABLE_CP_V2 = EnvBool(False)
    SGLANG_PROFILE_WITH_STACK = EnvBool(True)
    SGLANG_PROFILE_RECORD_SHAPES = EnvBool(True)
    SGLANG_PROFILE_V2 = EnvBool(False)
    SGLANG_ENABLE_NVTX_SCHEDULER = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_ENABLE_NVTX"
    )
    SGLANG_ENABLE_NVTX_OPERATIONS = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_OPERATIONS_ENABLE_PROFILE"
    )
    SGLANG_RECORD_STEP_TIME = EnvBool(False)
    SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE = EnvBool(False)
    SGLANG_FORCE_SHUTDOWN = EnvBool(False)
    SGLANG_DEBUG_MEMORY_POOL = EnvBool(False)
    SGLANG_DEBUG_REVERT_PR = EnvInt(0)
    SGLANG_PHASE_CHECKER_DEBUG = EnvBool(False)
    SGLANG_TEST_REQUEST_TIME_STATS = EnvBool(False)
    SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(False)
    SGLANG_SIMULATE_ACC_LEN = EnvFloat(-1)
    SGLANG_SIMULATE_ACC_METHOD = EnvStr("match-expected")
    SGLANG_SIMULATE_UNIFORM_EXPERTS = EnvBool(False)
    SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS = EnvBool(False)
    SGLANG_TORCH_PROFILER_DIR = EnvStr("/tmp")
    SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS = EnvInt(500)
    SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE = EnvInt(64)
    SGLANG_NATIVE_MOVE_KV_CACHE = EnvBool(False)
    SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK = EnvBool(True)
    SGLANG_TEST_DISAGG_FAILURE_PROB = EnvFloat(0.0)

    # Scheduler: memory leak test
    SGLANG_TEST_RETRACT = EnvBool(False)
    SGLANG_TEST_RETRACT_INTERVAL = EnvInt(3)
    SGLANG_TEST_RETRACT_NO_PREFILL_BS = EnvInt(2 ** 31)
    # Scheduler: force lazy extra_buffer prealloc to fail at decode boundaries
    SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL = EnvBool(False)
    # KL tests: skip the cache-hit count assertion (e.g. when alloc failure reduces hits)
    SGLANG_TEST_SKIP_CACHE_HIT_ASSERT = EnvBool(False)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY = EnvInt(0)
    SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE = EnvBool(True)
    # Physical KV-page checks: committed<=allocated + no page alias.
    SGLANG_CHECK_KV_PAGE_INVARIANTS = EnvBool(False)

    # Load snapshot backend
    SGLANG_LOAD_SNAPSHOT_USE_ZMQ = EnvBool(False)

    # Scheduler: new token ratio hyperparameters
    SGLANG_INIT_NEW_TOKEN_RATIO = EnvFloat(0.7)
    SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR = EnvFloat(0.14)
    SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS = EnvInt(600)
    SGLANG_RETRACT_DECODE_STEPS = EnvInt(20)
    SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION = EnvInt(4096)

    # Scheduler: recv interval
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DEFAULT = EnvInt(1000)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_DECODE = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_TARGET_VERIFY = EnvInt(1)
    SGLANG_SCHEDULER_RECV_SKIPPER_WEIGHT_NONE = EnvInt(1)

    # PD Disaggregation (runtime)
    # NOTE: For SGLANG_DISAGGREGATION_THREAD_POOL_SIZE, the effective default is
    # computed dynamically at runtime based on cpu_count; see disaggregation backends.
    SGLANG_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(None)
    SGLANG_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    SGLANG_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    SGLANG_DISAGGREGATION_NIXL_BACKEND = EnvStr("UCX")
    SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS = EnvStr("{}")
    SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX = EnvBool(True)
    SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER = EnvBool(False)
    SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK = EnvBool(False)

    # Attention–FFN Disaggregation (StepMesh AFD) — see srt/afd/RFC.md
    # null | attn | ffn
    SGLANG_AFD_MODE = EnvStr("null")
    # fake | stepmesh
    SGLANG_AFD_TRANSPORT = EnvStr("fake")
    SGLANG_AFD_NUM_MB = EnvInt(2)
    SGLANG_AFD_MAX_NUM_TOKEN = EnvInt(256)
    # P2: use StepMesh write_flag/wait_flag around push_pull (requires stepmesh + CUDA)
    SGLANG_AFD_USE_WAIT_FLAG = EnvBool(False)
    # P3: split tokens across NUM_MB slots and overlap A2F/F2A (disabled under TRUE_OVERLAP)
    SGLANG_AFD_PIPELINE = EnvBool(False)
    # P7: stagger dual-mb Attn↔FFN across layers (requires NUM_MB>=2).
    # Forced on with TRUE_OVERLAP (AFD decode default).
    SGLANG_AFD_LAYER_PIPELINE = EnvBool(True)
    # StepMesh-style AFD stages (recommended 2). When >=2: forces LAYER_PIPELINE,
    # NUM_MB=stages, disables IN_GRAPH_WAIT and token PIPELINE (P3).
    SGLANG_AFD_STEPMESH_STAGES = EnvInt(2)
    # Fake only: serve collocated ffn_compute on a background thread so
    # remote_ffn_async / pipeline can overlap A2F enqueue with FFN (CPU-safe).
    # Keep off for CUDA Fake smoke (background CUDA contexts hang).
    SGLANG_AFD_FAKE_ASYNC_FFN = EnvBool(False)
    # Fake async: simulated A2F transfer latency (ms) before FFN sees the batch.
    SGLANG_AFD_FAKE_TRANSFER_MS = EnvFloat(0.0)
    # P4: move unused role params off GPU after load
    SGLANG_AFD_RELEASE_UNUSED_PARAMS = EnvBool(False)
    # P4: allow AFD attn + PD prefill (off by default)
    SGLANG_AFD_ALLOW_PREFILL_ATTN = EnvBool(False)
    # P5: skip constructing unused modules (attn on FFN / experts on Attn)
    SGLANG_AFD_MODULE_STUBS = EnvBool(True)
    # P6: MoE routing on Attn (a) or FFN (b)
    SGLANG_AFD_ROUTING_SCHEME = EnvStr("a")
    # P6: A2F hidden wire dtype — auto|bf16|fp16|fp8
    SGLANG_AFD_A2F_DTYPE = EnvStr("auto")
    # MxN: Attn worker rank in StepMesh key space (default: tp_rank)
    SGLANG_AFD_WORKER_RANK = EnvInt(-1)
    # Emit A2F issue/wait timestamps for StepMesh timeline reports
    SGLANG_AFD_TIMELINE = EnvBool(False)
    # Fine split of post_to_ffn (poll vs a2f sync) and FFN compute (wall vs cuda)
    SGLANG_AFD_PROFILE_DETAIL = EnvBool(False)
    # FFN worker: capture CUDA graphs for MLP/MoE compute (not model-level decode CG)
    SGLANG_AFD_FFN_CUDA_GRAPH = EnvBool(True)
    # Same-host cuda_ipc transport: Unix socket for CUDA IPC handle exchange
    SGLANG_AFD_IPC_ENDPOINT = EnvStr("/tmp/afd_cuda_ipc.sock")
    # P8: GPU write_flag/wait_flag inside decode CUDA Graph (full backend);
    # CPU thread runs transport push_pull+wait. Works with cuda_ipc or stepmesh.
    SGLANG_AFD_IN_GRAPH_WAIT = EnvBool(False)
    # FFN: wait up to this many microseconds to gather more ready mb slots
    # (same-layer concat). 0 = disabled (return first ready slot).
    SGLANG_AFD_FFN_GATHER_US = EnvInt(0)
    # FFN: max mb slots to gather into one compute when gather_us > 0.
    SGLANG_AFD_FFN_GATHER_MAX = EnvInt(4)
    # FFN per-layer bounded queue. This gathers ready A2F hops before compute,
    # independent of the short transport-level gather above.
    SGLANG_AFD_FFN_QUEUE_ENABLE = EnvBool(False)
    # Dispatch a layer once it holds this many tokens (or max hops / deadline).
    SGLANG_AFD_FFN_QUEUE_TARGET_TOKENS = EnvInt(64)
    # Maximum time a head-of-layer hop may wait for more same-layer work.
    SGLANG_AFD_FFN_QUEUE_MAX_WAIT_US = EnvInt(300)
    # Per-layer and global bounds on hops held by the queue.
    SGLANG_AFD_FFN_QUEUE_MAX_HOPS = EnvInt(4)
    SGLANG_AFD_FFN_QUEUE_MAX_GLOBAL_HOPS = EnvInt(16)
    # Attn-side per-layer outbound queue. It snapshots post-Attn microbatches
    # and coalesces same-layer hops before consuming an A2F slot.
    SGLANG_AFD_ATTN_QUEUE_ENABLE = EnvBool(False)
    # Keep the first pipeline layer on the direct path to avoid serializing
    # the dependency wavefront before it has any same-layer peers.
    SGLANG_AFD_ATTN_QUEUE_BYPASS_FIRST_LAYER = EnvBool(True)
    # Flush a layer once it holds this many tokens (or max hops / deadline).
    SGLANG_AFD_ATTN_QUEUE_TARGET_TOKENS = EnvInt(64)
    # Maximum time the oldest outbound hop may wait for same-layer work.
    SGLANG_AFD_ATTN_QUEUE_MAX_WAIT_US = EnvInt(200)
    # Maximum hops coalesced into one A2F call.
    SGLANG_AFD_ATTN_QUEUE_MAX_HOPS = EnvInt(4)
    # Log queue statistics every N issued groups (0 = silent).
    SGLANG_AFD_ATTN_QUEUE_STATS_EVERY = EnvInt(0)
    # FFN: run ready microbatches on separate CUDA streams (cuts peer-mb poll wait).
    SGLANG_AFD_FFN_PARALLEL_MB = EnvBool(False)
    # Experimental: capture FFN CUDA graphs against IPC shared tensors (often unsafe).
    SGLANG_AFD_FFN_SHARED_IO = EnvBool(False)
    # Fewer RTTs: dense MLP stays on Attn (no remote for DeepseekV2MLP layers).
    SGLANG_AFD_REMOTE_MOE_ONLY = EnvBool(False)
    # Deprecated / ignored: must stay 0 (all layers remote). Values >0 are clamped.
    SGLANG_AFD_REMOTE_FROM_LAYER = EnvInt(0)
    # Required Attn∥FFN overlap: NUM_MB>=2 + LAYER_PIPELINE + breakable CG +
    # deferred wait. Forced on at AFD bootstrap; IN_GRAPH_WAIT is disabled.
    SGLANG_AFD_TRUE_OVERLAP = EnvBool(True)
    # Layer-group merge: K transformer layers per A2F/F2A. K=1 off.
    # Interior Attn+KV run on FFN; forces REMOTE_FROM_LAYER=0 and disables IN_GRAPH.
    SGLANG_AFD_LAYER_MERGE_K = EnvInt(1)
    # AfPool MxN (experimental): Attn/FFN work-pool over cuda_ipc. Default off —
    # keeps classic 1A1F TRUE_OVERLAP. When on, KPI is tok/s + FFN util.
    SGLANG_AFD_POOL = EnvBool(False)
    SGLANG_AFD_POOL_NUM_ATTN = EnvInt(1)
    SGLANG_AFD_POOL_NUM_FFN = EnvInt(1)
    SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN = EnvInt(4)
    # Preserve the conservative serve-loop behavior by default. Disable for
    # pipelined FFN batch issue; per-slot events still protect transport reuse.
    SGLANG_AFD_POOL_SERVE_SYNC = EnvBool(True)
    SGLANG_AFD_POOL_ENDPOINT_DIR = EnvStr("/tmp/afd_pool")
    # least_inflight | rr
    SGLANG_AFD_POOL_ROUTE = EnvStr("least_inflight")
    # Local rank inside Attn or FFN pool (0 .. NUM_* - 1). -1 → WORKER_RANK or 0.
    SGLANG_AFD_POOL_LOCAL_RANK = EnvInt(-1)
    # Decode farm (P0/P1): tokens at different layers in one decode step.
    # Replaces lockstep layer-pipeline. cuda_ipc (or Fake). Default off.
    SGLANG_AFD_FARM = EnvBool(False)
    # Max sequences dequeued per Attn window (B_step). Typical {8,16,32}.
    SGLANG_AFD_FARM_B_STEP = EnvInt(16)
    # Number of independent sequence-context wavefronts. Contexts may be
    # smaller than B_step so sequence groups can advance at different layers.
    SGLANG_AFD_FARM_NUM_CONTEXTS = EnvInt(2)
    # Layer lead before injecting the next sequence context. 0 restores the
    # legacy behavior where every context starts at layer 0 together.
    SGLANG_AFD_FARM_CONTEXT_STAGGER_LAYERS = EnvInt(1)
    # Contexts admitted together at one stage. Same-stage contexts are
    # coalesced into one A2F call when they occupy the same layer.
    SGLANG_AFD_FARM_CONTEXTS_PER_STAGE = EnvInt(1)
    # Sticky same-layer windows before re-picking a layer (B_win / K).
    SGLANG_AFD_FARM_B_WIN_K = EnvInt(8)
    # Max in-flight A2F hops (also used to bump NUM_MB).
    SGLANG_AFD_FARM_MAX_INFLIGHT = EnvInt(4)
    # Max in-flight A2F hops from one layer. 0 = auto (min(max_inflight, 4)).
    # A finite cap prevents one ready layer from occupying every MB slot and
    # starving the next layer in the dependency wavefront.
    SGLANG_AFD_FARM_MAX_INFLIGHT_PER_LAYER = EnvInt(0)
    # Bounded poll window when no farm work can be issued. 0 preserves the
    # original fully blocking wait on the oldest hop.
    SGLANG_AFD_FARM_WAIT_SLICE_US = EnvInt(250)
    # Persistent farm runtime: sequence contexts (hidden/residual/positions,
    # pending hop, child ForwardBatch) survive across ``model.forward()`` so
    # different sequences can sit at different layers/token steps. One forward
    # may return zero sampled tokens. 0 keeps the one-shot intra-forward farm.
    SGLANG_AFD_FARM_PERSISTENT = EnvBool(False)
    # Max consecutive forwards a single context may stay live before the
    # runtime stops admitting new work for it (0 = unbounded).
    SGLANG_AFD_FARM_PERSISTENT_MAX_AGE = EnvInt(0)
    # Poll budget (us) per forward for pending hops. 0 = pure non-blocking poll
    # so the forward never sleeps; >0 allows one bounded drain window.
    SGLANG_AFD_FARM_PERSISTENT_POLL_US = EnvInt(0)
    # Number of groups the persistent farm partitions a decode batch into. Each
    # group is one context whose members share a layer, so one A2F hop carries
    # the whole group. 0 keeps the one-context-per-row behaviour (1 token/hop).
    # G groups give G fat independent chains instead of `rows` thin ones; the win
    # is that a hop costs the FFN a fixed host dispatch regardless of token count.
    SGLANG_AFD_FARM_PERSISTENT_GROUPS = EnvInt(0)
    # If FFN_GATHER_US is 0, farm sets it to this (LPU-sim same-layer gather).
    SGLANG_AFD_FARM_LPU_GATHER_US = EnvInt(50)
    # MoE combine (moe_sum_reduce) via @torch.compile for small token counts.    # progress.md §14: on this box Dynamo's per-call re-guard makes it 9x slower
    # than the plain sgl_kernel op for the farm's ~4-16 token hops, and its pure
    # Python guard evaluation holds the GIL away from the FFN poll thread.
    # 1 = stock upstream behaviour (torch.compile for num_tokens <= 32).
    SGLANG_AFD_MOE_SUM_REDUCE_COMPILE = EnvInt(1)
    # Minimum interval (ms) between full ``Scheduler.on_idle`` housekeeping
    # passes. 0 keeps upstream behaviour (every idle loop iteration). The idle
    # path is pure Python that holds the GIL, and an AFD FFN server never runs
    # real requests, so it spins that path thousands of times a second and
    # starves the FFN compute thread (progress.md §15).
    SGLANG_IDLE_HOUSEKEEPING_INTERVAL_MS = EnvInt(0)
    # Log occupancy every N decode farm forwards (0 = silent).
    SGLANG_AFD_FARM_LOG_EVERY = EnvInt(32)
    # P3 soft-persistent proxy (8-card): first B_step in a B_win pays weight tax.
    SGLANG_AFD_FARM_SOFT_PERSISTENT = EnvBool(False)
    # Host busy-wait microseconds on win_index==0 (Lite-scale default).
    SGLANG_AFD_FARM_WEIGHT_TAX_US = EnvFloat(100.0)
    # Optional HBM copy burn bytes on first window (0 = off, saves memory).
    SGLANG_AFD_FARM_WEIGHT_TAX_BYTES = EnvInt(0)
    # P2 true amortize: dequeue up to COALESCE_K * B_step tokens into one Attn
    # launch (stock kernels see larger M). 1 = disabled (single B_step).
    SGLANG_AFD_FARM_COALESCE_K = EnvInt(1)
    # Optional cap on tokens dequeued from one layer per pick. 0 preserves the
    # legacy COALESCE_K * B_STEP burst; small values improve cross-layer overlap.
    SGLANG_AFD_FARM_LAYER_BURST = EnvInt(0)
    # max = legacy sticky/largest-queue; oldest = advance the lowest ready layer.
    SGLANG_AFD_FARM_SCHED = EnvStr("max")
    # True pipeline (1F1B): split the decode batch into this many microbatches
    # and inject them at layer 0 at staggered times, so different microbatches
    # occupy different layers concurrently (layers_peak > 1). Without this the
    # wavefront is flat (all tokens share a layer) and no cross-layer overlap is
    # possible. 0/1 = disabled (legacy: enqueue every sequence at layer 0).
    SGLANG_AFD_FARM_STAGGER_MB = EnvInt(0)
    # Layers the leading microbatch must advance before the next one is injected.
    SGLANG_AFD_FARM_STAGGER_SPAN = EnvInt(2)
    # Diagnostics: per-phase wall-clock breakdown of the farm decode loop.
    SGLANG_AFD_FARM_PHASE_TIMING = EnvBool(False)
    # P3: try per-(layer, n_tok) CUDAGraph replay of forward_pre_ffn. Falls
    # back to eager when capture fails (common with FlashInfer host launch).
    SGLANG_AFD_FARM_LAYER_CG = EnvBool(False)
    # P3: Triton weight-outer / mb-inner linear on MLA projections (FP16/BF16).
    SGLANG_AFD_FARM_PERSISTENT_LINEAR = EnvBool(False)
    # P3c: spin-wait style session — keep W resident, push B_step mbs without
    # coalescing wait (host-driven; see farm/spin_wait_linear.py).
    SGLANG_AFD_FARM_SPIN_WAIT = EnvBool(False)
    # P0: hard cap on tokens holding an A2F credit (running) across all layers.
    # Reservation is refused once the budget would be exceeded; waiting backlog
    # is untouched. 0 = auto (MAX_INFLIGHT * B_STEP, i.e. the natural ceiling
    # implied by the hop cap); negative disables the budget entirely.
    SGLANG_AFD_FARM_GLOBAL_TOKEN_BUDGET = EnvInt(0)
    # P0: scheduling-round age at which a waiting window stops being polite:
    # it bypasses COALESCE_K (single B_step burst) and outranks sticky layers
    # so it cannot starve behind a sticky layer. 0 = auto (B_WIN_K); negative
    # disables aging.
    SGLANG_AFD_FARM_MAX_AGE_STEPS = EnvInt(0)
    # P1: let the Attn send queue's per-layer continuous batching choose the
    # A2F group size instead of pre-packing COALESCE_K windows on the Attn
    # scheduler (avoids FFN seeing a singleton pair). Forces COALESCE_K=1.
    SGLANG_AFD_FARM_NATURAL_BATCH = EnvBool(False)
    # P1: keep the farm scheduler/queue alive across decode steps and allow a
    # new batch to begin while an older one is still draining. Removes per-step
    # teardown; true cross-step pipelining additionally needs the model runner
    # to defer its OutputBarrier (see farm/README.md).
    SGLANG_AFD_FARM_PERSISTENT = EnvBool(False)
    # Build a sliced SamplingBatchInfo on *intermediate* farm hops too. Only the
    # hop that finishes a sequence is ever handed to the sampler, so for the
    # other 26 layers this is dead work on the attn critical path (py-spy §19:
    # slice_sampling_info ~11% of farm time). Kept as an escape hatch.
    SGLANG_AFD_FARM_MID_SAMPLING_INFO = EnvBool(False)
    # P1: reserved for concurrent active batches. Not implemented yet: queue
    # entries are bare indices into one decode forward's tensors, so a second
    # live batch would slice garbage. Concurrent batches also cannot overlap
    # across steps without a deferred consumer (the model forward finalises
    # logits immediately after the farm). Kept at 1; see farm/README.md.
    SGLANG_AFD_FARM_MAX_ACTIVE_BATCHES = EnvInt(1)

    # --- Fake MoE EP communication (fair full-decode baseline vs AF hop) ---
    # When on, each DeepseekV2MoE.forward pays dispatch+combine traffic tax
    # sized like DeepEP: 2 * N * topk * H * elem * (1-1/ep).
    SGLANG_FAKE_EP_COMM = EnvBool(False)
    # Assumed expert-parallel world size for the volume model.
    SGLANG_FAKE_EP_SIZE = EnvInt(8)
    # Effective interconnect bandwidth (GB/s) for delay-mode timing model.
    SGLANG_FAKE_EP_BW_GBS = EnvFloat(150.0)
    # Fixed per-layer launch latency (microseconds), applied in delay mode.
    SGLANG_FAKE_EP_LAT_US = EnvFloat(10.0)
    # delay | copy | nvlink
    SGLANG_FAKE_EP_MODE = EnvStr("copy")
    # Peer CUDA index for nvlink mode (must be visible in this process).
    SGLANG_FAKE_EP_PEER_GPU = EnvInt(-1)
    # If true, only tax AFD mode=null (colocated full). AF attn/ffn skip the tax.
    SGLANG_FAKE_EP_FULL_ONLY = EnvBool(True)
    # Optional volume overrides (0 = use real model topk / hidden).
    # Use with Lite weights to mimic V3-sized dispatch/combine payloads.
    SGLANG_FAKE_EP_TOPK = EnvInt(0)
    SGLANG_FAKE_EP_HIDDEN = EnvInt(0)
    # Extra multiplier on top of the volume model (layer-density, etc.).
    SGLANG_FAKE_EP_BYTES_SCALE = EnvFloat(1.0)
    # Extra MoE FLOPs as GEMM burn: scale≈1 means ~one [T,H]@[H,H]; V3-proxy ~7.
    # Applies on every DeepseekV2MoE.forward (PD decode + AF FFN), not Attn stubs.
    SGLANG_FAKE_MOE_COMPUTE_SCALE = EnvFloat(0.0)
    # Named profile: "" | "v3_proxy" (sets EP/topk/hidden/bytes/compute for Lite).
    SGLANG_FAKE_V3_PROFILE = EnvStr("")

    # Scheduler: others:
    SGLANG_EMPTY_CACHE_INTERVAL = EnvFloat(-1)  # in seconds. Set if you observe high memory accumulation over a long serving period.
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP = EnvBool(False)
    # Force-enable the WAR (write-after-read) barrier for the overlap scheduler
    # even when is_cuda() is False (e.g. AMD/ROCm). On CUDA the barrier is
    # already enabled regardless of this flag (see start_event_loop).
    SGLANG_ENABLE_WAR_BARRIER = EnvBool(False)
    # PP: skip output send/recv when the entire batch consists of non-final chunked prefill requests,
    # since process_batch_result_prefill discards next_token_ids for those anyway.
    SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM = EnvBool(False)
    SGLANG_SCHEDULER_MAX_RECV_PER_POLL = EnvInt(-1)
    SGLANG_EXPERIMENTAL_CPP_RADIX_TREE = EnvBool(False)
    SGLANG_RADIX_FORCE_MISS = EnvBool(False)
    SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR = EnvFloat(0.75)
    SGLANG_SCHEDULER_SKIP_ALL_GATHER = EnvBool(False)
    SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE = EnvBool(False)
    SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION = EnvBool(False)
    SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES = EnvInt(None)
    SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK = EnvFloat(None)
    SGLANG_DATA_PARALLEL_BUDGET_INTERVAL = EnvInt(1)
    SGLANG_REQ_WAITING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH = EnvBool(False)
    SGLANG_REQ_RUNNING_TIMEOUT = EnvFloat(-1)  # in seconds
    SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL = EnvInt(120)
    # Decode batches between SWA out-of-window evictions.
    SGLANG_SWA_EVICTION_INTERVAL = EnvInt(128)
    # For non-streaming requests, the scheduler still flushes intermediate
    # output batches to the tokenizer manager every N decoded tokens so that
    # `first_token_time`/TTFT can be recorded. Lower this (e.g. to 1) to get
    # an accurate TTFT for benchmarking; the upstream default of 50 trades
    # off some TTFT-metric accuracy for less IPC overhead.
    SGLANG_FORCE_STREAM_INTERVAL = EnvInt(50)

    # Test: pd-disaggregation
    SGLANG_TEST_PD_DISAGG_BACKEND = EnvStr("mooncake")
    SGLANG_TEST_PD_DISAGG_DEVICES = EnvStr(None)
    SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB = EnvFloat(0.0)

    SGLANG_TEST_SCRIPTED_RUNTIME = EnvBool(False)
    SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH = EnvStr(None)
    SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY = EnvStr(None)

    # Model Parallel
    SGLANG_USE_MESSAGE_QUEUE_BROADCASTER = EnvBool(True)
    SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS = EnvBool(False)
    # Comma-separated bundle indices for Ray Custom PG mode (e.g., "0,1,2,7").
    SGLANG_RAY_BUNDLE_INDICES = EnvStr("")
    # Override the distributed init method used by torch.distributed.init_process_group.
    # Set to "env://" to use an externally-created TCPStore via MASTER_ADDR/MASTER_PORT.
    SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE = EnvStr(None)
    SGLANG_TCP_STORE_PORT = EnvInt(29600)

    # Tool Calling
    SGLANG_FORWARD_UNKNOWN_TOOLS = EnvBool(False)

    # Hi-Cache
    SGLANG_HICACHE_HF3FS_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_DECODE_OFFLOAD_STRIDE = EnvInt(None)
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = EnvStr(None)
    # File-backend LRU eviction (opt-in; sizes accept SI/IEC suffixes, "0" disables).
    SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE = EnvStr(None)
    SGLANG_HICACHE_FILE_BACKEND_EVICTION_RATIO = EnvFloat(0.9)
    SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE = EnvStr("0")
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = EnvStr(None)
    # Enable O_DIRECT when opening NIXL POSIX backend files (bypasses OS page cache).
    # Disable with SGLANG_HICACHE_NIXL_USE_DIRECT_IO=0 or via the
    # "use_direct_io": false key in --hicache-storage-backend-extra-config.
    SGLANG_HICACHE_NIXL_USE_DIRECT_IO = EnvBool(True)
    SGLANG_HUGEPAGE_SIZE = EnvStr("")
    # Staging buffer for heterogeneous TP KV transfer
    SGLANG_DISAGG_STAGING_BUFFER = EnvBool(False)
    SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB = EnvInt(64)
    SGLANG_DISAGG_STAGING_POOL_SIZE_MB = EnvInt(4096)
    # TODO(yangminl): remove SGLANG_STAGING_USE_TORCH and the torch fallback in
    # staging_buffer.py once Triton kernels are fully validated in production.
    SGLANG_STAGING_USE_TORCH = EnvBool(False)
    # Mooncake KV Transfer
    SGLANG_MOONCAKE_CUSTOM_MEM_POOL = EnvStr(None)
    ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE = EnvBool(False)
    ASCEND_NPU_PHY_ID = EnvInt(-1)
    SGLANG_MOONCAKE_SEND_AUX_TCP = EnvBool(False)
    SGLANG_ENABLE_FAILED_SESSION_PROBE = EnvBool(False)
    SGLANG_FAILED_SESSION_PROBE_INTERVAL_S = EnvFloat(30.0)

    # Mooncake Store
    SGLANG_HICACHE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    SGLANG_HICACHE_MOONCAKE_REUSE_TE = EnvBool(True)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_CLIENT = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("rdma")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    MOONCAKE_STANDALONE_STORAGE = EnvBool(False)
    MOONCAKE_ENABLE_SSD_OFFLOAD = EnvBool(False)
    MOONCAKE_OFFLOAD_FILE_STORAGE_PATH = EnvStr(None)

    # MoRI KV Transfer
    # Send CPU-resident AUX data via RDMA instead of ZMQ TCP (default: TCP).
    SGLANG_MORI_SEND_AUX_RDMA = EnvBool(False)
    # Number of RDMA Queue Pairs (QPs) used per transfer operation. Higher
    # values can increase parallelism and bandwidth utilization.
    SGLANG_MORI_QP_PER_TRANSFER = EnvInt(4)
    # Number of RDMA work requests posted in a single batch to each QP. Larger
    # batch sizes reduce per-operation overhead and improve throughput at the
    # cost of higher latency. -1 selects automatic sizing based on the number
    # of merged work requests and available endpoints.
    SGLANG_MORI_POST_BATCH_SIZE = EnvInt(-1)
    # Number of worker threads in the RDMA executor thread pool. More workers
    # can improve parallelism for large batch transfers across multiple QPs,
    # but excessive threads may cause contention.
    SGLANG_MORI_NUM_WORKERS = EnvInt(4)
    # Number of sharded synchronous worker threads that drain KV transfers.
    # Also the bound on outstanding (posted-but-not-completed) transfers, so it
    # is the primary throttle keeping the RDMA send queue from overflowing.
    SGLANG_MORI_TRANSFER_SHARDS = EnvInt(8)
    # Poll cadence (ms) at which a transfer worker wakes to check the SLA while
    # waiting for completion; real completion still wakes it immediately.
    SGLANG_MORI_WAIT_POLL_MS = EnvInt(1000)
    # Per-transfer SLA (ms) before a KV transfer is failed; 0 disables the SLA
    # and relies on the RDMA retry-exceeded timeout only.
    SGLANG_MORI_TRANSFER_TIMEOUT_MS = EnvInt(0)

    # AMD & ROCm
    SGLANG_USE_AITER = EnvBool(False)
    SGLANG_USE_AITER_AG = EnvBool(True)
    # Use reduce_scatter (instead of all_reduce + dp_scatter) for the equal-chunk
    # MAX_LEN DP-MoE combine. Default ON for ROCm/HIP (uses the aiter custom
    # symmetric-memory kernel), OFF elsewhere (would fall back to RCCL); override
    # explicitly to force on/off on any platform.
    SGLANG_DP_USE_REDUCE_SCATTER = EnvBool(_default_hip)
    SGLANG_USE_AITER_UNIFIED_ATTN = EnvBool(False)
    # Select the gate/up tile layout for AITER MoE: True -> interleave
    # (matches FlyDSL `gate_mode="interleave"` kernels), False -> separated
    # (matches `gate_mode="separated"`, the layout used by gptoss_fp4 tuned
    # configs and by Mxfp4MoEMethod's post-fix weight shuffle).
    SGLANG_USE_AITER_MOE_GU_ITLV = EnvBool(True)
    # Fuse the `residual_add + RMSNorm + zero-pad` triplet that appears
    # before the MoE block for models whose MoE input hidden_size must be
    # padded up to a stride (e.g. GPT-OSS MXFP4 needs pad to multiple of
    # 256). When False (default) the pad runs as a separate
    # torch.nn.functional.pad call inside the MoE method. When True, the
    # aiter Triton kernel `fused_add_rmsnorm_pad` produces a padded
    # post-attention layernorm output in one launch and the MoE method
    # skips the explicit pad. Currently only takes effect on the
    # post_attention_layernorm path with aiter backend and TP=1.
    SGLANG_AITER_FUSE_RMSNORM_PAD = EnvBool(False)
    # Physical layout for MHA KV cache. "nhd" (default) keeps the existing
    # (size, head_num, head_dim) per-token storage that
    # `aiter.mha.mha_batch_prefill_func`/`unified_attention` consume directly.
    # "vectorized_5d" allocates K as (num_blocks, H_kv, head_dim/x, page_size, x)
    # and V as (num_blocks, H_kv, page_size/x, head_dim, x) (x = 16 / dtype_size),
    # matching the SHUFFLE layout that aiter's CK FmhaBatchPrefill kernel and
    # `aiter.ops.triton.gluon.pa_decode_gluon` both consume natively. This is
    # the SHUFFLE KV layout that enables pa_decode_gluon for full-attn
    # decode without runtime permutes.
    SGLANG_AITER_KV_CACHE_LAYOUT = EnvStr("nhd")
    SGLANG_ROCM_FUSED_DECODE_MLA = EnvBool(False)
    SGLANG_ROCM_DISABLE_LINEARQUANT = EnvBool(False)
    SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(4096)
    # Enable dual-stream MoE (shared experts vs routed experts) on the
    # ROCm/AITER path. Requires GPU_MAX_HW_QUEUES>=5 to avoid HW-queue serialization.
    SGLANG_ROCM_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_HACK_FLASHMLA_BACKEND = EnvStr("tilelang")

    # MPS (Apple Silicon)
    SGLANG_USE_MLX = EnvBool(False)
    SGLANG_MLX_USE_CUSTOM_ROPE = EnvBool(False)
    SGLANG_MLX_FUSE_SWIGLU = EnvBool(False)
    # Number of decode steps between periodic mx.clear_cache() calls.
    # Set to 0 to disable cache clearing entirely.
    SGLANG_MLX_CLEAR_CACHE_STEPS = EnvInt(256)

    # NPU
    SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT = EnvBool(False)
    SGLANG_NPU_USE_MULTI_STREAM = EnvBool(False)
    SGLANG_NPU_USE_MLAPO = EnvBool(False)
    # Forward native implementation for activation gelu tanh for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GELUTANH = EnvBool(False)
    # Forward native implementation for gemma rms norm for model Skywork-Reward-Gemma-2-27B-v0.2
    SGLANG_NPU_FORWARD_NATIVE_GEMMA_RMS_NORM = EnvBool(False)
    # Delay all-gather after qlora for better performance for Deepseek v3.2
    SGLANG_USE_AG_AFTER_QLORA = EnvBool(False)
    # Master switch for the experimental TRT-LLM LoRA fast path; when OFF (default) every
    # fine-grained opt switch reads False, keeping non-experimental paths byte-identical.
    SGLANG_EXPERIMENTAL_LORA_OPTI = EnvBool(False)
    # Quantize x to int8 in the dispatch operator
    DEEP_NORMAL_MODE_USE_INT8_QUANT = EnvBool(False) # This argument is deprecated
    SGLANG_NPU_FUSED_MOE_MODE = EnvInt(1)

    # MTHREADS & MUSA
    SGLANG_MUSA_FA3_FORCE_UPDATE_METADATA = EnvBool(False)

    # Quantization
    SGLANG_INT4_WEIGHT = EnvBool(False)
    SGLANG_CPU_QUANTIZATION = EnvBool(False)
    SGLANG_USE_DYNAMIC_MXFP4_LINEAR = EnvBool(False)
    SGLANG_FORCE_FP8_MARLIN = EnvBool(False)
    SGLANG_MOE_NVFP4_DISPATCH = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_GEMM_IN_ATTN = EnvBool(False)
    SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE = EnvBool(False)
    SGLANG_QUANT_ALLOW_DOWNCASTING = EnvBool(False)
    SGLANG_FP8_IGNORED_LAYERS = EnvStr("")
    SGLANG_FP4_IGNORED_LAYERS = EnvStr("")

    # Flashinfer
    SGLANG_IS_FLASHINFER_AVAILABLE = EnvBool(True)
    SGLANG_FLASHINFER_USE_PAGED = EnvBool(False)
    # Default to the pick from flashinfer
    SGLANG_FLASHINFER_WORKSPACE_SIZE = EnvInt(384 * 1024 * 1024)
    # Enable NVFP4 per-token activation scaling path for FlashInfer TRT-LLM MoE.
    SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION = EnvBool(False)
    # SGLang needs to know FlashInfer NVFP4 4over6 config to compute the global scale factor.
    FLASHINFER_NVFP4_4OVER6 = EnvBool(False)
    FLASHINFER_NVFP4_4OVER6_E4M3_USE_256 = EnvBool(False)
    # Skip-softmax threshold scale factor for TRT-LLM attention (prefill and decode separately).
    # None = standard attention. See https://arxiv.org/abs/2512.12087
    SGLANG_SKIP_SOFTMAX_PREFILL_THRESHOLD_SCALE_FACTOR = EnvFloat(None)
    SGLANG_SKIP_SOFTMAX_DECODE_THRESHOLD_SCALE_FACTOR = EnvFloat(None)

    # Triton
    SGLANG_TRITON_DECODE_ATTN_STATIC_KV_SPLITS = EnvBool(False)
    SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE = EnvBool(False)

    # Torch Compile
    SGLANG_ENABLE_TORCH_COMPILE = EnvBool(False)

    # EPLB
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_INPUT = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_CANARY = EnvBool(False)
    SGLANG_EXPERT_LOCATION_UPDATER_LOG_METRICS = EnvBool(False)
    SGLANG_LOG_EXPERT_LOCATION_METADATA = EnvBool(False)
    SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")
    SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL = EnvInt(0)
    SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC = EnvBool(False)

    # TBO
    SGLANG_TBO_DEBUG = EnvBool(False)

    # DeepGemm
    SGLANG_ENABLE_JIT_DEEPGEMM = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_PRECOMPILE = EnvBool(True)
    SGLANG_JIT_DEEPGEMM_FAST_WARMUP = EnvBool(False)
    SGLANG_JIT_DEEPGEMM_COMPILE_WORKERS = EnvInt(4)
    SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE = EnvBool(False)
    SGLANG_DG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/deep_gemm"))
    SGLANG_DG_USE_NVRTC = EnvBool(False)
    SGLANG_USE_DEEPGEMM_BMM = EnvBool(False)
    SGLANG_DEEPGEMM_SANITY_CHECK = EnvBool(False)
    SGLANG_DEEPGEMM_PDL = EnvBool(True)
    SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP = EnvBool(False)

    # DeepSeek MHA Optimization
    SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = EnvInt(8192)
    SGLANG_MAX_KV_CHUNK_CAPACITY = EnvInt(128 * 1024)

    # DeepEP
    SGLANG_DEEPEP_BF16_DISPATCH = EnvBool(False) # This argument is deprecated
    SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)
    SGLANG_DEEPEP_LL_COMBINE_SEND_NUM_SMS = EnvInt(32)
    SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO = EnvBool(False)
    # Force dynamic DeepEP Waterfill with runtime EP all-reduce instead of the
    # default static local-batch path.
    SGLANG_DISABLE_STATIC_WATERFILL = EnvBool(False)

    # NIXL-EP
    SGLANG_NIXL_EP_BF16_DISPATCH = EnvBool(False)
    SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK = EnvInt(128)

    # DSA Backend (canonical names; fall back to SGLANG_NSA_* with deprecation warning)
    SGLANG_DSA_FUSE_TOPK = EnvBoolWithAlias(True, deprecated_name="SGLANG_NSA_FUSE_TOPK")
    SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC = EnvBool(False)
    SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK = EnvStr(None)
    SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA = EnvBoolWithAlias(
        True, deprecated_name="SGLANG_NSA_ENABLE_MTP_PRECOMPUTE_METADATA"
    )
    SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD = EnvIntWithAlias(
        2048, deprecated_name="SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD"
    )
    SGLANG_DSA_HIP_DISABLE_PRESHUFFLE = EnvBoolWithAlias(
        False, deprecated_name="SGLANG_NSA_HIP_DISABLE_PRESHUFFLE"
    )
    SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION = EnvFloat(0.2)
    SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM = EnvBool(False)
    SGLANG_USE_FUSED_METADATA_COPY = EnvBool(True)
    SGLANG_DSA_TOPK_BROADCAST = EnvBool(False)

    # sgl-kernel
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK = EnvBool(False)

    # Flash Attention
    SGLANG_USE_SGL_FA3_KERNEL = EnvBool(True)

    # Kernels
    USE_TRITON_W8A8_FP8_KERNEL = EnvBool(False)
    SGLANG_RETURN_ORIGINAL_LOGPROB = EnvBool(False)
    SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    SGLANG_MOE_PADDING = EnvBool(False)
    SGLANG_CUTLASS_MOE = EnvBool(False)
    HF_HUB_DISABLE_XET = EnvBool(False)
    DISABLE_OPENAPI_DOC = EnvBool(False)
    SGLANG_ENABLE_TORCH_INFERENCE_MODE = EnvBool(False)
    SGLANG_IS_FIRST_RANK_ON_NODE = EnvBool(True)
    SGLANG_SYNC_TOKEN_IDS_ACROSS_TP = EnvBool(False)
    SGLANG_ENABLE_COLOCATED_BATCH_GEN = EnvBool(False)

    # Deterministic inference
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE = EnvBool(False)
    # Use 1-stage all-reduce kernel on AMD (deterministic, fixed accumulation order)
    # If not set: auto (enabled when --enable-deterministic-inference is on)
    # Set to 1: force enable (even without --enable-deterministic-inference)
    # Set to 0: force disable (use default Aiter AR even with --enable-deterministic-inference)
    SGLANG_USE_1STAGE_ALLREDUCE = EnvBool(False)
    SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2 = EnvBool(True)
    SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE = EnvInt(4096)
    SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE = EnvInt(2048)
    SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE = EnvInt(4096)
    SGLANG_TRITON_DECODE_SPLIT_TILE_SIZE = EnvInt(256)

    # RoPE cache configuration
    SGLANG_SPEC_EXPANSION_SAFETY_FACTOR = EnvInt(2)
    SGLANG_ROPE_CACHE_SAFETY_MARGIN = EnvInt(256)
    SGLANG_ROPE_CACHE_ALIGN = EnvInt(128)

    # Overlap Spec V2
    SGLANG_ENABLE_OVERLAP_PLAN_STREAM = EnvBool(False)

    # Spec Config
    SGLANG_SPEC_ENABLE_STRICT_FILTER_CHECK = EnvBool(True)
    # Skip draft_extend while adaptive spec is at steps=0 (drafting disabled).
    # Saves the per-step draft forward, but the draft KV goes stale: an upshift
    # back to steps>0 starts from a cold draft state (low accept until it recovers).
    SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND = EnvBool(False)
    # Use the split-KV (flash-decode) kernel for EAGLE target-verify on the
    # Triton backend (ROCm). Only active at speculative topk == 1; falls back to
    # extend_attention_fwd for unsupported cases or when set false (e.g. for
    # debugging). Correctness is unaffected; this only changes performance.
    SGLANG_ENABLE_SPLITKV_VERIFY = EnvBool(True)
    # Master switch for all async-asserted invariant probes (NaN, Inf, OOB,
    # page alignment). Off in prod; tests turn it on to fail-fast on
    # numerical / index violations instead of getting silent NaN cascades.
    SGLANG_ENABLE_ASYNC_ASSERT = EnvBool(False)
    # Sanitize NaN logits before sampling kernels and log a throttled warning
    # (see sanitize_nan_logits).
    SGLANG_SANITIZE_NAN_LOGITS = EnvBool(False)

    # VLM
    SGLANG_VLM_CACHE_SIZE_MB = EnvInt(100)
    SGLANG_IMAGE_MAX_PIXELS = EnvInt(16384 * 28 * 28)
    SGLANG_RESIZE_RESAMPLE = EnvStr("")
    SGLANG_MM_BUFFER_SIZE_MB = EnvInt(0)
    SGLANG_MM_PRECOMPUTE_HASH = EnvBool(False)
    SGLANG_VIT_ENABLE_CUDA_GRAPH = EnvBool(False)
    # Use the fully-vectorized ViT position-embedding interpolation (no per-image
    # Python loop / CPU<->GPU sync). Bit-exact with the legacy implementation;
    # set False to fall back to the per-image loop.
    SGLANG_VIT_ENABLE_VECTORIZED_POS_EMBED = EnvBool(True)
    SGLANG_MM_SKIP_COMPUTE_HASH = EnvBool(False)
    # For pre-tokenized (list[int]) multimodal prompts,
    # preserve the user's original tokens to avoid retokenization drift.
    SGLANG_MM_AVOID_RETOKENIZE = EnvBool(True)


    # VLM Item CUDA IPC Transport
    SGLANG_USE_CUDA_IPC_TRANSPORT = EnvBool(False)
    SGLANG_USE_IPC_POOL_HANDLE_CACHE = EnvBool(False)
    SGLANG_MM_FEATURE_CACHE_MB = EnvInt(1 * 1024)
    SGLANG_MM_ITEM_MEM_POOL_RECYCLE_INTERVAL_SEC = EnvFloat(0.05)

    # Mamba
    SGLANG_MAMBA_CONV_DTYPE = EnvStr("bfloat16")
    SGLANG_MAMBA_SSM_DTYPE = EnvStr(None)

    # Unified Radix Tree
    SGLANG_ENABLE_UNIFIED_RADIX_TREE = EnvBool(False)

    # Breakable CUDA Graph
    SGLANG_USE_BREAKABLE_CUDA_GRAPH = EnvBool(False)

    # Release & Resume Memory
    SGLANG_MEMORY_SAVER_CUDA_GRAPH = EnvBool(False)

    # Sparse Embeddings
    SGLANG_EMBEDDINGS_SPARSE_HEAD = EnvStr(None)

    # Logits processor
    SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK = EnvBool(False)
    SGLANG_LOGITS_PROCESSER_CHUNK_SIZE = EnvInt(2048)

    # Tool-Call behavior
    SGLANG_TOOL_STRICT_LEVEL = EnvInt(ToolStrictLevel.OFF)

    # Think tokens budget: negative means unlimited, >= 0 caps thinking tokens
    SGLANG_MAX_THINK_TOKENS = EnvInt(-1)

    # Ngram
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY = EnvBool(False)

    # Warmup
    SGLANG_WARMUP_TIMEOUT = EnvFloat(-1) # in seconds. If a warmup forward batch takes longer than this, the server will crash to prevent hanging. Recommend to increase warmup timeout to 1800 to accommodate some kernel JIT precache e.g. deep gemm

    # HTTP Server
    SGLANG_TIMEOUT_KEEP_ALIVE = EnvInt(5)
    # Uvicorn multiprocess supervisor pings each worker on this interval; default 5s is
    # too short when many workers cold-start and load tokenizers in parallel.
    SGLANG_UVICORN_WORKER_HEALTHCHECK_TIMEOUT = EnvInt(10)

    # Health Check
    SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION = EnvBool(True)

    # Crash diagnostics
    SGLANG_PYSPY_DUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH = EnvBool(True)
    SGLANG_CUDA_COREDUMP_BEFORE_CRASH_WAIT_SECS = EnvFloat(60.0)

    # Encoder gRPC
    SGLANG_ENCODER_GRPC_TIMEOUT_SECS = EnvInt(60)
    # Encoder receiver selection: http|grpc (used by EPD paths).
    SGLANG_ENCODER_MM_RECEIVER_MODE = EnvStr("http")

    # Native gRPC server (internal, not yet user-facing)
    SGLANG_GRPC_PORT = EnvInt(None)
    SGLANG_ENABLE_GRPC = EnvBool(False)

    # External models
    SGLANG_EXTERNAL_MODEL_PACKAGE = EnvStr("")
    SGLANG_EXTERNAL_MM_MODEL_ARCH = EnvStr("")
    SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE = EnvStr("")

    # Numa
    SGLANG_NUMA_BIND_V2 = EnvBool(True)
    SGLANG_AUTO_NUMA_BIND = EnvBool(False)
    SGLANG_CRASH_ON_NUMA_BIND_FAILURE = EnvBool(False)

    # Metrics
    SGLANG_ENABLE_METRICS_DEVICE_TIMER = EnvBool(False)
    SGLANG_ENABLE_METRICS_DP_ATTENTION = EnvBool(False)

    # Tokenizer (Kimi tiktoken: cache all_special_tokens / all_special_ids; the ITL can differ by +10x under high batch size).
    SGLANG_PATCH_TOKENIZER = EnvBool(True)

    # TokenizerManager
    SGLANG_REQUEST_STATE_WAIT_TIMEOUT = EnvInt(4)

    # ZBAL, zero buffer accelerate library, currently worked only in npu
    SGLANG_ZBAL_LOCAL_MEM_SIZE = EnvInt(0)
    SGLANG_ZBAL_BOOTSTRAP_URL = EnvStr("")

    SGLANG_DEFAULT_THINKING = EnvBool(False)

    # ====================================================================
    # DeepSeek V4
    SGLANG_OPT_DPSK_V4_RADIX = EnvBool(True)
    SGLANG_OPT_USE_OLD_COMPRESSOR = EnvBool(False)
    SGLANG_OPT_USE_TRITON_SWA_PREPARE = EnvBool(True)
    SGLANG_OPT_USE_AITER_MHC_PRE = EnvBool(True)
    SGLANG_OPT_USE_AITER_MHC_POST = EnvBool(True)
    SGLANG_OPT_USE_AITER_SILU_MUL = EnvBool(False)
    SGLANG_OPT_USE_FUSED_COMPRESS = EnvBool(False)
    SGLANG_OPT_USE_FUSED_COMPRESS_TRITON = EnvBool(False)
    SGLANG_OPT_USE_FUSED_QK_NORM_ROPE = EnvBool(True)
    SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL = EnvBool(True)
    SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION = EnvBool(True)
    SGLANG_FIX_MTP_HC_HIDDEN = EnvBool(False)
    # ====================================================================

    # Set False when using FP4-to-FP8 converted DeepSeek V4 checkpoint.
    SGLANG_DSV4_FP4_EXPERTS = EnvBool(True)
    # Default reasoning_effort for dsv4 chat encoder when request doesn't set it.
    # Accepts "", "max", "high" (empty string means unset); other values filtered to None.
    SGLANG_DSV4_REASONING_EFFORT = EnvStr("")

    # CUDA kernels
    SGLANG_OPT_DEEPGEMM_HC_PRENORM = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_PRE = EnvBool(True)
    SGLANG_OPT_USE_TILELANG_MHC_POST = EnvBool(True)
    SGLANG_DSV4_MHC_PREWARM = EnvBool(True)
    SGLANG_OPT_USE_TRITON_FUSED_MHC = EnvBool(True)
    SGLANG_OPT_FUSE_MHC_POST_PRE = EnvBool(False)
    SGLANG_OPT_USE_TILELANG_INDEXER = EnvBool(False)
    SGLANG_OPT_USE_AITER_INDEXER = EnvBool(False)
    SGLANG_OPT_USE_JIT_INDEXER_METADATA = EnvBool(True)
    SGLANG_OPT_USE_ONLINE_COMPRESS = EnvBool(False)
    SGLANG_EXPERIMENTAL_ONLINE_C128_MTP = EnvBool(False)
    SGLANG_DSV4_COMPRESS_STATE_DTYPE = EnvStr("float32")
    SGLANG_OPT_USE_COMPRESSOR_V2 = EnvBool(True)
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH = EnvBool(False)
    SGLANG_TOPK_TRANSFORM_512_TORCH = EnvBool(False)
    SGLANG_OPT_FLASHMLA_SPARSE_PREFILL = EnvBool(False)

    # SWA radix cache
    # TODO(DSV4): @ispobock this has bug on main branch when retract
    SGLANG_OPT_SWA_RADIX_CACHE_COMPACT = EnvBool(False)
    SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT = EnvBool(False)
    SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW = EnvBool(False)
    SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN = EnvBool(False)

    # Unified radix cache
    SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS = EnvBool(False)

    # DeepGemm Mega MoE
    SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE = EnvBool(False)
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK = EnvInt(1024)

    # When set, the mega-MoE x slot is packed E2M1 (FP4) instead of FP8 E4M3.
    # Halves symm-buffer footprint and unlocks the MXF4 mainloop downstream.
    # Setting this also exports DG_USE_FP4_ACTS=1 so DeepGEMM's symm-buffer
    # sizing + fp8_fp4_mega_moe pick up the FP4 layout.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS = EnvBool(False)
    # Switches the L1+L2 mainloops from kind::mxf8f6f4 (K=32 with-padding) to
    # kind::mxf4 (K=64 dense) inside fp8_fp4_mega_moe. No effect unless
    # SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS is also set; DeepGEMM asserts
    # this combination on the host side.
    SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND = EnvBool(False)
    SGLANG_OPT_FIX_MEGA_MOE_MEMORY = EnvBool(False)

    # TopK
    SGLANG_OPT_USE_FUSED_HASH_TOPK = EnvBool(True)
    SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK = EnvBool(True)
    SGLANG_OPT_USE_TOPK_V2 = EnvBool(True)

    # Single-group grouped top-k (n_group == topk_group == 1) over a softmax collapses
    # to a plain top-k, so run it as one fused ``topk_softmax`` kernel instead of the
    # ~8-op eager chain inside compiled ``grouped_topk_gpu``. ~8x faster at decode
    # batch sizes and strictly more accurate (fp32 softmax instead of bf16).
    SGLANG_OPT_DEGENERATE_GROUPED_TOPK_FUSED = EnvBool(True)

    # Fuse ``q_proj`` + ``kv_a_proj_with_mqa`` into one GEMM on the no-LoRA MLA path
    # (``q_lora_rank is None``). Both read the same ``hidden_states``, so at decode
    # batch sizes the second launch is pure dispatch overhead. The checkpoint splits
    # them; the loader concatenates them on load. Only engaged when attn_tp_size == 1
    # and quantization is off -- ``q_proj`` is ColumnParallelLinear (output-sharded)
    # while ``kv_a_proj_with_mqa`` is replicated, so a single replicated weight is
    # only equivalent with no sharding. Default off: A/B only.
    SGLANG_OPT_FUSE_QKV_A_PROJ_NOLORA = EnvBool(False)

    # Temporary: one-shot check that the fused weight matches the concatenation
    # of the two checkpoint weights (verify aid for the flag above).
    SGLANG_OPT_FUSE_QKV_A_PROJ_NOLORA_VERIFY = EnvBool(False)

    # MiniMax-M3 sparse decode indexer: single JIT radix-select kernel replaces the 2-stage split-K Triton topk.
    SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX = EnvBool(True)

    # MiniMax-M3 MXFP8 MoE experimental fusion toggles (default off; A/B only).
    SGLANG_MINIMAX_M3_FUSED_SWIGLU_MXFP8 = EnvBool(False)
    SGLANG_MINIMAX_M3_FUSED_MOE_COMBINE = EnvBool(False)

    # GEMM / kernel fusion
    SGLANG_OPT_FP8_WO_A_GEMM = EnvBool(True)
    SGLANG_OPT_BF16_FP32_GEMM_ALGO = EnvStr("cublas")
    SGLANG_OPT_USE_JIT_EP_ACTIVATION = EnvBool(True)
    SGLANG_OPT_FUSE_WQA_WKV = EnvBool(True)
    SGLANG_OPT_SWIGLU_CLAMP_FUSION = EnvBool(True)

    # Cache / overlap
    SGLANG_OPT_USE_FUSED_STORE_CACHE = EnvBool(True)
    SGLANG_OPT_USE_JIT_NORM = EnvBool(True)
    SGLANG_OPT_USE_MULTI_STREAM_OVERLAP = EnvBool(True)

    # CUDA graph
    SGLANG_PREP_IN_CUDA_GRAPH = EnvBool(True)

    # Eager forward wraps the ForwardBatch's own tensors instead of copying them
    # into the CUDA graph buffer registry (no per-iter device-to-device copy).
    SGLANG_EAGER_INPUT_NO_COPY = EnvBool(False)

    # Distributed
    SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER = EnvBool(True)
    SGLANG_SHARED_EXPERT_TP1 = EnvBool(False)
    # Replicate the input embedding across TP ranks instead of sharding it
    # along the vocab dimension (saves an all-reduce/all-gather in the embed
    # lookup at the cost of replicated embedding weights). Drives both the
    # target and every draft that shares its embedding (see
    # get_embedding_tp_kwargs); they must stay in lock-step. Currently only
    # applies to the Deepseek-V2 family (Deepseek V3.1, Kimi K2.5) + drafts.
    SGLANG_ENABLE_EMBED_REPLICATION = EnvBool(False)
    # Symmetric Memory
    SGLANG_SYMM_MEM_PREALLOC_GB_SIZE = EnvInt(-1)
    SGLANG_DEBUG_SYMM_MEM = EnvBool(False)

    # Aiter
    SGLANG_USE_AITER_FP8_PER_TOKEN = EnvBool(False)
    # fmt: on

    # EPD
    SGLANG_ENCODER_RECV_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_SEND_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_HTTP_TIMEOUT = EnvFloat(1800.0)
    SGLANG_ENCODER_REQ_TIMEOUT = EnvFloat(180.0)
    SGLANG_ENCODER_DISPATCH_MIN_ITEMS = EnvInt(2)
    SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU = EnvBool(False)
    SGLANG_ENCODER_MAX_BATCH_SIZE = EnvInt(8)
    SGLANG_ENCODER_PREPROC_WORKERS = EnvInt(8)
    # EncoderBootstrapServer health-check tuning.  Interval == 0 disables it.
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL = EnvFloat(10.0)
    SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT = EnvFloat(2.0)
    # Persistent receiver-side GPU embedding pool size for mooncake EPD transport.
    # 0 disables (per-request register/deregister). 4096 = 4GB default per TP
    SGLANG_EMBEDDING_POOL_SIZE_MB = EnvInt(4096)
    SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT = EnvInt(64)

    # Elastic EP Backup Port
    SGLANG_BACKUP_PORT_BASE = EnvInt(10000)

    # Sglang Cache Dir
    SGLANG_CACHE_DIR = EnvStr(os.path.expanduser("~/.cache/sglang"))
    SGLANG_FLASHINFER_AUTOTUNE_CACHE = EnvBool(True)
    SGLANG_ENABLE_MOE_DEFERRED_FINALIZE = EnvBool(False)

    # Plugin system
    SGLANG_PLATFORM = EnvStr("")
    SGLANG_PLUGINS = EnvStr("")

    # ===================================================================
    # KV-Canary / Token-Oracle (testing-only)
    # ===================================================================
    SGLANG_KV_CANARY_RING_CAPACITY = EnvInt(1024)
    SGLANG_KV_CANARY_STATS_PRINT_EVERY_N_STEPS = EnvInt(100)
    SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_PERTURB_REQ_TO_TOKEN_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_WARMUP_STEPS = EnvInt(50)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_USED_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_UNUSED_CACHE_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_REAL_KV_POST_FORWARD_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_PERTURB_TARGET_GROUP = EnvStr(None)
    SGLANG_KV_CANARY_PERTURB_NEXT_TOKEN_SWAP_PROB = EnvFloat(0.0)
    SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE = EnvBool(False)
    SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT = EnvBool(False)
    SGLANG_KV_CANARY_SWA_DIVERGENCE_STATS_INTERVAL = EnvInt(0)
    SGLANG_KV_CANARY_ENABLE_MHA_V = EnvBool(False)


envs = Envs()
EnvField._allow_set_name = False


def _print_deprecated_env(old_name: str, new_name: Optional[str] = None):
    if old_name in os.environ:
        if new_name is None:
            warnings.warn(f"Environment variable {old_name} has been deprecated.")
        else:
            warnings.warn(
                f"Environment variable {old_name} will be deprecated, please use {new_name} instead"
            )
            os.environ[new_name] = os.environ[old_name]


def _warn_deprecated_env_to_cli_flag(env_name: str, suggestion: str):
    """Warn when a deprecated environment variable is used.

    This is for env vars that are deprecated in favor of CLI flags.
    """
    if env_name in os.environ:
        warnings.warn(f"Environment variable {env_name} is deprecated. {suggestion}")


def _convert_SGL_to_SGLANG():
    _print_deprecated_env("SGLANG_GC_LOG", "SGLANG_LOG_GC")
    _print_deprecated_env(
        "SGLANG_CUTEDSL_MOE_NVFP4_DISPATCH", "SGLANG_MOE_NVFP4_DISPATCH"
    )
    _print_deprecated_env(
        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK",
    )
    _print_deprecated_env("SGLANG_PER_TOKEN_GROUP_QUANT_8BIT_V2")
    _print_deprecated_env("SGLANG_ENABLE_THINKING", "SGLANG_DEFAULT_THINKING")
    _print_deprecated_env("SGLANG_REASONING_EFFORT", "SGLANG_DSV4_REASONING_EFFORT")
    _print_deprecated_env(
        "SGLANG_USE_JIT_ALL_REDUCE", "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"
    )
    _deprecated_ms_to_s = {
        "SGLANG_QUEUED_TIMEOUT_MS": "SGLANG_REQ_WAITING_TIMEOUT",
        "SGLANG_FORWARD_TIMEOUT_MS": "SGLANG_REQ_RUNNING_TIMEOUT",
    }
    for old_name, new_name in _deprecated_ms_to_s.items():
        if old_name in os.environ:
            ms_val = os.environ[old_name]
            warnings.warn(
                f"Environment variable {old_name} (in ms) is deprecated, "
                f"please use {new_name} (in seconds) instead"
            )
            os.environ[new_name] = str(float(ms_val) / 1000.0)

    for key, value in os.environ.items():
        if key.startswith("SGL_"):
            new_key = key.replace("SGL_", "SGLANG_", 1)
            warnings.warn(
                f"Environment variable {key} is deprecated, please use {new_key}"
            )
            os.environ[new_key] = value


_convert_SGL_to_SGLANG()
_warn_deprecated_env_to_cli_flag(
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE",
    "Please use '--enable-prefill-delayer' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES",
    "Please use '--prefill-delayer-max-delay-passes' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK",
    "Please use '--prefill-delayer-token-usage-low-watermark' instead.",
)
_warn_deprecated_env_to_cli_flag(
    "SGLANG_DFLASH_PREFILL_REFILL_TARGET",
    "DFlash now auto-enables the min-free-slots delay; unset this env. To "
    "override the threshold, use '--min-free-slots-delay'.",
)

# Import cuda_coredump to trigger auto-injection of CUDA env vars
# when SGLANG_CUDA_COREDUMP=1. Best-effort; for strict guarantees,
# set CUDA_* env vars in the shell before launching Python.
import sglang.srt.debug_utils.cuda_coredump  # noqa: F401, E402  # isort: skip


def example_with_exit_stack():
    # Use this style of context manager in unit test
    exit_stack = ExitStack()
    exit_stack.enter_context(envs.SGLANG_TEST_RETRACT.override(False))
    assert envs.SGLANG_TEST_RETRACT.get() is False
    exit_stack.close()
    assert envs.SGLANG_TEST_RETRACT.get() is None


def example_with_subprocess():
    command = ["python", "-c", "import os; print(os.getenv('SGLANG_TEST_RETRACT'))"]
    with envs.SGLANG_TEST_RETRACT.override(True):
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        process.wait()
        output = process.stdout.read().decode("utf-8").strip()
        assert output == "True"

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.stdout.read().decode("utf-8").strip()
    assert output == "None"


def example_with_implicit_bool_avoidance():
    @contextmanager
    def assert_throws(message_matcher: str):
        try:
            yield
        except Exception as e:
            assert message_matcher in str(e), f"{e=}"
            print(f"assert_throws find expected error: {e}")
            return
        raise AssertionError("assert_throws do not see exceptions")

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if (1 != 1) or envs.SGLANG_TEST_RETRACT:
            pass

    with assert_throws("Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"):
        if envs.SGLANG_TEST_RETRACT or (1 == 1):
            pass


def examples():
    # Example usage for envs
    envs.SGLANG_TEST_RETRACT.clear()
    assert envs.SGLANG_TEST_RETRACT.get() is False

    envs.SGLANG_TEST_RETRACT.set(None)
    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    envs.SGLANG_TEST_RETRACT.clear()
    assert not envs.SGLANG_TEST_RETRACT.is_set()

    envs.SGLANG_TEST_RETRACT.set(True)
    assert envs.SGLANG_TEST_RETRACT.get() is True

    with envs.SGLANG_TEST_RETRACT.override(None):
        assert (
            envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None
        )

    assert envs.SGLANG_TEST_RETRACT.get() is True

    envs.SGLANG_TEST_RETRACT.set(None)
    with envs.SGLANG_TEST_RETRACT.override(True):
        assert envs.SGLANG_TEST_RETRACT.get() is True

    assert envs.SGLANG_TEST_RETRACT.is_set() and envs.SGLANG_TEST_RETRACT.get() is None

    example_with_exit_stack()
    example_with_subprocess()
    example_with_implicit_bool_avoidance()


if __name__ == "__main__":
    examples()
