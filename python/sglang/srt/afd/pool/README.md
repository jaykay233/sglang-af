# AfPool — MxN Attn/FFN work pool (experimental)

## Goal

Independent Attn and FFN pools over **real `cuda_ipc`**, scheduled for
**throughput (tok/s) and FFN utilization**, not Lite single-stream TPOT vs PD.

Default remains classic 1A1F + TRUE_OVERLAP (`SGLANG_AFD_POOL=0`).

## Enable

```bash
export SGLANG_AFD_POOL=1
export SGLANG_AFD_TRANSPORT=cuda_ipc
export SGLANG_AFD_POOL_NUM_ATTN=1
export SGLANG_AFD_POOL_NUM_FFN=2
export SGLANG_AFD_POOL_ENDPOINT_DIR=/tmp/afd_pool
export SGLANG_AFD_POOL_MAX_INFLIGHT_PER_FFN=4
export SGLANG_AFD_POOL_ROUTE=least_inflight   # or rr
export SGLANG_AFD_POOL_LOCAL_RANK=0           # rank inside Attn or FFN pool
```

Each pair `(attn_i, ffn_j)` uses socket `$ENDPOINT_DIR/a{i}_f{j}.sock`.

## Bench

```bash
python -m sglang.srt.afd.bench_af_pool \
  --num-attn 1 --num-ffn 2 --gpus 6,7 \
  --reqs 8 --layers 26 --tokens 8 \
  --attn-us 200 --ffn-us 400 \
  --compare-1a1f
```

Read `tok_s` and `mean_ffn_util` — success is 1A2F beating 1A1F when FFN-bound.

## Layout

| Module | Role |
|--------|------|
| `types.py` / `credit.py` / `router.py` / `scheduler.py` | routing + credit |
| `topology.py` | env → Na×Nf endpoint matrix |
| `attn_client.py` / `ffn_worker.py` | multi-link cuda_ipc |
| `bootstrap_pool.py` | ModelRunner hook when `POOL=1` |
| `../bench_af_pool.py` | multi-process throughput bench |
