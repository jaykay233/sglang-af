# 面试证据包：SGLang Unified 调度下的 Decode Anti-Starvation

> 目标：把「改了调度 / 跑了数字」讲成**可复现的问题定义 → ablation → trade-off → 与 PD 的定位**，而不是「我调了个 flag」。

---

## 1. 一句话结论（开场 15 秒）

在 **非 PD、unified scheduler** 下，SGLang 默认是 **prefill-first**：只要 waiting 里还有 prefill，就优先跑 prefill，in-flight decode 会被饿死。  
我加了 **mixed chunk + stall-limit**（连续 N 步纯 prefill 后强制 decode-only），在「长 decode + 持续短/中 prefill」场景下压低 victim 的 **flood 窗口 ITL/p99**；代价是高并发时 **TTFT 变差**——每步都带上 decode，prefill 推进变慢。

---

## 2. 问题怎么构造（比随机压测更有说服力）

随机 `bench_serving` 只能看到平均 TTFT/TPOT，**看不清饥饿**。面试里要主动说你构造了：

| 角色 | 设定 | 目的 |
|------|------|------|
| Victim | 少量长 decode（例如 4–8 路，out=2k–3k） | 模拟在线长生成用户 |
| Quiet 窗 | victim 出 first token 后先空转 ~1–2s | 得到无干扰 ITL 基线 |
| Attacker flood | 高并发持续灌短/中 prefill（短 out） | 模拟持续到达的新请求 |
| 指标 | **victim 在 flood 窗内的 ITL/TPOT CDF** | 饥饿只发生在「有 decode 在跑 + 新 prefill 不断」时 |

脚本：`bench-results/starvation_bench.py`  
重启三模式：`bench-results/restart_sglang_mode.sh {baseline|mixed|budget}`

**Ablation 三档（必须能讲清每档差在哪）：**

1. **baseline**：默认 prefill-first（可有 chunked prefill，但 `enable_mixed_chunk=False` → chunk 之间仍可不穿插 decode）
2. **mixed only**：`--enable-mixed-chunk` → prefill chunk 与 running decode **同一步 mix**
3. **mixed + stall**：`--enable-decode-token-budget --decode-token-budget-stall-limit 2` → 在 mixed 之上，若连续 stall 步仍无 decode，**强制 decode-only**

代码落点（背这三处即可）：
- `get_next_batch_to_run`：prefill 优先；`force_decode_only` 时跳过 `get_new_batch_prefill`
- `mix_with_running` / `is_mixed_chunk`：同一步带 decode
- `decode_stall_steps`：纯 prefill 步累加，超限清零并强制 decode

---

## 3. 数字怎么讲（主推 heavy flood）

**环境**：Qwen2.5-0.5B / TP=1 / A800 / triton attn / unified（非 PD）

**主场景（heavy）**：8 victims × out=3072；attacker c=48，in≈6000 chars（中长 prefill），out=8；flood=25s

| Mode | Quiet mean ITL | Flood mean ITL | Flood p99 ITL | Flood max ITL |
|------|---------------:|---------------:|--------------:|--------------:|
| Baseline | ~1.85 ms | ~13.4 ms | **~174 ms** | ~474 ms |
| Mixed only | ~1.88 ms | ~14.5 ms | **~60 ms** | ~379 ms |
| Mixed + stall | ~1.83 ms | ~14.6 ms | **~59 ms** | ~379 ms |

**辅场景（light，短 prefill）**：4 victims；attacker c=64，in≈400 chars → baseline flood p99 ~66ms / max ~692ms；mixed/budget 把 p99 压到 ~32–36ms。

**口述要点：**
- Quiet 三者接近 → 优化**不是**让单步 decode 变快，而是**砍掉被 prefill 打断的长尾**。
- Heavy 下 baseline flood **p99 ≈ 174ms（约 quiet 的 90×）**；mixed/stall 压到 **~60ms（约 3× 改善）** → 饥饿证据够硬。
- Mean 三者接近甚至略升：因为 mix 后每步更“重”，平均 ITL 会抬一点；**看 p99/CDF 尾部才对**。
- 本机上 mixed ≈ budget：budget 会强制打开 mixed；当每步都能 mix 成功时 stall 很少触发。**Stall = mix 失败时的上界兜底**，面试里主动说清，避免被问成「多此一举」。
- 图：`plots/starvation_itl_cdf.png`、`plots/starvation_flood_bars.png`

---

## 4. 代价：为何 64–128 并发 TTFT 变差（必背）

同一套 random 压测（in=128, out=1024）：

| Conc | Baseline mean TTFT | Budget mean TTFT | Baseline out tok/s | Budget out tok/s |
|-----:|-------------------:|-----------------:|-------------------:|-----------------:|
| 16 | 251 ms | **138 ms** | 7.2k | **7.7k** |
| 32 | 174 ms | **55 ms** | 13.6k | **14.4k** |
| 64 | **134 ms** | 243 ms | 17.7k | 17.1k |
| 128 | **188 ms** | 619 ms | 17.5k | 16.6k |

**因果链（面试官最爱听）：**

1. 高并发时 waiting prefill **几乎永远非空** → 默认几乎一直在「推进 prefill」。
2. Mixed/stall 让 **几乎每一步（或每隔 N 步）都要带着 running decode 做一步**。
3. 单步算力被 decode tokens 分走 → **每个 prefill 请求凑齐 first token 的墙钟变长** → TTFT↑。
4. 中等并发（16–32）有时反而更好：减少「排队里被饿很久才轮到」的尾部，平均 TTFT/QPS 可改善。
5. 这是 **公平性（decode ITL）vs 准入延迟（TTFT）** 的显式权衡，不是实现 bug。

一句话：**Stall/mixed 买的是在线 decode 的平滑度，付的是高负载下 prefill 的日历时间。**

---

## 5. 若有时间：更大模型 / PD 对比（各一句）

**更大模型：**  
0.5B 上 prefill 极短，饥饿窗容易被「请求瞬间做完」稀释；更大模型（7B/9B）单次 prefill 更重，baseline 的 ITL 尖峰通常更夸张，mixed/stall 的相对收益更好看。  
（本机 `Qwen3.5-9B` 权重未下全，仅 ~1GB 元数据，故未强行扩模型；面试可说「方法与实验脚本已具备，换完整权重即可复跑」。）

**vs PD 分离：**  
- **PD**：物理拆容量，prefill/decode 互不抢同一 GPU 时间片，运维与调度更重，扩展性好。  
- **本方案**：单实例、改调度策略，**零额外部署**，用 mixed+stall 做「软隔离」。  
定位：**PD 是架构解耦；anti-starvation 是 unified 路径上的低成本 fairness 旋钮。** 二者不互斥——PD 内部 decode 池仍可能要类似策略。

---

## 6. 推荐口述结构（2–3 分钟）

1. **背景**：unified 默认 prefill-first，持续到达会饿 decode。  
2. **方法**：mixed chunk 同批混跑；stall-limit 强制 decode-only。  
3. **实验**：饥饿场景画 quiet vs flood ITL；三档 ablation。  
4. **结果**：flood p99/max 明显下降；随机负载中等并发受益、高并发 TTFT 变差。  
5. **Trade-off + 定位**：公平 vs TTFT；相对 PD 是轻量 soft fairness。

---

## 7. 可能被问的硬问题（短答）

**Q: 和 continuous batching / Sarathi 什么关系？**  
A: 思想同源（chunked prefill + 与 decode 交织）。我做的是在 SGLang 现有 `ScheduleBatch` / `PrefillAdder` 上接 `mix_with_running`，并加 stall 兜底，偏工程落地与测量，不是新论文算法。

**Q: stall-limit 怎么选？**  
A: 越小越偏 decode 公平、TTFT 越容易变差；2 是「允许短暂纯 prefill 突发、但不让饥饿拉长」的折中。可用 flood p99 ITL vs 高并发 TTFT 做曲线选型。

**Q: 为啥有时 mixed ≈ budget？**  
A: budget 路径会强制 `is_mixed_chunk=True`；当每步 prefill 都能成功 mix 进 decode 时，stall 计数经常被清零，stall 很少触发。Stall 的价值是 **mix 失败/不可用时的上界**。

---

## 8. 产物路径

- 饥饿脚本：`/root/.cuda/sglang/bench-results/starvation_bench.py`
- 结果：`.../ablation/starvation_{baseline,mixed,budget}.json`（及 `_heavy`）
- 随机扫：`bench-results/` vs `bench-results-budget/`
- 图与笔记：`bench-results/plots/`
