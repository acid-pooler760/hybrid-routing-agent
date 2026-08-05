# Baseline comparison report (5-run error-bar edition)

> Dataset: test_all_no_internet.json (309 tasks) | temp=0.0 | max_steps=50 | screen recording off
> Token counts come from each trajectory.json `meta` (precomputed at inference time with the Qwen3-VL-8B tokenizer, vision tokens included)
>
> **Auto-generated report**: every setting was run 5 times (orig + repeats/rep1–4);
> all numbers are **mean ± std** (±std shown on the acc column; other columns are means), denominator 309.
> Turn/token stats are averaged over completed tasks within each run, then equal-weighted across the 5 runs.
> ΔAcc significance uses the pooled SE (`✅` = |Δ|>2·SE; `≈noise` = within 2·SE).
> Per-run accuracies are listed in `REPEATS.md`; the generator script and raw trajectory trees are not shipped.

## 1. Overall comparison

| Baseline | Action | Model | ImgHist | Skip | n | **Acc (mean±std)** | AvgTurn | SuccTurn | FailTurn | AvgResp(chars) |
|----------|--------|-------|---------|------|---|--------------------|---------|----------|----------|----------------|
| B1-Instruct | GUI | Instruct | img4 | no | 5 | **25.4±1.5%** | 30.1 | 15.8 | 35.1 | 219 |
| B1-Thinking | GUI | Thinking | img4 | no | 5 | **30.5±1.0%** | 32.3 | 19.5 | 38.0 | 1503 |
| B2-Instruct | Hybrid | Instruct | img4 | no | 5 | **19.5±1.2%** | 31.1 | 13.6 | 35.4 | 217 |
| B2-Thinking | Hybrid | Thinking | img4 | no | 5 | **34.5±2.1%** | 33.2 | 19.9 | 40.2 | 1501 |
| B2-Inst-img2 | Hybrid | Instruct | img2 | no | 5 | **20.5±1.1%** | 37.1 | 19.3 | 41.7 | 215 |
| B2-Think-img2 | Hybrid | Thinking | img2 | no | 5 | **30.6±0.9%** | 36.2 | 21.9 | 42.5 | 1500 |
| B2-Inst-skip | Hybrid | Instruct | img4 | mcp | 5 | **20.9±0.7%** | 30.7 | 14.1 | 35.1 | 217 |
| B2-Think-skip | Hybrid | Thinking | img4 | mcp | 5 | **32.3±1.8%** | 34.0 | 21.1 | 40.1 | 1510 |
| B2-Inst-skip-img2 | Hybrid | Instruct | img2 | mcp | 5 | **19.5±0.5%** | 37.3 | 19.7 | 41.6 | 214 |
| B2-Think-skip-img2 | Hybrid | Thinking | img2 | mcp | 5 | **30.6±1.2%** | 35.7 | 22.5 | 41.5 | 1545 |
| B2-Think-nochange | Hybrid | Thinking | img4 | hash | 5 | **32.6±0.5%** | 33.1 | 20.2 | 39.4 | 1504 |

> `hash` = `skip_on_no_change` (drop a frame that is pixel-identical to the immediately preceding one, at most 2 consecutive skips); `mcp` = `skip_on_mcp_success`.

## 2. Token statistics (avg per task, 5-run means)

| Baseline | Acc | AvgSteps | **TotalIn(K)** | **TotalOut(K)** | AvgStepIn | AvgStepOut | PeakIn(avg) | PeakIn(p95) | PeakIn(max) |
|----------|-----|---------|----------------|-----------------|-----------|-----------|-------------|-------------|-------------|
| B1-Instruct | 25.4% | 30.1 | **287.3K** | **1.8K** | 9531 | 58 | 9902 | 10966 | 11689 |
| B1-Thinking | 30.5% | 32.3 | **313.1K** | **10.9K** | 9684 | 337 | 10302 | 11385 | 12435 |
| B2-Instruct | 19.5% | 31.1 | **316.4K** | **1.8K** | 10163 | 57 | 10367 | 11441 | 11841 |
| B2-Thinking | 34.5% | 33.2 | **337.1K** | **11.3K** | 10126 | 341 | 10485 | 11544 | 13752 |
| B2-Inst-img2 | 20.5% | 37.1 | **232.4K** | **2.1K** | 6263 | 56 | 6446 | 7254 | 7529 |
| B2-Think-img2 | 30.6% | 36.2 | **226.1K** | **12.6K** | 6234 | 347 | 6501 | 7314 | 8018 |
| B2-Inst-skip | 20.9% | 30.7 | **310.1K** | **1.7K** | 10103 | 56 | 10217 | 11437 | 11615 |
| B2-Think-skip | 32.3% | 34.0 | **342.4K** | **12.0K** | 10048 | 351 | 10340 | 11487 | 12037 |
| B2-Inst-skip-img2 | 19.5% | 37.3 | **231.1K** | **2.1K** | 6192 | 56 | 6382 | 7231 | 7624 |
| B2-Think-skip-img2 | 30.6% | 35.7 | **219.5K** | **14.4K** | 6095 | 401 | 6399 | 7243 | 7896 |
| B2-Think-nochange | 32.6% | 33.1 | **305.1K** | **11.5K** | 9185 | 345 | 10409 | 11486 | 12372 |

## 3. Dimension-wise deltas (Δ with pooled-SE significance)

### 3.1 Thinking vs Instruct (A=Instruct, B=Thinking)

| Config | A Acc | B Acc | ΔAcc (B−A, ±SE) | Verdict | Think TotalIn | Inst TotalIn |
|------|-------|-------|-----------------|------|------|------|
| GUI, img4 | 25.4% | 30.5% | +5.1 (±0.8) | ✅ | 313.1K | 287.3K |
| Hybrid, img4 | 19.5% | 34.5% | +15.0 (±1.1) | ✅ | 337.1K | 316.4K |
| Hybrid, img2 | 20.5% | 30.6% | +10.2 (±0.6) | ✅ | 226.1K | 232.4K |
| Hybrid, img4, skip | 20.9% | 32.3% | +11.4 (±0.9) | ✅ | 342.4K | 310.1K |
| Hybrid, img2, skip | 19.5% | 30.6% | +11.1 (±0.6) | ✅ | 219.5K | 231.1K |

### 3.2 MCP gain (A=pure GUI, B=GUI+MCP, same model)

| Config | A Acc | B Acc | ΔAcc (B−A, ±SE) | Verdict | GUI TotalIn | MCP TotalIn |
|------|-------|-------|-----------------|------|------|------|
| Thinking (img4) | 30.5% | 34.5% | +4.0 (±1.0) | ✅ | 313.1K | 337.1K |
| Instruct (img4) | 25.4% | 19.5% | -5.9 (±0.8) | ✅ | 287.3K | 316.4K |
| Thinking (img2) | 30.5% | 30.6% | +0.1 (±0.6) | ≈noise | 313.1K | 226.1K |
| Instruct (img2) | 25.4% | 20.5% | -4.9 (±0.8) | ✅ | 287.3K | 232.4K |

### 3.3 Effect of skip_on_mcp_success (A=standard, B=skip)

| Config | A Acc | B Acc | ΔAcc (B−A, ±SE) | Verdict | Std TotalIn | Skip TotalIn | ΔInput |
|------|-------|-------|-----------------|------|------|------|------|
| Thinking, img4 | 34.5% | 32.3% | -2.2 (±1.2) | ≈noise | 337.1K | 342.4K | +1.6% |
| Instruct, img4 | 19.5% | 20.9% | +1.4 (±0.6) | ✅ | 316.4K | 310.1K | -2.0% |
| Thinking, img2 | 30.6% | 30.6% | -0.1 (±0.7) | ≈noise | 226.1K | 219.5K | -2.9% |
| Instruct, img2 | 20.5% | 19.5% | -1.0 (±0.5) | ≈noise | 232.4K | 231.1K | -0.5% |

### 3.4 Effect of img_history (A=img4, B=img2)

| Config | A Acc | B Acc | ΔAcc (B−A, ±SE) | Verdict | img4 TotalIn | img2 TotalIn | ΔInput | ΔPeak |
|------|-------|-------|-----------------|------|------|------|------|------|
| Thinking | 34.5% | 30.6% | -3.9 (±1.0) | ✅ | 337.1K | 226.1K | -32.9% | -38.0% |
| Instruct | 19.5% | 20.5% | +1.0 (±0.7) | ≈noise | 316.4K | 232.4K | -26.6% | -37.8% |
| Thinking+skip | 32.3% | 30.6% | -1.7 (±1.0) | ≈noise | 342.4K | 219.5K | -35.9% | -38.1% |
| Instruct+skip | 20.9% | 19.5% | -1.4 (±0.4) | ✅ | 310.1K | 231.1K | -25.5% | -37.5% |

### 3.5 Effect of skip_on_no_change (byte-identical hash frame skipping)

| Config | Baseline Acc | NoChange Acc | ΔAcc (±SE) | Verdict | Baseline TotalIn | NoChange TotalIn | ΔInput |
|------|-------------|--------------|------------|------|------------------|------------------|--------|
| Thinking, img4 | 34.5% | 32.6% | -1.9 (±1.0) | ≈noise | 337.1K | 305.1K | -9.5% |

**Mechanism (verified)**: the skip only triggers on screenshots pixel-identical to the immediately preceding frame. Measured on adjacent frame pairs, only **13.4%** even match in byte size (pixel-identical is rarer), and the skip only applies inside the 4-image window with ≤2 consecutive skips → opportunity rate 0.134×4 ≈ 0.54 frames/prompt, measured actual saving ≈ 0.47 frames/prompt → total input only drops ~9.5%.
Far less efficient than img2's unconditional halving (−28% input). A single run once showed −4.5pp acc; over 5 runs this shrinks to **-1.9pp (±1.0), ≈noise**.

## 4. Token-efficiency ranking (token cost per 1% acc, 5-run means)

| Rank | Baseline | Acc | TotalTok/task(K) | **TotalTok / 1%Acc (K)** |
|------|----------|-----|-----------------|------------------------|
| 🥇 | B2-Think-skip-img2 | 30.6% | 233.9K | **7.7K** |
| 🥈 | B2-Think-img2 | 30.6% | 238.7K | **7.8K** |
| 🥉 | B2-Think-nochange | 32.6% | 316.5K | **9.7K** |
| 4 | B2-Thinking | 34.5% | 348.5K | **10.1K** |
| 5 | B1-Thinking | 30.5% | 324.0K | **10.6K** |
| 6 | B2-Think-skip | 32.3% | 354.4K | **11.0K** |
| 7 | B1-Instruct | 25.4% | 289.1K | **11.4K** |
| 8 | B2-Inst-img2 | 20.5% | 234.5K | **11.5K** |
| 9 | B2-Inst-skip-img2 | 19.5% | 233.2K | **12.0K** |
| 10 | B2-Inst-skip | 20.9% | 311.8K | **14.9K** |
| 11 | B2-Instruct | 19.5% | 318.2K | **16.3K** |

## 5. Per-app accuracy (%, 5-run means)

| Baseline | calc | impress | writer | vs_code | os | vlc | gimp | thunder | chrome | multi |
|------|------|------|------|------|------|------|------|------|------|------|
| B1-Instruct | 17.0 | 18.5 | 29.9 | 49.5 | 41.7 | 27.1 | 57.7 | 6.7 | 62.9 | 7.0 |
| B1-Thinking | 17.9 | 28.5 | 51.3 | 57.1 | 51.7 | 24.7 | 60.8 | 6.7 | 64.3 | 8.3 |
| B2-Instruct | 7.2 | 15.3 | 14.8 | 38.1 | 45.0 | 16.5 | 34.6 | 4.0 | 54.3 | 9.9 |
| B2-Thinking | 18.3 | 26.8 | 45.2 | 68.6 | 57.5 | 40.0 | 67.7 | 10.7 | 68.6 | 14.9 |
| B2-Inst-img2 | 3.8 | 16.6 | 5.2 | 43.8 | 46.7 | 25.9 | 50.8 | 6.7 | 44.3 | 9.6 |
| B2-Think-img2 | 12.8 | 21.7 | 37.4 | 62.9 | 53.3 | 31.8 | 69.2 | 8.0 | 64.3 | 13.6 |
| B2-Inst-skip | 5.5 | 17.9 | 13.9 | 39.0 | 43.3 | 25.9 | 38.5 | 4.0 | 61.4 | 10.9 |
| B2-Think-skip | 14.9 | 20.9 | 38.3 | 64.8 | 57.5 | 34.1 | 70.0 | 12.0 | 74.3 | 14.1 |
| B2-Inst-skip-img2 | 4.7 | 10.6 | 9.6 | 45.7 | 47.5 | 21.2 | 46.9 | 6.7 | 44.3 | 9.1 |
| B2-Think-skip-img2 | 12.3 | 20.0 | 38.3 | 64.8 | 54.2 | 31.8 | 66.9 | 12.0 | 64.3 | 13.6 |
| B2-Think-nochange | 17.0 | 22.6 | 44.3 | 62.6 | 55.8 | 32.9 | 69.2 | 9.3 | 64.3 | 15.5 |

## 6. Per-app token cost (avg total tok/task, K, 5-run means)

| Baseline | calc | impress | writer | vs_code | os | vlc | gimp | thunder | chrome | multi |
|------|------|------|------|------|------|------|------|------|------|------|
| B1-Instruct | 346.5 | 248.6 | 207.1 | 290.3 | 143.8 | 216.5 | 223.6 | 480.2 | 224.1 | 362.8 |
| B1-Thinking | 337.2 | 231.0 | 260.5 | 280.9 | 255.7 | 326.1 | 282.5 | 481.6 | 223.2 | 429.1 |
| B2-Instruct | 427.2 | 389.1 | 279.1 | 307.2 | 93.8 | 193.8 | 183.6 | 334.2 | 179.0 | 390.0 |
| B2-Thinking | 410.2 | 361.7 | 242.9 | 287.0 | 214.9 | 318.3 | 272.9 | 465.6 | 204.9 | 430.3 |
| B2-Inst-img2 | 296.6 | 270.3 | 245.6 | 185.9 | 104.2 | 210.9 | 161.6 | 264.3 | 150.3 | 265.2 |
| B2-Think-img2 | 297.3 | 257.4 | 202.2 | 189.2 | 139.6 | 229.5 | 172.0 | 268.6 | 140.3 | 284.5 |
| B2-Inst-skip | 417.2 | 368.5 | 274.8 | 271.3 | 95.2 | 202.8 | 194.0 | 346.6 | 169.7 | 387.4 |
| B2-Think-skip | 394.4 | 372.6 | 255.6 | 303.8 | 215.7 | 318.5 | 286.5 | 471.0 | 199.8 | 443.8 |
| B2-Inst-skip-img2 | 295.3 | 266.4 | 229.4 | 176.5 | 119.7 | 217.4 | 168.6 | 256.8 | 151.3 | 263.5 |
| B2-Think-skip-img2 | 281.1 | 253.2 | 174.9 | 202.3 | 150.6 | 229.7 | 180.9 | 256.7 | 143.7 | 277.5 |
| B2-Think-nochange | 354.5 | 322.6 | 222.6 | 243.2 | 189.9 | 263.7 | 241.0 | 456.3 | 215.4 | 407.7 |

## 7. Per-app token efficiency (token cost per 1% app-acc, K, 5-run means)

> Cell = the app's avg total tok/task ÷ the app's acc(%); smaller is cheaper. `-` = app acc=0 (no success, efficiency undefined).
> Same convention as §4 (cross-run averaged tok divided by cross-run averaged acc).

| Baseline | calc | impress | writer | vs_code | os | vlc | gimp | thunder | chrome | multi |
|------|------|------|------|------|------|------|------|------|------|------|
| B1-Instruct | 20.4 | 13.5 | 6.9 | 5.9 | 3.5 | 8.0 | 3.9 | 72.0 | 3.6 | 52.0 |
| B1-Thinking | 18.9 | 8.1 | 5.1 | 4.9 | 4.9 | 13.2 | 4.6 | 72.2 | 3.5 | 51.7 |
| B2-Instruct | 59.1 | 25.4 | 18.9 | 8.1 | 2.1 | 11.8 | 5.3 | 83.5 | 3.3 | 39.5 |
| B2-Thinking | 22.4 | 13.5 | 5.4 | 4.2 | 3.7 | 8.0 | 4.0 | 43.7 | 3.0 | 28.8 |
| B2-Inst-img2 | 77.4 | 16.3 | 47.1 | 4.2 | 2.2 | 8.1 | 3.2 | 39.6 | 3.4 | 27.6 |
| B2-Think-img2 | 23.3 | 11.9 | 5.4 | 3.0 | 2.6 | 7.2 | 2.5 | 33.6 | 2.2 | 20.9 |
| B2-Inst-skip | 75.4 | 20.6 | 19.8 | 6.9 | 2.2 | 7.8 | 5.0 | 86.7 | 2.8 | 35.4 |
| B2-Think-skip | 26.5 | 17.9 | 6.7 | 4.7 | 3.8 | 9.3 | 4.1 | 39.3 | 2.7 | 31.4 |
| B2-Inst-skip-img2 | 63.1 | 25.0 | 24.0 | 3.9 | 2.5 | 10.3 | 3.6 | 38.5 | 3.4 | 29.1 |
| B2-Think-skip-img2 | 22.8 | 12.7 | 4.6 | 3.1 | 2.8 | 7.2 | 2.7 | 21.4 | 2.2 | 20.4 |
| B2-Think-nochange | 20.8 | 14.3 | 5.0 | 3.9 | 3.4 | 8.0 | 3.5 | 48.9 | 3.4 | 26.4 |

## 8. Per-app completion steps (avg steps/task, 5-run means)

> Average trajectory length per app (successes + failures; max_steps=50). Smaller = faster convergence or earlier give-up.

| Baseline | calc | impress | writer | vs_code | os | vlc | gimp | thunder | chrome | multi |
|------|------|------|------|------|------|------|------|------|------|------|
| B1-Instruct | 35.8 | 26.1 | 22.0 | 30.3 | 15.7 | 23.1 | 23.6 | 49.3 | 23.8 | 37.4 |
| B1-Thinking | 33.4 | 23.5 | 26.0 | 28.6 | 26.0 | 32.6 | 28.5 | 47.7 | 23.0 | 42.2 |
| B2-Instruct | 39.9 | 36.1 | 26.3 | 30.1 | 10.9 | 19.8 | 20.6 | 36.3 | 20.1 | 38.1 |
| B2-Thinking | 37.1 | 32.8 | 22.5 | 27.5 | 21.7 | 31.4 | 28.9 | 48.6 | 22.1 | 40.6 |
| B2-Inst-img2 | 43.8 | 39.5 | 36.3 | 29.4 | 18.0 | 34.6 | 30.7 | 49.7 | 28.4 | 41.7 |
| B2-Think-img2 | 41.8 | 36.0 | 28.5 | 28.8 | 23.3 | 36.5 | 31.1 | 48.1 | 25.1 | 42.6 |
| B2-Inst-skip | 38.9 | 34.8 | 26.4 | 27.0 | 10.9 | 20.7 | 21.7 | 37.6 | 19.1 | 37.8 |
| B2-Think-skip | 36.0 | 34.0 | 24.0 | 29.2 | 22.1 | 31.4 | 30.3 | 49.1 | 21.5 | 41.9 |
| B2-Inst-skip-img2 | 43.7 | 40.4 | 34.1 | 29.2 | 20.1 | 35.8 | 32.0 | 48.3 | 28.6 | 41.7 |
| B2-Think-skip-img2 | 40.1 | 35.6 | 24.8 | 30.8 | 25.1 | 36.4 | 32.6 | 46.2 | 25.8 | 41.9 |
| B2-Think-nochange | 35.8 | 32.7 | 22.3 | 26.3 | 21.4 | 30.0 | 29.1 | 48.7 | 24.0 | 41.5 |

## 9. TIR tool-invocation rate (MCP-steps / total-steps, 5-run means)

> `TIR_real` = share of steps with `action_type==mcp` AND `exec_ok` (tool calls that actually executed);
> `TIR_any` = share of all `action_type==mcp` steps (including hallucinated/failed MCP-shaped actions); `Gap = any − real` = hallucinated tool calls.
> Convention: per-task ratio → averaged over completed tasks within a run → averaged across 5 runs (macro). B1 is pure GUI, so identically 0.

| Baseline | **TIR_real(%)** | TIR_any(%) | Halluc. gap(pp) |
|----------|-----------------|------------|-------------|
| B1-Instruct | **0.0** | 0.0 | +0.0 |
| B1-Thinking | **0.0** | 0.0 | +0.0 |
| B2-Instruct | **2.0** | 2.3 | +0.4 |
| B2-Thinking | **2.8** | 2.9 | +0.0 |
| B2-Inst-img2 | **2.4** | 2.7 | +0.3 |
| B2-Think-img2 | **3.1** | 3.2 | +0.2 |
| B2-Inst-skip | **2.7** | 2.8 | +0.1 |
| B2-Think-skip | **3.3** | 3.4 | +0.0 |
| B2-Inst-skip-img2 | **2.9** | 3.0 | +0.1 |
| B2-Think-skip-img2 | **4.4** | 4.5 | +0.1 |
| B2-Think-nochange | **3.0** | 3.1 | +0.1 |

### Per-app TIR_real (%, 5-run means)

| Baseline | calc | impress | writer | vs_code | os | vlc | gimp | thunder | chrome | multi |
|------|------|------|------|------|------|------|------|------|------|------|
| B1-Instruct | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| B1-Thinking | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| B2-Instruct | 0.6 | 7.2 | 5.2 | 3.3 | 1.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 |
| B2-Thinking | 3.0 | 3.0 | 18.8 | 2.7 | 2.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.6 |
| B2-Inst-img2 | 0.2 | 6.8 | 7.6 | 6.0 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7 |
| B2-Think-img2 | 2.8 | 3.0 | 17.8 | 3.8 | 3.4 | 0.0 | 0.0 | 0.1 | 0.0 | 1.3 |
| B2-Inst-skip | 0.9 | 8.7 | 6.0 | 5.8 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 |
| B2-Think-skip | 6.1 | 3.4 | 16.9 | 2.5 | 3.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.7 |
| B2-Inst-skip-img2 | 1.1 | 7.7 | 7.0 | 9.1 | 2.1 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| B2-Think-skip-img2 | 6.2 | 6.0 | 24.5 | 1.8 | 2.9 | 0.0 | 0.0 | 0.0 | 0.0 | 1.6 |
| B2-Think-nochange | 3.2 | 3.4 | 17.9 | 3.2 | 3.6 | 0.1 | 0.1 | 0.0 | 0.0 | 0.7 |

## 10. Key findings (re-judged with error bars)

### Accuracy

1. **B2-Thinking (Hybrid, img4, no skip) stays on top: 34.5±2.1%** (the single-run orig 37.9% was the best of the 5; the mean regresses). Still the only configuration worth using as the RL starting point.
2. **The MCP gain holds only for Thinking**: Thinking +4.0pp (±1.0, ✅), Instruct -5.9pp (±0.8, ✅, negative).
3. **Thinking ≫ Instruct is amplified under MCP**: pure GUI +5.1pp; Hybrid img4 +15.0pp (±1.1, ✅).
4. **Most skip / img knob deltas fall inside noise** (see the verdict columns of §3.3–3.5): img2 remains a real loss for Thinking; the acc effects of skip_on_mcp_success / skip_on_no_change are mostly within 2·SE and must not be cited as firm conclusions.
5. **The skip_on_no_change controversy is settled**: ΔAcc -1.9pp (±1.0), ≈noise — the single-run −4.5pp does not replicate; it was sampling noise.

### Tokens (5-run means; conclusions stable)

1. **Thinking emits ~340–400 tok/step, Instruct ~56** (6–7×).
2. **img2 cuts ~30% total input and ~38% peak** (§3.4) — the most reliable token lever.
3. **skip_on_mcp_success *increases* input under Thinking img4** (extra steps offset the saved frames); it only truly saves under img2.
4. **Best token efficiency: B2-Think-skip-img2 (7.7K per 1% acc)**; the highest-acc B2-Thinking ranks 4th (10.1K).
5. **All configurations peak below 14K input tokens** (model max_length = 262K) — nowhere near the context limit.
6. **skip_on_no_change saves little (−9.5% input) and is mechanistically limited** (§3.5): it can only drop pixel-identical frames, which are only ~13% of adjacent pairs for a GUI agent.

---
> Per-run (orig + rep1–4) accuracies are listed in `REPEATS.md`.
