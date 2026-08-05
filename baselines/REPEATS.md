# Baseline repeated runs — error bars

> Denominator fixed at 309 tasks (test_all_no_internet.json) | temp=0.0 | screen recording off

> Samples = the original single run (orig) + repeats/rep* | acc = strict/309

## Strict acc per setting (mean ± std, %)

| Setting | n | **Acc mean±std** | min | max | Per-run samples (acc% / done) |
|---------|---|------------------|-----|-----|----------------------|
| b2_think_base | 5 | ** 34.5 ±  2.1** | 32.4 | 37.9 | orig=37.9/309, rep1=32.4/309, rep2=35.0/309, rep3=33.7/309, rep4=33.7/309 |
| b2_think_nochange | 5 | ** 32.6 ±  0.5** | 32.0 | 33.3 | orig=33.3/309, rep1=32.7/309, rep2=32.0/309, rep3=32.4/308, rep4=32.7/309 |
| b2_think_img2 | 5 | ** 30.6 ±  0.9** | 29.8 | 32.0 | orig=29.8/309, rep1=32.0/309, rep2=30.4/309, rep3=30.7/309, rep4=30.1/309 |
| b2_think_skipmcp | 5 | ** 32.3 ±  1.8** | 29.4 | 34.3 | orig=32.7/309, rep1=33.0/309, rep2=32.0/309, rep3=29.4/309, rep4=34.3/309 |
| b2_think_skipmcp_img2 | 5 | ** 30.6 ±  1.2** | 29.1 | 31.7 | orig=31.7/309, rep1=29.1/309, rep2=29.4/309, rep3=31.4/309, rep4=31.1/309 |
| b2_inst_base | 5 | ** 19.5 ±  1.2** | 17.8 | 21.0 | orig=21.0/309, rep1=19.7/309, rep2=19.4/309, rep3=17.8/309, rep4=19.4/309 |
| b2_inst_img2 | 5 | ** 20.5 ±  1.1** | 19.4 | 22.3 | orig=20.1/309, rep1=20.1/309, rep2=22.3/309, rep3=19.4/309, rep4=20.4/309 |
| b2_inst_skipmcp | 5 | ** 20.9 ±  0.7** | 20.1 | 21.7 | orig=20.4/309, rep1=21.0/309, rep2=21.7/309, rep3=21.4/309, rep4=20.1/309 |
| b2_inst_skipmcp_img2 | 5 | ** 19.5 ±  0.5** | 18.8 | 20.1 | orig=20.1/309, rep1=19.1/309, rep2=19.7/309, rep3=18.8/309, rep4=19.7/309 |
| b1_thinking | 5 | ** 30.5 ±  1.0** | 29.4 | 32.0 | orig=30.4/308, rep1=32.0/309, rep2=29.4/309, rep3=29.8/309, rep4=30.7/309 |
| b1_instruct | 5 | ** 25.4 ±  1.5** | 23.0 | 26.5 | orig=26.5/304⚠, rep1=24.9/309, rep2=26.2/309, rep3=26.2/309, rep4=23.0/309 |

## Disputed point: baseline vs skip_on_no_change

- baseline      :  34.5 ±  2.1%  (n=5)
- skip_no_change:  32.6 ±  0.5%  (n=5)
- **ΔAcc = -1.9pp**, pooled SE ≈ ±1.0pp → **below 2·SE, inside noise**
