# OSWorld 上的 GUI + MCP 混合动作空间 RL

[English](README.md) | **中文**

在 [OSWorld](https://github.com/xlang-ai/OSWorld) 桌面环境上训练混合动作空间的智能体：既能用 pyautogui 点屏幕，也能调用 MCP 工具。评测用 [OSWorld-MCP](https://github.com/X-PLUG/OSWorld-MCP) 的 309 个任务，策略模型是 **Qwen3-VL-8B-Thinking**。

本仓库是论文 **[Screenshots or Tools? Eliciting Tool Use and Managing Multimodal Context in Hybrid GUI–MCP Computer-Use Agents](https://arxiv.org/abs/2608.03327)**（[arXiv:2608.03327](https://arxiv.org/abs/2608.03327)）的配套代码：推理、评测、RL 训练的完整流水线，论文每张表对应的实验配置，以及任务切分。模型权重、VM 镜像等外部资源见[环境准备](#环境准备)。

> **两个核心结论。**
> **(1) 动作层面：行为学得会，能力学不会。** 加一个 dense 工具 bonus，表格类工具的调用率从 **0.03 涨到 0.33**，并能保持到 greedy 解码；但 held-out 精度没有变化。瓶颈不是是否调用工具，而是调用得对不对。
> **(2) 上下文层面：压缩的代价可以靠重训消掉。** 工具调用成功后，下一张截图往往是多余的。丢掉它、再把图像历史从 4 张减到 2 张，输入 token 省约三分之一，精度略降；用同样的观测规则重新训练后，这个代价消失。压缩后的智能体达到 **37.8%**，超过未压缩工作点的 **33.0%**，输入成本只有 **53%**，预注册退化子集上的 rich–lean 差距收敛到**零**（分布内）。
>
> 总结：动作层的问题是工具可用但模型很少调用，调用了也常出错；上下文层的问题是工具结果已经以文本形式在上下文里了，之后那张截图是冗余的，但默认仍然保留。两者的共同原因是训练里没有对应的信号。每个数字的出处见 [`results/context_rl/PROVENANCE.md`](results/context_rl/PROVENANCE.md) 和 [`results/dense_bonus/`](results/dense_bonus/README.md)，复现命令见 [`REPRODUCE.md`](REPRODUCE.md)。

## 一图看懂

**同一个任务、同一个模型，唯一的区别是给不给 MCP 工具。** 纯 GUI 50 步都耗在滚动字体列表上；混合智能体全选文本、调用一次工具，3 步完成：

![纯 GUI：50/50 步，失败](docs/figs/demo_gui_only.png)
![GUI+MCP：3/50 步，成功](docs/figs/demo_gui_mcp.png)

一条完整的成功轨迹（LibreOffice Writer，"给每一页左下角加页码"）：先用 GUI 打开页脚，然后一次 MCP 调用完成：

![完整混合轨迹](docs/figs/demo_full_rollout.png)

---

## 从哪开始

- **想核对论文数字** → [`REPRODUCE.md`](REPRODUCE.md)：论文每个结论对应哪条命令、哪个结果文件。
- **想跑起来** → 下面的[快速上手](#快速上手)；每个 RL 实验就是 `configs/experiments/` 下的一个 YAML。
- **切分文件太多看不懂** → [`data/README.md`](data/README.md) 开头的速查表。运行时真正用到的只有三个集合：训练 74、held-out 48、评测 309。
- **想在这个仓库上二次开发**（不管用不用 AI 编程助手）→ [`AGENTS.md`](AGENTS.md) 是写给编码 agent 的仓库地图：架构、约定、常见的坑、由轻到重的验证方法。建议的工作流：拿 `REPRODUCE.md` 当规格说明，所有改动写成 `configs/experiments/` 下的新 YAML（别直接改 base 配置），跑正式实验前先过 compileall → 冒烟测试 → probe。

## 仓库结构

```
osworld_rl.py            # 顶层 RL 编排器 (serve→rollout→reward→preproc→train)
train/                   # accelerate + DeepSpeed 训练 worker (osworld_train.py, preproc)
reward/                  # reward / advantage 计算 (outcome、dense MCP bonus、pos-adv-only)
agents/                  # BM25 工具检索器
prompts/                 # policy / reward prompt 模板
tools/                   # tools_registry.json (检索器排序的 MCP 工具语料)
mcp_tools/               # 随仓库发布的 MCP 服务端+工具 (注入进 VM;论文用的工具集)
configs/
  osworld_rl.yaml        # 基础配置 (路径用 ${oc.env:...})
  experiments/           # 每个论文 run 一个 YAML (RL run 系列 + 探针/profile)
  env.example.sh         # 本机环境变量 (拷成 env.sh 后 source)
data/splits/             # 任务 id 切分 + 元数据
scripts/                 # run_mcp_eval.sh、run_puregui_eval.sh、prepare_splits_v2.py ...
OSWorld-main/            # 随仓库发布、有改动的 OSWorld (改动见 OSWorld-main/PATCHES.md)
docs/                    # ENVIRONMENT.md + 研究笔记
results/outcome_only/, results/context_rl/  # 论文数字、逐 ckpt 格点、溯源、重算脚本
```

## 环境准备

我们的实验环境（论文所有数字的来源）：8 张 80 GB 级 GPU、支持 **/dev/kvm** 的 Docker、一块快速本地盘做临时空间，Python 3.10。脚本默认按这套配置布置（8 个 vLLM 实例 + 96 个并发 VM）；Qwen3-VL-8B 单卡即可放下，卡少也能跑，相应调低脚本里的实例数和 `NUM_ENVS` 即可，只是并行度和速度下降。

```bash
# 1. 环境
conda env create -f setup/environment_rlanything.yml   # 或: pip install -r requirements.txt
pip install -r OSWorld-main/requirements.txt

# 2. 本机路径 / 密钥 (仓库里没有任何硬编码)
cp configs/env.example.sh env.sh && $EDITOR env.sh && source env.sh
cp OSWorld-main/.env.example OSWorld-main/.env         # 本地 vLLM: key 保持 "dummy" 即可

# 3. 外部资源 (不在仓库里) — 见 docs/ENVIRONMENT.md
#    - 模型: Qwen/Qwen3-VL-8B-Thinking  -> $MODEL_DIR/
#    - VM:   happysixd/osworld-docker + 打过补丁的 Ubuntu-MCP.qcow2 -> /dev/shm/
#    - MCP:  已随仓库发布 (mcp_tools/);设 MCP_SRC_ROOT=0 可换成原版 OSWorld-MCP
```

Docker+QEMU、MCP 注入 VM 的架构、镜像制作、硬件要求，详见 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)。

## 已发布 checkpoint

两个 RL checkpoint 已发布到 HuggingFace，都是完整、可直接加载的 Qwen3-VL-8B 模型目录（config + `model.safetensors` + tokenizer/processor）。

| Checkpoint | Run | 是什么 | HF repo |
|-----------|-----|-------|---------|
| **outcome-only** | `outcome_only_rl` | 纯 outcome-only RLVR：工具调用率 0.02→0.33，held-out 不涨（[results/outcome_only](results/outcome_only/README.md)） | [`redai-infra/hybrid-routing-outcome-only`](https://huggingface.co/redai-infra/hybrid-routing-outcome-only) |
| **context-RL** | `context_rl` | 训推一致的上下文压缩 RL：**epoch-40** 论文工作点，37.8% @ 51% 输入成本，img2 + `skip_on_mcp_success`（[results/context_rl](results/context_rl/PROVENANCE.md)） | [`redai-infra/hybrid-routing-context-rl`](https://huggingface.co/redai-infra/hybrid-routing-context-rl) |

```bash
# 下载
huggingface-cli download redai-infra/hybrid-routing-context-rl --local-dir ckpts/context_rl

# 评测某个 checkpoint (把 MODEL 指向下载目录)
MODEL=ckpts/context_rl bash scripts/run_mcp_eval.sh
```

## 快速上手

**冒烟测试（几个任务，端到端跑通）：**
```bash
TEST_META=evaluation_examples/smoke_test.json bash scripts/run_mcp_eval.sh
```

**复现两个推理 baseline（309 任务）：**
```bash
# B1 —— 纯 GUI (pyautogui,无 MCP)   -> overall 30.5% (5 次均值)
bash scripts/run_puregui_eval.sh
python OSWorld-main/show_result.py --result_dir baselines/b1_puregui_thinking

# B2 —— GUI + MCP (工具检索 + MCP 调用)   -> overall 34.5% (5 次均值;单次最佳 37.9%)
bash scripts/run_mcp_eval.sh
python OSWorld-main/show_result.py --result_dir baselines/b2_mcp_thinking
```
两个脚本加 `MODEL_TYPE=instruct` 就是 Instruct 版（B1/B2-Instruct）。

**推理侧 context-policy 扫描**（baseline / `skip_on_mcp_success` / `skip_on_no_change`，img4 / img2）用同一个脚本的环境变量控制，例如 `CONTEXT_POLICY=skip_on_mcp_success MAX_IMAGE_HISTORY_LENGTH=2 bash scripts/run_mcp_eval.sh`，详见 [`REPRODUCE.md`](REPRODUCE.md) §A。

**跑一个 RL 实验（上下文压缩 RL）：**
```bash
OSWORLD_LOCAL_TEMP="$OSWORLD_LOCAL_TEMP" TQDM_DISABLE=1 \
  nohup python osworld_rl.py config=configs/experiments/context_rl.yaml \
  > logs/context_rl.log 2>&1 &
```
每个实验 YAML 的头注都写清了它的假设、判读标准和触发即停的红线。

## Headline 结果（309 任务，temp=0，max_steps=50，5 次均值）

| 模型 | 纯 GUI | GUI + MCP (win.4) | Δ(MCP) |
|------|-------:|------------------:|-------:|
| **Thinking** | 30.5% | **34.5%** | **+4.0pp** |
| **Instruct** | 25.4% | 19.5% | **−5.9pp** |

同一套 MCP 注入，Thinking 涨 4.0pp，Instruct 反而掉 5.9pp，都超过 2 SE——论文回答的正是"MCP 注入什么时候有用"。headline 数字都是 5 次运行的均值（[`baselines/COMPARISON.md`](baselines/COMPARISON.md)、[`baselines/REPEATS.md`](baselines/REPEATS.md)），Thinking 单次最好的一次到 37.9%。完整表格、采用—能力解耦、上下文压缩的差中差分析见 [`REPRODUCE.md`](REPRODUCE.md)。

### 两个 RL 探针，各一张图

**动作层面：bonus 推得动调用率，推不动能力**（论文图 2，训练记录在 [`results/dense_bonus/`](results/dense_bonus/README.md)）。表格类工具调用率 0.03→0.33，greedy 下 0.02→0.29；held-out 精度始终停在 base 水平，没有一个任务持续从 fail 翻成 pass：

![采用—能力解耦](docs/figs/fig_decoupling.png)

**上下文层面：同观测规则重训，消除压缩的精度代价**（论文图 4，曲线数据在 [`results/context_rl/train-reward.csv`](results/context_rl/train-reward.csv) 和 [`results/outcome_only/train-reward.csv`](results/outcome_only/train-reward.csv)，格点在 [`results/context_rl/cells.json`](results/context_rl/cells.json)）。(a) 压缩训练与富观测对照的训练奖励；(b) D13 子集上 rich–lean 差距在 step 30 归零：

![同观测重训恢复](docs/figs/fig_recovery.png)

## 案例

**RL 前后对比**：同一个 VS Code 保存文件的任务。base checkpoint（img4 未压缩）50 步内反复循环，始终没有保存文件；RL 后的 step-40 checkpoint（img2 压缩）走完保存对话框，12 步完成：

![RL 前：base checkpoint 失败](docs/figs/rl_before_vscode.png)
![RL 后：step-40 checkpoint 成功](docs/figs/rl_after_vscode.png)

一个贯穿三个发现的例子：LibreOffice Calc 矩阵转置（"把 B2:F5 转置，粘到 B8"），完整过程见 [`docs/CASES.md`](docs/CASES.md)。

**base 用纯 GUI 失败**：9 步 pyautogui 在 Paste Special 菜单里反复尝试，最后 B8 只粘进一个数值 `30`：

![base fail](results/cases/calc_transpose/base_fail.png)

**context-RL 一次 MCP 调用成功**：完整的转置表出现在 B8：

![success](results/cases/calc_transpose/s40_success.png)

```json
{ "action_type": "mcp", "tool_name": "libreoffice_calc.transpose_range",
  "params": { "source_range": "B2:F5", "target_cell": "B8" } }
```

- **能力翻转**：base 0/3 → context-RL 3/3（这样翻转的任务共 15 个）。
- **工具采用**：base 全程只用 GUI；RL 后的模型会主动选 MCP 工具。
- **上下文压缩**：在一个两边都能做对的任务上，img2 每步只带 2 张图（img4 带 4 张），输入只有 61%（省 39%），结果相同；工具成功后的那张截图被跳过（`skip_on_mcp_success`）。

## 引用

如果这个仓库或我们的论文对你的研究有帮助，欢迎考虑引用：

```bibtex
@article{fan2026screenshots,
  title   = {Screenshots or Tools? Eliciting Tool Use and Managing Multimodal
             Context in Hybrid GUI-MCP Computer-Use Agents},
  author  = {Fan, Siqi and Li, Minghao and Ma, Xiaoqian and Tan, Wenhui and
             Huang, Xiusheng and Wu, Juntong and Zhang, Liujie and Shang, Shuo
             and Chen, Weihang},
  journal = {arXiv preprint arXiv:2608.03327},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.03327}
}
```

## 许可与归属

Apache-2.0（见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)）。基于 [verl](https://github.com/volcengine/verl)（训练脚手架）、[OSWorld](https://github.com/xlang-ai/OSWorld)、[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) 构建。

## 致谢

由衷感谢以下开源工作，没有它们就没有这个项目：

- [**OSWorld**](https://github.com/xlang-ai/OSWorld)：本项目运行所在的桌面基准和 VM 环境。
- [**OSWorld-MCP**](https://github.com/X-PLUG/OSWorld-MCP)：带已验证 MCP 工具的基准，本仓库自带的工具集由它衍生。
- [**ToolCUA**](https://github.com/X-PLUG/ToolCUA)：GUI–工具路径编排方向最接近的同期工作，它对 MCP 注入的观察对我们的实验设计帮助很大。
- [**RLAnything** (Open-AgentRL)](https://github.com/Gen-Verse/Open-AgentRL)：面向 LLM 和智能体场景的开源 RL。
- [**Relax**](https://github.com/redai-infra/Relax)：多模态大模型后训练的高性能分布式 RL 框架。
