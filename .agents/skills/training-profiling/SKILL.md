---
name: training-profiling
description: Assists with profiling data collection for MindSpeed-LLM model training on Ascend NPU hardware. Given a user's training script, creates a profiling-enabled copy with configurable scope (CPU activity, NPU memory, call stacks, tensor shapes), collection level (level0/level1/level2), and step range. Validates the script environment, executes the profiling run, and verifies the output artifacts. Activated by keywords such as profiling, 性能采集, Profiling采集, 训练性能分析, MindSpeed-LLM profiling.
---

# Profiling 采集 — MindSpeed-LLM on Ascend NPU

> **Scope:** This skill only collects profiling data. It does not set up environments, download models, or convert checkpoints.

## What the user must provide

A working MindSpeed-LLM training script (`.sh`). Without it, reply:

> 需要一个可运行的训练脚本（.sh）才能进行 Profiling 采集。请先准备好训练脚本。

## Interaction flow

```
User provides script path (and optional .yaml config)
        ↓
Read script → detect backend (Mcore or FSDP2)
        ↓
Extract model, NPU count, paths
        ↓
Show config table → wait for user OK
        ↓
Check NPU available (npu-smi info)
        ↓
Validate script paths exist
        ↓
Create profiling-enabled copy (never touch originals)
        ↓
Run → Verify output → Report
```

## Backend detection

Read the `.sh` script to determine which backend it uses:

| Script contains | Backend | Profiling method |
|-----------------|---------|-----------------|
| `pretrain_gpt.py` or `posttrain_gpt.py` | Mcore | Add CLI args to `.sh` script |
| `train_fsdp2.py` + a `.yaml` config | FSDP2 | Add profiling fields to a copy of the `.yaml` config |

This determines how profiling args are injected. **Get this right — CLI args are silently ignored by FSDP2.**

## Config table template

After parsing the script and user request, present:

> | 项目 | 值 |
> |------|-----|
> | 模型 | *(parsed from script/yaml)* |
> | 脚本 | *(user provided path)* |
> | 后端 | Mcore / FSDP2 |
> | NPU 数 | *(parsed NPUS_PER_NODE)* |
> | Level | level1 *(recommended; framework default is level0)* |
> | Step 范围 | 2 ~ 4 *(end exclusive, actual: step 2, 3)* |
> | Rank | 0 |
> | CPU | on |
> | Memory | off |
> | Stack | off |
> | Shapes | off |
>
> 如需修改请说明，确认后开始。

### How to parse user intent

| 用户表述 | 对应参数 |
|----------|----------|
| CPU / 采集CPU / CPU数据 | cpu profiling |
| 内存 / memory / 显存 | memory profiling |
| 堆栈 / stack / 调用栈 | stack profiling |
| shape / 维度 / tensor形状 | record shapes |
| level0 / level1 / level2 | profiling level |
| step N 到 M / step N to M | step range |
| 所有卡 / all ranks | profile all ranks *(only if explicit)* |

## Execution rules

**Environment:**
- Run `npu-smi info` first. No NPU → stop immediately.
- Source CANN if `ASCEND_HOME_PATH` is unset: `source /usr/local/Ascend/ascend-toolkit/set_env.sh`

**Script validation:**
- Read the script and check referenced paths (checkpoint, data, tokenizer) exist on disk.
- Missing paths → stop, list what's missing, ask user to fix.

**Never modify the user's original files.** Create copies for profiling.

---

### Mcore backend (pretrain_gpt.py / posttrain_gpt.py)

1. Copy the `.sh` script to a new file with a timestamped name.
2. Insert profiling CLI args into the `torchrun` command:
   - Before `| tee` if piped
   - At the end of the `torchrun` line if no pipe
   - If the pattern cannot be parsed, show args and ask user
3. If `--profile` args already exist, show them and ask: keep or replace?
4. Create the output directory, then run the new script.

**CLI args to inject:**
```
--profile --profile-step-start 2 --profile-step-end 4 \
--profile-ranks 0 --profile-level level1 --profile-with-cpu \
--profile-save-path ./profiling_output
```

### FSDP2 backend (train_fsdp2.py)

**FSDP2 does NOT accept profiling args via CLI. They must go in the YAML config.**

1. Read the `.sh` script to find the `.yaml` config path it references.
2. Copy the `.yaml` config to a new file with a timestamped name (e.g. `config_profiling_20260325.yaml`).
3. Add profiling fields under the `training:` section of the new YAML:

```yaml
training:
  profile: true
  profile_step_start: 2
  profile_step_end: 4
  profile_ranks: [0]
  profile_level: level1
  profile_with_cpu: true
  profile_save_path: ./profiling_output
```

4. If profiling fields already exist in the YAML, show them and ask: keep or replace?
5. Copy the `.sh` script to a new timestamped file. Change the YAML path in the copy to point to the new profiling YAML.
6. Create the output directory, then run the new `.sh` script.

---

**On failure:**
- Training exits non-zero → report exit code, stop. Do not retry.

## Output verification

After training, check:
1. Output directory contains `*_ascend_pt/` subdirectory
2. Inside it: `PROF_*/device_*/data/` exists (NPU profiling data)

**If profiling output is missing but training succeeded, the profiling args were likely not applied.** Check:
- Mcore: did the CLI args actually appear in the `torchrun` command?
- FSDP2: did the YAML config contain `profile: true` under `training:`?

Report: output path, total size, directory tree. Recommend MindStudio Insight for visualization.

## Error reference

| 错误 | 处理 |
|------|------|
| NPU out of memory | 减小 batch size |
| Profiling 目录为空 | step-start 必须 >= 1，训练需执行到采集步骤 |
| Address already in use | 更换 MASTER_PORT |
| 训练成功但无 profiling 数据 | 检查后端类型：FSDP2 需要在 YAML 中配置，不接受 CLI 参数 |
| 权重形状不匹配 | 通知用户后停止 |

## Hard constraints

1. Never modify the user's original training script or YAML config
2. Never install, download, or convert anything
3. `profile_step_start` must be >= 1
4. `profile_ranks: [-1]` only on explicit user request
5. No training script = no profiling. Stop.
6. Large runs: 2-3 profiling steps max
7. FSDP2 profiling goes in YAML, never CLI

## Parameter reference

See [reference/mindspeed-profiling-args.md](reference/mindspeed-profiling-args.md) — covers Mcore CLI args and FSDP2 YAML config.
