# MindSpeed-LLM Profiling Parameter Reference

## Mcore Backend (CLI args for pretrain_gpt.py)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--profile` | bool | false | Enable profiling |
| `--profile-step-start` | int | *(none)* | Start step (inclusive). **Must be >= 1.** Framework default is 0 but rejected at runtime. |
| `--profile-step-end` | int | -1 | End step (exclusive). -1 = until end |
| `--profile-ranks` | int list | [0] | Ranks to profile. -1 = all |
| `--profile-level` | str | level0 | `level0` / `level1` (recommended) / `level2` |
| `--profile-with-cpu` | bool | false | Include CPU activity |
| `--profile-with-memory` | bool | false | Include NPU memory events |
| `--profile-with-stack` | bool | false | Include call stack |
| `--profile-record-shapes` | bool | false | Record tensor shapes |
| `--profile-save-path` | str | ./profile | Output directory |
| `--profile-export-type` | str | text | `text` or `db` |

## FSDP2 Backend (YAML config under training:)

Entry point: `train_fsdp2.py <config.yaml>`. Add profiling fields under `training:`:

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

See `MindSpeed-LLM/examples/fsdp2/` for full config templates.

## Profiling Levels

| Level | Captures | Use case |
|-------|----------|----------|
| `level0` | Basic operator timing | Quick overview |
| `level1` | + AICore, comm operators | **Recommended** |
| `level2` | + Cache, memory counters | Deep debugging |

## Output Structure

```
<hostname>_<pid>_<timestamp>_ascend_pt/
├── ASCEND_PROFILER_OUTPUT/    # Parsed results
├── PROF_<id>/
│   ├── device_0/data/         # NPU data
│   └── host/data/             # CPU data
└── logs/
```

Visualize with **MindStudio Insight**: https://gitcode.com/Ascend/msinsight
