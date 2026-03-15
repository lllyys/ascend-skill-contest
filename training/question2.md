# 题目 2：verl部署

**Verl 官方仓库：**

[https://github.com/verl-project/verl](https://github.com/verl-project/verl)

## 题目概览

|   项目   |  说明   |
| :------: |:-----:|
|   难度   | ⭐⭐ 中等 |
| 预估时长 | 90 分钟 |

------

## 使用场景

- 帮助对于 Verl 版本依赖不熟悉的同学，快速部署 Verl 训练环境
- 通过 AI 端到端完成 Verl 训练环境的验证

## 任务描述

- 将 Verl 框架及其依赖安装到 NPU 环境中， vllm + vllm_ascend 为推理后端，megatron + mindspeed 为训练后端
- 以 Qwen3-0.6B 模型为示例，完成训练流程验证

**基本要求：**

1. 适配 NPU 环境：安装对应的torch/torch_npu环境（推荐2.7版本以上）
2. 依赖安装：安装 Verl 框架及其依赖（包括 vllm、vllm_ascend、megatron、mindspeed 等）
3. 验证训练流程：第一次训练完成后，可保存checkpoints；第二次训练可以加载checkpoints

**加分项：**

能够根据卡数和节点数自动配置训练参数，例如根据 2 卡节点，自动设置 nnodes = 2，nproc_per_node = 8
加分占比：20%

## 具体要求

|   项目   |                         说明                          |
| :------: |:---------------------------------------------------:|
|  Prompt  | 需包含引导 Agent 完成 “VerL环境部署→训练启动→checkpoint 验证” 的全流程指令 |
| 执行时间 |                       30分钟以内                        |

## 输出要求

参赛者需提交：

**目录结构**

```
skill-name/
├── SKILL.md        # 必须
├── reference/      # 可选
└── scripts/        # 可选
```

## 评分标准

参考 [Agent Skill 创作最佳实践](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)：

| 维度             | 权重  | 说明                                                         |
| ---------------- |-----| ------------------------------------------------------------ |
| 功能完整性       | 60% | 是否能成功引导完成 Qwen3-0.6B 训练流程    |
| description 质量 | 20% | 是否包含 WHAT（做什么）和 WHEN（何时触发）；是否具体、含关键词；是否用第三人称 |
| 指令与结构       | 10% | SKILL.md 是否简洁（建议 500 行以内）；指令步骤是否清晰可执行；是否合理使用渐进式披露 |
| 代码与脚本       | 10% | 脚本是否明确列出依赖；路径是否使用正斜杠；错误处理是否清晰；是否避免推卸给 Agent |