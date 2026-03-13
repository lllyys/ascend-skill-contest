# 题目1: 训练框架Profiling采集

## 题目概览

| 项目 | 说明 |
|------|------|
| 难度 | ⭐ 初等 |
| 预估时长 | 60 分钟 |

---

## 使用场景

- 对于训练框架Profiling不熟悉的开发者，用于指导完成性能数据采集
- 在Agentic Coding场景下，profiling采集是自动化执行训练性能分析流程的前置步骤
- 帮助开发者快速定位训练瓶颈，优化模型训练效率

## 任务描述

选择一个已有profiling功能适配的主流训练仓（VeRL/MindSpeed-LLM/MindSpeed-MM等）中的**其中一个**，制作一个Agent Skill，用于指导或自动化执行该框架的Profiling采集流程。

具体要求：

| 项目 | 说明 |
|------|------|
| Prompt | 使用[所选框架]完成模型(具体模型不限)训练的Profiling数据采集 |
| 执行时间 | 30 分钟以内 |
| 采集内容 | Agent 需能正常识别采集CPU、内存、不同level、不同采集step范围的采集需求 |
| 框架选择 | VeRL/MindSpeed-LLM/MindSpeed-MM等主流训练仓 |

## 输出要求

参赛者需提交：

**目录结构**
   ```
   skill-name/
   ├── SKILL.md        # 必须
   ├── reference/      # 可选（包含参考文档、命令等）
   └── scripts/        # 可选（包含自动化脚本等）
   ```

## 评分标准

参考 [Agent Skill 创作最佳实践](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices)：

| 维度 | 权重 | 说明 |
|------|------|------|
| 功能完整性 | 60% | 是否能成功引导完成Profiling采集流程，覆盖关键采集项； |
| description 质量 | 20% | 是否包含WHAT（做什么）和WHEN（何时触发）；是否具体、含关键词；是否用第三人称 |
| 指令与结构 | 10% | SKILL.md是否简洁（建议500行以内）；指令步骤是否清晰可执行；是否合理使用渐进式披露 |
| 代码与脚本 | 10% | 脚本是否明确列出依赖；路径是否使用正斜杠；错误处理是否清晰；是否避免推卸给Agent |

## PR模板

### 题目5: 训练框架Profiling采集

#### 训练框架
VeRL/MindSpeed-LLM/MindSpeed-MM等选一个

#### Prompt

#### 测试结果（截图）