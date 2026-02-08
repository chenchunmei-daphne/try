# Level 2 — Project Constraints（当前阶段硬约束）

本文件描述 `wireless-cax` 项目在**当前阶段（项目启动期与基础结构搭建阶段）**已经明确、必须遵守、不可违反的硬约束事实。  
本文件仅陈述“是什么”，不包含原因、解释、规划或判断。

## 一、阶段性范围硬约束

- 当前阶段仅限于**仓库结构、文档体系与协作机制的建立**
- 当前阶段不开展任何具体无线信道模型、电磁算法或数值方法的实现
- 当前阶段不以算法正确性、性能指标或实验结果作为交付目标
- 当前阶段不进行大规模工程开发或功能性扩展


## 二、仓库结构硬约束

- 项目工作必须基于已确认的仓库顶层结构开展
- 不允许在当前阶段随意新增或调整顶层目录职责
- 所有新增内容必须放置于其职责已明确的目录下
- `agents/`、`kb/`、`examples/`、`tests/` 在当前阶段仅作为占位与边界声明存在
- 截至当前阶段，项目仓库的整体结构已经建立，结构如下：

```text
.
+---agents
|       README.md
|
+---docs
|   |   README.md
|   |
|   +---collaboration
|   |       ai_collaboration.md
|   |       README.md
|   |
|   \---context
|           level0_project_identity.md
|           level1_project_stage.md
|           level2_project_constraints.md
|           README.md
|
+---examples
|       README.md
|
+---kb
|   |   README.md
|   |
|   \---retrospective
|           2026_01_06_project_bootstrap_and_ai_collaboration_retrospective.md
|           README.md
|
+---src
|   |   README.md
|   |
|   +---apps
|   |       README.md
|   |
|   \---libs
|       +---fealpy_ext
|       |   \---wireless
|       |           README.md
|       |
|       \---ofx_ext
|           \---wireless
|                   README.md
|
\---tests
        README.md
```

## 三、文档职责边界硬约束

- `docs/context/`  
  - 仅用于存放项目事实层上下文  
  - 内容必须为 facts-only  
  - 禁止出现推理、判断、评价、规划、TODO 或路线描述

- `docs/collaboration/`  
  - 用于存放协作流程与 AI 协作规则  
  - 具体操作细节仅能存在于该目录及其子文档中

- `kb/`  
  - 用于认知、推导、分析与探索性内容  
  - 当前阶段不要求填充内容，但不得混入事实层文档

上述职责边界在当前阶段视为强约束。


## 四、协作流程硬约束

- 项目协作统一采用 **Issue + PR** 工作流
- 所有变更（代码、文档、结构）必须通过 PR 进入主分支
- 不允许绕过 Issue 或 PR 直接修改主分支内容
- Issue 与 PR 是当前阶段唯一被认可的协作记录形式

## 五、AI 使用硬约束

- AI 仅作为辅助工具参与当前阶段工作
- AI 生成的任何内容不得直接进入主分支
- 所有 AI 产出必须经过人工审查并通过 PR 合并
- AI 不作为事实来源、不作为决策主体

## 六、实现与依赖硬约束

- 当前阶段不引入复杂第三方依赖
- 当前阶段不进行 FEALPy 或 OpenFiniteX 主仓库的代码回流
- `src/` 目录在当前阶段仅确认结构，不开展实质性实现

## 七、变更与冻结约束

- 本文件描述的约束在当前阶段视为有效且必须遵守
- 对本文件的任何修改必须通过 Issue + PR 完成
- 未被明确修改的约束默认持续有效


