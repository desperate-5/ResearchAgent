# 方案 B 升级计划：检索-评估-再检索的受控反馈闭环

> 目标：把 reviewer 从"事后点评"升级为"事中控制器"，构建带反馈闭环的受控 agent 循环。
> 核心拆分原则：**评估由 LLM 驱动（判断检索质量、产出缺口描述），循环的触发与终止由规则兜底（缺口数 > 0 且补搜次数 < 上限）。**
> 检索质量评估：**规则信号（域名/类型/时效/可达性）+ LLM 语义信号（相关/一致）混合评分**，输出可解释评分卡，替代纯 LLM 主观评级。

---

## 1. 背景与现状

现有图结构（[builder.py](src/graph/builder.py)）：

```
START → load_context → supervisor → researcher → reviewer → supervisor → ... → generate_response → END
```

现状的关键断点：

- [reviewer_node](src/graph/nodes.py) 只输出每条来源的高/中/低评级（`source_ratings`），评审完即结束，**意见不回注任何后续决策**。
- [supervisor_node](src/graph/nodes.py) 的补搜是**硬编码禁止**的（prompt 里写明"reviewer 意见仅供参考，不可自动补搜"），只在用户明确要求时才重派 researcher。
- 评级结果仅用于前端展示 + 回答措辞提示，不参与检索重排与引用权重。

方案 B 要做的，是把这条"断头"的评审结果接上，形成：**检索 → 评估 →（缺口判断）→ 定向补搜 → 重排 → 分层回答** 的闭环。

---

## 2. 目标架构（闭环全景）

```
用户问题
   │
   ▼
[researcher] 粗检索 ──► [reviewer 评估]
   ▲                        │
   │                        ├─ 每条来源评级（保留）
   │                        └─ 质量小结 + 缺口列表（新增）
   │                              │
   │                        [supervisor 规则门控]
   │                              │
   │               缺口数 > 0 且 补搜次数 < 上限？
   │                    │是                │否
   │                    ▼                  ▼
   └────── 定向补搜（换 query/换工具）   [重排 + 分层回答]
                                        (generate_response)
```

---

## 3. 功能点与实现方式

### 功能点 1：检索质量评估（核心）—— 规则信号 + LLM 语义混合评分

**要实现什么**

把"纯 LLM 打高/中/低"升级为**可解释的多维度评估**：每个来源从 4 个维度打分，规则信号负责可核查的事实，LLM 只负责需要语义理解的维度，加权聚合出综合可信度，并输出带证据的评分卡。

**1.1 评估维度设计**

| 维度 | 含义 | 主要信号 | 分制 |
|---|---|---|---|
| 权威性 | 来源渠道的可信程度 | 规则为主 | 0-5 |
| 时效性 | 内容是否满足问题时效要求 | 规则 | 0-5 |
| 相关性 | 内容与用户问题子主题的匹配度 | LLM 为主 | 0-5 |
| 一致性 | 与其他来源结论的共识/冲突 | LLM | 0-5 |

**1.2 规则信号（可核查事实，零 LLM 成本）**

1. 域名规则表 `SOURCE_DOMAIN_RULES`（新增 `src/graph/reviewer/rules.py`）：
   - 加分域：doi.org、arxiv.org、期刊/出版社官网、edu.cn、.gov 官方机构
   - 减分域：个人博客、低质聚合站、已知不可信域名
2. 来源类型权重：paper > 官方机构 > news > blog（复用 `parse_tool_sources` 的 `source_type`）
3. URL 可达性：复用现有 `_verify_source_urls` 的存活结果，死链权威性直接 0 分
4. 时效解析：web 来源 `published` 字段目前为空，需新增发布日期解析；问题含"最新"等时效词时要求近 1-2 年

**1.3 LLM 语义信号（两阶段调用：N 次并行小调用 + 1 次全局调用）**

相关性、一致性、缺口对上下文需求不同，不能全部塞 1 次调用，也不宜全部拆 N 次：

| 判断 | 需要看什么 | 性质 | 调用方式 |
|---|---|---|---|
| 相关性分 0-5 | 单条来源 vs 用户问题 | 逐来源独立 | **N 次并行**（`asyncio.gather`），每次上下文 = 单条来源内容 + 用户问题，小且便宜 |
| 一致性分 0-5 | 该来源 vs 其他所有来源 | 跨来源对比 | **1 次全局调用**（单条视角看不到其他来源） |
| 缺口 + 小结 | 用户问题 vs 整个检索批次 | 全局视野 | **1 次全局调用**（与一致性合并） |

**阶段 1（并行）**：每条来源 1 次调用 → 出相关性分；N 个调用同时跑，总耗时 ≈ 1 次调用，避免单次塞全文时 LLM 遗漏部分来源。

**阶段 2（全局）**：1 次调用 → 出一致性分 + gaps + summary；上下文用所有来源的**压缩信息**（标题 + 一句话结论），不用全文。

**可省钱变体（可选）**：一致性不逐条输出 0-5 分，只标注存在冲突的 0-N 条来源，无冲突来源一致性取默认中分，牺牲粒度换成本。

**1.4 评分聚合模型（可配置权重）**

```
维度分: s_权威, s_时效, s_相关, s_一致 ∈ [0, 5]
权重:   w = [0.35, 0.15, 0.30, 0.20]   # 常量放 prompts.py，可调
score = Σ w_i × s_i
映射:   score ≥ 3.8 → 高    ≥ 2.5 → 中    否则 → 低
```

**1.5 结构化输出（可解释评分卡）**

每个来源产出 `source_assessments`：

```json
{"source_number": 1,
 "dimension_scores": {"authority": 4, "timeliness": 3, "relevance": 5, "consistency": 4},
 "score": 4.15, "credibility": "高",
 "evidence": "域名 doi.org 白名单+1；近2年发表满足时效；完整覆盖子主题X"}
```

- 用 `llm.with_structured_output(PydanticModel)` 替代裸 `ainvoke` + 括号计数解析
- **降级策略**：解析失败时仅用规则信号计算综合分（LLM 维度取中性分 2.5），不阻塞主流程
- 同时保留 `summary`（整体质量小结）与 `gaps`（缺口列表）两个输出
- 新增 Pydantic 模型：`SourceAssessment` / `ReviewerOutput(assessments/summary/gaps)`

---

### 功能点 2：扩展 AgentState 状态字段

**要实现什么**

让"缺口信息"和"补搜计数"能在节点间流转。

**怎么实现**

在 [state.py](src/graph/state.py) 的 `AgentState` 中新增字段：

```python
retrieval_gaps: list[str]    # 缺口子问题列表
source_assessments: list[dict]  # 每条来源评分卡（dimension_scores/score/credibility/evidence）
search_round: int            # 已补搜次数（0 表示首轮，每次补搜 +1）
```

默认值在 [server.py](src/api/server.py) 构造 `initial_state` 时给出（`search_round=0`、`retrieval_gaps=[]`）。

---

### 功能点 3：supervisor 补搜决策 —— 规则兜底的门控

**要实现什么**

用规则替代现在"禁止自动补搜"的硬约束：**`len(retrieval_gaps) > 0` 且 `search_round < MAX_SEARCH_ROUNDS` 时，路由回 researcher 做定向补搜；否则 FINISH。**

- LLM 负责产出"缺口是什么"（功能点 1），规则负责"要不要再搜、最多搜几次"。
- 常量 `MAX_SEARCH_ROUNDS = 1`（首轮 + 最多补搜 1 次），放在 [prompts.py](src/graph/prompts.py) 顶部与现有 `MAX_CONTEXT_TURNS` 并列。

**怎么实现**

- 在 [supervisor_node](src/graph/nodes.py) 的最前面（在 `len(log) >= 5` 判断之后、LLM 调用之前）插入一段确定性门控逻辑：

```python
gaps = state.get("retrieval_gaps", [])
search_round = state.get("search_round", 0)
if "researcher" in agent_outputs and gaps and search_round < MAX_SEARCH_ROUNDS:
    return {
        "next_agent": "researcher",
        "supervisor_log": log + [{"next": "researcher", "reason": f"存在 {len(gaps)} 个信息缺口，触发第 {search_round + 1} 次补搜"}],
    }
```

- 保留现有 `len(log) >= 5` 的循环上限作为最后兜底，双重防发散。
- 同时精简 [SUPERVISOR_PROMPT_MINIMAL](src/graph/prompts.py)：去掉"不可自动补搜"的硬性文字，改为说明"补搜已由规则门控，无需你手动判断"。

---

### 功能点 4：researcher 定向补搜 —— 换 query / 换工具

**要实现什么**

补搜时不再拿原始用户问题原样重搜，而是针对缺口子问题定向搜索，避免浪费。

**怎么实现**

- 修改 [researcher_node](src/graph/nodes.py) 的 query 来源：补搜时以 `retrieval_gaps` 拼接的文本作为搜索 query，而非 `_extract_user_query`。
- 新增一个"是否补搜"的判断：`state.get("search_round", 0) > 0` 即视为补搜轮。
- 补搜轮的工具决策仍走现有"LLM 决策工具 + 并行执行"，但把缺口作为新 query 注入。
- 返回值中额外更新 `search_round = search_round + 1`，并把本轮结果**追加**进 `agent_outputs["researcher"]`（或单独存 `researcher_round_2`），供后续合并。

---

### 功能点 5：评级驱动的重排 / 过滤（轻量 rerank）

**要实现什么**

把 reviewer 的每条来源评级作为重排权重，让低可信来源在最终上下文里沉底或剔除，高可信来源置顶。

**怎么实现**

- 新增一个纯函数（可放在 [context.py](src/graph/context.py) 或 [nodes.py](src/graph/nodes.py)），输入 `reference_sources` + `source_assessments`，输出重排后的来源列表：

```python
重排分 = 归一化(source_assessments.score) × 0.5 + 归一化(相关性维度分) × 0.5
# 相关性维度分 = source_assessments[i].dimension_scores.relevance（评审阶段 1 已产出，无需新增向量相似度）
低可信（score < 2.5）来源沉底并追加「（待验证）」标记
```

- 在 [build_generate_context](src/graph/context.py) 中，用重排后的来源列表构建"参考文献编号对照"，低可信来源追加 `（待验证）` 标记。
- 不引入 cross-encoder 等重型重排模型——当前检索池规模（约 10-20 条）不足以支撑其收益，规则 + 评级加权已足够。

---

### 功能点 6：回答引用强度控制

**要实现什么**

让评级影响最终回答的**确定性表达**：高可信来源支撑"已确认结论"，低可信来源降级为"待验证观点"。

**怎么实现**

- 修改 [GENERATE_PROMPT](src/graph/prompts.py)，明确要求：
  - 核心结论只由高可信来源支撑；
  - 引用低可信来源时使用"据 XX 报道，有待核实"等降级措辞；
  - 参考文献列表按可信度降序排列。
- 依赖功能点 5 重排后的来源列表，`generate_response` 直接消费。

---

### 功能点 7：终止保障与循环安全

**要实现什么**

确保闭环不会死循环、不会成本失控、行为可复现。

**怎么实现**

三重兜底，均不依赖 LLM 自律：

| 兜底 | 机制 | 位置 |
|---|---|---|
| 补搜上限 | `search_round < MAX_SEARCH_ROUNDS` | supervisor 门控 |
| 调度上限 | `len(supervisor_log) >= 5` 强制 FINISH（已有） | supervisor_node |
| 缺口消解 | 补搜后重新走 reviewer，若仍无缺口则自然 FINISH | 图循环 |

---

## 4. 涉及改动的文件清单

| 文件 | 改动 |
|---|---|
| [src/graph/state.py](src/graph/state.py) | 新增 `source_assessments` / `retrieval_gaps` / `search_round` 字段 |
| [src/graph/prompts.py](src/graph/prompts.py) | 新增 `MAX_SEARCH_ROUNDS`；改 `REVIEWER_PROMPT` / `SUPERVISOR_PROMPT_MINIMAL` / `GENERATE_PROMPT` |
| [src/graph/nodes.py](src/graph/nodes.py) | reviewer_node 变薄，改为调用 `reviewer.assess_sources()`；supervisor 补搜门控；researcher 定向补搜；新增重排函数 |
| [src/graph/reviewer/](src/graph/reviewer/__init__.py)（新增包） | 评审节点领域逻辑：规则信号、评分聚合、LLM 两阶段评估、schema 定义 |
| [src/graph/context.py](src/graph/context.py) | 重排后来源注入 `build_generate_context` |
| [src/api/server.py](src/api/server.py) | 初始化新字段默认值（`search_round=0`、`retrieval_gaps=[]`） |

`src/graph/reviewer/` 包内文件划分：

| 文件 | 职责 |
|---|---|
| `__init__.py` | 对外暴露 `assess_sources(state)` 编排入口 |
| `schemas.py` | Pydantic 模型（`SourceAssessment` / `ReviewerOutput`） |
| `rules.py` | 规则信号：域名表、来源类型权重、时效解析（纯函数，可单测） |
| `scoring.py` | 评分聚合：综合分计算、高低中映射、评分卡组装（纯函数，可单测） |
| `llm_assess.py` | LLM 两阶段调用：阶段 1 并行相关性 / 阶段 2 全局一致性+缺口+小结 |

---

## 5. 实施顺序（建议里程碑）

1. **状态字段 + 常量**（功能点 2）：先铺好数据通道，改动最小、无风险。
2. **规则信号层**（功能点 1）：域名表、来源类型权重、时效解析，纯函数可单测。
3. **LLM 维度分 + 评分聚合**（功能点 1）：接 `with_structured_output`，产出评分卡与缺口。
4. **supervisor 门控 + researcher 定向补搜**（功能点 3、4）：核心闭环打通。
5. **重排 + 引用强度**（功能点 5、6）：评分卡驱动回答质量。
6. **循环安全验证**（功能点 7）：构造"低质量检索"用例验证补搜只触发一次、能正常终止。

---

## 6. 风险与权衡

| 风险 | 说明 | 缓解 |
|---|---|---|
| 补搜引入额外延迟与 token 成本 | 每次补搜 = 1 次 LLM 工具决策 + 1 轮工具调用 | 上限 1 次；缺口为空时零成本 |
| reviewer 结构化输出解析失败 | LLM 偶尔不按 schema 输出 | 解析失败降级为 `gaps=[]`，退化为现状 |
| 补搜与原始结果合并导致上下文膨胀 | 两轮结果叠加变长 | 复用 `_truncate_output` 或按轮次截断 |
| 低可信过滤误删高相关来源 | 只看质量会误伤 | 综合分 = 相关 × 质量，非纯质量过滤 |
| LLM 维度分不稳定 | 相关性/一致性分每次推理有波动 | 规则维度权重占比 50% 托底；LLM 维度分多次采样取中位数（可选） |

---

## 7. 验收标准

1. 给定"AI Agent 综述"类问题，reviewer 能输出非空的 `gaps` 列表。
2. 首轮结果存在缺口时，supervisor 触发恰好 1 次定向补搜，且补搜 query 针对缺口而非原始问题。
3. 补搜后无论结果如何，流程能正常 FINISH 并生成回答（不进入死循环）。
4. 最终回答中，低可信来源被降级措辞或从核心结论中剔除。
5. 每条来源的 `source_assessments` 含 4 个维度分、综合分、证据描述，评级可追溯到具体依据。
6. 相同来源在域名/类型/时效相同时，规则维度得分完全一致（可复现性验证）。
