# Research Assistant Agent

多智能体科研助手：基于 LangGraph Supervisor 编排架构，集成文献检索（Web / AMiner / 上传文档 RAG）、来源评审、研究方案人机协同设计、报告生成。

## 系统工作流编排图

```mermaid
flowchart TD
    START([用户消息]) -->|"POST /chat · SSE 流式<br/>initial_state 注入 messages/project_id/summary/required_tools"| LC

    subgraph LC["load_context 节点"]
        LC1["读取: preferences 偏好<br/>+ summary 历史摘要<br/>+ 项目已上传文件清单<br/>+ 最新研究方案"] --> LC2["拼接 system_prompt<br/>→ 写入 state.system_prompt<br/>+ has_prior_research 标记"]
    end

    LC2 --> QT

    subgraph QTN["query_triage 节点（检索前澄清门卫）"]
        Q1["规则预检: 空/乱码/废话<br/>→ query_invalid=true（零 LLM 成本）"] -->|"invalid"| ENDQ([END])
        Q1 -->|"通过"| Q2["LLM 分类 + 概览型规则匹配<br/>→ ambiguous?"]
        Q2 -->|"清晰 → 直通"| Q3["effective_query = 原始输入"]
        Q2 -->|"模糊/概览型"| Q4["_ensure_directions<br/>LLM 补足 + 模板兜底 → 3 个方向"]
        Q4 --> Q5["interrupt() 挂起图执行<br/>→ SSE 推送 query_clarification"]
        Q5 -. "POST /chat/resume<br/>selected_direction / use_original" .-> Q6["Command(resume) 恢复执行<br/>→ effective_query = 用户选择方向"]
    end

    Q3 -->|"route_after_triage: supervisor"| SUP
    Q6 -->|"route_after_triage: supervisor"| SUP

    subgraph SUPN["supervisor 节点（可多次进入，调度次数上限 5）"]
        SUP["决策优先级（确定性规则优先于 LLM）:<br/>① 调度次数 ≥ 5 → FINISH<br/>② planner 已有输出 → FINISH<br/>③ 方案关键词 ∧ (researcher 输出 ∨ has_prior_research) → planner<br/>④ researcher 已执行 ∧ gaps 非空 ∧ search_round<1 → researcher（补搜）<br/>⑤ LLM 自主决策:<br/>├─ 未知 agent → 回退 FINISH<br/>└─ planner 但无内容 → 回退 researcher"]
    end

    SUP -->|"③⑤ next=planner"| PLN1
    SUP -->|"④ next=researcher<br/>search_round+1"| RS2
    SUP -->|"⑤ next=researcher"| RS1
    SUP -->|"①②⑤ next=FINISH"| GEN

    subgraph RS1["researcher 节点 · 首轮"]
        R1["检索模式三选一:<br/>a) required_tools 显式指定 → 跳过 LLM 直接并行<br/>b) 提及文件 ∧ 有上传 → search_uploaded_docs + web_search 并行<br/>c) 默认: LLM bind_tools 决策（并行 ≤ 3 个）"]
        R1 --> R2["执行工具:<br/>web_search → Bocha API<br/>aminer_search_papers → AMiner API(JWT)<br/>search_uploaded_docs → ChromaDB RAG"]
        R2 --> R3["parse_tool_sources → 统一结构<br/>+ 来源编号去重"]
        R3 --> R4["truncate_output 压缩工具结果<br/>+ _verify_source_urls 死链过滤"]
        R4 --> R5["→ state: agent_outputs.researcher<br/>reference_sources"]
    end

    subgraph RS2["researcher 节点 · 补搜轮（MAX_SEARCH_ROUNDS=1，仅 1 次）"]
        R6["query = gaps 拼接（定向补搜）<br/>继承上轮来源继续编号"] --> R7["同上 a/b/c 三模式执行"] --> R5
    end

    R5 -->|"固定边"| REV

    subgraph REVN["reviewer 节点"]
        REV["assess_sources 编排:<br/>权威性/时效性 → 纯规则信号（零 LLM 成本）<br/>相关性 → LLM 阶段1 N 次并行<br/>一致性/缺口/小结 → LLM 阶段2 一次全局调用"] --> RV2["→ state: source_ratings<br/>source_assessments<br/>retrieval_gaps<br/>agent_outputs.reviewer"]
    end

    RV2 -->|"固定边，回到 supervisor（可能触发④补搜闭环，上限 1 次）"| SUP

    subgraph PLNN["planner 节点（人机协同中断点）"]
        PLN1["build_planner_context<br/>（researcher 输出 + 原始问题 + 历史方案）"] --> PLN2["LLM 生成候选方案 → _parse_plan_options"]
        PLN2 --> PLN3["interrupt() 挂起图执行<br/>→ SSE 推送 plan_options 给前端"]
        PLN3 -. "POST /chat/resume<br/>chosen_plan_id / custom_plan_text" .-> PLN4["Command(resume) 恢复执行<br/>→ state: agent_outputs.planner<br/>chosen_plan_id / custom_plan_text<br/>（同时 save_project_plan 持久化）"]
    end

    PLN4 -->|"固定边，回到 supervisor（②触发 FINISH）"| SUP

    subgraph GENN["generate_response 节点"]
        GEN["build_generate_context:<br/>rerank_sources 按可信度重排<br/>+ 参考文献编号对照（低分标注「待验证」）"] --> G2["流式 LLM 生成最终回答"]
    end

    G2 --> ENDN["SSE: response / source / source_ratings / done<br/>save_message 保存对话 + save_project_sources<br/>后台任务 _background_compress → 写 summaries"]
```

## 图说明

- **入口**：`POST /chat`，SSE 流式返回；`query_triage` 与 `planner` 两个节点会通过 `interrupt()` 挂起，经 `POST /chat/resume` 恢复。
- **调度核心**：`supervisor` 可被多次进入，调度次数上限 5；确定性规则（关键词、缺口补搜）优先于 LLM 决策。
- **补搜闭环**：reviewer 产出信息缺口后，supervisor 最多触发 1 次定向补搜（`MAX_SEARCH_ROUNDS=1`）。
- **人机协同**：模糊提问在 `query_triage` 澄清方向，方案设计在 `planner` 让用户选方案/自定义，均通过 interrupt/resume 实现。
- **技术栈**：LangGraph（Supervisor 编排）+ FastAPI（SSE）+ DeepSeek LLM + ChromaDB RAG + SQLite（消息/摘要/方案持久化 + LangGraph checkpoint）。
