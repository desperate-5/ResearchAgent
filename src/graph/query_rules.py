"""检索前规则层：通用乱输入识别（词素存在性 + 重复单元检测）。

设计要点（为什么不用元音/辅音等统计猜测）：
- 统计特征（元音占比、连续辅音数）是针对"某一种乱码"的拟合，换个随机串就失效；
  本模块改用**词素存在性**：输入中只要含任一中文词素或英文真实单词/术语即放行，
  一个都没有才判定为无意义输入。这是语言学的确定性判据，不是统计猜测。
- 英文词判定：
  * 长度 >= 4 的常见词做**子串匹配**（容忍拼写变形与前后缀：research → researched/researching）；
    选择长度 >= 4 的词做子串是关键——2-3 字母词（is/an/at）出现在随机串里的概率太高，
    而 >= 4 的常见词出现在随机串里的概率极低。
  * 长度 <= 3 的短词只做**整词**匹配，避免误放行（xnisjdl 含 "is" 也不能算有词）。
  * 全大写字母串（>= 2 字符）视为术语/缩写（LLM / RAG / GPT）。
- 重复单元检测：文本可分解为长度 >= 2 的单元重复 >= 2 次 → 判垃圾
  （覆盖 gfdgfdgfd、锟斤拷锟斤拷 这类"有字符但无单词"的串；单字符重复由原有规则处理）。
- 宁漏勿杀：规则层负责拦截明显的垃圾；边界情况（如公式、生僻缩写）漏过时，
  由 LLM 分类层（query_triage 的 invalid 判定）兜底，绝不阻塞主流程。
"""

import re

# ── 长度 >= 4 的常见英语词（子串匹配）───────────────────────────────
# 通用高频词 + 学术/科研词。词根化存储（research 即可命中 researcher/researched）。
_COMMON_WORDS = frozenset(
    """
    about above accept access across action active actual address after again
    against agent agents ahead allow almost alone along already also although
    always among amount analysis analyze another answer appear area around asked
    assume attention away back based basic because become before began begin
    being below best better between beyond both bring brought build called came
    cannot care carry case cause center certain change check choose class clear
    close code cognitive come common compare comparison complete computation
    computer concept conclusion consider contain content continue control corpus
    course create criterion current daily data date deal decided deep define
    degree design detail develop development different digital direct discuss
    discussion distance does doing down during each early earth easily easy edge
    effect effort either else end enough enter entire equal even event ever
    every everyone everything example expect experience explain explanation
    express extract extraction eye face fact fail fall family far fast feel
    field figure final find fine first follow force form found four free from
    full function further future game general give given global goal good great
    green ground group grow growth half hand happen hard head health hear help
    hello high history hold home hope hour house human hypothesis idea important
    include increase indeed individual industry information inside instance
    instead interest into issue itself join just keep kind know known language
    large last late later learn learning least leave left less level life light
    like line list little live local long look lose loss lot love main major
    make man many matter mean means measure meet member method methods might
    mind model models month more most move much must name national nature need
    never next nice none normal north note nothing notice number object occur
    offer often once only open operate order other others over own page paper
    papers part party pass past pay people perhaps period person physical place
    plan plans play point possible power practice present press pretty price please
    problem problems process produce product program project projects provide
    public purpose put quality question questions quick quite race raise range
    rather reach read ready real reason receive record reduce region relate
    remember report research result results return review right road role room
    rule run same save say school science search second section see seem sense
    serve service set several share short show shown side similar simple since
    small social some something sometimes south space speak special specific
    stand start state states still stop story strong study subject such suggest
    support sure system systems table take talk team tell term test than thank
    their them then there these they thing things think third this those though
    thought three through time together told took top total toward town track
    trade train travel treat true try turn two type under understand unit until
    upon use used useful using usual value various very view want warm way ways
    week well went were what when where whether which while white whole whose
    will within without woman women work world would write wrong wrote year
    years young your morning night today tomorrow yesterday internet email phone

    abstract academic accuracy achieve algorithm analysis annotation application
    approach architecture artificial assess assessment benchmark citation
    classify cluster compress dataset domain dynamic efficiency embedding
    empirical evaluation evidence experiment experimental feature framework
    generation generate graph graphs implement implementation inference infer
    intelligent knowledge latent literature machine mechanism memory metric
    methodology multimodal network neural objective ontology optimize parameter
    performance planning precision prediction probability prompt reasoning
    recall retrieval semantic sentence sentiment software statistical summary
    survey symbolic syntactic task token transfer transformer training
    translation unsupervised validation vector vision vocabulary
    """.split()
)

# ── 长度 <= 3 的短词 / 常见缩写（仅整词匹配）────────────────────────
_SHORT_WORDS = frozenset(
    """
    the and for are but not you all can how why who what when where did get has
    her him his its let may new now old our out put say she too use was way ask
    run set try big few off own via per vs ok hi hey ai llm rag gpt api pdf ui
    nlp cnn lstm bert dna rna cpu gpu ram os io db id ml cv dl kg qa faq url
    http www org com edu gov cn us uk 3d 5g
    """.split()
)

# ── 领域术语 / 产品名（长度 >= 4，子串匹配）─────────────────────────
_DOMAIN_TERMS = frozenset(
    """
    langchain langgraph deepseek openai anthropic chatgpt chroma ragas faiss
    pytorch tensorflow huggingface transformers matplotlib numpy pandas scikit
    sklearn spacy nltk streamlit fastapi sqlite postgres mongodb redis docker
    kubernetes github gitlab vscode jupyter colab arxiv scholar semantic
    """.split()
)


def has_any_morpheme(text: str) -> bool:
    """词素存在性检查：文本中是否含任一中文词素或英文真实单词/术语。

    有中文字符 → 一定有词素（中文是表意文字，每个字都是实义字）。
    纯英文/混合文本 → 逐段检查英文字母串，命中短词（整词）、常见词（子串）、
    全大写缩写、领域术语（子串）任一即视为有词。
    """
    if re.search(r"[\u4e00-\u9fff]", text):
        return True
    for run in re.findall(r"[A-Za-z]+", text):
        if len(run) >= 2 and run.isupper():  # 全大写：LLM / RAG / GPT / COVID
            return True
        low = run.lower()
        if low in _SHORT_WORDS:              # 短词仅整词匹配
            return True
        if any(w in low for w in _COMMON_WORDS):   # 长词子串匹配，容忍变形
            return True
        if any(w in low for w in _DOMAIN_TERMS):   # 领域术语子串匹配
            return True
    return False


def is_repetitive_pattern(text: str) -> bool:
    """重复单元检测：文本是否为某个长度 >= 2 的单元重复 >= 2 次。

    覆盖 gfdgfdgfd、锟斤拷锟斤拷、123123123 等"有字符但无词素"的串。
    单字符重复（啊啊啊啊）由上层 len(set)==1 规则处理，不在此重复。
    """
    n = len(text)
    for u in range(2, n // 2 + 1):
        unit = text[:u]
        if (unit * (n // u + 1))[:n] == text:
            return True
    return False


def rule_reject(text: str) -> bool:
    """规则层总入口：返回 True 表示输入无效（乱码/废话/无意义串）。

    判定顺序（零 LLM 成本）：
    1. 空 / 过短（< 2 字符）
    2. 有效字符占比 < 30%（乱码字节 / 纯符号）
    3. 整串仅一个字符重复（啊啊啊啊 / !!!!!）
    4. 重复单元（gfdgfdgfd / 锟斤拷锟斤拷）
    5. 无任何词素（ndbajdkla566 / xnisjdl / 12345）
    """
    s = text.strip()
    if not s or len(s) < 2:
        return True
    meaningful = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", s))
    if meaningful / len(s) < 0.3:
        return True
    if len(set(s)) == 1:
        return True
    if is_repetitive_pattern(s):
        return True
    if not has_any_morpheme(s):
        return True
    return False
