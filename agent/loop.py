# -*- coding: utf-8 -*-
"""六步自主循环: observe -> hypothesize -> design -> execute -> evaluate -> reflect
设计原则:
  - LLM 权重冻结, Nidus 权重冻结; 进化的只是记忆/策略/技能三层上下文
  - 每轮全量落库 (可审计), verdict 三态: confirmed / refuted / inconclusive
  - 同教训出现 >=2 次自动提升为 strategy (策略层固化的最简实现)
"""
import json, re, time
from collections import Counter
from .llm import FatalQuotaError

SYS = """You are the planning brain of Nidus-Agent, a discovery agent operating a virtual
bone-marrow niche (Nidus, a frozen generative model; 132 cell types, 19 conditions,
zero-shot KO of any of 16,105 panel genes, 2,730 ligand-receptor pairs).

Hard capability boundaries (never over-claim in hypotheses):
- Readout is a binned expression proxy: direction and magnitude fidelity, NOT
  perturbation discrimination, NOT clinical severity.
- Rare lineages (osteocyte) are lower-bound only.
- Ligand KOs must target the PRODUCER cell type; receptor KOs target the receiver.
- Compound claims are mechanism-level only (target + IC50 -> virtual KO/agonism).
- ko_gene currently serves osteoblast / mesenchymal stem cell / osteoclast only;
  for other cell types use expr_query / lr_query and biological knowledge.

Tools (executor.run(tool, args) — use EXACTLY these names):
- ko_gene(cell_type, condition, gene, n_cells, seeds): hard-mask KO. Returns
  top gene deltas PLUS program-level readouts under "programs":
  osteoblast_formation_net / osteoclast_resorption_net (mean delta of the 19
  marker genes), per-marker deltas, and "calibration" with formation_z /
  resorption_z (z-score vs 49 control-gene KOs; |z|>1 means the effect exceeds
  the generic perturbation response) and formation_call / resorption_call
  ("increase"/"decrease"/"minimal"). ALWAYS report program direction from the
  calibration call, not from the top-gene lists.
- expr_query(cell_type, condition, genes): baseline expression in the real corpus.
- lr_query(gene_or_pair): ligand-receptor edges from the curated LR database.
- cond_generate(condition, cell_type, n_cells, seeds): condition generation.
- drug_lookup(name): is a compound covered by the target-annotation library
  (4277 ChEMBL compounds)? Returns in_library, phase, n_targets.

IMPORTANT behavioral rules:
- For any question about the effect of knocking out a gene (direction, magnitude,
  program changes), you MUST actually run ko_gene. The tool's hard-mask KO and
  null calibration are valid even for genes with low baseline expression — a
  near-zero effect will read as "minimal", which is itself a valid answer.
  NEVER skip the KO just because baseline expression looks low.
- Answer from tool evidence, not from general knowledge alone. If a tool fails
  (unsupported cell type, unknown condition), adjust the arguments and retry
  once; only give up after a retry also fails.
- Paracrine / cross-cell questions: ko_gene measures the effect WITHIN the
  chosen cell type only. If a question asks how a factor secreted by cell type
  A affects cell type B (e.g. osteocyte sclerostin acting on osteoblasts),
  do NOT KO the ligand in cell type B and report 'minimal' — the ligand is not
  produced there. Use lr_query to identify the receptor, then reason about the
  downstream effect in B from the known signaling logic, stating clearly that
  direct cross-cell simulation is beyond the current tool.

Always reply in JSON when asked. Be concrete: one hypothesis, one decisive experiment."""

HYP_PROMPT = """Task: {task}

Relevant past experience:
{memory}

Propose ONE testable mechanistic hypothesis for this task.
JSON: {{"hypothesis": "...", "rationale": "...", "expected": "...", "tags": [...]}}"""

DESIGN_PROMPT = """Task: {task}
Hypothesis: {hypothesis}
Past strategies: {strategies}

Design the single most decisive in-silico experiment. Respect the capability boundaries
(baseline expression check first when relevant; ligand KO in producer cells).
JSON: {{"tool": "ko_gene|expr_query|lr_query|cond_generate", "args": {{...}},
"readout": "what result would confirm/refute"}}"""

EVAL_PROMPT = """Task: {task}
Hypothesis: {hypothesis}
Experiment: {experiment}
Result: {result}

Judge the outcome. When the result contains a calibration block or
DIRECTION_ANSWER field, that field is the ONLY valid basis for program-direction
statements: a "minimal" call means no significant change, regardless of how the
top_up/top_down gene lists look (they include the generic knockout response).
JSON: {{"verdict": "confirmed|refuted|inconclusive",
"lesson": "one-sentence reusable lesson (empty string if nothing reusable)",
"quality": 0.0-1.0, "tags": ["gene:...", "celltype:...", "condition:..."]}}"""

ANS_PROMPT = """Task: {task}

Your investigation log (hypotheses, experiments run, tool results, verdicts):
{log}

Give your FINAL answer to the task, based on the experimental evidence above.
Rules:
- Answer the question DIRECTLY and SPECIFICALLY (direction, gene/cell-type names,
  numbers from the tool results when available).
- Tool results override prior knowledge: if an experiment contradicts your initial
  hypothesis, the experiment wins.
- For KO direction questions, cite the program-level calibration call and z value.
- HARD RULE on calibration: if the tool returns formation_call/resorption_call
  = "minimal" (|z| <= 1), your answer MUST be that the program shows no
  significant change ("minimal/no effect"). NEVER override a "minimal"
  calibration call by reading raw marker-delta signs as increase/decrease —
  those are within generic perturbation noise.
- COMMIT to the best-supported answer. Only answer "inconclusive" when tool
  failures made it impossible to gather ANY relevant evidence — an imperfect
  or surprising tool result is still evidence; interpret it.
- For "which gene ... more" comparison questions: compare the calibration
  z VALUES numerically (larger z = stronger increase; more negative z =
  stronger decrease) and ALWAYS name one gene. A "minimal" call on both sides
  is NOT a reason to refuse — pick the one with the more extreme z.
JSON: {{"answer": "...", "confidence": 0.0-1.0,
"basis": "which tool results support this answer"}}"""


class Agent:
    def __init__(self, llm, executor, memory, critic=None, skills=None,
                 max_rounds_per_task=3, verbose=True, bench_mode=False,
                 vocab_text=None):
        self.llm, self.ex, self.mem = llm, executor, memory
        self.critic = critic      # None = 无评审消融
        self.skills = skills      # None = 无技能库消融
        self.max_rounds = 2 if bench_mode else max_rounds_per_task
        self.verbose = verbose
        self.bench_mode = bench_mode  # 跑分模式: 少轮次/轻评审/跳过复核, 省调用
        self.sys = SYS + ("\n\n" + vocab_text if vocab_text else "")
        # 从 vocab_text 解析细胞型/条件词表, 供实体路由召回使用
        self.vocab = ([l[2:].strip() for l in vocab_text.splitlines()
                       if l.startswith("- ")] if vocab_text else [])
        self.log = []

    def _entities(self, task):
        """从任务文本提取生物学实体 (基因/细胞型/条件), 用于路由召回。
        不依赖基准的 type 标签, 真实部署同样可用。"""
        ents = {t.lower() for t in re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", task)}
        low = task.lower()
        for v in self.vocab:
            if v.lower() in low:
                ents.add(v.lower())
        return ents

    def _say(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _chat(self, messages, fallback=None, required=()):
        """LLM 调用容错: 失败/缺关键字段重试 1 次, 再失败返回 fallback。"""
        for attempt in range(2):
            try:
                r = self.llm.chat_json(messages)
                # GLM 偶发返回 [{...}] 数组包裹, 归一化为 dict
                if isinstance(r, list) and r and isinstance(r[0], dict):
                    r = r[0]
                if not isinstance(r, dict):
                    raise ValueError(f"unexpected json type {type(r)}")
                missing = [k for k in required if k not in r]
                if missing:
                    raise ValueError(f"missing keys {missing}")
                return r
            except FatalQuotaError:
                raise  # 欠费/封停: 不重试不兜底, 直接中止
            except Exception as e:
                self._say(f"[llm 重试 {attempt + 1}/2] {type(e).__name__}: {e}")
                time.sleep(3)
        self._say("[llm 放弃] 使用 fallback")
        return fallback

    def run_task(self, task):
        """对一个任务跑六步循环, 返回最终结论与轮数。"""
        t0 = time.time()
        ents = self._entities(task)
        ctx = self.mem.recall(task, entities=ents)
        self._say(f"[observe] 检索到 {len(ctx['episodes'])} 条记忆, "
                  f"{len(ctx['strategies'])} 条策略")
        final = None
        tool_calls = []
        for rnd in range(1, self.max_rounds + 1):
            hyp = self._chat([
                {"role": "system", "content": self.sys},
                {"role": "user", "content": HYP_PROMPT.format(
                    task=task,
                    memory=json.dumps(ctx["episodes"][:5], ensure_ascii=False) or "none")}],
                fallback={"hypothesis": task, "rationale": "llm failure",
                          "expected": "", "tags": []},
                required=["hypothesis"])
            self._say(f"[hypothesize r{rnd}] {str(hyp['hypothesis'])[:80]}")

            exp = self._chat([
                {"role": "system", "content": self.sys},
                {"role": "user", "content": DESIGN_PROMPT.format(
                    task=task, hypothesis=hyp["hypothesis"],
                    strategies=json.dumps(ctx["strategies"][:5], ensure_ascii=False) or "none")}],
                required=["tool", "args"])
            if exp is None or "tool" not in exp or "args" not in exp:
                self._say(f"[design r{rnd}] 设计失败, 任务记为 inconclusive")
                eid = self.mem.add_episode(
                    task=task, hypothesis=hyp["hypothesis"], experiment={},
                    result_summary="design stage failed (llm)",
                    verdict="inconclusive", lesson="", tags=[], quality=0.0)
                final = {"task": task, "rounds": rnd,
                         "hypothesis": hyp["hypothesis"], "verdict": "inconclusive",
                         "quality": 0.0, "elapsed_s": round(time.time() - t0, 1)}
                break
            self._say(f"[design r{rnd}] {exp['tool']}({json.dumps(exp['args'])[:60]})")

            # 跑分模式下 critic 只在第 1 轮预审 (省调用), 且跳过结论复核
            if self.critic is not None and (not self.bench_mode or rnd == 1):
                try:
                    exp, issues = self.critic.pre_check(hyp["hypothesis"], exp)
                except Exception as e:
                    issues = []
                    self._say(f"[critic r{rnd}] 评审失败跳过: {type(e).__name__}")
                if issues:
                    self._say(f"[critic r{rnd}] {len(issues)} 个意见, "
                              f"设计{'已修正' if exp else '保留'}")
            if exp is None:
                final = {"task": task, "rounds": rnd,
                         "hypothesis": hyp["hypothesis"], "verdict": "inconclusive",
                         "quality": 0.0, "elapsed_s": round(time.time() - t0, 1)}
                break

            try:
                result = self.ex.run(exp["tool"], exp["args"])
            except Exception as e:
                result = {"ok": False, "error": f"executor: {type(e).__name__}: {e}"}
            tool_calls.append({"tool": exp["tool"], "args": exp["args"],
                               "ok": bool(result.get("ok")),
                               "result": json.dumps(result, ensure_ascii=False)[:1200]})
            self._say(f"[execute r{rnd}] ok={result.get('ok')}")

            ev = self._chat([
                {"role": "system", "content": self.sys},
                {"role": "user", "content": EVAL_PROMPT.format(
                    task=task, hypothesis=hyp["hypothesis"],
                    experiment=json.dumps(exp, ensure_ascii=False),
                    result=json.dumps(result, ensure_ascii=False)[:3000])}],
                fallback={"verdict": "inconclusive", "lesson": "",
                          "quality": 0.0, "tags": []},
                required=["verdict"])
            self._say(f"[evaluate r{rnd}] verdict={ev['verdict']} q={ev.get('quality')}")

            try:
                quality = float(ev.get("quality", 0))
            except (TypeError, ValueError):
                quality = 0.0
            raw_tags = ev.get("tags", [])
            if not isinstance(raw_tags, list):
                raw_tags = []
            tags = []
            for t in raw_tags:
                t = str(t)
                tags.append(t.split(":", 1)[1] if ":" in t else t)
            eid = self.mem.add_episode(
                task=task, hypothesis=hyp["hypothesis"], experiment=exp,
                result_summary=json.dumps(result, ensure_ascii=False)[:2000],
                verdict=ev["verdict"], lesson=ev.get("lesson", ""),
                tags=tags, quality=quality)
            self._maybe_promote_strategy(ev.get("lesson", ""), eid)

            self.log.append({"task": task, "round": rnd, "hyp": hyp, "exp": exp,
                             "result": json.dumps(result, ensure_ascii=False)[:1500],
                             "verdict": ev["verdict"], "quality": ev.get("quality", 0)})
            final = {"task": task, "rounds": rnd, "hypothesis": hyp["hypothesis"],
                     "verdict": ev["verdict"], "quality": quality,
                     "elapsed_s": round(time.time() - t0, 1)}
            if ev["verdict"] == "confirmed":
                break
            ctx = self.mem.recall(task, entities=ents)  # 复盘后带着新记忆进入下一轮

        # 最终结论综合: judge 打分的对象从"实验前假设"改为"基于证据的结论"
        if final is not None:
            ans = self._chat([
                {"role": "system", "content": self.sys},
                {"role": "user", "content": ANS_PROMPT.format(
                    task=task,
                    log=json.dumps(self.log[-4:], ensure_ascii=False)[:4500])}],
                fallback={"answer": final["hypothesis"], "confidence": 0.0,
                          "basis": "fallback to initial hypothesis (llm failure)"},
                required=["answer"])
            final["answer"] = ans["answer"]
            final["answer_confidence"] = ans.get("confidence")
            final["answer_basis"] = ans.get("basis", "")
            final["tool_calls"] = tool_calls
            self._say(f"[answer] {str(ans['answer'])[:80]}")

        # 结论复核 (防自我确认; 跑分模式跳过以省调用)
        if (final and self.critic is not None and not self.bench_mode
                and final["verdict"] == "confirmed"):
            try:
                review = self.critic.post_review(task, final["hypothesis"],
                                                 self.log[-3:])
            except Exception as e:
                review = {"error": f"{type(e).__name__}: {e}"}
            final["post_review"] = review
            if review.get("overclaim_flags"):
                self._say(f"[critic 复核] overclaim: {review['overclaim_flags']}")
        return final

    def _maybe_promote_strategy(self, lesson, eid):
        if not lesson or len(lesson) < 12:
            return
        # 只从经确认的高质量剧集里提炼策略, 失败经验的教训不进策略库
        rows = self.mem.db.execute(
            "SELECT lesson FROM episodes WHERE lesson != ''"
            " AND verdict='confirmed' AND quality>=0.6").fetchall()
        cnt = Counter(r[0] for r in rows)
        for text, n in cnt.items():
            if n >= 2:
                exists = self.mem.db.execute(
                    "SELECT 1 FROM strategies WHERE rule=?", (text,)).fetchone()
                if not exists:
                    sid = self.mem.add_strategy(
                        name=text[:40], rule=text, evidence_ids=[eid])
                    self._say(f"[strategy] 固化为策略: {text[:60]}... ({sid})")
