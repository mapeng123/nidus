# -*- coding: utf-8 -*-
"""一夜任务模式: 开放式使命 -> 子问题分解 -> 假设树搜索 -> 排序报告
run_mission(goal, agent, budget):
  1. decompose: LLM 把使命拆成 <=6 个子问题
  2. 每个子问题: 假设树 + 六步循环, 预算内直到 confirmed 或子树耗尽
  3. 汇总 findings, LLM 评 novelty (vs 文献先验), 输出 ranked report
预算以"实验次数"计 (每次 execute 计 1)。
"""
import json, time
from .search import HypothesisTree

DECOMPOSE_PROMPT = """Research mission: {goal}

Decompose into at most 6 concrete, independently testable sub-questions about the
bone-marrow niche (mechanisms, cell types, ligand-receptor axes, perturbations).
JSON: {{"sub_questions": ["...", "..."]}}"""

NOVELTY_PROMPT = """Finding: {finding}
Supporting experiments: {evidence}

Rate how novel this finding is against established bone biology literature
(0 = textbook knowledge, 1 = not reported anywhere you know). Be strict:
RANKL-RANK axis = 0.0. JSON: {{"novelty": 0.0-1.0, "known_as": "...", "reason": "..."}}"""


def run_mission(goal, agent, llm, budget=100, out_prefix="mission", verbose=True):
    def say(m):
        if verbose:
            print(m, flush=True)

    t0 = time.time()
    say(f"[mission] 目标: {goal}  预算: {budget} 次实验")
    dec = llm.chat_json([{"role": "user", "content": DECOMPOSE_PROMPT.format(goal=goal)}])
    subs = dec["sub_questions"][:6]
    say(f"[mission] 分解为 {len(subs)} 个子问题")

    spent = 0
    report = {"goal": goal, "started": time.strftime("%F %T"), "sub_questions": []}
    for qi, q in enumerate(subs):
        if spent >= budget:
            say("[mission] 预算耗尽, 提前收尾")
            break
        say(f"[mission] 子问题 {qi+1}/{len(subs)}: {q[:70]}")
        tree = HypothesisTree(q)
        root = agent.llm.chat_json([
            {"role": "user", "content":
             f"Sub-question: {q}\nPropose an initial hypothesis. "
             'JSON: {"hypothesis": "..."}'}])
        tree.add(root["hypothesis"])
        sub_rec = {"question": q, "findings": [], "experiments": 0}

        while spent < budget:
            nid = tree.select()
            if nid is None:
                break
            node = tree.nodes[nid]
            final = agent.run_task(f"{q} | hypothesis under test: {node['text']}")
            spent += final["rounds"]
            sub_rec["experiments"] += final["rounds"]
            tree.record(nid, final["verdict"], final["quality"])
            if final["verdict"] == "confirmed":
                nov = llm.chat_json([{"role": "user", "content": NOVELTY_PROMPT.format(
                    finding=final["hypothesis"], evidence=json.dumps(final)[:1500])}])
                sub_rec["findings"].append({**final, **nov})
                say(f"  [finding] novelty={nov.get('novelty')} {final['hypothesis'][:60]}")
                break
            # 失败 -> 变异出新分支
            mem = agent.mem.recall(q)
            tree.mutate(nid, llm, final, mem["episodes"][:3])
        report["sub_questions"].append(sub_rec)

    # 排序: 质量 x 新颖度
    ranked = sorted(
        [f for sq in report["sub_questions"] for f in sq["findings"]],
        key=lambda f: -(f.get("quality", 0) * f.get("novelty", 0)))
    report["ranked_findings"] = ranked
    report["experiments_total"] = spent
    report["elapsed_min"] = round((time.time() - t0) / 60, 1)

    json.dump(report, open(f"{out_prefix}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
        f.write(f"# Mission: {goal}\n\n实验 {spent} 次, 耗时 {report['elapsed_min']} 分钟\n\n")
        for i, r in enumerate(ranked, 1):
            f.write(f"## {i}. {r['hypothesis']}\n"
                    f"- quality={r['quality']} novelty={r.get('novelty')}"
                    f" ({r.get('known_as', '')})\n"
                    f"- verdict={r['verdict']}, rounds={r['rounds']}\n\n")
    say(f"[mission] 完成: {len(ranked)} 个 finding, 报告 -> {out_prefix}.json/.md")
    return report
