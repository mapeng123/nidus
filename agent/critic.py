# -*- coding: utf-8 -*-
"""Critic 角色: 执行前评审 (防止浪费实验) + 结论复核 (防止自我确认)
两个用途:
  pre_check(hypothesis, design) -> {verdict: accept|revise, issues, fixed_design?}
  post_review(task, conclusion, evidence) -> {confidence, overclaim_flags}
"""
import json

CRITIC_SYS = """You are the hostile reviewer inside Nidus-Agent. Your job is to catch
mistakes BEFORE an experiment spends GPU and AFTER a conclusion is drawn.
Check, in order:
1. Producer/receiver logic: is a ligand KO targeted at the producing cell type?
2. Baseline expression: does the design verify the target gene is expressed?
3. Readout validity: is the claimed readout within Nidus capability boundaries
   (direction/magnitude only; no severity, no discrimination claims)?
4. Statistical adequacy: seeds >= 3 for any claim; 10 for figure-grade claims.
5. Novelty honesty: would this result be trivially known from literature? Flag it.
Be terse and specific. JSON only."""

PRE_PROMPT = """Hypothesis: {hypothesis}
Proposed experiment: {design}

Review. JSON: {{"verdict": "accept|revise", "issues": ["..."],
"fixed_design": {{...}} or null}}"""

POST_PROMPT = """Task: {task}
Conclusion: {conclusion}
Evidence trail: {evidence}

Review for overclaiming. JSON: {{"confidence": 0.0-1.0,
"overclaim_flags": ["..."], "needed_followups": ["..."]}}"""


class Critic:
    def __init__(self, llm, strict=True):
        self.llm = llm
        self.strict = strict  # strict=False 时仅记录不拦截 (消融用)

    def pre_check(self, hypothesis, design):
        r = self.llm.chat_json([
            {"role": "system", "content": CRITIC_SYS},
            {"role": "user", "content": PRE_PROMPT.format(
                hypothesis=hypothesis, design=json.dumps(design, ensure_ascii=False))}])
        if (self.strict and r.get("verdict") == "revise" and r.get("fixed_design")
                and isinstance(r["fixed_design"], dict)
                and "tool" in r["fixed_design"] and "args" in r["fixed_design"]):
            return r["fixed_design"], r.get("issues", [])
        return design, r.get("issues", [])

    def post_review(self, task, conclusion, evidence):
        return self.llm.chat_json([
            {"role": "system", "content": CRITIC_SYS},
            {"role": "user", "content": POST_PROMPT.format(
                task=task, conclusion=conclusion,
                evidence=json.dumps(evidence, ensure_ascii=False)[:3000])}])
