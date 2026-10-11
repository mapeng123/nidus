# -*- coding: utf-8 -*-
"""BoneBench 跑分器: 进化曲线 + 记忆开/关消融 + 并行执行
用法:
  from nidus_agent.bench import BoneBench
  bb = BoneBench("bonebench_v3.json")   # [{"task": ..., "gold": ...}, ...]
  curve = bb.run(make_agent, batches=5, memory_on=True, workers=16)
    make_agent(mem) -> Agent   mem 由跑分器提供
进化曲线: 题目按顺序切 batches, 批内并行, 批间串行 (后批吃前批固化的记忆)。
对照组 (memory_on=False): 每题独立空记忆库。
"""
import json, os, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .memory import Memory


JUDGE_PROMPT = """Question: {task}
Gold standard: {gold}
Agent's final answer/hypothesis: {answer}

Does the agent's answer match the gold standard in substance?
JSON: {{"match": 0 or 1, "reason": "one sentence"}}"""


class BoneBench:
    def __init__(self, path):
        self.items = json.load(open(path, encoding="utf-8"))

    def _run_item(self, bi, item, make_agent, memory_on, shared,
                  judge_llm, log_path, log_lock):
        if memory_on:
            mem = shared or Memory(os.path.join(tempfile.gettempdir(),
                                                "bonebench_shared.db"))
        else:
            mem = Memory(os.path.join(tempfile.mkdtemp(), "fresh.db"))
            mem.enabled = False
        agent = make_agent(mem)
        final = agent.run_task(item["task"])
        judge_match = None
        if judge_llm is not None and item.get("gold"):
            judge_match = 0
            # 优先判"基于证据的最终结论"; 老日志无 answer 字段时退回 hypothesis
            graded = final.get("answer") or final["hypothesis"]
            for _attempt in range(2):
                try:
                    j = judge_llm.chat_json([{"role": "user", "content":
                        JUDGE_PROMPT.format(task=item["task"], gold=item["gold"],
                                            answer=graded)}])
                    judge_match = int(j.get("match", 0))
                    break
                except Exception:
                    time.sleep(3)
            final["judge_match"] = judge_match
            final["graded_field"] = "answer" if final.get("answer") else "hypothesis"
        final["qtype"] = item.get("type", "?")
        if log_path:
            with log_lock, open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"batch": bi, **final},
                                   ensure_ascii=False) + "\n")
        return float(final["quality"]), final["rounds"], judge_match

    def run(self, make_agent, batches=5, memory_on=True, log_path=None,
            shared_mem_path=None, judge_llm=None, workers=1, resume_log=None,
            freeze_mem=False):
        """judge_llm 提供时: 用金标客观判分 (accuracy); 否则退回 agent 自评 quality。
        resume_log: 之前跑过一部分的 jsonl; 按 task 文本匹配, 已跑题目直接复用
        旧记录的成绩, 不重跑也不重写日志 (仅对无状态臂 mem_off/bare 严格正确)。"""
        resumed = {}
        if resume_log and os.path.exists(resume_log):
            for line in open(resume_log, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    if r.get("task") and r.get("quality") is not None:
                        resumed[r["task"]] = (float(r["quality"]),
                                              r.get("rounds", 0),
                                              r.get("judge_match"))
                except Exception:
                    pass
            if resumed:
                print(f"[bench] 断点续跑: 复用 {len(resumed)} 条已跑记录",
                      flush=True)
        n = len(self.items)
        per = max(1, (n + batches - 1) // batches)
        shared = Memory(shared_mem_path) if (memory_on and shared_mem_path) else None
        if shared is not None and freeze_mem:
            shared.freeze = True  # 评估期记忆只读, 防止自评噪声回灌
        log_lock = threading.Lock()
        batch_scores, batch_rounds, batch_acc = [], [], []
        for bi in range(batches):
            chunk = self.items[bi * per:(bi + 1) * per]
            if not chunk:
                continue
            results = [None] * len(chunk)
            todo = []
            for i, item in enumerate(chunk):
                if item.get("task") in resumed:
                    results[i] = resumed[item["task"]]
                else:
                    todo.append((i, item))
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futs = {pool.submit(self._run_item, bi, item, make_agent,
                                        memory_on, shared, judge_llm,
                                        log_path, log_lock): i
                            for i, item in todo}
                    for fut in as_completed(futs):
                        results[futs[fut]] = fut.result()
            else:
                for i, item in todo:
                    results[i] = self._run_item(bi, item, make_agent, memory_on,
                                                shared, judge_llm, log_path, log_lock)
            scores = [r[0] for r in results]
            rounds = [r[1] for r in results]
            accs = [r[2] for r in results if r[2] is not None]
            batch_scores.append(round(sum(scores) / len(scores), 4))
            batch_rounds.append(round(sum(rounds) / len(rounds), 3))
            if accs:
                batch_acc.append(round(sum(accs) / len(accs), 4))
            msg = (f"[bench] batch {bi + 1}/{batches}: quality={batch_scores[-1]:.3f} "
                   f"rounds={batch_rounds[-1]:.2f}")
            if batch_acc:
                msg += f" acc={batch_acc[-1]:.3f}"
            print(msg, flush=True)
        return {"batch_scores": batch_scores, "batch_rounds": batch_rounds,
                "batch_accuracy": batch_acc}
