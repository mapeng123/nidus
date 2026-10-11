# -*- coding: utf-8 -*-
"""实验记忆库: episode(单次实验) / strategy(固化的策略) / skill(自创工具)
SQLite 存储, 检索用 token 重叠打分 (依赖零, 后续可换 embedding)。
"""
import json, os, sqlite3, threading, time, uuid
from collections import Counter

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
  id TEXT PRIMARY KEY, ts REAL, task TEXT, hypothesis TEXT,
  experiment TEXT, result_summary TEXT, verdict TEXT,
  lesson TEXT, tags TEXT, quality REAL);
CREATE TABLE IF NOT EXISTS strategies (
  id TEXT PRIMARY KEY, ts REAL, name TEXT, rule TEXT,
  evidence TEXT, uses INTEGER DEFAULT 0, successes INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS skills (
  id TEXT PRIMARY KEY, ts REAL, name TEXT, description TEXT,
  code TEXT, uses INTEGER DEFAULT 0);
"""

def _tok(s):
    s = (s or "").lower()
    for ch in ",.;:!?()[]{}<>/\\\"'\"`\n\t":
        s = s.replace(ch, " ")
    return [t for t in s.split() if len(t) > 1]

class Memory:
    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.db.executescript(SCHEMA)
            self.db.commit()
        self.enabled = True  # False = 无记忆消融 (写入但不检索)

    # ---------- 写入 ----------
    def add_episode(self, task, hypothesis, experiment, result_summary,
                    verdict, lesson, tags, quality=0.0):
        if getattr(self, "freeze", False):
            return "frozen"  # 冻结模式 (评估期只读): 写入静默丢弃
        eid = uuid.uuid4().hex[:12]
        with self._lock:
            self.db.execute(
                "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?)",
                (eid, time.time(), task, hypothesis, json.dumps(experiment, ensure_ascii=False),
                 result_summary, verdict, lesson,
                 json.dumps(list(tags), ensure_ascii=False), quality))
            self.db.commit()
        return eid

    def add_strategy(self, name, rule, evidence_ids):
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self.db.execute("INSERT INTO strategies (id,ts,name,rule,evidence) VALUES (?,?,?,?,?)",
                            (sid, time.time(), name, rule, json.dumps(evidence_ids)))
            self.db.commit()
        return sid

    def bump_strategy(self, sid, success):
        with self._lock:
            self.db.execute(
                "UPDATE strategies SET uses=uses+1, successes=successes+? WHERE id=?",
                (1 if success else 0, sid))
            self.db.commit()

    def add_skill(self, name, description, code):
        sid = uuid.uuid4().hex[:12]
        with self._lock:
            self.db.execute("INSERT INTO skills VALUES (?,?,?,?,?,0)",
                            (sid, time.time(), name, description, code))
            self.db.commit()
        return sid

    # ---------- 检索 ----------
    def _score(self, qtok, text):
        tt = Counter(_tok(text))
        return sum(tt[t] for t in qtok if t in tt)

    def recall(self, task, k=5, entities=None):
        """返回相关 episodes + strategies。
        门控 (v2): 只召回 confirmed 且 quality>=0.6 的剧集。
        路由 (v3): entities 提供时优先只保留共享实体的剧集 (防跨题型串扰),
        无共享实体时回退全局 token 召回。"""
        if not self.enabled:
            return {"episodes": [], "strategies": []}
        qtok = set(_tok(task))
        with self._lock:
            # 门控: 只召回经确认的高质量剧集, 防止失败/错误经验污染后续任务
            rows = self.db.execute(
                "SELECT id,task,hypothesis,verdict,lesson,tags,quality,experiment"
                " FROM episodes WHERE verdict='confirmed' AND quality>=0.6").fetchall()
            strat_rows = self.db.execute(
                "SELECT id,name,rule,successes,uses FROM strategies").fetchall()
        scored = []
        for r in rows:
            text = f"{r[1]} {r[2]} {r[4]} {r[5]}"
            s = self._score(qtok, text)
            if s > 0:
                scored.append((s, r))
        scored.sort(key=lambda x: -x[0])
        if entities:
            routed = []
            for s, r in scored:
                hay = f"{r[1]} {r[2]} {r[7]}".lower()  # task+hypothesis+experiment
                if any(e in hay for e in entities):
                    routed.append((s, r))
            if routed:
                scored = routed
        eps = [{"id": r[0], "task": r[1], "hypothesis": r[2], "verdict": r[3],
                "lesson": r[4], "tags": json.loads(r[5]), "quality": r[6]}
               for _, r in scored[:k]]
        strats = [{"id": r[0], "name": r[1], "rule": r[2],
                   "success_rate": (r[3] / r[4]) if r[4] else None}
                  for r in strat_rows]
        strats.sort(key=lambda x: -(x["success_rate"] or 0))
        return {"episodes": eps, "strategies": strats}

    def stats(self):
        with self._lock:
            c = self.db
            return {
                "episodes": c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                "strategies": c.execute("SELECT COUNT(*) FROM strategies").fetchone()[0],
                "skills": c.execute("SELECT COUNT(*) FROM skills").fetchone()[0],
            }
