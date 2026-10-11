# -*- coding: utf-8 -*-
"""技能库: agent 自创分析工具 (Voyager/STELLA 式能力层进化)
技能 = 一段受限 Python: def run(result: dict) -> dict
执行环境: 仅 math/json/statistics + 白名单内建函数, 禁 import/文件/网络 (原型级沙箱)
生产环境应换 subprocess 隔离; 这里先保证可用与可审计 (代码全文入库)。
"""
import json, math, statistics

ALLOWED_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "round": round, "sorted": sorted, "enumerate": enumerate,
    "zip": zip, "range": range, "float": float, "int": int, "str": str,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "any": any,
    "all": all, "isinstance": isinstance, "getattr": getattr,
}


class SkillRunner:
    def __init__(self, memory):
        self.mem = memory

    def create(self, name, description, code):
        """校验语法 + 必须定义 run(), 然后入库。"""
        compile(code, f"<skill:{name}>", "exec")
        if "def run(" not in code:
            raise ValueError("技能必须定义 run(result)")
        return self.mem.add_skill(name, description, code)

    def run(self, skill_id, result):
        row = self.mem.db.execute(
            "SELECT name, code FROM skills WHERE id=?", (skill_id,)).fetchone()
        if not row:
            raise KeyError(f"skill {skill_id} 不存在")
        name, code = row
        g = {"__builtins__": ALLOWED_BUILTINS,
             "math": math, "json": json, "statistics": statistics}
        exec(code, g)  # noqa: S102 (原型级沙箱, 代码已全文入库可审计)
        out = g["run"](result)
        self.mem.db.execute("UPDATE skills SET uses=uses+1 WHERE id=?", (skill_id,))
        self.mem.db.commit()
        return out

    def list_skills(self):
        return [{"id": r[0], "name": r[1], "description": r[2], "uses": r[3]}
                for r in self.mem.db.execute(
                    "SELECT id,name,description,uses FROM skills").fetchall()]
