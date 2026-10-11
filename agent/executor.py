# -*- coding: utf-8 -*-
"""Nidus 执行器: 服务器队列客户端 + mock
服务器协议 (server_worker.py):
  任务: 写 JSON 到 /root/autodl-tmp/agent_queue/todo/<job_id>.json
  结果: worker 写到 /root/autodl-tmp/agent_queue/done/<job_id>.json
工具集 (worker 端实现):
  ko_gene        硬掩码 KO 生成 -> 全面板 delta vs WT
  expr_query     真实语料表达查询 (cell_type, condition, genes)
  lr_query       L-R 库查询 (基因/细胞型对的配受体边)
  cond_generate  条件生成 + 程序读出
"""
import base64, json, os, sys, time, uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bone_ssh import client  # noqa: E402

QUEUE = "/root/autodl-tmp/agent_queue"


class ServerExecutor:
    """通过 SSH 与骨机上的常驻 worker 通信。模型在 worker 里常驻, 无需每次重载。"""

    def __init__(self, poll_interval=15, timeout=3600):
        self.poll = poll_interval
        self.timeout = timeout

    def submit(self, tool, args):
        job_id = uuid.uuid4().hex[:12]
        payload = json.dumps({"job_id": job_id, "tool": tool, "args": args})
        b64 = base64.b64encode(payload.encode()).decode()
        c = client()
        c.exec_command(f"mkdir -p {QUEUE}/todo {QUEUE}/done && "
                       f"echo {b64} | base64 -d > {QUEUE}/todo/{job_id}.json",
                       timeout=30)
        c.close()
        return job_id

    def wait(self, job_id):
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            c = client()
            _, out, _ = c.exec_command(f"cat {QUEUE}/done/{job_id}.json 2>/dev/null",
                                       timeout=30)
            txt = out.read().decode()
            c.close()
            if txt.strip():
                return json.loads(txt)
            time.sleep(self.poll)
        raise TimeoutError(f"job {job_id} 超时 ({self.timeout}s)")

    def run(self, tool, args):
        return self.wait(self.submit(tool, args))

    # 便捷封装
    def ko_gene(self, cell_type, condition, gene, n_cells=1000, seeds=1):
        return self.run("ko_gene", {"cell_type": cell_type, "condition": condition,
                                    "gene": gene, "n_cells": n_cells, "seeds": seeds})


class MockExecutor:
    """管道测试: 规则化的伪真值 (覆盖 BoneBench 开发集用到的经典事实)。"""

    # (cell_type, gene) -> {下游基因: 方向}  仅用于跑通管道, 不是真数据
    TRUTH = {
        ("osteoblast", "TNFSF11"): {"Tnfsf11_rankl_axis": -1},
        ("mesenchymal stem cell", "TNFSF11"): {"rankl_axis": -1},
    }

    def run(self, tool, args):
        time.sleep(0.05)
        if tool == "ko_gene":
            key = (args.get("cell_type", ""), args.get("gene", ""))
            eff = self.TRUTH.get(key, {})
            return {"job_id": "mock", "tool": tool, "ok": True,
                    "delta": {g: float(d) * 0.42 for g, d in eff.items()},
                    "ko_self_delta": -1.0, "seeds": args.get("seeds", 1)}
        if tool == "expr_query":
            return {"job_id": "mock", "tool": tool, "ok": True,
                    "mean": {g: 1.0 for g in args.get("genes", [])}}
        if tool == "lr_query":
            return {"job_id": "mock", "tool": tool, "ok": True, "edges": []}
        return {"job_id": "mock", "tool": tool, "ok": False, "error": "unknown tool"}


def get_executor(mode="auto"):
    if mode == "mock":
        return MockExecutor()
    return ServerExecutor()
