# -*- coding: utf-8 -*-
"""服务器本地队列执行器: 直接写 /root/autodl-tmp/agent_queue (不走 SSH)
仅在骨机上使用。bench 模式自动 clamp GPU 参数: n_cells<=250, seeds=1。
"""
import json, os, time, uuid

QUEUE = "/root/autodl-tmp/agent_queue"


class LocalQueueExecutor:
    def __init__(self, poll_interval=5, timeout=1800, clamp=True):
        self.poll = poll_interval
        self.timeout = timeout
        self.clamp = clamp
        os.makedirs(f"{QUEUE}/todo", exist_ok=True)
        os.makedirs(f"{QUEUE}/done", exist_ok=True)

    def run(self, tool, args):
        args = dict(args)
        if self.clamp and tool in ("ko_gene", "cond_generate"):
            args["n_cells"] = min(int(args.get("n_cells", 250)), 250)
            args["seeds"] = 10
        job_id = uuid.uuid4().hex[:12]
        tmp = f"{QUEUE}/todo/{job_id}.tmp"
        json.dump({"job_id": job_id, "tool": tool, "args": args},
                  open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, f"{QUEUE}/todo/{job_id}.json")
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            p = f"{QUEUE}/done/{job_id}.json"
            if os.path.exists(p):
                try:
                    return json.load(open(p, encoding="utf-8"))
                except json.JSONDecodeError:
                    time.sleep(1)
                    return json.load(open(p, encoding="utf-8"))
            time.sleep(self.poll)
        return {"ok": False, "error": f"timeout {self.timeout}s"}
