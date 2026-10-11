# -*- coding: utf-8 -*-
"""bench_runner_bare.py — bare 对照臂: 同一 LLM, 工具全部禁用
用法: python bench_runner_bare.py full
"""
import json, sys, time

sys.path.insert(0, '/root/autodl-tmp')
from nidus_agent.llm import LLM
from nidus_agent.loop import Agent
from nidus_agent.critic import Critic
from nidus_agent.bench import BoneBench

OUT = '/root/autodl-tmp/bench_out'

class BareExecutor:
    def run(self, tool, args):
        return {'ok': False, 'error': 'tools disabled in this arm'}

llm = LLM()
assert llm.available, 'no API key'
print(f'LLM: {llm.model} (bare arm, no tools)', flush=True)
ex = BareExecutor()

vocab = json.load(open('/root/autodl-tmp/agent_queue/vocab.json'))
VOCAB_TEXT = ''

def make_agent(mem):
    return Agent(llm, ex, mem, critic=Critic(llm), bench_mode=True,
                 verbose=False, vocab_text=VOCAB_TEXT)

bb = BoneBench('/root/autodl-tmp/nidus_agent/bonebench_v4.json')
print(f'题库 {len(bb.items)} 题', flush=True)

log_path = f'{OUT}/log_v6_bare.jsonl'
t0 = time.time()
res = bb.run(make_agent, batches=5, memory_on=False, workers=2,
             judge_llm=llm, log_path=log_path, resume_log=log_path)
res['arm'] = 'bare_v6'
res['wall_min'] = round((time.time() - t0) / 60, 1)
json.dump(res, open(f'{OUT}/curve_v6_bare.json', 'w'), ensure_ascii=False, indent=1)
print(f'=== bare 完成 {res["wall_min"]}min ===', flush=True)
print('BARE_DONE', flush=True)
