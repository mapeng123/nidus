# -*- coding: utf-8 -*-
"""bench_runner_v10.py — 只重跑答错的 pairwise_compare 题(比较规则生效后)"""
import json, sys, time
sys.path.insert(0, '/root/autodl-tmp')
from nidus_agent.llm import LLM
from nidus_agent.local_executor import LocalQueueExecutor
from nidus_agent.loop import Agent
from nidus_agent.critic import Critic
from nidus_agent.bench import BoneBench

OUT = '/root/autodl-tmp/bench_out'
llm = LLM()
assert llm.available
print(f'LLM: {llm.model}', flush=True)
ex = LocalQueueExecutor(clamp=True)

vocab = json.load(open('/root/autodl-tmp/agent_queue/vocab.json'))
VOCAB_TEXT = ('Valid cell types (use EXACTLY one of these names):\n- '
              + '\n- '.join(vocab['ct_vocab'])
              + '\n\nValid conditions (use EXACTLY one of these):\n- '
              + '\n- '.join(vocab['cond_vocab'])
              + '\nNote: not every cell-type x condition pair has real cells; '
                'if a query returns an error, pick a nearby lineage or another condition.')

def make_agent(mem):
    return Agent(llm, ex, mem, critic=Critic(llm), bench_mode=True,
                 verbose=False, vocab_text=VOCAB_TEXT)

bb = BoneBench('/root/autodl-tmp/nidus_agent/bonebench_v4.json')

# resume = 除'答错的pairwise'以外的全部记录
keep = f'{OUT}/log_v10_keep.jsonl'
n_keep = n_drop = 0
with open(keep, 'w', encoding='utf-8') as w:
    for line in open(f'{OUT}/log_v9_merged_final.jsonl', encoding='utf-8'):
        r = json.loads(line)
        if r.get('judge_match') == 0 and r.get('qtype') == 'pairwise_compare':
            n_drop += 1
            continue
        w.write(json.dumps(r, ensure_ascii=False) + '\n')
        n_keep += 1
print(f'保留 {n_keep}, 重跑 pairwise 错题 {n_drop}', flush=True)

log_path = f'{OUT}/log_v10_pairwise.jsonl'
t0 = time.time()
res = bb.run(make_agent, batches=1, memory_on=False, workers=2,
             judge_llm=llm, log_path=log_path, resume_log=keep)
res['arm'] = 'tools_v10_pairwise_rule'
res['wall_min'] = round((time.time() - t0) / 60, 1)
json.dump(res, open(f'{OUT}/curve_v10.json', 'w'), ensure_ascii=False, indent=1)
print(f'=== v10 完成 {res["wall_min"]}min ===', flush=True)
print('V10_DONE', flush=True)
