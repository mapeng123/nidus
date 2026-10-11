# -*- coding: utf-8 -*-
"""bench_runner_mem2.py — 自进化Stage2: 冻结验证库, 只跑batch4-5 (items[297:]), 与mem_off配对比较
判据: 同一批题 mem_off(log_v10_merged_final) vs frozen-mem(本脚本), McNemar。
"""
import json, shutil, sqlite3, sys, time
sys.path.insert(0, '/root/autodl-tmp')
from nidus_agent.llm import LLM
from nidus_agent.local_executor import LocalQueueExecutor
from nidus_agent.loop import Agent
from nidus_agent.critic import Critic
from nidus_agent.bench import BoneBench

OUT = '/root/autodl-tmp/bench_out'
SRC_DB = f'{OUT}/memory_stage1.db'
FROZEN_DB = f'{OUT}/memory_frozen.db'

# ---- 1. 构建冻结库: 仅保留有定论(confirmed/refuted)且quality>=0.6的剧集 ----
shutil.copy(SRC_DB, FROZEN_DB)
db = sqlite3.connect(FROZEN_DB)
n0 = db.execute('select count(*) from episodes').fetchone()[0]
db.execute("delete from episodes where verdict not in ('confirmed','refuted') or quality < 0.6")
db.commit()
n1 = db.execute('select count(*) from episodes').fetchone()[0]
db.close()
print(f'[frozen] episodes {n0} -> {n1} (confirmed/refuted, quality>=0.6)', flush=True)

# ---- 2. 切出 batch4-5 题目 (与 bench.py 切分一致: per=ceil(495/5)=99) ----
items = json.load(open('/root/autodl-tmp/nidus_agent/bonebench_v4.json'))
if isinstance(items, dict):
    items = items.get('items', items.get('questions'))
sub = items[297:]
assert len(sub) == 198, len(sub)
SUB_JSON = '/root/autodl-tmp/nidus_agent/bonebench_v4_b45.json'
json.dump(sub, open(SUB_JSON, 'w'), ensure_ascii=False)
print(f'[subset] {len(sub)} items (batch4-5)', flush=True)

# ---- 3. 跑 frozen-mem 臂 ----
llm = LLM()
assert llm.available
print(f'LLM: {llm.model} (mem_on stage2 frozen)', flush=True)
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

bb = BoneBench(SUB_JSON)
log_path = f'{OUT}/log_mem_stage2.jsonl'
t0 = time.time()
res = bb.run(make_agent, batches=2, memory_on=True,
             shared_mem_path=FROZEN_DB, freeze_mem=True,
             workers=2, judge_llm=llm, log_path=log_path, resume_log=log_path)
res['arm'] = 'tools_v10_mem_on_stage2_frozen_b45'
res['n_frozen_episodes'] = n1
res['wall_min'] = round((time.time() - t0) / 60, 1)
json.dump(res, open(f'{OUT}/curve_mem_stage2.json', 'w'), ensure_ascii=False, indent=1)
print(f'=== mem stage2 完成 {res["wall_min"]}min ===', flush=True)
print('MEM2_DONE', flush=True)
