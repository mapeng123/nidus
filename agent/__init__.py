# -*- coding: utf-8 -*-
"""Nidus-Agent: 自进化发现智能体原型
结构:
  memory.py    实验记忆库 (episode / strategy / skill 三层)
  llm.py       LLM 后端 (Moonshot / OpenAI 兼容 / mock)
  executor.py  Nidus 执行器 (服务器队列客户端 + mock)
  loop.py      六步自主循环 observe->hypothesize->design->execute->evaluate->reflect
  bench.py     BoneBench 跑分 + 进化曲线 (记忆开/关消融)
  server_worker.py  部署到骨机的常驻 worker (模型只加载一次)
"""
__version__ = "0.1.0"
