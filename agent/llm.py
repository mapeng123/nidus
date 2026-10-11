# -*- coding: utf-8 -*-
"""LLM 后端: Moonshot / OpenAI 兼容接口 + mock (管道测试用)
环境变量:
  NIDUS_LLM_BASE   默认 https://api.moonshot.cn/v1
  NIDUS_LLM_KEY    API key
  NIDUS_LLM_MODEL  默认 kimi-k2-0905-preview
"""
import json, os, time, urllib.error, urllib.request


class FatalQuotaError(Exception):
    """账户欠费/封停类错误: 重试无意义, 应立即中止整个跑批。"""

class LLM:
    def __init__(self, base=None, key=None, model=None, temperature=0.3):
        cfg = {}
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path, encoding="utf-8"))
        self.base = (base or os.environ.get("NIDUS_LLM_BASE")
                     or cfg.get("NIDUS_LLM_BASE")
                     or "https://api.moonshot.cn/v1").rstrip("/")
        self.key = (key or os.environ.get("NIDUS_LLM_KEY")
                    or cfg.get("NIDUS_LLM_KEY") or os.environ.get("MOONSHOT_API_KEY"))
        self.model = (model or os.environ.get("NIDUS_LLM_MODEL")
                      or cfg.get("NIDUS_LLM_MODEL") or "kimi-k3")
        self.temperature = temperature
        if "k3" in self.model:
            self.temperature = 1  # kimi-k3 仅允许 temperature=1
        # 混合思考模型 (GLM-4.7 等): 可通过配置关闭思考省 token
        self.thinking = (os.environ.get("NIDUS_LLM_THINKING")
                         or cfg.get("NIDUS_LLM_THINKING"))  # None=默认, "disabled"=关

    @property
    def available(self):
        return bool(self.key)

    def chat(self, messages, json_mode=False, max_tokens=2000):
        if "k3" in self.model:
            max_tokens = max(max_tokens, 6000)  # 推理模型需要额外 token 余量
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.thinking:
            body["thinking"] = {"type": self.thinking}
        last_err = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(
                    f"{self.base}/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.key}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                body_txt = ""
                try:
                    body_txt = e.read().decode()
                except Exception:
                    pass
                if ("insufficient balance" in body_txt or "suspended" in body_txt
                        or "余额不足" in body_txt or "无可用资源包" in body_txt):
                    raise FatalQuotaError(body_txt[:200])
                last_err = e
                if e.code in (429, 500, 502, 503, 504):
                    wait = 15 * (2 ** attempt)
                    print(f"[llm] HTTP {e.code}, {wait}s 后重试 "
                          f"({attempt + 1}/4) {body_txt[:80]}", flush=True)
                    time.sleep(wait)
                    continue
                raise
            except (TimeoutError, ConnectionError, urllib.error.URLError) as e:
                last_err = e
                wait = 15 * (2 ** attempt)
                print(f"[llm] {type(e).__name__}, {wait}s 后重试 "
                      f"({attempt + 1}/4)", flush=True)
                time.sleep(wait)
        raise last_err

    def chat_json(self, messages, max_tokens=2000):
        txt = self.chat(messages, json_mode=True, max_tokens=max_tokens)
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            i, j = txt.find("{"), txt.rfind("}")
            return json.loads(txt[i:j + 1])


class MockLLM(LLM):
    """管道测试: 不联网, 返回规则化输出。"""
    def __init__(self):
        self.temperature = 0

    @property
    def available(self):
        return True

    def chat(self, messages, json_mode=False, max_tokens=2000):
        return "MOCK"

    def chat_json(self, messages, max_tokens=2000):
        user = messages[-1]["content"].lower()
        if "propose one testable" in user:
            return {"hypothesis": "mock hypothesis", "rationale": "mock"}
        if "most decisive" in user:
            return {"tool": "ko_gene",
                    "args": {"cell_type": "osteoblast", "condition": "healthy",
                             "gene": "TNFSF11", "n_cells": 1000, "seeds": 1}}
        if "review. json" in user:  # critic pre_check
            return {"verdict": "accept", "issues": [], "fixed_design": None}
        if "overclaiming" in user:  # critic post_review
            return {"confidence": 0.8, "overclaim_flags": [], "needed_followups": []}
        if "decompose into at most" in user:  # mission
            return {"sub_questions": ["q1 RANKL axis", "q2 Wnt axis"]}
        if "rate how novel" in user:
            return {"novelty": 0.5, "known_as": "mock", "reason": "mock"}
        if "mutated hypothesis" in user:
            return {"hypothesis": "mutated mock", "change_vs_parent": "mock"}
        if "initial hypothesis" in user:
            return {"hypothesis": "root mock hypothesis"}
        return {"verdict": "confirmed", "lesson": "mock lesson padding here",
                "quality": 0.6, "tags": ["mock"]}
