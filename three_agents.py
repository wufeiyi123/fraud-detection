#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于结构化工作底稿的三智能体协同分析系统
=========================================

实现“指控—辩护—裁决”三智能体顺序博弈流程，用于识别上市公司
资产与利润舞弊风险（利润舞弊 / 资产舞弊两个子方向）。

设计严格遵循《基于结构化工作底稿的三智能体构建思路》：
  指控智能体(prosecution)  -> 发现风险，避免漏报
  辩护智能体(defense)      -> 解释合理性，降低误报
  裁决智能体(verdict)      -> 综合裁决，形成最终风险等级与核查建议

事实基础：workpaper_generator.py 生成的结构化工作底稿（JSON / dict）。
三智能体不重新计算指标，只基于底稿中的结构化证据进行推理。

依赖：
    pip install openai>=1.0
    （DeepSeek 兼容 OpenAI SDK 接口）

用法：
    # 1) 已有底稿 JSON 文件
    python three_agents.py --workpaper workpaper_000651_2023.json --out result.json

    # 2) 直接从底稿生成器生成底稿再分析（需要数据 CSV + 模型文件在同目录）
    python three_agents.py --stock_code 000651 --year 2023 --out result.json
    python three_agents.py --stock_code 002740 --year 2020 --out result.json

    # 设置 API Key（二选一）
    export DEEPSEEK_API_KEY="sk-aa8a11c2f4724dbabc8b7eb84051d99b"
    # 或运行时 --api_key sk-aa8a11c2f4724dbabc8b7eb84051d99b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── DeepSeek 配置 ──────────────────────────────────────────────
BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"     # 也可换 deepseek-v4-pro
DEFAULT_TEMPERATURE = 0.2           # 低温度，减少自由发挥


# ════════════════════════════════════════════════════════════════
#  一、可调软参数（AgentConfig）：系统设计的敏感度、证据要求、风险认定等参数，三智能体共用
# ════════════════════════════════════════════════════════════════
class AgentConfig:
    """三智能体可调软参数配置类，包含指控、辩护、裁决智能体的参数。"""

    # ── 指控智能体 ──
    prosecution = {
        "max_accusations_per_type": 3,        # 每类风险最多输出几个主要指控点
        "allow_single_high_rule_to_high": False,  # 单一高权重规则能否直接判高风险
        "allow_non_rule_anomaly": True,       # 是否允许规则外异常（须有指标/文本证据）
        "high_risk_min_evidence": 2,          # 高风险指控至少需几条证据共同支持
        "allow_cross_dim_escalation": True,   # 是否允许跨维度风险提升等级
    }

    # ── 辩护智能体 ──
    defense = {
        "strength_levels": ["无有效辩护", "弱辩护", "中等辩护", "强辩护"],
        "commonsense_max_strength": "弱辩护",   # 常识性解释最高只能给到的强度
        "allow_downgrade_without_evidence": False,  # 无底稿证据能否下调风险
        "strong_defense_can_reject": False,    # 强辩护是否允许驳回指控
        "multi_high_rule_no_low": False,       # 多项高权重规则协,可以直接降为低风险
    }

    # ── 裁决智能体 ──
    verdict = {
        "high_risk_need_multi_strong": True,  # 高风险需多项强证据相互印证且辩护失败
        "insufficient_evidence_to_watchlist": True,  # 证据不足 -> 保留关注而非直接判高
        "keep_minority_opinion": True,        # 辩护成功但仍有疑点 -> 写入少数意见
        "ml_weight": "重要参考",               # ML 模型在裁决中的地位（重要参考，不机械决定）
    }


# ════════════════════════════════════════════════════════════════
#  二、底稿切片：根据不同智能体的分析重点，从完整底稿中提取相关模块
# ════════════════════════════════════════════════════════════════
def slice_for_prosecution(wp: Dict[str, Any]) -> Dict[str, Any]:
    """指控智能体重点读取的底稿模块。"""
    return {
        "basic_info": wp.get("basic_info"),
        "risk_summary": wp.get("risk_summary"),
        "core_financial_metrics": wp.get("core_financial_metrics"),
        "hit_rules_detail": wp.get("hit_rules_detail"),
        "evidence_chain": wp.get("evidence_chain"),
        "key_risk_signals": wp.get("key_risk_signals"),
        "fraud_triangle_analysis": wp.get("fraud_triangle_analysis"),
        "engine_status_brief": _engine_brief(wp.get("engine_status", {})),
    }


def slice_for_defense(wp: Dict[str, Any]) -> Dict[str, Any]:
    """辩护智能体重点读取的底稿模块。"""
    return {
        "basic_info": wp.get("basic_info"),
        "peer_comparison": wp.get("peer_comparison"),
        "non_financial_indicators": wp.get("non_financial_indicators"),
        "fraud_triangle_analysis": wp.get("fraud_triangle_analysis"),
        "core_financial_metrics": wp.get("core_financial_metrics"),
        "engine_status_brief": _engine_brief(wp.get("engine_status", {})),
        "comprehensive_assessment": wp.get("comprehensive_assessment"),
    }


def slice_for_verdict(wp: Dict[str, Any]) -> Dict[str, Any]:
    """裁决智能体重点读取的底稿模块。"""
    rs = wp.get("risk_summary", {}) or {}
    return {
        "basic_info": wp.get("basic_info"),
        "risk_summary": rs,
        "ml_score": rs.get("ml_score"),
        "overall_risk_level_ml": rs.get("overall_risk_level_ml"),
        "evidence_chain": wp.get("evidence_chain"),
        "key_risk_signals": wp.get("key_risk_signals"),
        "questions_for_verification": wp.get("questions_for_verification"),
        "comprehensive_assessment": wp.get("comprehensive_assessment"),
        "engine_status_brief": _engine_brief(wp.get("engine_status", {})),
    }


def _engine_brief(engine_status: Dict[str, Any]) -> Dict[str, Any]:
    """提取引擎状态中“不可作为主要依据”的规则清单。"""
    details = (engine_status or {}).get("details", {}) or {}
    unusable = []
    for code, d in details.items():
        status = d.get("status", "")
        if status in ("停用", "未实现", "计算异常", "数据不足"):
            unusable.append({"rule_code": code, "status": status, "name": d.get("name", "")})
    return {
        "total_rules": engine_status.get("total_rules"),
        "normal_calculated": engine_status.get("normal_calculated"),
        "unusable_rules": unusable,
        "note": "unusable_rules 中的规则因停用/未实现/计算异常/数据不足，不得作为主要指控或裁决依据。",
    }


# ════════════════════════════════════════════════════════════════
#  三、LLM 客户端封装（DeepSeek，OpenAI 兼容）
# ════════════════════════════════════════════════════════════════
class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = BASE_URL,
                 model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("[ERROR] 缺少 openai 库，请先运行: pip install openai")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def chat_json(self, system_prompt: str, user_prompt: str,
                  max_retries: int = 3) -> Dict[str, Any]:
        """调用模型并强制返回 JSON（解析失败自动重试）。"""
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = resp.choices[0].message.content
                return _safe_parse_json(content)
            except Exception as e:
                last_err = e
                print(f"[WARN] 第 {attempt} 次调用失败: {e}")
                time.sleep(1.5 * attempt)
        raise RuntimeError(f"LLM 调用多次失败: {last_err}")


def _safe_parse_json(text: str) -> Dict[str, Any]:
    """从模型输出中稳健地解析 JSON。"""
    if text is None:
        raise ValueError("模型返回为空")
    text = text.strip()
    # 去掉可能的 ```json 围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 退而求其次：截取第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


# ════════════════════════════════════════════════════════════════
#  四、提示词（System Prompt）：固化角色、规则映射与调参约束
# ════════════════════════════════════════════════════════════════
COMMON_BACKGROUND = """你是上市公司财务舞弊风险识别系统中的一个智能体。
本系统识别“资产与利润舞弊风险”，分为两个子方向：
  - 利润舞弊风险：关注收入、利润、应收账款、毛利率与经营现金流之间是否异常背离。
  - 资产舞弊风险：关注存货、在建工程、预付款、其他应收款等是否异常增长、长期挂账、周转偏弱或减值不足。

底稿中的 17 条规则按底层风险来源分三类：
  R1 = 收入舞弊；R2 = 存货异常；R3 = 资金异常。
风险类别映射：
  - 利润舞弊风险：主要由 R1 类规则 + 部分 R3 类（现金流、关联交易、资金压力相关）规则支持。
  - 资产舞弊风险：主要由 R2 类规则 + 部分 R3 类（如 R3-02 预付款、资金占用）规则支持。
  - 资金异常(R3)不单独作为一级舞弊类别，而是作为利润/资产舞弊的辅助证据。

铁律：
  1. 所有判断必须来源于给定的结构化工作底稿，不得编造底稿中不存在的指标、规则或证据。
  2. engine_status_brief.unusable_rules 中的规则（停用/未实现/计算异常/数据不足）不得作为主要依据。
  3. 严格只输出 JSON，不要任何额外解释、markdown 或前后缀文字。
"""

PROSECUTION_SYSTEM = COMMON_BACKGROUND + """
== 你的角色：指控智能体（prosecution）==
你是风险发现者，站在审计与监管的怀疑视角，从底稿中寻找资产与利润舞弊迹象。
你只负责发现与归类风险，不负责最终裁决。

可调敏感度参数（必须严格遵守）：
  - 每类风险最多输出 {max_acc} 个主要指控点（main_accusations）。
  - 没有底稿证据支持的内容不得列为主要指控，只能放入 attention_items。
  - 单一低权重规则命中只能列为关注事项(attention_items)。
  - 单一高权重规则命中但无其他证据印证，risk_level 最高只能判为“中风险”。
  - 两个及以上规则或指标共同指向同一风险，可判“中高风险”。
  - 只有多项高权重规则 + 关键指标 + 行业对比共同支持时，才可判“高风险”。
  - 不允许为凑数量而输出证据不足的指控。
  - {cross_dim}

对利润舞弊，重点分析：收入舞弊类规则是否命中；收入/利润/经营现金流是否匹配；
应收账款与营收增长是否背离；毛利率是否显著偏离行业；经营现金流是否支撑净利润；
是否存在关联交易辅助利润调节。
对资产舞弊，重点分析：存货异常类规则是否命中；存货周转率是否显著低于行业；
存货占比是否异常偏高；存货增速是否高于成本增速；在建工程是否长期挂账不转固；
预付款或其他资产科目是否异常。

只输出如下 JSON 结构（字段名严格一致）：
{{
  "agent_role": "prosecution",
  "profit_fraud": {{
    "risk_level": "低风险/中风险/中高风险/高风险",
    "confidence": "低/中/高",
    "main_accusations": [
      {{"accusation_id": "P1", "risk_point": "", "related_rules": [], "related_indicators": [], "evidence": "", "fraud_meaning": "", "strength": "弱/中/强"}}
    ]
  }},
  "asset_fraud": {{
    "risk_level": "低风险/中风险/中高风险/高风险",
    "confidence": "低/中/高",
    "main_accusations": [
      {{"accusation_id": "A1", "risk_point": "", "related_rules": [], "related_indicators": [], "evidence": "", "fraud_meaning": "", "strength": "弱/中/强"}}
    ]
  }},
  "attention_items": [],
  "data_limitations": []
}}
"""

DEFENSE_SYSTEM = COMMON_BACKGROUND + """
== 你的角色：辩护智能体（defense）==
你是误报过滤者，站在企业解释与审计复核视角，对指控逐条进行合理性解释与反向检验。
你不是为否定所有指控，而是判断：哪些异常有合理商业解释、哪些指控证据不足、
哪些即使有解释仍需保留关注、哪些需进一步人工核查。

可寻找的辩护方向（reason_type 取值）：
  行业合理性 / 业务模式合理性 / 经营周期合理性 / 数据质量与重要性 / 反向证据。

辩护强度（defense_strength）四档：无有效辩护 / 弱辩护 / 中等辩护 / 强辩护。
风险调整建议（suggested_adjustment）：维持原风险 / 轻微下调 / 下调一个等级 / 驳回该指控。

调参约束（必须严格遵守）：
  - 仅行业常识推测，最高只能给“弱辩护”。
  - 没有底稿证据支持，不得给“强辩护”，也不建议下调风险。
  - 多个高权重规则共同命中，不得直接降为低风险。
  - 收入、利润、现金流或资产质量之间存在协同异常时，辩护必须谨慎。
  - 标准无保留审计意见可作辅助辩护，但不能单独否定财务异常。
  - 数据缺失不能作为强辩护理由。
  - 辩护建议只供裁决参考，不直接决定最终等级。
  - supporting_evidence 必须引用底稿中的具体指标或模块；若无证据，evidence_available=false。

只输出如下 JSON 结构：
{
  "agent_role": "defense",
  "defense_results": [
    {
      "target_accusation_id": "",
      "target_risk_type": "利润舞弊/资产舞弊",
      "original_risk_point": "",
      "defense_strength": "无有效辩护/弱辩护/中等辩护/强辩护",
      "suggested_adjustment": "维持原风险/轻微下调/下调一个等级/驳回该指控",
      "defense_reasons": [
        {"reason_type": "行业合理性/业务模式合理性/经营周期合理性/数据质量与重要性/反向证据", "reason": "", "supporting_evidence": "", "evidence_available": true}
      ],
      "remaining_concerns": [],
      "defense_summary": ""
    }
  ],
  "overall_defense": {"main_defense_points": [], "unresolved_risks": [], "data_limitations": []}
}
"""

VERDICT_SYSTEM = COMMON_BACKGROUND + """
== 你的角色：裁决智能体（verdict）==
你是最终判断者，综合底稿、指控结果与辩护结果，对资产与利润舞弊风险作出最终判断。
你既不能简单全盘采纳指控，也不能简单全盘采纳辩护。

裁决依据：ML 模型得分与初筛等级、规则命中数量、高/中/低权重规则结构、
风险是否跨维度、证据链是否集中指向某类舞弊、辩护是否有底稿证据支持、
引擎是否存在停用/未实现/计算异常、底稿的 questions_for_verification。
ML 模型结果是“重要参考”，但不机械决定最终裁决。

风险等级认定（参数）：
  - 高风险：多项强证据相互印证且辩护未能有效解释；或 ML 为高风险且命中多个关键规则/跨维度信号。
  - 中高风险：存在高权重规则或多个中等风险点但仍需核查；或 ML 中高风险且证据链集中指向某类舞弊。
  - 中风险：存在单项异常或弱证据且辩护给出一定解释；或 ML 中风险、规则命中少但有关注事项。
  - 低风险：未命中主要规则、异常较弱，或辩护提供充分反向证据；或 ML 低风险且证据链无明显异常。
  - 证据不足时：放入 watchlist_items（保留关注），不直接判高风险。
  - 辩护成功但仍有疑点：写入 minority_opinions（少数意见）。

将风险点分为三类：confirmed_risks（采纳）/ rejected_risks（驳回）/ watchlist_items（保留关注）。

只输出如下 JSON 结构：
{
  "agent_role": "verdict",
  "final_risk_level": "低风险/中风险/中高风险/高风险",
  "asset_profit_fraud_risk_level": "低风险/中风险/中高风险/高风险",
  "profit_fraud_risk_level": "低风险/中风险/中高风险/高风险",
  "asset_fraud_risk_level": "低风险/中风险/中高风险/高风险",
  "main_risk_type": "利润舞弊/资产舞弊/两者均存在/未发现明显风险",
  "confirmed_risks": [{"risk_type": "利润舞弊/资产舞弊", "risk_point": "", "basis": "", "accepted_reason": ""}],
  "rejected_risks": [{"risk_type": "利润舞弊/资产舞弊", "risk_point": "", "rejected_reason": ""}],
  "watchlist_items": [{"risk_type": "利润舞弊/资产舞弊", "reason": "", "required_additional_data": ""}],
  "evidence_chain": [{"risk_type": "利润舞弊/资产舞弊", "evidence_summary": "", "related_rules": [], "related_indicators": [], "strength": "弱/中/强"}],
  "minority_opinions": [],
  "audit_suggestions": [{"risk_type": "利润舞弊/资产舞弊", "suggestion": "", "priority": "高/中/低"}],
  "verdict_summary": ""
}
"""


# ════════════════════════════════════════════════════════════════
#  五、三智能体实现
# ════════════════════════════════════════════════════════════════
class ProsecutionAgent:
    def __init__(self, llm: DeepSeekClient, cfg: AgentConfig):
        self.llm = llm
        self.cfg = cfg

    def run(self, workpaper: Dict[str, Any]) -> Dict[str, Any]:
        p = self.cfg.prosecution
        cross = ("两个及以上不同维度(R1/R2/R3)同时命中，可在证据支持下适度提升风险等级。"
                 if p["allow_cross_dim_escalation"]
                 else "不允许仅因跨维度命中就提升风险等级。")
        system = PROSECUTION_SYSTEM.format(
            max_acc=p["max_accusations_per_type"],
            cross_dim=cross,
        )
        evidence = slice_for_prosecution(workpaper)
        user = ("以下是结构化工作底稿（指控智能体视角）。请基于它发现并归类资产与利润舞弊风险，"
                "严格按系统要求的 JSON 输出。\n\n"
                + json.dumps(evidence, ensure_ascii=False, indent=2, default=str))
        return self.llm.chat_json(system, user)


class DefenseAgent:
    def __init__(self, llm: DeepSeekClient, cfg: AgentConfig):
        self.llm = llm
        self.cfg = cfg

    def run(self, workpaper: Dict[str, Any], prosecution_result: Dict[str, Any]) -> Dict[str, Any]:
        evidence = slice_for_defense(workpaper)
        user = (
            "以下是结构化工作底稿（辩护智能体视角）与指控智能体结果。\n"
            "请逐条对应指控中的 main_accusations（按 accusation_id），判断每个风险点是否有合理解释，"
            "区分可解释与不可解释异常，并给出辩护强度与风险调整建议。严格按系统要求的 JSON 输出。\n\n"
            "【工作底稿（辩护视角）】\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
            + "\n\n【指控智能体结果】\n"
            + json.dumps(prosecution_result, ensure_ascii=False, indent=2, default=str)
        )
        return self.llm.chat_json(DEFENSE_SYSTEM, user)


class VerdictAgent:
    def __init__(self, llm: DeepSeekClient, cfg: AgentConfig):
        self.llm = llm
        self.cfg = cfg

    def run(self, workpaper: Dict[str, Any],
            prosecution_result: Dict[str, Any],
            defense_result: Dict[str, Any]) -> Dict[str, Any]:
        evidence = slice_for_verdict(workpaper)
        user = (
            "以下是结构化工作底稿（裁决智能体视角）、指控结果与辩护结果。\n"
            "请判断指控是否成立、辩护是否充分，综合形成最终风险等级、证据链、少数意见与核查建议。"
            "严格按系统要求的 JSON 输出。\n\n"
            "【工作底稿（裁决视角）】\n"
            + json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
            + "\n\n【指控智能体结果】\n"
            + json.dumps(prosecution_result, ensure_ascii=False, indent=2, default=str)
            + "\n\n【辩护智能体结果】\n"
            + json.dumps(defense_result, ensure_ascii=False, indent=2, default=str)
        )
        return self.llm.chat_json(VERDICT_SYSTEM, user)


# ════════════════════════════════════════════════════════════════
#  六、协同编排器：负责整体流程控制，依次调用三智能体，并整合输出结果
# ════════════════════════════════════════════════════════════════
class ThreeAgentOrchestrator:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE,
                 config: Optional[AgentConfig] = None):
        self.cfg = config or AgentConfig()
        self.llm = DeepSeekClient(api_key=api_key, model=model, temperature=temperature)
        self.prosecution = ProsecutionAgent(self.llm, self.cfg)
        self.defense = DefenseAgent(self.llm, self.cfg)
        self.verdict = VerdictAgent(self.llm, self.cfg)

    def analyze(self, workpaper: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        if not workpaper.get("success", True):
            return {"success": False, "error": workpaper.get("error", "底稿无效")}

        info = workpaper.get("basic_info", {}) or {}
        tag = f"{info.get('company_name', '')} {info.get('stock_code', '')} {info.get('year', '')}"

        # 第二步：指控
        if verbose:
            print(f"\n[1/3] 指控智能体分析中…  {tag}")
        pros = self.prosecution.run(workpaper)

        # 第三步：辩护
        if verbose:
            print("[2/3] 辩护智能体分析中…")
        defe = self.defense.run(workpaper, pros)

        # 第四步：裁决
        if verbose:
            print("[3/3] 裁决智能体分析中…")
        verd = self.verdict.run(workpaper, pros, defe)

        return {
            "success": True,
            "basic_info": info,
            "ml_reference": {
                "ml_score": (workpaper.get("risk_summary", {}) or {}).get("ml_score"),
                "overall_risk_level_ml": (workpaper.get("risk_summary", {}) or {}).get("overall_risk_level_ml"),
            },
            "prosecution": pros,
            "defense": defe,
            "verdict": verd,
        }


# ════════════════════════════════════════════════════════════════
#  七、底稿加载（文件 or 实时调用底稿生成器）
# ════════════════════════════════════════════════════════════════
def load_workpaper_from_file(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_workpaper(stock_code: str, year: int) -> Dict[str, Any]:
    """调用同目录下的 workpaper_generator 生成底稿。"""
    try:
        import workpaper_generator as wg
    except ImportError:
        sys.exit("[ERROR] 未找到 workpaper_generator.py，请将本脚本与底稿生成器、数据、模型放在同目录，"
                 "或改用 --workpaper 指定已生成的底稿 JSON。")
    gen = wg.WorkpaperGenerator()
    return gen.generate(stock_code, year)


# ════════════════════════════════════════════════════════════════
#  八、命令行入口
# ════════════════════════════════════════════════════════════════
def _sep(char: str = "─", width: int = 64) -> str:
    return char * width


def _print_accusations(accusations: List[Dict[str, Any]], label: str) -> None:
    """打印一类（利润/资产）指控点列表。"""
    if not accusations:
        print(f"    {label}: 未发现主要指控点")
        return
    for acc in accusations:
        aid   = acc.get("accusation_id", "?")
        rp    = acc.get("risk_point", "")
        rules = ", ".join(acc.get("related_rules", [])) or "—"
        indic = ", ".join(acc.get("related_indicators", [])) or "—"
        evid  = acc.get("evidence", "")
        mean  = acc.get("fraud_meaning", "")
        stre  = acc.get("strength", "")
        print(f"    [{aid}] {rp}  (强度: {stre})")
        print(f"          相关规则: {rules}")
        print(f"          相关指标: {indic}")
        if evid:
            print(f"          证据: {evid}")
        if mean:
            print(f"          舞弊含义: {mean}")


def _print_defense_results(defense_results: List[Dict[str, Any]]) -> None:
    """打印辩护逐条结果。"""
    if not defense_results:
        print("    （无辩护条目）")
        return
    for dr in defense_results:
        tid    = dr.get("target_accusation_id", "?")
        rtype  = dr.get("target_risk_type", "")
        rp     = dr.get("original_risk_point", "")
        dstr   = dr.get("defense_strength", "")
        adj    = dr.get("suggested_adjustment", "")
        summ   = dr.get("defense_summary", "")
        remain = dr.get("remaining_concerns", [])
        print(f"    [{tid}|{rtype}] {rp}")
        print(f"          辩护强度: {dstr}  →  建议: {adj}")
        # 辩护理由
        for r in dr.get("defense_reasons", []):
            rtype_r = r.get("reason_type", "")
            reason  = r.get("reason", "")
            evail   = "✓有证据" if r.get("evidence_available") else "✗无底稿证据"
            supev   = r.get("supporting_evidence", "")
            print(f"          [{rtype_r} {evail}] {reason}")
            if supev:
                print(f"            └ 证据: {supev}")
        if remain:
            print(f"          仍存疑点: {'; '.join(str(x) for x in remain)}")
        if summ:
            print(f"          辩护总结: {summ}")


def print_summary(result: Dict[str, Any]) -> None:
    """展示三智能体各自的完整结论，最后汇总裁决。"""
    if not result.get("success"):
        print(f"\n[失败] {result.get('error')}")
        return

    info = result.get("basic_info", {}) or {}
    ml   = result.get("ml_reference", {}) or {}
    pros = result.get("prosecution", {}) or {}
    defe = result.get("defense", {}) or {}
    verd = result.get("verdict", {}) or {}

    header = (f"{info.get('company_name','')}  "
              f"{info.get('stock_code','')}  {info.get('year','')}")

    # ── 标题栏 ────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(f"  三智能体协同分析报告  |  {header}")
    print(f"  ML参考: 得分 {ml.get('ml_score','N/A')} / 等级 {ml.get('overall_risk_level_ml','N/A')}")
    print("═" * 64)

    # ════════════════════════════════════════════════════════════
    #  ① 指控智能体结论
    # ════════════════════════════════════════════════════════════
    print("\n【一】指控智能体结论")
    print(_sep())

    # 利润舞弊
    pf = pros.get("profit_fraud", {}) or {}
    print(f"  ▶ 利润舞弊风险  等级: {pf.get('risk_level','—')}  置信度: {pf.get('confidence','—')}")
    _print_accusations(pf.get("main_accusations", []), "主要指控")

    # 资产舞弊
    af = pros.get("asset_fraud", {}) or {}
    print(f"\n  ▶ 资产舞弊风险  等级: {af.get('risk_level','—')}  置信度: {af.get('confidence','—')}")
    _print_accusations(af.get("main_accusations", []), "主要指控")

    # 关注事项
    attention = pros.get("attention_items", []) or []
    if attention:
        print(f"\n  ▶ 关注事项 ({len(attention)} 条)")
        for i, item in enumerate(attention, 1):
            print(f"    {i}. {item}")

    # 数据局限
    pros_limits = pros.get("data_limitations", []) or []
    if pros_limits:
        print(f"\n  ▶ 数据局限: {'; '.join(str(x) for x in pros_limits)}")

    # ════════════════════════════════════════════════════════════
    #  ② 辩护智能体结论
    # ════════════════════════════════════════════════════════════
    print("\n\n【二】辩护智能体结论")
    print(_sep())

    _print_defense_results(defe.get("defense_results", []))

    # 整体辩护摘要
    od = defe.get("overall_defense", {}) or {}
    main_pts = od.get("main_defense_points", []) or []
    unresolved = od.get("unresolved_risks", []) or []
    defe_limits = od.get("data_limitations", []) or []

    if main_pts:
        print(f"\n  ▶ 主要辩护论点:")
        for pt in main_pts:
            print(f"    • {pt}")
    if unresolved:
        print(f"\n  ▶ 未能化解的风险:")
        for ur in unresolved:
            print(f"    ✗ {ur}")
    if defe_limits:
        print(f"\n  ▶ 数据局限: {'; '.join(str(x) for x in defe_limits)}")

    # ════════════════════════════════════════════════════════════
    #  ③ 裁决智能体结论
    # ════════════════════════════════════════════════════════════
    print("\n\n【三】裁决智能体结论")
    print(_sep())

    print(f"  ★ 最终总体风险等级   : {verd.get('final_risk_level','—')}")
    print(f"  ★ 利润舞弊风险等级   : {verd.get('profit_fraud_risk_level','—')}")
    print(f"  ★ 资产舞弊风险等级   : {verd.get('asset_fraud_risk_level','—')}")
    print(f"  ★ 主要风险类型       : {verd.get('main_risk_type','—')}")

    # 采纳风险
    confirmed = verd.get("confirmed_risks", []) or []
    print(f"\n  ▶ 采纳风险点 ({len(confirmed)} 个)")
    for cr in confirmed:
        print(f"    [采纳|{cr.get('risk_type','')}] {cr.get('risk_point','')}")
        print(f"          依据: {cr.get('basis','')}")
        print(f"          采纳原因: {cr.get('accepted_reason','')}")

    # 驳回风险
    rejected = verd.get("rejected_risks", []) or []
    print(f"\n  ▶ 驳回风险点 ({len(rejected)} 个)")
    for rr in rejected:
        print(f"    [驳回|{rr.get('risk_type','')}] {rr.get('risk_point','')}")
        print(f"          驳回原因: {rr.get('rejected_reason','')}")

    # 保留关注
    watchlist = verd.get("watchlist_items", []) or []
    print(f"\n  ▶ 保留关注 ({len(watchlist)} 项)")
    for wi in watchlist:
        print(f"    [关注|{wi.get('risk_type','')}] {wi.get('reason','')}")
        if wi.get("required_additional_data"):
            print(f"          需补充: {wi.get('required_additional_data')}")

    # 核心证据链
    ev_chain = verd.get("evidence_chain", []) or []
    if ev_chain:
        print(f"\n  ▶ 核心证据链")
        for ev in ev_chain:
            rules = ", ".join(ev.get("related_rules", [])) or "—"
            indic = ", ".join(ev.get("related_indicators", [])) or "—"
            print(f"    [{ev.get('risk_type','')} | 强度:{ev.get('strength','')}]")
            print(f"      {ev.get('evidence_summary','')}")
            print(f"      规则: {rules}  指标: {indic}")

    # 少数意见
    minority = verd.get("minority_opinions", []) or []
    if minority:
        print(f"\n  ▶ 少数意见")
        for mo in minority:
            print(f"    △ {mo}")

    # 核查建议
    suggestions = verd.get("audit_suggestions", []) or []
    if suggestions:
        print(f"\n  ▶ 核查建议 ({len(suggestions)} 条)")
        for sg in suggestions:
            pri = sg.get("priority", "")
            pri_icon = {"高": "🔴", "中": "🟡", "低": "🟢"}.get(pri, "•")
            print(f"    {pri_icon}[{pri}|{sg.get('risk_type','')}] {sg.get('suggestion','')}")

    # 裁决总结
    print(f"\n  ▶ 裁决总结:")
    print(f"    {verd.get('verdict_summary','')}")

    # ── 尾部汇总一览 ─────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(f"  汇总一览  |  {header}")
    print(_sep())
    pf_lvl = pf.get("risk_level", "—")
    af_lvl = af.get("risk_level", "—")
    print(f"  指控  → 利润舞弊: {pf_lvl:<8}  资产舞弊: {af_lvl}")

    # 辩护整体强度统计
    strengths = [dr.get("defense_strength", "") for dr in (defe.get("defense_results") or [])]
    strong_cnt = strengths.count("强辩护")
    mid_cnt    = strengths.count("中等辩护")
    weak_cnt   = strengths.count("弱辩护")
    no_cnt     = strengths.count("无有效辩护")
    print(f"  辩护  → 强:{strong_cnt} 中:{mid_cnt} 弱:{weak_cnt} 无:{no_cnt}  "
          f"未化解风险: {len(unresolved)} 项")

    print(f"  裁决  → 总体: {verd.get('final_risk_level','—'):<8}  "
          f"采纳:{len(confirmed)} 驳回:{len(rejected)} 保留:{len(watchlist)}")
    print("═" * 64)


def main():
    parser = argparse.ArgumentParser(description="基于结构化工作底稿的三智能体协同分析")
    parser.add_argument("--workpaper", type=str, help="已生成的工作底稿 JSON 文件路径")
    parser.add_argument("--stock_code", type=str, help="股票代码（与 --year 一起实时生成底稿）")
    parser.add_argument("--year", type=int, help="年份")
    parser.add_argument("--out", type=str, default=None, help="结果输出 JSON 路径")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API Key（或用环境变量 DEEPSEEK_API_KEY）")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="模型名（deepseek-chat / deepseek-reasoner）")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("[ERROR] 未提供 API Key。请用 --api_key 或设置环境变量 DEEPSEEK_API_KEY。")

    # 加载底稿
    if args.workpaper:
        workpaper = load_workpaper_from_file(args.workpaper)
    elif args.stock_code and args.year:
        workpaper = generate_workpaper(args.stock_code, args.year)
    else:
        sys.exit("[ERROR] 请提供 --workpaper 文件，或同时提供 --stock_code 与 --year。")

    orchestrator = ThreeAgentOrchestrator(
        api_key=api_key, model=args.model, temperature=args.temperature,
    )
    result = orchestrator.analyze(workpaper)

    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n[完成] 结果已保存至: {args.out}")

    print_summary(result)


if __name__ == "__main__":
    main()
