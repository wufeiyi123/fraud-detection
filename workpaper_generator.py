#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
工作底稿生成器 — 财务舞弊风险识别系统核心模块
=============================================================================
功能说明：
    基于最终版规则库（无金融业 + 虚构利润/虚列资产 + RF_RUS_1to3），
    输入股票代码和年份，自动生成结构化工作底稿（JSON格式），供后续三智能体分析使用。

技术路线：
    1. 完全复用 rule_base_threshold_optimizer.py 的 load_and_prepare()
       进行数据预处理（单位转换、增长率计算、行业对比等），确保与建模阶段100%一致。
    2. 完全复用 rule_base_threshold_optimizer.py 的 build_candidates()
       进行17条规则的触发判断，使用最终版训练脚本选定的最优阈值。
    3. 加载最终版训练好的随机森林模型（RF_RUS_1to3），计算综合舞弊风险得分。
    4. 综合财务指标、非财务指标、同业对比、舞弊三角分析，生成完整工作底稿。

依赖文件：
    - rule_base_threshold_optimizer.py                      # 规则引擎核心
    - 汇总数据.csv                                           # 本地数据库
    - final_rf_rus_1to3_model_2007_2023.pkl                 # 训练好的RF模型
    - final_feature_columns.json                            # 模型特征列清单
    - 05_rule_weights_final_2007_2023.csv                   # 规则权重表
    - 06_risk_score_level_ranges_final.csv                  # 风险等级阈值

输入：
    股票代码（6位字符串）+ 年份（整数）

输出：
    结构化工作底稿 JSON，包含12个模块：
    - basic_info             公司基本信息
    - risk_summary           风险汇总（ML评分、规则命中统计）
    - core_financial_metrics 核心财务指标
    - peer_comparison        同业对比
    - non_financial_indicators 非财务指标
    - hit_rules_detail       17条规则逐条诊断明细
    - evidence_chain         证据链
    - fraud_triangle_analysis 舞弊三角分析
    - key_risk_signals       关键风险信号
    - questions_for_verification 核查问题清单
    - engine_status          规则引擎运行状态
    - comprehensive_assessment 综合评估结论

使用方法：
    # 交互模式
    python workpaper_generator.py

    # 命令行模式
    python workpaper_generator.py --stock_code 000651 --year 2023

    # 作为模块导入
    from workpaper_generator import WorkpaperGenerator
    gen = WorkpaperGenerator()
    result = gen.generate("000651", 2023)

作者：智能财务机器人课程项目组
版本：v3.0 — 最终版（完整12模块 + 动态加载配置）
=============================================================================
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── 引入规则引擎核心模块 ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_base_threshold_optimizer as dg


# ═══════════════════════════════════════════════════════════════
# 一、全局配置（文件路径）
# ═══════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "汇总数据.csv"
FINAL_OUT_DIR = ROOT
MODEL_PATH = FINAL_OUT_DIR / "final_rf_rus_1to3_model_2007_2023.pkl"
FEATURE_COLS_PATH = FINAL_OUT_DIR / "final_feature_columns.json"
RULE_WEIGHTS_PATH = FINAL_OUT_DIR / "05_rule_weights_final_2007_2023.csv"
RISK_RANGES_PATH = FINAL_OUT_DIR / "06_risk_score_level_ranges_final.csv"


# ═══════════════════════════════════════════════════════════════
# 二、辅助工具函数
# ═══════════════════════════════════════════════════════════════

def safe_div(a, b):
    if b is None or b == 0 or (isinstance(b, float) and pd.isna(b)):
        return np.nan
    if a is None or (isinstance(a, float) and pd.isna(a)):
        return np.nan
    return a / b


def nan_to_none(v):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def fmt_billion(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(v / 1e8, 2)


def get_rule_category(rule_code):
    prefix = rule_code.split("-")[0]
    categories = {"R1": "收入舞弊", "R2": "存货异常", "R3": "资金异常"}
    return categories.get(prefix, "其他")


# ═══════════════════════════════════════════════════════════════
# 三、工作底稿生成器主类
# ═══════════════════════════════════════════════════════════════

class WorkpaperGenerator:
    """工作底稿生成器"""

    def __init__(self):
        self.df_raw = None
        self.df_prepared = None
        self.model = None
        self.feature_cols = None
        self.rule_config = None
        self.risk_thresholds = None
        self._loaded = False

    # ── 配置加载 ──

    def _load_rule_weights(self) -> dict:
        if not RULE_WEIGHTS_PATH.exists():
            print(f"[WARN] 规则权重表不存在: {RULE_WEIGHTS_PATH}，将使用空配置")
            return {}
        df = pd.read_csv(RULE_WEIGHTS_PATH)
        config = {}
        for _, row in df.iterrows():
            code = row["rule_code"]
            config[code] = {
                "level": row.get("level", "低"),
                "score": int(row.get("score_point", 1)),
                "threshold_desc": row.get("selected_threshold", ""),
                "calibrated_weight": row.get("calibrated_weight", 0.5),
            }
        print(f"[INFO] 规则权重表加载成功，共 {len(config)} 条规则")
        return config

    def _load_risk_thresholds(self) -> dict:
        if not RISK_RANGES_PATH.exists():
            print(f"[WARN] 风险等级表不存在，将使用默认阈值")
            return {"low_medium": 14.0, "alert": 42.5, "high": 53.7}
        df = pd.read_csv(RISK_RANGES_PATH)
        thresholds = {"low_medium": 14.0, "alert": 42.5, "high": 53.7}
        for _, row in df.iterrows():
            range_str = row.get("风险得分范围", "")
            level = row.get("风险等级", "")
            if "低风险" in level:
                match = re.search(r'<\s*(\d+\.?\d*)', range_str)
                if match:
                    thresholds["low_medium"] = float(match.group(1))
            elif "中高风险" in level:
                match = re.search(r'(\d+\.?\d*)\s*<=', range_str)
                if match:
                    thresholds["alert"] = float(match.group(1))
            elif "高风险" in level:
                match = re.search(r'>=\s*(\d+\.?\d*)', range_str)
                if match:
                    thresholds["high"] = float(match.group(1))
        print(f"[INFO] 风险阈值加载成功: low_medium={thresholds['low_medium']}, alert={thresholds['alert']}, high={thresholds['high']}")
        return thresholds

    # ── 数据与模型加载 ──

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")
        self.df_raw = pd.read_csv(DATA_PATH)
        self.df_raw["stock_code"] = self.df_raw["stock_code"].astype(str).str.zfill(6)
        self.df_raw["year_num"] = pd.to_datetime(self.df_raw["year"], errors="coerce").dt.year
        print(f"[INFO] 数据加载成功，共 {len(self.df_raw)} 条记录")

        self.df_prepared = dg.load_and_prepare()
        self.df_prepared["stock_code"] = self.df_prepared["stock_code"].astype(str).str.zfill(6)
        print(f"[INFO] 派生字段计算完成，共 {len(self.df_prepared)} 条")

        if MODEL_PATH.exists():
            import joblib
            self.model = joblib.load(MODEL_PATH)
            print(f"[INFO] 模型加载成功: {MODEL_PATH}")
        else:
            print(f"[WARN] 模型文件不存在: {MODEL_PATH}")
            self.model = None

        if FEATURE_COLS_PATH.exists():
            self.feature_cols = json.loads(FEATURE_COLS_PATH.read_text(encoding="utf-8"))
        else:
            self.feature_cols = None

        self.rule_config = self._load_rule_weights()
        self.risk_thresholds = self._load_risk_thresholds()
        self._loaded = True

    # ── 规则触发判断 ──

    def _compute_rules(self, prepared: dict) -> tuple:
        all_candidates = dg.build_candidates()
        lookup = {(c.rule_code, c.description): c for c in all_candidates}
        DISABLED = {"R3-03", "R3-05"}
        df_single = pd.DataFrame([prepared])
        results = []
        hits = {}
        for code in dg.RULE_CODES:
            cfg = self.rule_config.get(code, {})
            level = cfg.get("level", "低")
            score = cfg.get("score", 1)
            threshold_desc = cfg.get("threshold_desc", "")
            if code in DISABLED:
                hit = False
                evidence = f"规则{code}已停用"
                status = "停用"
            else:
                cand = lookup.get((code, threshold_desc))
                if cand is None and threshold_desc:
                    for c in all_candidates:
                        if c.rule_code == code and threshold_desc in c.description:
                            cand = c
                            break
                if cand is None:
                    for c in all_candidates:
                        if c.rule_code == code:
                            cand = c
                            break
                if cand is None:
                    hit = False
                    evidence = f"未找到规则{code}的计算逻辑"
                    status = "未实现"
                else:
                    try:
                        hit_series = cand.condition(df_single)
                        hit = bool(hit_series.iloc[0]) if len(hit_series) > 0 else False
                        evidence = self._build_evidence(code, prepared, hit)
                        status = "正常计算"
                    except Exception as e:
                        hit = False
                        evidence = f"计算异常: {str(e)}"
                        status = "计算异常"
            hits[code] = hit
            results.append({
                "rule_code": code,
                "rule_name": dg.RULE_NAMES.get(code, code),
                "category": get_rule_category(code),
                "weight": level,
                "score": score if hit else 0,
                "hit": hit,
                "evidence": evidence,
                "threshold_desc": threshold_desc,
                "status": status,
                "interpretation": self._get_interpretation(code, hit),
            })
        return results, hits

    def _build_evidence(self, code: str, d: dict, hit: bool) -> str:
        try:
            if code == "R1-01":
                rec = d.get("receivable_growth_model")
                rev = d.get("revenue_growth_model")
                if rec is not None and rev is not None:
                    gap = rec - rev
                    return f"应收账款增速{fmt_pct(rec)}，营业收入增速{fmt_pct(rev)}，差值{fmt_pct(gap)}，阈值>20个百分点。"
                return "数据不足"
            elif code == "R1-02":
                cr = d.get("cash_collection_ratio")
                rg = d.get("revenue_growth_model")
                return f"销售回款率{fmt_pct(cr)}，营收增速{fmt_pct(rg)}，阈值<80%且营收增长>0%。"
            elif code == "R1-03":
                m = d.get("gross_profit_margin")
                im = d.get("industry_gross_profit_margin")
                if m is not None and im is not None:
                    return f"毛利率{fmt_pct(m)}，行业均值{fmt_pct(im)}，高出{fmt_pct(m - im)}，阈值>15个百分点。"
                return f"毛利率{fmt_pct(m)}，行业均值{fmt_pct(im)}"
            elif code == "R1-04":
                npg = d.get("net_profit_growth_model")
                ocf = d.get("ocf_to_profit")
                return f"净利润增速{fmt_pct(npg)}，OCF/利润={ocf:.4f}" if ocf is not None else f"净利润增速{fmt_pct(npg)}"
            elif code == "R1-05":
                rg = d.get("revenue_growth_model")
                ind = d.get("industry_revenue_growth_median")
                return f"营收增速{fmt_pct(rg)}，行业中位数{fmt_pct(ind)}"
            elif code == "R1-06":
                rg = d.get("revenue_growth_model")
                drop = d.get("selling_expense_ratio_drop")
                return f"营收增速{fmt_pct(rg)}，销售费用率下降{fmt_pct(drop)}"
            elif code == "R2-01":
                ig = d.get("inventory_growth_model")
                cg = d.get("cost_growth_model")
                if ig is not None and cg is not None:
                    return f"存货增速{fmt_pct(ig)}，成本增速{fmt_pct(cg)}，差值{fmt_pct(ig - cg)}，阈值>25个百分点。"
                return f"存货增速{fmt_pct(ig)}，成本增速{fmt_pct(cg)}"
            elif code == "R2-02":
                t = d.get("inventory_turnover_ratio")
                it = d.get("industry_inventory_turnover_median")
                if t is not None and it is not None:
                    th = it * 0.40
                    return f"公司存货周转率{t:.4f}次，行业均值{it:.4f}次，阈值<{th:.4f}次。"
                return f"公司存货周转率{t}，行业均值{it}"
            elif code == "R2-03":
                ir = d.get("inventory_to_assets")
                t = d.get("inventory_turnover_ratio")
                return f"存货占比{fmt_pct(ir)}，周转率{t:.4f}次" if t is not None else f"存货占比{fmt_pct(ir)}"
            elif code == "R2-04":
                ir = d.get("inventory_to_assets")
                return f"存货占比{fmt_pct(ir)}，阈值>行业中位数×2.5且自身>40%。"
            elif code == "R2-05":
                cg = d.get("cips_growth")
                cl = d.get("cips_growth_lag")
                fg = d.get("fixed_assets_growth", 0)
                return f"在建工程增速{fmt_pct(cg)}（上年{fmt_pct(cl)}），固定资产增速{fmt_pct(fg)}，阈值连续两年>20%且固资增长<5%。"
            elif code == "R3-01":
                cr = d.get("cash_to_assets")
                lr = d.get("short_loan_to_liab")
                return f"货币资金/资产={fmt_pct(cr)}，短借/负债={fmt_pct(lr)}，阈值>40%且>30%。"
            elif code == "R3-02":
                pr = d.get("prepayment_to_assets")
                ip = d.get("industry_prepayment_to_assets_p75")
                return f"预付款/资产={fmt_pct(pr)}，行业P75={fmt_pct(ip)}，阈值>20%且超过行业P75。" if pr is not None else "数据不足"
            elif code == "R3-03":
                ocf = d.get("ocf_to_profit")
                return f"OCF/利润={ocf:.4f}" if ocf is not None else "数据不足"
            elif code == "R3-04":
                rp = d.get("related_party_to_assets")
                return f"关联交易金额/资产={fmt_pct(rp)}，阈值>30%。"
            elif code == "R3-05":
                cr = d.get("cash_to_assets")
                iir = d.get("interest_income_to_cash")
                return f"现金/资产={fmt_pct(cr)}，利息收入/现金={fmt_pct(iir)}"
            elif code == "R3-06":
                fr = d.get("financing_to_assets")
                rg = d.get("revenue_growth_model")
                ocf = d.get("operating_cash_flow_net")
                ocf_str = f"{ocf/1e8:.2f}亿" if ocf is not None else "N/A"
                return f"筹资现金流/资产={fmt_pct(fr)}，营收增速{fmt_pct(rg)}，经营现金流={ocf_str}"
            else:
                return f"规则{code}触发" if hit else f"规则{code}未触发"
        except Exception as e:
            return f"证据生成异常: {str(e)}"

    def _get_interpretation(self, code: str, hit: bool) -> str:
        if not hit:
            return ""
        interpretations = {
            "R1-01": "应收账款增速远超营收增速，可能意味着公司通过放宽信用政策、提前确认收入甚至虚构收入来粉饰报表。",
            "R1-02": "销售回款率偏低表明收入质量较差，现金回收不足，可能存在虚增收入或客户付款能力恶化的风险。",
            "R1-03": "毛利率显著高于行业均值，可能是产品竞争力强的体现，但也可能是虚增收入、少计成本的结果。",
            "R1-04": "净利润大幅增长但经营现金流并未同步改善，盈利质量较差，存在利用应计项目操纵利润的可能。",
            "R1-05": "营收增速远超行业水平，需要关注增长的可持续性和真实性。",
            "R1-06": "营收增长但销售费用率下降，可能表明公司削减了必要的市场投入，或收入确认存在异常。",
            "R2-01": "存货增速显著高于营业成本增速，可能意味着产品滞销、存货积压或存货计价存在异常。",
            "R2-02": "存货周转率远低于行业水平，说明存货流动性差，存在跌价风险或存货真实性存疑。",
            "R2-03": "存货占比畸高且周转缓慢，是存货积压的典型特征，需关注存货质量和可变现净值。",
            "R2-04": "存货占总资产比例异常偏高，资产结构失衡，可能存在虚增存货或滞销积压问题。",
            "R2-05": "在建工程连续大幅增长但迟迟不转入固定资产，可能存在延迟转固以少提折旧、虚增利润的动机。",
            "R3-01": "存贷双高现象不符合商业逻辑，高额现金与高额借款并存，资金使用效率存疑，可能存在资金占用或虚构货币资金。",
            "R3-02": "预付款项异常高企或大幅增长，可能存在资金被关联方占用、虚假采购或利益输送的风险。",
            "R3-03": "净利润为正但经营现金流远低于利润，盈利的现金保障程度极低，需关注利润质量。",
            "R3-04": "关联交易金额占比过高，可能通过关联方进行利益输送或虚构交易。",
            "R3-05": "账面资金充裕但利息收入极低，货币资金的真实性存疑，可能存在虚构或限制用途的银行存款。",
            "R3-06": "持续筹资但业务未见扩张，可能依赖外部融资维持运营，资金链紧张风险较高。",
        }
        return interpretations.get(code, "")

    # ── ML模型评分 ──

    def _ml_score(self, row: dict) -> tuple:
        if self.model is None or self.feature_cols is None:
            return None, "N/A"
        try:
            X = pd.DataFrame([row])
            for c in self.feature_cols:
                if c not in X.columns:
                    X[c] = 0
            prob = self.model.predict_proba(X[self.feature_cols])[:, 1][0]
            score = round(prob * 100, 2)
            th = self.risk_thresholds
            if score >= th["high"]:
                level = "高风险"
            elif score >= th["alert"]:
                level = "中高风险"
            elif score >= th["low_medium"]:
                level = "中风险"
            else:
                level = "低风险"
            return score, level
        except Exception as e:
            print(f"[ERROR] 模型推理失败: {e}")
            return None, "N/A"

    # ── 主入口 ──

    def generate(self, stock_code: str, year: int) -> dict:
        self._ensure_loaded()
        stock_code = str(stock_code).zfill(6)
        year = int(year)
        print(f"\n[INFO] 正在生成工作底稿: {stock_code} {year}")
        raw_mask = (self.df_raw["stock_code"] == stock_code) & (self.df_raw["year_num"] == year)
        if raw_mask.sum() == 0:
            return {"success": False, "error": f"原始数据中未找到 {stock_code} {year}"}
        row = self.df_raw[raw_mask].iloc[0].to_dict()
        prep_mask = (self.df_prepared["stock_code"] == stock_code) & (self.df_prepared["year_num"] == year)
        if prep_mask.sum() == 0:
            return {"success": False, "error": f"预处理数据中未找到 {stock_code} {year}"}
        prepared = self.df_prepared[prep_mask].iloc[0].to_dict()
        rule_results, hits_dict = self._compute_rules(prepared)
        ml_score, ml_level = self._ml_score(row)
        workpaper = self._build_workpaper(row, prepared, rule_results, hits_dict, ml_score, ml_level)
        return workpaper

    # ═══════════════════════════════════════════════════════════
    # 工作底稿各模块构建
    # ═══════════════════════════════════════════════════════════

    def _build_workpaper(self, row, prepared, rule_results, hits_dict, ml_score, ml_level):
        stock_code = str(row.get("stock_code", "")).zfill(6)
        year = row.get("year_num")
        company_name = row.get("company_name", "")
        industry_code = row.get("industry_code", "")
        industry_name = row.get("industry_name", "")

        basic_info = {
            "stock_code": stock_code,
            "year": int(year) if pd.notna(year) else None,
            "company_name": str(company_name),
            "industry_code": str(industry_code),
            "industry_name": str(industry_name),
            "report_date": str(row.get("year", "")),
        }

        hit_rules = [r for r in rule_results if r["hit"]]
        total_score = sum(r["score"] for r in rule_results)
        max_score = sum(self.rule_config.get(r["rule_code"], {}).get("score", 1) for r in rule_results) or 26
        weight_count = {"高": 0, "中": 0, "低": 0}
        for r in hit_rules:
            w = r["weight"]
            if w in weight_count:
                weight_count[w] += 1

        by_category = {}
        for cat_name in ["收入舞弊", "存货异常", "资金异常"]:
            cat_hit = [r for r in hit_rules if r["category"] == cat_name]
            cat_score = sum(r["score"] for r in cat_hit)
            by_category[cat_name] = {
                "hit_count": len(cat_hit),
                "total_score": cat_score,
                "rules": [{"code": r["rule_code"], "name": r["rule_name"], "weight": r["weight"], "score": r["score"]} for r in cat_hit],
                "risk_note": f"{cat_name}存在{len(cat_hit)}个风险信号" if cat_hit else f"{cat_name}未发现明显异常",
            }

        if len(hit_rules) == 0:
            trend_note = "未命中任何风险规则，财务表现相对健康。"
        elif len(hit_rules) <= 2:
            trend_note = "命中少量风险规则，建议关注相关科目变化趋势。"
        elif len(hit_rules) <= 5:
            trend_note = f"命中{len(hit_rules)}条风险规则，存在较为明显的风险信号，建议深入核查。"
        else:
            trend_note = f"命中{len(hit_rules)}条风险规则，风险信号密集，需重点关注。"

        risk_summary = {
            "overall_risk_level_ml": ml_level if ml_level else "N/A",
            "ml_score": ml_score,
            "total_hit_score": total_score,
            "max_possible_score": max_score,
            "score_ratio": round(total_score / max_score * 100, 1) if max_score > 0 else 0,
            "hit_count": len(hit_rules),
            "total_rules": 17,
            "hit_rate": round(len(hit_rules) / 17 * 100, 1),
            "by_weight": weight_count,
            "by_category": by_category,
            "risk_trend_note": trend_note,
            "risk_level_thresholds": self.risk_thresholds,
        }

        core_metrics = {
            "scale_billion": {
                "total_assets": fmt_billion(row.get("total_assets")),
                "revenue": fmt_billion(row.get("operating_revenue")),
                "net_profit": fmt_billion(row.get("net_profit")),
                "operating_cash_flow": fmt_billion(row.get("operating_cash_flow_net")),
            },
            "growth_rates_pct": {
                "revenue": nan_to_none(prepared.get("revenue_growth_model")),
                "receivable": nan_to_none(prepared.get("receivable_growth_model")),
                "inventory": nan_to_none(prepared.get("inventory_growth_model")),
                "cost": nan_to_none(prepared.get("cost_growth_model")),
                "net_profit": nan_to_none(prepared.get("net_profit_growth_model")),
                "cash_received": nan_to_none(prepared.get("cash_received_growth_model")),
                "prepayments": nan_to_none(prepared.get("prepayment_growth_model")),
                "cips": nan_to_none(prepared.get("cips_growth")),
                "fixed_assets": nan_to_none(prepared.get("fixed_assets_growth")),
            },
            "profitability": {
                "gross_margin_pct": nan_to_none(prepared.get("gross_profit_margin")),
                "selling_expense_ratio_pct": nan_to_none(prepared.get("selling_expense_ratio")),
                "ocf_to_profit": nan_to_none(prepared.get("ocf_to_profit")),
                "selling_expense_drop_pct": nan_to_none(prepared.get("selling_expense_ratio_drop")),
            },
            "asset_structure_pct": {
                "cash_to_assets": nan_to_none(prepared.get("cash_to_assets")),
                "inventory_to_assets": nan_to_none(prepared.get("inventory_to_assets")),
                "prepayments_to_assets": nan_to_none(prepared.get("prepayment_to_assets")),
                "short_loan_to_liab": nan_to_none(prepared.get("short_loan_to_liab")),
                "financing_to_assets": nan_to_none(prepared.get("financing_to_assets")),
            },
            "operational": {
                "inventory_turnover": nan_to_none(row.get("inventory_turnover_ratio")),
                "receivables_turnover": nan_to_none(row.get("receivables_turnover_ratio")),
            },
            "cash_management": {
                "interest_income_rate_pct": nan_to_none(prepared.get("interest_income_to_cash")),
            },
            "related_party": {
                "amount": nan_to_none(row.get("related_party_transaction_amount")),
                "to_assets_pct": nan_to_none(prepared.get("related_party_to_assets")),
            },
        }

        margin = prepared.get("gross_profit_margin")
        ind_margin = prepared.get("industry_gross_profit_margin")
        turnover = row.get("inventory_turnover_ratio")
        ind_turnover = prepared.get("industry_inventory_turnover_median")
        rev_g = prepared.get("revenue_growth_model")
        ind_rev_g = prepared.get("industry_revenue_growth_median")

        peer_comparison = {
            "gross_margin": {
                "company_pct": nan_to_none(margin),
                "industry_avg_pct": nan_to_none(ind_margin),
                "deviation_pct": nan_to_none(margin - ind_margin) if margin is not None and ind_margin is not None else None,
                "assessment": "毛利率处于行业合理范围" if (margin is not None and ind_margin is not None and abs(margin - ind_margin) <= 0.15) else (
                    "毛利率显著高于行业" if (margin is not None and ind_margin is not None and margin - ind_margin > 0.15) else "毛利率低于行业"),
            },
            "inventory_turnover": {
                "company": nan_to_none(turnover),
                "industry_avg": nan_to_none(ind_turnover),
                "deviation_pct": nan_to_none((turnover - ind_turnover) / ind_turnover * 100) if turnover is not None and ind_turnover is not None and ind_turnover != 0 else None,
                "assessment": "周转率低于行业，存货积压风险高" if (turnover is not None and ind_turnover is not None and turnover < ind_turnover * 0.55) else "周转率处于行业合理范围",
            },
            "revenue_growth": {"company_pct": nan_to_none(rev_g), "industry_avg_pct": nan_to_none(ind_rev_g)},
            "inventory_ratio": {"company_pct": nan_to_none(prepared.get("inventory_to_assets")), "industry_avg_pct": nan_to_none(prepared.get("industry_inventory_to_assets_median"))},
            "prepayment_ratio": {"company_pct": nan_to_none(prepared.get("prepayment_to_assets")), "industry_p75_pct": nan_to_none(prepared.get("industry_prepayment_to_assets_p75"))},
        }

        non_financial = self._build_non_financial(row)
        hit_rules_detail = [r for r in rule_results if r["hit"]]

        evidence_chain = {}
        for cat_name in ["收入舞弊", "存货异常", "资金异常"]:
            cat_hit = [r for r in hit_rules_detail if r["category"] == cat_name]
            if len(cat_hit) == 0:
                assessment = "未发现明显异常"
            elif len(cat_hit) <= 1:
                assessment = "存在少量异常信号，建议关注"
            elif len(cat_hit) <= 3:
                assessment = f"存在{len(cat_hit)}个异常信号，需进一步核查"
            else:
                assessment = f"风险信号较多({len(cat_hit)}个)，建议优先核查"
            evidence_chain[cat_name] = {
                "hit_rules": [{"code": r["rule_code"], "name": r["rule_name"], "weight": r["weight"], "evidence": r["evidence"]} for r in cat_hit],
                "hit_count": len(cat_hit),
                "risk_assessment": assessment,
            }

        fraud_triangle = self._fraud_triangle_analysis(rule_results, row, prepared, non_financial)

        key_signals = []
        high_weight_hits = [r for r in hit_rules_detail if r["weight"] == "高"]
        if high_weight_hits:
            key_signals.append({"type": "高风险规则命中", "severity": "高", "detail": f"命中{len(high_weight_hits)}条高权重规则", "rules": [r["rule_code"] for r in high_weight_hits]})
        multi_cat = list(set(r["category"] for r in hit_rules_detail))
        if len(multi_cat) >= 2:
            key_signals.append({"type": "跨维度风险", "severity": "高" if len(multi_cat) >= 3 else "中", "detail": f"{len(multi_cat)}个维度存在风险信号", "categories": multi_cat})
        if ml_score and ml_score >= self.risk_thresholds["alert"]:
            key_signals.append({"type": "ML模型预警", "severity": "高" if ml_score >= self.risk_thresholds["high"] else "中", "detail": f"ML评分为{ml_score}分"})

        questions = []
        priority_map = {"高": "重要", "中": "重要", "低": "一般"}
        for r in hit_rules_detail:
            questions.append({
                "priority": priority_map.get(r["weight"], "一般"),
                "category": r["category"],
                "rule_code": r["rule_code"],
                "question": self._generate_question(r),
            })

        engine_details = {}
        for r in rule_results:
            engine_details[r["rule_code"]] = {
                "name": r.get("rule_name", ""),
                "category": r.get("category", ""),
                "hit": r.get("hit", False),
                "status": r.get("status", ""),
                "missing_fields": r.get("missing_fields", []),
            }
        engine_status = {
            "total_rules": 17,
            "normal_calculated": sum(1 for r in rule_results if r["status"] == "正常计算"),
            "data_insufficient": sum(1 for r in rule_results if r["status"] in ("停用", "计算异常", "未实现")),
            "details": engine_details,
        }

        comprehensive = self._comprehensive_assessment(rule_results, ml_score, ml_level, non_financial, row, prepared)

        workpaper = {
            "success": True,
            "basic_info": basic_info,
            "risk_summary": risk_summary,
            "core_financial_metrics": core_metrics,
            "peer_comparison": peer_comparison,
            "non_financial_indicators": non_financial,
            "hit_rules_detail": hit_rules_detail,
            "evidence_chain": evidence_chain,
            "fraud_triangle_analysis": fraud_triangle,
            "key_risk_signals": key_signals,
            "questions_for_verification": questions,
            "engine_status": engine_status,
            "comprehensive_assessment": comprehensive,
        }
        return workpaper

    def _build_non_financial(self, row):
        audit_non_std = row.get("audit_non_standard", 0)
        audit_risk = row.get("audit_opinion_risk", 0)
        pledge_risk = row.get("pledge_high_risk", 0)
        core_tech = row.get("core_tech_turnover", 0)
        customer_conc = row.get("customer_concentration")
        supplier_conc = row.get("supplier_concentration")
        supply_chain_conc = row.get("supply_chain_concentration")

        audit_opinion_map = {0: "标准无保留意见", 1: "带强调事项段无保留意见", 2: "保留意见", 3: "无法表示意见", 4: "否定意见"}
        audit_risk_int = int(audit_risk) if pd.notna(audit_risk) else 0
        audit_desc = audit_opinion_map.get(audit_risk_int, "标准无保留意见")

        risk_items = []
        if audit_non_std == 1 or audit_risk_int > 0:
            risk_items.append({"field": "审计意见", "value": audit_desc, "risk": "非标审计意见，财务报表可信度存疑"})
        if pledge_risk == 1:
            risk_items.append({"field": "控股股东质押", "value": "高质押风险", "risk": "控股股东质押比例过高，存在控制权转移风险"})
        if core_tech == 1:
            risk_items.append({"field": "关键技术人员", "value": "发生变动", "risk": "核心技术人员离职，影响技术稳定性"})
        if customer_conc is not None and pd.notna(customer_conc) and customer_conc > 50:
            risk_items.append({"field": "客户集中度", "value": f"{customer_conc:.1f}%", "risk": "客户集中度过高"})
        if supplier_conc is not None and pd.notna(supplier_conc) and supplier_conc > 50:
            risk_items.append({"field": "供应商集中度", "value": f"{supplier_conc:.1f}%", "risk": "供应商集中度过高"})

        non_financial = {
            "audit_quality": {
                "field_name": "审计意见类型",
                "audit_non_standard": "是" if audit_non_std == 1 else "否",
                "audit_opinion_risk_code": audit_risk_int,
                "audit_opinion_desc": audit_desc,
                "risk_flag": audit_non_std == 1 or audit_risk_int > 0,
                "note": "审计意见为标准无保留意见，财务报表可信度较高" if (audit_non_std == 0 and audit_risk_int == 0) else f"审计意见为{audit_desc}，需关注",
            },
            "ownership_risk": {
                "field_name": "控股股东质押风险",
                "pledge_high_risk": "是" if pledge_risk == 1 else "否",
                "risk_flag": pledge_risk == 1,
                "note": "控股股东质押比例在可控范围内" if pledge_risk == 0 else "控股股东存在高比例质押风险",
            },
            "management_stability": {
                "field_name": "关键技术人员变动",
                "value": "是" if core_tech == 1 else "否",
                "risk_flag": core_tech == 1,
                "note": "关键技术人员未发生变动，经营团队稳定" if core_tech == 0 else "关键技术人员发生变动，需关注管理层稳定性",
            },
            "supply_chain": {
                "customer_concentration": nan_to_none(customer_conc),
                "supplier_concentration": nan_to_none(supplier_conc),
                "supply_chain_concentration": nan_to_none(supply_chain_conc),
                "risk_flag": (pd.notna(customer_conc) and customer_conc > 50) or (pd.notna(supplier_conc) and supplier_conc > 50),
                "note": "供应链集中度在合理范围内" if not ((pd.notna(customer_conc) and customer_conc > 50) or (pd.notna(supplier_conc) and supplier_conc > 50)) else "供应链集中度偏高，需关注依赖风险",
            },
            "summary": {
                "risk_level": "高" if len(risk_items) >= 2 else ("中" if len(risk_items) == 1 else "低"),
                "risk_count": len(risk_items),
                "risk_items": risk_items,
                "note": "非财务指标未发现明显风险信号" if len(risk_items) == 0 else f"非财务指标发现{len(risk_items)}个风险信号",
            },
        }
        return non_financial

    def _fraud_triangle_analysis(self, rule_results, row, prepared, non_financial):
        pressure_signals = []
        rev_g = prepared.get("revenue_growth_model")
        if rev_g is not None and rev_g < 0:
            pressure_signals.append("营收出现负增长，面临业绩压力")
        roe = row.get("weighted_avg_roe")
        if roe is not None and roe < 0.05:
            pressure_signals.append("ROE偏低，盈利压力较大")
        ocf = row.get("operating_cash_flow_net")
        if ocf is not None and ocf < 0:
            pressure_signals.append("经营现金流为负，资金压力明显")

        opportunity_signals = []
        hit_categories = set(r["category"] for r in rule_results if r["hit"])
        if "收入舞弊" in hit_categories:
            opportunity_signals.append("收入确认存在操纵空间（应收异常或回款率低）")
        if "存货异常" in hit_categories:
            opportunity_signals.append("存货管理存在操纵空间（周转异常或占比过高）")
        if "资金异常" in hit_categories:
            opportunity_signals.append("资金管理存在异常信号（存贷双高或关联交易）")
        if non_financial["audit_quality"]["risk_flag"]:
            opportunity_signals.append("审计意见异常，内控可能存在缺陷")
        if non_financial["ownership_risk"]["risk_flag"]:
            opportunity_signals.append("控股股东高比例质押，治理结构存在隐患")

        rationalization_signals = []
        if non_financial["management_stability"]["risk_flag"]:
            rationalization_signals.append("管理层变动频繁，可能存在不合理决策")

        def assess(signals):
            if len(signals) >= 3:
                return "高"
            elif len(signals) >= 1:
                return "中"
            return "低"

        return {
            "pressure": {"description": "舞弊压力/动机 - 迫使管理层进行财务舞弊的内外部压力", "signals": pressure_signals, "assessment": assess(pressure_signals)},
            "opportunity": {"description": "舞弊机会 - 内部控制缺陷或治理问题为舞弊提供的可乘之机", "signals": opportunity_signals, "assessment": assess(opportunity_signals)},
            "rationalization": {"description": "舞弊借口/态度 - 管理层可能合理化舞弊行为的态度或倾向", "signals": rationalization_signals, "assessment": assess(rationalization_signals)},
        }

    def _generate_question(self, rule):
        questions_map = {
            "R1-01": "应收账款增速显著高于营收增速，是否存在提前确认收入、虚构销售或放宽信用政策？建议核查大额应收账款明细、账龄结构和期后回款情况。",
            "R1-02": "销售回款率偏低，收入质量是否存在问题？建议核查销售合同回款条款、客户信用政策及大额应收账款的回收情况。",
            "R1-03": "毛利率显著高于行业均值，是否存在虚增收入或少计成本的迹象？建议核查收入确认时点、成本归集方法和同行业可比公司数据。",
            "R1-04": "净利润增长但经营现金流走弱，盈利质量存疑。建议核查应计项目明细、经营性应收应付变动原因。",
            "R1-05": "营收增速远超行业水平，需关注增长可持续性。建议核查前五大客户销售真实性、新客户拓展情况和订单储备。",
            "R1-06": "营收增长但销售费用率下降，是否存在费用资本化或不当压缩？建议核查销售费用明细和变动原因。",
            "R2-01": "存货增速远超成本增速，是否存在产品滞销或存货积压？建议核查存货库龄结构和可变现净值。",
            "R2-02": "存货周转率远低于行业水平，存货是否存在滞销、过时或虚构？建议核查存货库龄结构和跌价准备计提政策。",
            "R2-03": "存货占比畸高且周转缓慢，建议核查存货实物盘点记录、库龄超过一年的存货明细及跌价准备充分性。",
            "R2-04": "存货占总资产比例异常偏高，资产结构是否合理？建议核查存货构成、周转情况和同行业对比。",
            "R2-05": "在建工程长期挂账不转固，是否存在延迟转固以少提折旧？建议核查工程进度报告、验收单据和转固时点。",
            "R3-01": "存贷双高现象不符合商业逻辑，建议核查货币资金真实性（银行函证）、受限资金情况及借款合理性。",
            "R3-02": "预付款项异常高企，建议核查大额预付对象及其与公司的关联关系、采购合同执行情况和期后到货记录。",
            "R3-03": "经营现金流与利润严重背离，建议核查经营性应收应付变动、存货变动对现金流的影响。",
            "R3-04": "关联交易金额占比过高，建议核查关联交易的公允性、必要性及审批程序合规性。",
            "R3-05": "货币资金充裕但利息收入极低，建议核查银行存款真实性、是否存在限制用途或虚构的货币资金。",
            "R3-06": "持续筹资但业务未见扩张，建议核查募集资金使用情况、项目进展和资金实际流向。",
        }
        return questions_map.get(rule["rule_code"], f"规则{rule['rule_code']}（{rule['rule_name']}）触发，建议人工核查相关科目。")

    def _comprehensive_assessment(self, rule_results, ml_score, ml_level, non_financial, row, prepared):
        hit_count = sum(1 for r in rule_results if r["hit"])
        high_weight_hits = [r for r in rule_results if r["hit"] and r["weight"] == "高"]

        strengths = []
        concerns = []

        if hit_count == 0:
            strengths.append("未命中任何财务风险规则，财务表现相对健康")
        elif hit_count <= 2:
            strengths.append("财务规则命中较少")

        rev_g = prepared.get("revenue_growth_model")
        if row.get("operating_revenue") and rev_g is not None and rev_g > 0:
            strengths.append("营收保持增长")
        if row.get("net_profit") and row.get("net_profit", 0) > 0:
            strengths.append("公司保持正向净利润")
        if not non_financial["audit_quality"]["risk_flag"]:
            strengths.append("审计意见为标准无保留意见，外部审计未发现重大错报")

        if hit_count > 0:
            concerns.append(f"命中{hit_count}条财务风险规则")
        if high_weight_hits:
            concerns.append(f"命中{len(high_weight_hits)}条高权重规则：{', '.join(r['rule_code'] for r in high_weight_hits)}")
        if ml_score is not None and ml_score >= self.risk_thresholds["alert"]:
            concerns.append(f"ML模型评分{ml_score}分，处于{ml_level}区间")
        if non_financial["summary"]["risk_count"] > 0:
            concerns.append(f"非财务指标发现{non_financial['summary']['risk_count']}个风险信号")

        if ml_score is not None:
            overall_level = ml_level
        else:
            overall_level = "N/A（模型未加载）"

        if overall_level == "低风险":
            assessment = "ML模型风险得分<{:.1f}，处于低风险区间。公司财务表现整体健康，未触发预警阈值。".format(self.risk_thresholds["low_medium"])
            recommendation = "保持常规监控，定期跟踪关键财务指标。"
        elif overall_level == "中风险":
            assessment = f"ML模型风险得分{ml_score}分，处于中风险区间，未达预警阈值但存在一定异常信号，建议关注。"
            recommendation = "建议关注相关异常科目变化趋势，定期跟踪关键财务指标。"
        elif overall_level == "中高风险":
            assessment = f"ML模型风险得分{ml_score}分，处于中高风险区间，已达到预警阈值，建议进入人工复核。"
            recommendation = "建议安排人工复核，重点核查命中规则的对应科目。"
        elif overall_level == "高风险":
            assessment = f"ML模型风险得分{ml_score}分，处于高风险区间，属于优先核查对象。"
            recommendation = "建议优先安排专项审计或现场核查。"
        else:
            assessment = "无法计算ML风险得分。"
            recommendation = "请检查模型文件是否正确加载。"

        return {
            "overall_risk_level": overall_level,
            "overall_assessment": assessment,
            "strengths": strengths,
            "concerns": concerns,
            "recommendation": recommendation,
        }


# ═══════════════════════════════════════════════════════════════
# 四、命令行入口
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="生成财务舞弊风险工作底稿")
    parser.add_argument("--stock_code", type=str, help="股票代码（如 000651）")
    parser.add_argument("--year", type=int, help="年份（如 2023）")
    parser.add_argument("--output", type=str, default=None, help="输出JSON文件路径")
    args = parser.parse_args()

    if args.stock_code and args.year:
        stock_code = args.stock_code
        year = args.year
    else:
        print("=" * 50)
        print("  财务舞弊风险识别 - 工作底稿生成器")
        print("  最终版：无金融业 + 虚构利润/虚列资产 + RF_RUS_1to3")
        print("=" * 50)
        stock_code = input("请输入股票代码（如 000651）: ").strip()
        year = int(input("请输入年份（如 2023）: ").strip())

    gen = WorkpaperGenerator()
    result = gen.generate(stock_code, year)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n工作底稿已保存至: {args.output}")
    else:
        print("\n" + "=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()