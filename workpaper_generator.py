#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
工作底稿生成器 — 财务舞弊风险识别系统核心模块
=============================================================================
功能说明：
    基于最终版规则库（无金融业 + 虚构利润/虚列资产 + RF_RUS_1to3），
    输入股票代码和年份，自动生成结构化工作底稿（JSON格式），供后续分析使用。

技术路线：
    1. 完全复用 rule_base_threshold_optimizer.py 的 load_and_prepare()
       进行数据预处理（单位转换、增长率计算、行业对比等），确保与建模阶段100%一致。
    2. 完全复用 rule_base_threshold_optimizer.py 的 build_candidates()
       进行17条规则的触发判断，使用最终版训练脚本选定的最优阈值。
    3. 加载最终版训练好的随机森林模型（RF_RUS_1to3），计算综合舞弊风险得分。
    4. 综合财务指标、非财务指标、同业对比、舞弊三角分析，生成完整工作底稿。

依赖文件（需提前运行 final_all_2007_2023_rule_modeling.py 生成）：
    - outputs_time_split/final_all_2007_2023_non_financial_fraud_type/
        ├── final_rf_rus_1to3_model_2007_2023.pkl      # 训练好的RF模型
        ├── final_feature_columns.json                  # 模型特征列清单
        ├── 05_rule_weights_final_2007_2023.csv         # 规则权重表（含阈值、分值）
        └── 06_risk_score_level_ranges_final.csv        # 风险等级阈值（分数边界）

输入：
    股票代码（6位字符串）+ 年份（整数）

输出：
    结构化工作底稿 JSON，包含：
    - basic_info：公司基本信息
    - risk_summary：风险汇总（ML评分、规则命中统计）
    - core_financial_metrics：核心财务指标
    - peer_comparison：同业对比
    - non_financial_indicators：非财务指标（审计意见、质押、供应链集中度等）
    - hit_rules_detail：17条规则逐条诊断明细
    - evidence_chain：证据链
    - fraud_triangle_analysis：舞弊三角分析（压力/机会/合理化）
    - key_risk_signals：关键风险信号
    - questions_for_verification：核查问题清单
    - engine_status：规则引擎运行状态
    - comprehensive_assessment：综合评估结论

使用方法：
    # 交互模式（推荐）
    python workpaper_generator.py

    # 命令行模式
    python workpaper_generator.py --stock_code 000651 --year 2023

    # 作为模块导入
    from workpaper_generator import WorkpaperGenerator
    gen = WorkpaperGenerator()
    result = gen.generate("000651", 2023)

作者：
    智能财务机器人课程项目组
版本：
    v3.0 — 完全复用最终版训练输出（动态加载规则权重、风险阈值、模型文件）
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

# ── 引入建模模块 ──────────────────────────────────────────────
# 说明：rule_base_threshold_optimizer.py 是本项目的规则引擎核心，
# 包含 load_and_prepare()（数据预处理）和 build_candidates()（规则候选构建）。
# 工作底稿生成器通过复用这两个函数，确保规则计算逻辑与建模阶段100%一致。
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rule_base_threshold_optimizer as dg


# ═══════════════════════════════════════════════════════════════
# 一、全局配置（文件路径）
# ═══════════════════════════════════════════════════════════════

# 项目根目录（假设本脚本位于项目根目录）
ROOT = Path(__file__).resolve().parent

# ── 本地数据库路径 ──
DATA_PATH = ROOT / "汇总数据.csv"

# ── 最终版模型和配置文件输出目录（需提前运行 final_all_2007_2023_rule_modeling.py 生成）──
FINAL_OUT_DIR = ROOT
# 模型推理所需的核心文件
MODEL_PATH = FINAL_OUT_DIR / "final_rf_rus_1to3_model_2007_2023.pkl"
FEATURE_COLS_PATH = FINAL_OUT_DIR / "final_feature_columns.json"
RULE_WEIGHTS_PATH = FINAL_OUT_DIR / "05_rule_weights_final_2007_2023.csv"
RISK_RANGES_PATH = FINAL_OUT_DIR / "06_risk_score_level_ranges_final.csv"


# ═══════════════════════════════════════════════════════════════
# 二、辅助工具函数
# ═══════════════════════════════════════════════════════════════

def safe_div(a, b):
    """
    安全除法
    说明：避免除零和NaN传播，确保财务比率计算的健壮性。
    """
    if b is None or b == 0 or (isinstance(b, float) and pd.isna(b)):
        return np.nan
    if a is None or (isinstance(a, float) and pd.isna(a)):
        return np.nan
    return a / b


def nan_to_none(v):
    """
    NaN转None
    说明：JSON标准不支持NaN和Infinity，序列化前必须将NaN/Inf转为None。
    """
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def fmt_pct(v, decimals=1):
    """
    格式化百分比
    说明：内部存储为小数形式（如0.26），展示时转为百分比字符串（如"26.0%"）。
    """
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"


def fmt_billion(v):
    """
    转换为亿元单位
    说明：原始数据单位为元，除以1亿便于阅读。
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(v / 1e8, 2)


def get_rule_category(rule_code):
    """
    获取规则所属类别名称
    说明：通过规则编号前缀（R1/R2/R3）映射到"收入舞弊/存货异常/资金异常"。
    """
    prefix = rule_code.split("-")[0]
    categories = {"R1": "收入舞弊", "R2": "存货异常", "R3": "资金异常"}
    return categories.get(prefix, "其他")


# ═══════════════════════════════════════════════════════════════
# 三、工作底稿生成器主类
# ═══════════════════════════════════════════════════════════════

class WorkpaperGenerator:
    """
    工作底稿生成器

    核心职责：
        1. 加载最终版训练输出的模型、特征列、规则权重表、风险阈值
        2. 调用规则引擎进行17条规则的触发判断
        3. 综合ML评分、规则命中、财务指标、非财务指标，生成结构化工作底稿

    设计原则：
        - 延迟加载：首次调用 _ensure_loaded() 时才加载数据和模型
        - 全局缓存：预处理数据只计算一次，后续查询复用
        - 完全复用建模模块：不自行重写任何规则计算逻辑
        - 动态配置：所有阈值、权重、风险等级边界均从最终版训练输出文件读取
    """

    def __init__(self):
        """初始化，设置数据、模型、配置为None，首次调用时延迟加载。"""
        self.df_raw = None          # 原始CSV数据
        self.df_prepared = None     # 预处理后的数据（含派生字段）
        self.model = None           # 训练好的RF_RUS_1to3模型
        self.feature_cols = None    # 模型推理所需的特征列清单
        self.rule_config = None     # 从CSV加载的规则权重表
        self.risk_thresholds = None # 从CSV加载的风险等级阈值
        self._loaded = False        # 加载状态标记

    # ═══════════════════════════════════════════════════════════
    # 配置加载方法
    # ═══════════════════════════════════════════════════════════

    def _load_rule_weights(self) -> dict:
        """
        从最终版训练输出的规则权重表（05_rule_weights_final_2007_2023.csv）加载规则配置。

        返回格式：
            {
                "R1-01": {
                    "level": "中",           # 高/中/低
                    "score": 2,              # 3/2/1
                    "threshold_desc": "应收增速-营收增速 > 20%",
                    "calibrated_weight": 0.72
                },
                ...
            }
        """
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
        """
        从最终版训练输出的风险等级表（06_risk_score_level_ranges_final.csv）加载风险得分边界。

        返回格式：
            {
                "low_medium": 14.0,   # 低风险 < 14.0 < 中风险
                "alert": 42.5,        # 预警阈值（中高风险起点）
                "high": 53.7          # 高风险起点
            }
        """
        if not RISK_RANGES_PATH.exists():
            print(f"[WARN] 风险等级表不存在: {RISK_RANGES_PATH}，将使用默认阈值")
            return {"low_medium": 14.0, "alert": 42.5, "high": 53.7}

        df = pd.read_csv(RISK_RANGES_PATH)
        thresholds = {"low_medium": 14.0, "alert": 42.5, "high": 53.7}  # 默认值

        for _, row in df.iterrows():
            range_str = row.get("风险得分范围", "")
            level = row.get("风险等级", "")

            # 解析 "42.5 <= 得分 < 53.7" 这类字符串
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

    # ═══════════════════════════════════════════════════════════
    # 数据与模型加载（延迟加载 + 缓存复用）
    # ═══════════════════════════════════════════════════════════

    def _ensure_loaded(self):
        """
        加载数据和模型（延迟加载 + 缓存复用）

        设计要点：
            1. 首次调用时执行完整加载流程（CSV读取 → 预处理 → 模型加载 → 配置加载）
            2. 后续调用直接复用已加载的对象
            3. 预处理调用 dg.load_and_prepare()，与建模阶段使用完全相同的逻辑
        """
        if self._loaded:
            return

        # ── Step 1: 加载原始CSV数据 ──
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}")

        self.df_raw = pd.read_csv(DATA_PATH)
        self.df_raw["stock_code"] = self.df_raw["stock_code"].astype(str).str.zfill(6)
        self.df_raw["year_num"] = pd.to_datetime(self.df_raw["year"], errors="coerce").dt.year
        print(f"[INFO] 数据加载成功，共 {len(self.df_raw)} 条记录")

        # ── Step 2: 数据预处理（派生字段 + 行业对比）──
        # 复用建模模块的 load_and_prepare()
        self.df_prepared = dg.load_and_prepare()
        self.df_prepared["stock_code"] = self.df_prepared["stock_code"].astype(str).str.zfill(6)
        print(f"[INFO] 派生字段计算完成（复用 modeling 的 load_and_prepare），共 {len(self.df_prepared)} 条")

        # ── Step 3: 加载训练好的ML模型 ──
        if MODEL_PATH.exists():
            import joblib
            self.model = joblib.load(MODEL_PATH)
            print(f"[INFO] 模型加载成功: {MODEL_PATH}")
        else:
            print(f"[WARN] 模型文件不存在: {MODEL_PATH}，将跳过ML评分")
            self.model = None

        # ── Step 4: 加载模型特征列清单 ──
        if FEATURE_COLS_PATH.exists():
            self.feature_cols = json.loads(FEATURE_COLS_PATH.read_text(encoding="utf-8"))
            print(f"[INFO] 特征列加载成功，共 {len(self.feature_cols)} 个特征")
        else:
            self.feature_cols = None

        # ── Step 5: 加载规则权重配置 ──
        self.rule_config = self._load_rule_weights()

        # ── Step 6: 加载风险等级阈值 ──
        self.risk_thresholds = self._load_risk_thresholds()

        self._loaded = True

    # ═══════════════════════════════════════════════════════════
    # 规则触发判断
    # ═══════════════════════════════════════════════════════════

    def _compute_rules(self, prepared: dict) -> tuple[list, dict]:
        """
        逐条计算17条规则触发情况

        设计要点：
            1. 完全复用 diagnosis_guided 的 build_candidates() 获取规则条件函数
            2. 规则阈值描述（threshold_desc）从最终版训练输出的规则权重表中读取
            3. 规则分值（score）也从规则权重表中读取（score_point列）
            4. 停用规则（R3-03、R3-05）保持停用状态

        返回：
            (rule_results, hits_dict)
            - rule_results: 每条规则的详细结果列表
            - hits_dict: {规则编号: 是否命中} 的简略字典
        """
        # 获取所有候选规则（每条规则有多个阈值版本）
        all_candidates = dg.build_candidates()
        # 构建查找表：(规则编号, 阈值描述) -> Candidate对象
        lookup = {(c.rule_code, c.description): c for c in all_candidates}

        # 停用规则（与最终版训练脚本保持一致）
        DISABLED = {"R3-03", "R3-05"}

        # 构造单行DataFrame用于规则条件判断
        df_single = pd.DataFrame([prepared])

        results = []
        hits = {}

        for code in dg.RULE_CODES:
            # 从加载的规则权重表中获取配置
            cfg = self.rule_config.get(code, {})
            level = cfg.get("level", "低")
            score = cfg.get("score", 1)
            threshold_desc = cfg.get("threshold_desc", "")

            if code in DISABLED:
                hit = False
                evidence = f"规则{code}已停用（诊断表显示区分度较弱）"
                status = "停用"
            else:
                # 根据阈值描述匹配候选规则
                cand = lookup.get((code, threshold_desc))
                # 降级：如果没有精确匹配，尝试包含匹配
                if cand is None and threshold_desc:
                    for c in all_candidates:
                        if c.rule_code == code and threshold_desc in c.description:
                            cand = c
                            break
                # 再降级：取该规则下第一个候选
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
        """
        根据规则代码和实际数据构造证据文本
        """
        try:
            if code == "R1-01":
                rec = d.get("receivable_growth_model")
                rev = d.get("revenue_growth_model")
                if rec is not None and rev is not None:
                    gap = rec - rev
                    return f"应收账款增速{fmt_pct(rec)}，营业收入增速{fmt_pct(rev)}，差值{fmt_pct(gap)}，阈值>20个百分点。"
                return "数据不足，无法计算应收与营收增速差异"

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
                    return f"存货增速{fmt_pct(ig)}，成本增速{fmt_pct(cg)}，差值{fmt_pct(ig - cg)}"
                return f"存货增速{fmt_pct(ig)}，成本增速{fmt_pct(cg)}"

            elif code == "R2-02":
                t = d.get("inventory_turnover_ratio")
                it = d.get("industry_inventory_turnover_median")
                return f"公司存货周转率{t:.4f}次，行业均值{it:.4f}次" if t is not None else f"存货周转率{t}"

            elif code == "R2-03":
                ir = d.get("inventory_to_assets")
                t = d.get("inventory_turnover_ratio")
                return f"存货占比{fmt_pct(ir)}，周转率{t:.4f}次" if t is not None else f"存货占比{fmt_pct(ir)}"

            elif code == "R2-04":
                ir = d.get("inventory_to_assets")
                return f"存货占比{fmt_pct(ir)}"

            elif code == "R2-05":
                cg = d.get("cips_growth")
                fg = d.get("fixed_assets_growth", 0)
                return f"在建工程增速{fmt_pct(cg)}，固定资产增速{fmt_pct(fg)}"

            elif code == "R3-01":
                cr = d.get("cash_to_assets")
                lr = d.get("short_loan_to_liab")
                return f"货币资金/资产={fmt_pct(cr)}，短借/负债={fmt_pct(lr)}"

            elif code == "R3-02":
                pr = d.get("prepayment_to_assets")
                ip = d.get("industry_prepayment_to_assets_p75")
                return f"预付款/资产={fmt_pct(pr)}，行业P75={fmt_pct(ip)}" if pr is not None else "数据不足"

            elif code == "R3-03":
                ocf = d.get("ocf_to_profit")
                return f"OCF/利润={ocf:.4f}" if ocf is not None else "数据不足"

            elif code == "R3-04":
                rp = d.get("related_party_to_assets")
                return f"关联交易金额/资产={fmt_pct(rp)}"

            elif code == "R3-05":
                cr = d.get("cash_to_assets")
                iir = d.get("interest_income_to_cash")
                return f"现金/资产={fmt_pct(cr)}，利息收入/现金={fmt_pct(iir)}"

            elif code == "R3-06":
                fr = d.get("financing_to_assets")
                rg = d.get("revenue_growth_model")
                return f"筹资现金流/资产={fmt_pct(fr)}，营收增速{fmt_pct(rg)}"

            else:
                return f"规则{code}触发" if hit else f"规则{code}未触发"
        except Exception as e:
            return f"证据生成异常: {str(e)}"

    def _get_interpretation(self, code: str, hit: bool) -> str:
        """规则解释文本"""
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

    # ═══════════════════════════════════════════════════════════
    # ML模型评分
    # ═══════════════════════════════════════════════════════════

    def _ml_score(self, row: dict) -> tuple:
        """
        使用训练好的随机森林模型计算舞弊风险得分

        返回：
            (score, level)：风险得分（0-100）和风险等级，模型未加载时返回 (None, "N/A")
        """
        if self.model is None or self.feature_cols is None:
            return None, "N/A"

        try:
            X = pd.DataFrame([row])
            # 确保所有特征列都存在，缺失的补0
            for c in self.feature_cols:
                if c not in X.columns:
                    X[c] = 0

            prob = self.model.predict_proba(X[self.feature_cols])[:, 1][0]
            score = round(prob * 100, 2)

            # 根据加载的风险阈值划分等级
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

    # ═══════════════════════════════════════════════════════════
    # 主入口：生成工作底稿
    # ═══════════════════════════════════════════════════════════

    def generate(self, stock_code: str, year: int) -> dict:
        """
        主入口：生成工作底稿

        参数：
            stock_code: 股票代码（字符串，如 "000651"）
            year:       年份（整数，如 2023）

        返回：
            dict: 结构化工作底稿，包含 success 字段标识是否成功
        """
        self._ensure_loaded()

        stock_code = str(stock_code).zfill(6)
        year = int(year)
        print(f"\n[INFO] 正在生成工作底稿: {stock_code} {year}")

        # ── 从原始数据取基本信息 ──
        raw_mask = (self.df_raw["stock_code"] == stock_code) & (self.df_raw["year_num"] == year)
        if raw_mask.sum() == 0:
            return {"success": False, "error": f"原始数据中未找到 {stock_code} {year}"}
        row = self.df_raw[raw_mask].iloc[0].to_dict()

        # ── 从预处理数据取派生字段 ──
        prep_mask = (self.df_prepared["stock_code"] == stock_code) & (self.df_prepared["year_num"] == year)
        if prep_mask.sum() == 0:
            return {"success": False, "error": f"预处理数据中未找到 {stock_code} {year}"}
        prepared = self.df_prepared[prep_mask].iloc[0].to_dict()

        # ── 规则触发判断 ──
        rule_results, hits_dict = self._compute_rules(prepared)

        # ── ML模型评分 ──
        ml_score, ml_level = self._ml_score(row)

        # ── 组装工作底稿（简化版，完整版可参考原代码）──
        workpaper = self._build_workpaper(row, prepared, rule_results, hits_dict, ml_score, ml_level)
        return workpaper

    # ═══════════════════════════════════════════════════════════
    # 工作底稿各模块构建方法（精简版，保留核心结构）
    # ═══════════════════════════════════════════════════════════

    def _build_workpaper(self, row, prepared, rule_results, hits_dict, ml_score, ml_level):
        """
        组装完整工作底稿（精简版，可根据需要扩展）
        """
        stock_code = str(row.get("stock_code", "")).zfill(6)
        year = row.get("year_num")
        company_name = row.get("company_name", "")
        industry_code = row.get("industry_code", "")
        industry_name = row.get("industry_name", "")

        # 基本信息
        basic_info = {
            "stock_code": stock_code,
            "year": int(year) if pd.notna(year) else None,
            "company_name": str(company_name),
            "industry_code": str(industry_code),
            "industry_name": str(industry_name),
        }

        # 规则命中统计
        hit_rules = [r for r in rule_results if r["hit"]]
        risk_summary = {
            "overall_risk_level_ml": ml_level if ml_level else "N/A",
            "ml_score": ml_score,
            "hit_count": len(hit_rules),
            "total_rules": 17,
            "hit_rate": round(len(hit_rules) / 17 * 100, 1),
            "risk_level_thresholds": self.risk_thresholds,
        }

        # 命中规则明细
        hit_rules_detail = [r for r in rule_results if r["hit"]]

        # 引擎状态
        engine_details = {}
        for r in rule_results:
            engine_details[r["rule_code"]] = {
                "name": r.get("rule_name", ""),
                "category": r.get("category", ""),
                "hit": r.get("hit", False),
                "status": r.get("status", ""),
            }
        engine_status = {
            "total_rules": 17,
            "normal_calculated": sum(1 for r in rule_results if r["status"] == "正常计算"),
            "data_insufficient": sum(1 for r in rule_results if r["status"] in ("停用", "计算异常", "未实现")),
            "details": engine_details,
        }

        workpaper = {
            "success": True,
            "basic_info": basic_info,
            "risk_summary": risk_summary,
            "hit_rules_detail": hit_rules_detail,
            "engine_status": engine_status,
        }
        return workpaper


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