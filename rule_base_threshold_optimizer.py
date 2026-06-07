#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断建议驱动的规则库阈值再校准

本脚本用于修复“规则阈值过于经验化、单规则误报过多”的问题。

方法：
1. 使用 2007-2019 训练集 + 2020-2021 验证集，对每条规则的候选阈值做全样本回测；
2. 每条规则不再只看舞弊样本，而是同时比较舞弊样本与非舞弊样本的命中差异；
3. 对 R3-03、R3-05 等诊断表中表现较弱的规则做降权/停用；
4. 高风险判定采用“稳定规则组合”和“强证据规则”，取消“高权重规则命中即高风险”的兜底；
5. 输出优化后的规则阈值、规则命中明细、分期效果、预测集结果和报告所需数据。
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List
import tempfile

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = Path("汇总数据.csv")
OUT = Path(tempfile.gettempdir()) / "fraud_detection_temp"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


RULE_NAMES = {
    "R1-01": "应收账款增速显著背离营收增速",
    "R1-02": "销售回款率偏低",
    "R1-03": "毛利率显著高于行业均值",
    "R1-04": "净利润增长但经营现金流走弱",
    "R1-05": "营收增速显著高于行业均值",
    "R1-06": "销售费用率逆势下降",
    "R2-01": "存货增速显著高于营业成本增速",
    "R2-02": "存货周转率显著低于行业水平",
    "R2-03": "存货高企且周转偏弱",
    "R2-04": "存货占总资产比例异常偏高",
    "R2-05": "在建工程长期挂账不转固",
    "R3-01": "存贷双高",
    "R3-02": "往来款/预付款异常占用",
    "R3-03": "经营现金流与利润背离",
    "R3-04": "关联交易金额占比偏高",
    "R3-05": "货币资金与利息收入不匹配",
    "R3-06": "筹资现金流持续流入但业务无明显扩张",
}

RULE_CODES = list(RULE_NAMES)

# 诊断表显示这些规则单独或组合区分度相对更稳。
STABLE_RULES = ["R1-04", "R2-01", "R2-02", "R3-04", "R3-06"]
STRONG_EVIDENCE_RULES = ["R1-04", "R3-06"]
DISABLED_RULES = {"R3-03", "R3-05"}

RULE_CATEGORY = {code: code.split("-")[0] for code in RULE_CODES}


@dataclass
class Candidate:
    rule_code: str
    description: str
    condition: Callable[[pd.DataFrame], pd.Series]


def safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def normalize_unit(series: pd.Series, kind: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    sample = s.dropna()
    if sample.empty:
        return s
    median_abs = sample.abs().median()
    q95_abs = sample.abs().quantile(0.95)
    if kind in {"growth", "ratio"} and median_abs > 2 and q95_abs > 5:
        return s / 100.0
    return s


def pct_change_by_company(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return (
        df.groupby("stock_code", sort=False)[col]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
    )


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["year_dt"] = pd.to_datetime(df["year"], errors="coerce")
    df["year_num"] = df["year_dt"].dt.year
    df = df.sort_values(["stock_code", "year_num"]).reset_index(drop=True)

    keep_text = {"stock_code", "company_name", "industry_code", "industry_name", "year", "year_dt"}
    for col in df.columns:
        if col not in keep_text:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for c in ["gross_profit_margin", "industry_gross_profit_margin", "selling_expense_ratio", "revenue_growth"]:
        if c in df.columns:
            df[c] = normalize_unit(df[c], "ratio" if c != "revenue_growth" else "growth")
    for c in ["revenue_growth_yoy", "receivable_growth_yoy", "inventory_growth_yoy", "cost_of_revenue_growth_yoy"]:
        if c in df.columns:
            df[c] = normalize_unit(df[c], "growth")

    df["revenue_growth_model"] = df["revenue_growth_yoy"].fillna(df.get("revenue_growth"))
    df["receivable_growth_model"] = df["receivable_growth_yoy"].fillna(pct_change_by_company(df, "accounts_receivable_net"))
    df["inventory_growth_model"] = df["inventory_growth_yoy"].fillna(pct_change_by_company(df, "inventory_net"))
    df["cost_growth_model"] = df["cost_of_revenue_growth_yoy"].fillna(pct_change_by_company(df, "operating_cost"))
    df["net_profit_growth_model"] = pct_change_by_company(df, "net_profit")
    df["cash_received_growth_model"] = pct_change_by_company(df, "cash_received_from_sales")

    df["selling_expense_ratio_lag"] = df.groupby("stock_code", sort=False)["selling_expense_ratio"].shift(1)
    df["selling_expense_ratio_drop"] = df["selling_expense_ratio_lag"] - df["selling_expense_ratio"]

    df["cash_collection_ratio"] = safe_div(df["cash_received_from_sales"], df["operating_revenue"])
    df["ocf_to_profit"] = safe_div(df["operating_cash_flow_net"], df["net_profit"].abs())
    df["inventory_to_assets"] = safe_div(df["inventory_net"], df["total_assets"])
    df["cash_to_assets"] = safe_div(df["cash"], df["total_assets"])
    df["short_loan_to_liab"] = safe_div(df["short_term_borrowings"], df["total_liabilities"])
    df["prepayment_to_assets"] = safe_div(df["prepayments_net"], df["total_assets"])
    df["prepayment_growth_model"] = pct_change_by_company(df, "prepayments_net")
    df["related_party_to_assets"] = safe_div(df["related_party_transaction_amount"], df["total_assets"])
    df["interest_income_to_cash"] = safe_div(df["interest_income"], df["cash"])
    df["financing_to_assets"] = safe_div(df["financing_cash_flow_net"], df["total_assets"])
    df["cips_growth"] = pct_change_by_company(df, "construction_in_progress_net")
    df["fixed_assets_growth"] = pct_change_by_company(df, "fixed_assets_net")
    df["cips_growth_lag"] = df.groupby("stock_code", sort=False)["cips_growth"].shift(1)

    group = df.groupby(["industry_code", "year_num"], dropna=False)
    df["industry_revenue_growth_median"] = group["revenue_growth_model"].transform("median")
    df["industry_inventory_to_assets_median"] = group["inventory_to_assets"].transform("median")
    df["industry_prepayment_to_assets_p75"] = group["prepayment_to_assets"].transform(lambda s: s.quantile(0.75))
    df["industry_prepayment_to_assets_p90"] = group["prepayment_to_assets"].transform(lambda s: s.quantile(0.90))
    df["industry_inventory_turnover_median"] = group["inventory_turnover_ratio"].transform("median")
    df["industry_inventory_turnover_p25"] = group["inventory_turnover_ratio"].transform(lambda s: s.quantile(0.25))

    df["period"] = df["year_num"].apply(label_period)
    return df


def label_period(year: float) -> str:
    try:
        y = int(year)
    except Exception:
        return "out_of_scope"
    if 2007 <= y <= 2019:
        return "train_2007_2019"
    if 2020 <= y <= 2021:
        return "validation_2020_2021"
    if 2022 <= y <= 2023:
        return "test_2022_2023"
    if 2024 <= y <= 2025:
        return "prediction_2024_2025"
    return "out_of_scope"


def bool_series(condition: pd.Series, required: Iterable[pd.Series] | None = None) -> pd.Series:
    out = condition.fillna(False)
    if required:
        valid = pd.Series(True, index=condition.index)
        for s in required:
            valid &= s.notna()
        out &= valid
    return out.astype(int)


def add_candidate(cands: List[Candidate], rule: str, desc: str, fn: Callable[[pd.DataFrame], pd.Series]) -> None:
    cands.append(Candidate(rule, desc, fn))


def build_candidates() -> List[Candidate]:
    cands: List[Candidate] = []

    for gap in [0.20, 0.30, 0.40, 0.50]:
        add_candidate(cands, "R1-01", f"应收增速-营收增速 > {gap:.0%}", lambda d, gap=gap: bool_series(d["receivable_growth_model"] - d["revenue_growth_model"] > gap, [d["receivable_growth_model"], d["revenue_growth_model"]]))

    for cash_th, rev_th in product([0.50, 0.60, 0.70, 0.80], [0.00, 0.10, 0.20]):
        add_candidate(cands, "R1-02", f"销售回款率 < {cash_th:.0%} 且营收增长 > {rev_th:.0%}", lambda d, cash_th=cash_th, rev_th=rev_th: bool_series((d["cash_collection_ratio"] < cash_th) & (d["revenue_growth_model"] > rev_th), [d["cash_collection_ratio"], d["revenue_growth_model"]]))

    for gap in [0.15, 0.20, 0.25, 0.30]:
        add_candidate(cands, "R1-03", f"毛利率高于行业均值 {gap:.0%} 以上", lambda d, gap=gap: bool_series(d["gross_profit_margin"] - d["industry_gross_profit_margin"] > gap, [d["gross_profit_margin"], d["industry_gross_profit_margin"]]))

    for profit_g, ocf_th in product([0.20, 0.30, 0.50], [0.30, 0.40, 0.50]):
        add_candidate(cands, "R1-04", f"净利润增长 > {profit_g:.0%} 且 OCF/利润 < {ocf_th:.0%}", lambda d, profit_g=profit_g, ocf_th=ocf_th: bool_series((d["net_profit_growth_model"] > profit_g) & (d["ocf_to_profit"] < ocf_th), [d["net_profit_growth_model"], d["ocf_to_profit"]]))

    for gap, own in product([0.30, 0.40, 0.50], [0.30, 0.40, 0.50]):
        add_candidate(cands, "R1-05", f"营收增速 > 行业中位数+{gap:.0%} 且自身 > {own:.0%}", lambda d, gap=gap, own=own: bool_series((d["revenue_growth_model"] > d["industry_revenue_growth_median"] + gap) & (d["revenue_growth_model"] > own), [d["revenue_growth_model"], d["industry_revenue_growth_median"]]))

    for rev_g, drop in product([0.20, 0.30, 0.40], [0.05, 0.08, 0.10]):
        add_candidate(cands, "R1-06", f"营收增长 > {rev_g:.0%} 且销售费用率下降 > {drop:.0%}", lambda d, rev_g=rev_g, drop=drop: bool_series((d["revenue_growth_model"] > rev_g) & (d["selling_expense_ratio_drop"] > drop), [d["revenue_growth_model"], d["selling_expense_ratio_drop"]]))

    for gap in [0.25, 0.35, 0.50, 0.70]:
        add_candidate(cands, "R2-01", f"存货增速-成本增速 > {gap:.0%}", lambda d, gap=gap: bool_series(d["inventory_growth_model"] - d["cost_growth_model"] > gap, [d["inventory_growth_model"], d["cost_growth_model"]]))

    for mult in [0.40, 0.50, 0.60, 0.70, 0.80]:
        add_candidate(cands, "R2-02", f"存货周转率 < 行业中位数 × {mult:.0%}", lambda d, mult=mult: bool_series(d["inventory_turnover_ratio"] < d["industry_inventory_turnover_median"] * mult, [d["inventory_turnover_ratio"], d["industry_inventory_turnover_median"]]))
    add_candidate(cands, "R2-02", "存货周转率 < 行业P25", lambda d: bool_series(d["inventory_turnover_ratio"] < d["industry_inventory_turnover_p25"], [d["inventory_turnover_ratio"], d["industry_inventory_turnover_p25"]]))

    for inv_mult, turn_mult in product([1.50, 2.00, 2.50], [0.60, 0.80]):
        add_candidate(cands, "R2-03", f"存货占比 > 行业中位数×{inv_mult:.1f} 且周转率 < 行业中位数×{turn_mult:.0%}", lambda d, inv_mult=inv_mult, turn_mult=turn_mult: bool_series((d["inventory_to_assets"] > d["industry_inventory_to_assets_median"] * inv_mult) & (d["inventory_turnover_ratio"] < d["industry_inventory_turnover_median"] * turn_mult), [d["inventory_to_assets"], d["industry_inventory_to_assets_median"], d["inventory_turnover_ratio"], d["industry_inventory_turnover_median"]]))

    for inv_mult, fixed_th in product([1.50, 2.00, 2.50], [0.25, 0.30, 0.40]):
        add_candidate(cands, "R2-04", f"存货占比 > 行业中位数×{inv_mult:.1f} 且自身 > {fixed_th:.0%}", lambda d, inv_mult=inv_mult, fixed_th=fixed_th: bool_series((d["inventory_to_assets"] > d["industry_inventory_to_assets_median"] * inv_mult) & (d["inventory_to_assets"] > fixed_th), [d["inventory_to_assets"], d["industry_inventory_to_assets_median"]]))

    for cips_g, fa_g in product([0.20, 0.40], [0.05, 0.10]):
        add_candidate(cands, "R2-05", f"在建工程连续增长 > {cips_g:.0%} 且固定资产增长 < {fa_g:.0%}", lambda d, cips_g=cips_g, fa_g=fa_g: bool_series((d["cips_growth"] > cips_g) & (d["cips_growth_lag"] > cips_g) & (d["fixed_assets_growth"].fillna(0) < fa_g), [d["cips_growth"], d["cips_growth_lag"]]))

    for cash_th, loan_th in product([0.30, 0.40, 0.50], [0.20, 0.30, 0.40]):
        add_candidate(cands, "R3-01", f"货币资金/资产 > {cash_th:.0%} 且短借/负债 > {loan_th:.0%}", lambda d, cash_th=cash_th, loan_th=loan_th: bool_series((d["cash_to_assets"] > cash_th) & (d["short_loan_to_liab"] > loan_th), [d["cash_to_assets"], d["short_loan_to_liab"]]))

    for growth in [0.50, 0.80, 1.00]:
        add_candidate(cands, "R3-02", f"预付款增长 > {growth:.0%}", lambda d, growth=growth: bool_series(d["prepayment_growth_model"] > growth, [d["prepayment_growth_model"]]))
    for fixed in [0.10, 0.15, 0.20]:
        add_candidate(cands, "R3-02", f"预付款/资产 > {fixed:.0%} 且超过行业P75", lambda d, fixed=fixed: bool_series((d["prepayment_to_assets"] > fixed) & (d["prepayment_to_assets"] > d["industry_prepayment_to_assets_p75"]), [d["prepayment_to_assets"], d["industry_prepayment_to_assets_p75"]]))

    for ocf_th in [0.20, 0.30]:
        add_candidate(cands, "R3-03", f"OCF/利润 < {ocf_th:.0%} 且净利润为正", lambda d, ocf_th=ocf_th: bool_series((d["ocf_to_profit"] < ocf_th) & (d["net_profit"] > 0), [d["ocf_to_profit"], d["net_profit"]]))

    for related_th in [0.10, 0.15, 0.20, 0.30]:
        add_candidate(cands, "R3-04", f"关联交易金额/资产 > {related_th:.0%}", lambda d, related_th=related_th: bool_series(d["related_party_to_assets"] > related_th, [d["related_party_to_assets"]]))

    for cash_th, interest_th in product([0.20, 0.30, 0.40], [0.002, 0.005]):
        add_candidate(cands, "R3-05", f"现金/资产 > {cash_th:.0%} 且利息收入/现金 < {interest_th:.1%}", lambda d, cash_th=cash_th, interest_th=interest_th: bool_series((d["cash_to_assets"] > cash_th) & (d["interest_income_to_cash"] < interest_th), [d["cash_to_assets"], d["interest_income_to_cash"]]))

    for fin_th, rev_th in product([0.05, 0.10, 0.15], [0.00, 0.05, 0.10]):
        add_candidate(cands, "R3-06", f"筹资现金流/资产 > {fin_th:.0%} 且营收增长 < {rev_th:.0%} 且经营现金流<=0", lambda d, fin_th=fin_th, rev_th=rev_th: bool_series((d["financing_to_assets"] > fin_th) & (d["revenue_growth_model"].fillna(0) < rev_th) & (d["operating_cash_flow_net"].fillna(0) <= 0), [d["financing_to_assets"]]))

    return cands


def binary_metrics(y: pd.Series, pred: pd.Series) -> Dict[str, float]:
    y = y.astype(int).to_numpy()
    p = pred.astype(int).to_numpy()
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
        "hit_rate": float(p.mean()) if total else 0.0,
    }


def eval_candidate(df: pd.DataFrame, y: pd.Series, pred: pd.Series) -> Dict[str, float]:
    m = binary_metrics(y, pred)
    base = float(y.mean()) if len(y) else 0.0
    m["base_rate"] = base
    m["lift"] = m["precision"] / base if base > 0 else 0.0
    return m


def choose_best_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df["period"] == "train_2007_2019"
    val = df["period"] == "validation_2020_2021"
    y_train = df.loc[train, "fraud"].fillna(0).astype(int)
    y_val = df.loc[val, "fraud"].fillna(0).astype(int)

    rows = []
    for cand in build_candidates():
        pred_all = cand.condition(df)
        train_m = eval_candidate(df.loc[train], y_train, pred_all.loc[train])
        val_m = eval_candidate(df.loc[val], y_val, pred_all.loc[val])

        # 规则阈值评分：偏好验证集 precision/lift，同时约束命中率，避免一条规则覆盖半个市场。
        hit_penalty = max(0.0, val_m["hit_rate"] - 0.30) * 0.8
        tiny_penalty = max(0.0, 0.005 - val_m["hit_rate"]) * 3.0
        disabled_penalty = 0.40 if cand.rule_code in DISABLED_RULES else 0.0
        stability_bonus = 0.05 if cand.rule_code in STABLE_RULES else 0.0
        objective = (
            0.38 * val_m["f1"]
            + 0.26 * val_m["precision"]
            + 0.20 * val_m["recall"]
            + 0.10 * max(val_m["lift"] - 1.0, 0.0)
            + stability_bonus
            - hit_penalty
            - tiny_penalty
            - disabled_penalty
        )
        rows.append({
            "rule_code": cand.rule_code,
            "rule_name": RULE_NAMES[cand.rule_code],
            "candidate_description": cand.description,
            "objective": objective,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"validation_{k}": v for k, v in val_m.items()},
        })

    grid = pd.DataFrame(rows)
    best_rows = []
    for rule, grp in grid.groupby("rule_code", sort=False):
        best = grp.sort_values(["objective", "validation_lift", "validation_precision"], ascending=False).iloc[0].copy()
        if rule in DISABLED_RULES:
            best["recommended_action"] = "停用/仅保留人工说明"
            best["score_point"] = 0
        elif best["validation_lift"] >= 1.45 and best["validation_precision"] >= best["validation_base_rate"] * 1.35:
            best["recommended_action"] = "强证据规则"
            best["score_point"] = 3
        elif best["validation_lift"] >= 1.05 or rule in STABLE_RULES:
            best["recommended_action"] = "稳定/辅助规则"
            best["score_point"] = 2 if rule in STABLE_RULES else 1
        else:
            best["recommended_action"] = "降权为辅助规则"
            best["score_point"] = 1
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows).reset_index(drop=True)
    return grid, best_df


def build_hits_from_best(df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    all_candidates = build_candidates()
    lookup = {(c.rule_code, c.description): c for c in all_candidates}
    hits = pd.DataFrame(index=df.index)
    for row in best_df.itertuples():
        cand = lookup[(row.rule_code, row.candidate_description)]
        hits[row.rule_code] = cand.condition(df)
        if int(row.score_point) == 0:
            hits[row.rule_code] = 0
    return hits[RULE_CODES].astype(int)


def apply_diagnosis_guided_levels(hits: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    score_points = dict(zip(best_df["rule_code"], best_df["score_point"].astype(int)))
    adjusted_score = sum(hits[c] * score_points.get(c, 0) for c in RULE_CODES)
    stable_count = hits[STABLE_RULES].sum(axis=1)
    strong_count = hits[STRONG_EVIDENCE_RULES].sum(axis=1)
    stable_category_count = hits[STABLE_RULES].apply(
        lambda row: len({RULE_CATEGORY[c] for c in STABLE_RULES if row[c] == 1}), axis=1
    )
    cross_category_stable2 = (stable_count >= 2) & (stable_category_count >= 2)

    high = (stable_count >= 3) | ((strong_count >= 1) & (stable_count >= 2))
    medium_high = (~high) & ((stable_count >= 2) | (cross_category_stable2 & (adjusted_score >= 4)))
    medium = (~high) & (~medium_high) & ((adjusted_score >= 2) | (stable_count >= 1))

    level = np.where(high, "高风险", np.where(medium_high, "中高风险", np.where(medium, "中风险", "低风险")))
    alert = np.isin(level, ["中高风险", "高风险"]).astype(int)

    return pd.DataFrame({
        "adjusted_score": adjusted_score.astype(int),
        "stable_rule_count": stable_count.astype(int),
        "strong_evidence_count": strong_count.astype(int),
        "stable_category_count": stable_category_count.astype(int),
        "cross_category_stable2": cross_category_stable2.astype(int),
        "optimized_risk_level": level,
        "optimized_alert_flag": alert,
    }, index=hits.index)


def period_summary(out: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, sub in out.groupby("period", sort=False):
        y = sub["fraud"].fillna(0).astype(int)
        pred = sub["optimized_alert_flag"].astype(int)
        m = binary_metrics(y, pred)
        item = {
            "period": period,
            "rows": len(sub),
            "fraud_count": int(y.sum()),
            "fraud_rate": float(y.mean()) if len(y) else 0.0,
            "low_count": int((sub["optimized_risk_level"] == "低风险").sum()),
            "medium_count": int((sub["optimized_risk_level"] == "中风险").sum()),
            "medium_high_count": int((sub["optimized_risk_level"] == "中高风险").sum()),
            "high_count": int((sub["optimized_risk_level"] == "高风险").sum()),
            "low_share": float((sub["optimized_risk_level"] == "低风险").mean()),
            "medium_share": float((sub["optimized_risk_level"] == "中风险").mean()),
            "medium_high_share": float((sub["optimized_risk_level"] == "中高风险").mean()),
            "high_share": float((sub["optimized_risk_level"] == "高风险").mean()),
            "alert_share": float(pred.mean()),
            **m,
        }
        if period == "prediction_2024_2025":
            for k in ["tp", "fp", "tn", "fn", "precision", "recall", "f1", "accuracy"]:
                item[k] = np.nan
        rows.append(item)
    return pd.DataFrame(rows)


def rule_diagnostics(hits: pd.DataFrame, df: pd.DataFrame, best_df: pd.DataFrame) -> pd.DataFrame:
    test = df["period"] == "test_2022_2023"
    val = df["period"] == "validation_2020_2021"
    rows = []
    for rule in RULE_CODES:
        row = {
            "rule_code": rule,
            "rule_name": RULE_NAMES[rule],
            "selected_threshold": best_df.loc[best_df.rule_code == rule, "candidate_description"].iloc[0],
            "score_point": int(best_df.loc[best_df.rule_code == rule, "score_point"].iloc[0]),
            "recommended_action": best_df.loc[best_df.rule_code == rule, "recommended_action"].iloc[0],
        }
        for name, mask in [("validation", val), ("test", test)]:
            m = eval_candidate(df.loc[mask], df.loc[mask, "fraud"].fillna(0).astype(int), hits.loc[mask, rule])
            for k, v in m.items():
                row[f"{name}_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = load_and_prepare()
    grid, best_df = choose_best_candidates(df)
    hits = build_hits_from_best(df, best_df)
    levels = apply_diagnosis_guided_levels(hits, best_df)

    base_cols = ["stock_code", "company_name", "year_num", "industry_code", "industry_name", "fraud", "period"]
    out = pd.concat([df[base_cols].copy(), levels, hits.add_prefix("hit_")], axis=1)
    summary = period_summary(out)
    diagnostics = rule_diagnostics(hits, df, best_df)

    grid.to_csv(OUT / "01_rule_threshold_candidate_grid.csv", index=False, encoding="utf-8-sig")
    best_df.to_csv(OUT / "02_selected_rule_thresholds.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(OUT / "03_rule_diagnostics_after_recalibration.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "04_period_summary_after_recalibration.csv", index=False, encoding="utf-8-sig")
    out.to_csv(OUT / "05_all_samples_optimized_risk_results.csv", index=False, encoding="utf-8-sig")
    out[out["period"] == "validation_2020_2021"].to_csv(OUT / "06_validation_2020_2021_optimized_risk_results.csv", index=False, encoding="utf-8-sig")
    out[out["period"] == "prediction_2024_2025"].sort_values(
        ["optimized_alert_flag", "optimized_risk_level", "adjusted_score", "stable_rule_count"],
        ascending=[False, False, False, False],
    ).to_csv(OUT / "07_prediction_2024_2025_optimized_risk_results.csv", index=False, encoding="utf-8-sig")

    payload = {
        "method": "全样本规则阈值校准 + 诊断建议驱动风险分层",
        "data_path": str(DATA_PATH),
        "output_dir": str(OUT),
        "stable_rules": STABLE_RULES,
        "strong_evidence_rules": STRONG_EVIDENCE_RULES,
        "disabled_rules": sorted(DISABLED_RULES),
        "risk_level_logic": {
            "高风险": "稳定规则命中数>=3，或强证据规则命中且稳定规则命中数>=2",
            "中高风险": "未达高风险，但稳定规则命中数>=2，或跨类别稳定规则>=2且调整得分>=4",
            "中风险": "未达中高/高风险，但调整得分>=2或稳定规则命中数>=1",
            "低风险": "其余样本",
            "预警口径": "中高风险 + 高风险",
        },
        "period_summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "selected_thresholds": best_df.replace({np.nan: None}).to_dict(orient="records"),
    }
    (OUT / "08_recalibration_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("诊断建议驱动的规则库阈值再校准完成")
    print("输出目录:", OUT)
    print("\n分期表现:")
    print(summary[["period", "rows", "fraud_count", "alert_share", "high_share", "precision", "recall", "f1", "accuracy"]].to_string(index=False))
    print("\n选择后的规则阈值:")
    print(best_df[["rule_code", "rule_name", "candidate_description", "recommended_action", "score_point", "validation_precision", "validation_recall", "validation_lift"]].to_string(index=False))


if __name__ == "__main__":
    main()
