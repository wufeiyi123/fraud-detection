#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask 后端 API
提供股票查询、工作底稿生成、三智能体分析接口
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import sys
import json
import pandas as pd
from pathlib import Path

# 确保项目目录在路径中
from os import getcwd
SCRIPT_DIR = Path(getcwd())
sys.path.insert(0, str(SCRIPT_DIR))

app = Flask(__name__, static_folder=str(SCRIPT_DIR), static_url_path='')
CORS(app)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(Path(SCRIPT_DIR) / '.env')

DATA_PATH = Path(SCRIPT_DIR) / '汇总数据.csv'

# ── 全局缓存 ──
_df_raw = None
_workpaper_gen = None


def get_df():
    global _df_raw
    if _df_raw is None:
        _df_raw = pd.read_csv(DATA_PATH, encoding='utf-8')
        _df_raw['stock_code'] = _df_raw['stock_code'].astype(str).str.zfill(6)
        _df_raw['year_num'] = pd.to_datetime(_df_raw['year'], errors='coerce').dt.year
    return _df_raw


def get_workpaper_generator():
    global _workpaper_gen
    if _workpaper_gen is None:
        import workpaper_generator as wg
        _workpaper_gen = wg.WorkpaperGenerator()
    return _workpaper_gen


# ═══════════════════════════════════════════════════════════════
#  API 路由R
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return send_file(str(SCRIPT_DIR / 'index.html'))


@app.route('/api/stock/search')
def stock_search():
    """搜索股票：支持按代码或名称模糊搜索"""
    keyword = request.args.get('q', '').strip()
    if not keyword:
        return jsonify({"error": "请输入搜索关键词"}), 400

    df = get_df()
    mask = (
        df['stock_code'].str.contains(keyword) |
        df['company_name'].str.contains(keyword, na=False)
    )
    results = df[mask][['stock_code', 'company_name', 'industry_name']].drop_duplicates()
    results['industry_name'] = results['industry_name'].fillna('').astype(str)
    results = results.head(20)
    return jsonify(results.to_dict('records'))


@app.route('/api/stock/info')
def stock_info():
    """获取指定股票的可用年份列表和基本信息"""
    stock_code = request.args.get('code', '').strip().zfill(6)
    if not stock_code:
        return jsonify({"error": "请输入股票代码"}), 400

    df = get_df()
    mask = df['stock_code'] == stock_code
    if mask.sum() == 0:
        return jsonify({"error": f"未找到股票代码 {stock_code}"}), 404

    company = df[mask].iloc[0]
    years = sorted(df[mask]['year_num'].dropna().unique().tolist(), reverse=True)

    return jsonify({
        "stock_code": stock_code,
        "company_name": str(company['company_name']),
        "industry_name": str(company.get('industry_name', '')),
        "available_years": years,
    })


@app.route('/api/stock/basic_info')
def stock_basic_info():
    """获取指定股票某年的详细基本信息和财务数据"""
    stock_code = request.args.get('code', '').strip().zfill(6)
    year = request.args.get('year', '').strip()

    if not stock_code or not year:
        return jsonify({"error": "请输入股票代码和年份"}), 400

    df = get_df()
    year_int = int(year)
    mask = (df['stock_code'] == stock_code) & (df['year_num'] == year_int)
    if mask.sum() == 0:
        return jsonify({"error": f"未找到 {stock_code} {year} 年数据"}), 404

    row = df[mask].iloc[0]

    def fmt_b(v):
        if pd.isna(v) or v is None:
            return None
        return round(float(v) / 1e8, 2)

    def fmt_pct(v):
        if pd.isna(v) or v is None:
            return None
        return round(float(v) * 100, 2)

    info = {
        "stock_code": stock_code,
        "company_name": str(row['company_name']),
        "industry_name": str(row.get('industry_name', '')),
        "industry_code": str(row.get('industry_code', '')),
        "year": year_int,
        "report_date": str(row.get('year', '')),
        # 规模指标（亿元）
        "total_assets": fmt_b(row.get('total_assets')),
        "total_revenue": fmt_b(row.get('operating_revenue')),
        "net_profit": fmt_b(row.get('net_profit')),
        "total_liabilities": fmt_b(row.get('total_liabilities')),
        "owners_equity": fmt_b(row.get('total_owners_equity')),
        "operating_cash_flow": fmt_b(row.get('operating_cash_flow_net')),
        # 盈利能力
        "gross_margin": fmt_pct(row.get('gross_profit_margin')),
        "net_profit_margin": fmt_pct(row.get('net_profit') / row.get('total_operating_revenue')) if row.get('total_operating_revenue') and row.get('total_operating_revenue') != 0 else None,
        "roe": float(row['weighted_avg_roe']) if pd.notna(row.get('weighted_avg_roe')) else None,
        "basic_eps": float(row['basic_eps']) if pd.notna(row.get('basic_eps')) else None,
        # 营运能力
        "receivables_turnover": float(row['receivables_turnover_ratio']) if pd.notna(row.get('receivables_turnover_ratio')) else None,
        "inventory_turnover": float(row['inventory_turnover_ratio']) if pd.notna(row.get('inventory_turnover_ratio')) else None,
        "current_ratio": float(row['CurrentRatio']) if pd.notna(row.get('CurrentRatio')) else None,
        # 增长率
        "revenue_growth": fmt_pct(row.get('revenue_growth')),
        "receivable_growth": fmt_pct(row.get('receivable_growth_yoy')),
        "inventory_growth": fmt_pct(row.get('inventory_growth_yoy')),
        # 现金流
        "ocf": fmt_b(row.get('operating_cash_flow_net')),
        "investing_cf": fmt_b(row.get('investing_cash_flow_net')),
        "financing_cf": fmt_b(row.get('financing_cash_flow_net')),
        # 非财务
        "audit_opinion": str(row.get('audit_opinion_risk', '')),
        "customer_concentration": float(row['customer_concentration']) if pd.notna(row.get('customer_concentration')) else None,
        "supplier_concentration": float(row['supplier_concentration']) if pd.notna(row.get('supplier_concentration')) else None,
        "related_party_amount": float(row['related_party_transaction_amount']) if pd.notna(row.get('related_party_transaction_amount')) else None,
        # 费用率
        "selling_expense_ratio": fmt_pct(row.get('selling_expense_ratio')),
        "interest_liab_ratio": float(row['interest_liab_ratio']) if pd.notna(row.get('interest_liab_ratio')) else None,
    }

    return jsonify(info)


@app.route('/api/workpaper')
def generate_workpaper():
    """生成工作底稿（不调用LLM）"""
    stock_code = request.args.get('code', '').strip().zfill(6)
    year = request.args.get('year', '').strip()

    if not stock_code or not year:
        return jsonify({"error": "请输入股票代码和年份"}), 400

    try:
        gen = get_workpaper_generator()
        wp = gen.generate(stock_code, int(year))
        if not wp.get('success'):
            return jsonify({"error": wp.get('error', '底稿生成失败')}), 500
        return jsonify(wp)
    except Exception as e:
        return jsonify({"error": f"底稿生成失败: {str(e)}"}), 500


@app.route('/api/analyze', methods=['POST'])
def full_analysis():
    """完整三智能体分析（工作底稿生成 + 指控 + 辩护 + 裁决）"""
    data = request.get_json()
    stock_code = str(data.get('code', '')).strip().zfill(6)
    year = str(data.get('year', '')).strip()

    if not stock_code or not year:
        return jsonify({"error": "请输入股票代码和年份"}), 400

    api_key = os.environ.get('DEEPSEEK_API_KEY', '')
    if not api_key:
        return jsonify({"error": "未配置 DEEPSEEK_API_KEY"}), 500

    try:
        import three_agents as ta

        # 1. 生成工作底稿
        workpaper = ta.generate_workpaper(stock_code, int(year))
        if not workpaper.get('success'):
            return jsonify({"error": workpaper.get('error', '底稿生成失败')}), 500

        # 2. 三智能体分析
        orchestrator = ta.ThreeAgentOrchestrator(api_key=api_key)
        result = orchestrator.analyze(workpaper, verbose=False)

        if not result.get('success'):
            return jsonify({"error": result.get('error', '分析失败')}), 500

        # 附加工作底稿摘要信息
        result['workpaper_summary'] = {
            'risk_summary': workpaper.get('risk_summary'),
            'core_financial_metrics': workpaper.get('core_financial_metrics'),
            'peer_comparison': workpaper.get('peer_comparison'),
            'non_financial_indicators': workpaper.get('non_financial_indicators'),
            'fraud_triangle_analysis': workpaper.get('fraud_triangle_analysis'),
            'key_risk_signals': workpaper.get('key_risk_signals'),
            'hit_rules_detail': workpaper.get('hit_rules_detail'),
            'evidence_chain': workpaper.get('evidence_chain'),
            'comprehensive_assessment': workpaper.get('comprehensive_assessment'),
        }

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"分析失败: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    print("=" * 50)
    print("  财务舞弊风险识别 - 三智能体分析系统")
    print(f"  访问 http://0.0.0.0:{port}")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
