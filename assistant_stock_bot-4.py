import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine, text
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io
import base64
import time
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30

DB_USER = "root"
DB_PASSWORD = "cute1nan"
DB_HOST = "localhost:3306"
DB_NAME = "ai"
DB_CONN_STR = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}'

SQLITE_DB_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.db')

def init_sqlite_from_mysql():
    if os.path.exists(SQLITE_DB_PATH):
        return
    print("SQLite数据库不存在，正在从MySQL导入数据...")
    mysql_engine = create_engine(DB_CONN_STR, connect_args={'connect_timeout': 10})
    df = pd.read_sql('SELECT * FROM stock_history', mysql_engine)
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    df.to_sql('stock_history', conn, if_exists='replace', index=False)
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ts_code ON stock_history(ts_code)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_trade_date ON stock_history(trade_date)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_ts_code_date ON stock_history(ts_code, trade_date)')
    conn.commit()
    conn.close()
    print(f"已导入 {len(df)} 条记录到 {SQLITE_DB_PATH}")

init_sqlite_from_mysql()

system_prompt = """我是股票查询助手，以下是关于股票历史行情表相关的字段，我可能会编写对应的SQL，对数据进行查询
-- 股票历史行情表
CREATE TABLE stock_history (
    id BIGINT NOT NULL AUTO_INCREMENT,
    ts_code VARCHAR(20) NOT NULL COMMENT '股票代码(如600519.SH)',
    trade_date DATE NOT NULL COMMENT '交易日期',
    open DECIMAL(12,2) DEFAULT NULL COMMENT '开盘价',
    high DECIMAL(12,2) DEFAULT NULL COMMENT '最高价',
    low DECIMAL(12,2) DEFAULT NULL COMMENT '最低价',
    close DECIMAL(12,2) DEFAULT NULL COMMENT '收盘价',
    pre_close DECIMAL(12,2) DEFAULT NULL COMMENT '昨收价',
    `change` DECIMAL(12,2) DEFAULT NULL COMMENT '涨跌额',
    pct_chg DECIMAL(10,4) DEFAULT NULL COMMENT '涨跌幅(%)',
    vol DECIMAL(16,2) DEFAULT NULL COMMENT '成交量(手)',
    amount DECIMAL(20,2) DEFAULT NULL COMMENT '成交额(千元)',
    stock_name VARCHAR(20) DEFAULT NULL COMMENT '股票名称',
    PRIMARY KEY (id),
    KEY idx_ts_code (ts_code),
    KEY idx_trade_date (trade_date),
    KEY idx_ts_code_date (ts_code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票历史行情数据';

数据说明：
- 数据范围：2020-01-01至今
- 包含股票：贵州茅台(600519.SH)、五粮液(000858.SZ)、广发证券(000776.SZ)、中芯国际(688981.SH)
- change是MySQL保留字，查询时需要用反引号包裹：`change`
- pct_chg为涨跌幅百分比，如-4.48表示跌4.48%
- vol为成交量(手)，amount为成交额(千元)

常用查询示例：
1. 查某只股票某段时间的行情：
   SELECT trade_date, open, high, low, close, vol FROM stock_history WHERE ts_code='600519.SH' AND trade_date BETWEEN '2024-01-01' AND '2024-12-31' ORDER BY trade_date

2. 查某日所有股票涨跌幅排名：
   SELECT stock_name, ts_code, close, `change`, pct_chg FROM stock_history WHERE trade_date='2024-12-31' ORDER BY pct_chg DESC

3. 计算某只股票的月度平均收盘价：
   SELECT DATE_FORMAT(trade_date, '%Y-%m') AS month, AVG(close) AS avg_close, SUM(vol) AS total_vol FROM stock_history WHERE ts_code='600519.SH' GROUP BY month ORDER BY month

4. 计算某只股票的日收益率：
   SELECT trade_date, close, pct_chg FROM stock_history WHERE ts_code='600519.SH' ORDER BY trade_date

5. 对比多只股票某段时间的收盘价走势：
   SELECT trade_date, stock_name, close FROM stock_history WHERE trade_date BETWEEN '2024-01-01' AND '2024-06-30' ORDER BY trade_date

我将回答用户关于股票行情相关的问题

每当 exc_sql 工具返回 markdown 表格和图片时，你必须原样输出工具返回的全部内容（包括图片 markdown），不要只总结表格，也不要省略图片。这样用户才能直接看到表格和图片。

当用户要求对股票价格进行预测时，请使用 arima_stock 工具。该工具可以基于ARIMA模型对未来N天的收盘价进行预测。

当用户要求检测股票异常点（超买/超卖）时，请使用 boll_detection 工具。该工具基于布林带(20日周期+2σ)检测收盘价突破上轨(超买)或下穿下轨(超卖)的日期，默认检测过去1年，也可自定义时间范围。
"""

functions_desc = [
    {
        "name": "exc_sql",
        "description": "对于生成的SQL，进行SQL查询",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_input": {
                    "type": "string",
                    "description": "生成的SQL语句",
                }
            },
            "required": ["sql_input"],
        },
    },
    {
        "name": "arima_stock",
        "description": "使用ARIMA模型对指定股票的未来N天收盘价进行预测",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，如600519.SH",
                },
                "n": {
                    "type": "integer",
                    "description": "预测未来天数，默认5",
                }
            },
            "required": ["ts_code"],
        },
    },
    {
        "name": "boll_detection",
        "description": "使用布林带(20日周期+2σ)检测股票的超买和超卖日期",
        "parameters": {
            "type": "object",
            "properties": {
                "ts_code": {
                    "type": "string",
                    "description": "股票代码，如600519.SH",
                },
                "start_date": {
                    "type": "string",
                    "description": "检测起始日期，格式YYYY-MM-DD，默认为今天前1年",
                },
                "end_date": {
                    "type": "string",
                    "description": "检测结束日期，格式YYYY-MM-DD，默认为今天",
                }
            },
            "required": ["ts_code"],
        },
    },
]

_last_df_dict = {}

def get_session_id(kwargs):
    messages = kwargs.get('messages')
    if messages is not None:
        return id(messages)
    return None

@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    description = '对于生成的SQL，进行SQL查询，并自动可视化'
    parameters = [{
        'name': 'sql_input',
        'type': 'string',
        'description': '生成的SQL语句',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        session_id = get_session_id(kwargs)

        args = json.loads(params)
        sql_input = args['sql_input']
        print('sql_input=', sql_input)

        engine = create_engine(
            DB_CONN_STR,
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        df = pd.read_sql(text(sql_input), engine)
        print('df=', df)

        if session_id:
            _last_df_dict[session_id] = df

        n = len(df)
        if n <= 10:
            md = df.to_markdown(index=False)
        else:
            md = df.head(5).to_markdown(index=False)
            md += '\n\n... (省略中间数据) ...\n\n'
            md += df.tail(5).to_markdown(index=False)

        if n > 1:
            md += '\n\n**描述统计:**\n\n'
            md += df.describe().to_markdown()

        if len(df) <= 1:
            return md

        save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
        os.makedirs(save_dir, exist_ok=True)
        filename = f'chart_{int(time.time() * 1000)}.png'
        save_path = os.path.join(save_dir, filename)
        generate_stock_chart(df, save_path)
        img_path = os.path.join('image_show', filename)
        img_md = f'![图表]({img_path})'
        return f"{md}\n\n{img_md}"

@register_tool('arima_stock')
class ARIMAStockTool(BaseTool):
    description = '使用ARIMA模型对指定股票的未来N天收盘价进行预测'
    parameters = [{
        'name': 'ts_code',
        'type': 'string',
        'description': '股票代码，如600519.SH',
        'required': True
    }, {
        'name': 'n',
        'type': 'integer',
        'description': '预测未来天数，默认5',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        ts_code = args['ts_code']
        n = args.get('n', 5)

        conn = sqlite3.connect(SQLITE_DB_PATH)
        today = datetime.now().date()
        one_year_ago = today - timedelta(days=365)
        query = """
            SELECT trade_date, close, stock_name
            FROM stock_history
            WHERE ts_code = ? AND trade_date >= ?
            ORDER BY trade_date ASC
        """
        df = pd.read_sql_query(query, conn, params=(ts_code, one_year_ago.isoformat()))
        conn.close()

        if df.empty:
            return f"未找到股票 {ts_code} 的历史数据，请检查股票代码是否正确。"

        stock_name = df['stock_name'].iloc[0] if 'stock_name' in df.columns else ts_code
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)

        close_prices = df['close'].values.astype(float)

        if len(close_prices) < 30:
            return f"股票 {ts_code} 的历史数据不足（仅{len(close_prices)}条），无法进行ARIMA建模，至少需要30条数据。"

        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        try:
            model = ARIMA(close_prices, order=(5, 1, 5))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model_fit = model.fit()
            forecast_result = model_fit.forecast(steps=n)
            forecast_values = np.array(forecast_result).flatten()
        except Exception as e:
            return f"ARIMA建模失败: {str(e)}"

        last_date = df['trade_date'].iloc[-1]
        forecast_dates = []
        current = last_date
        for _ in range(n):
            current = current + timedelta(days=1)
            while current.weekday() >= 5:
                current = current + timedelta(days=1)
            forecast_dates.append(current)

        forecast_df = pd.DataFrame({
            '日期': forecast_dates,
            '预测收盘价': [round(v, 2) for v in forecast_values]
        })

        md = f"**{stock_name}({ts_code}) ARIMA(5,1,5) 预测结果**\n\n"
        md += f"- 历史数据范围: {df['trade_date'].iloc[0].strftime('%Y-%m-%d')} ~ {df['trade_date'].iloc[-1].strftime('%Y-%m-%d')} ({len(df)}个交易日)\n"
        md += f"- 预测未来 {n} 个交易日收盘价:\n\n"
        md += forecast_df.to_markdown(index=False)
        md += '\n\n> 注意: ARIMA模型基于历史价格趋势进行统计预测，仅供参考，不构成投资建议。'

        save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
        os.makedirs(save_dir, exist_ok=True)
        filename = f'arima_{int(time.time() * 1000)}.png'
        save_path = os.path.join(save_dir, filename)
        _plot_arima_forecast(df, forecast_dates, forecast_values, stock_name, ts_code, save_path)
        img_path = os.path.join('image_show', filename)
        img_md = f'![预测图表]({img_path})'

        return f"{md}\n\n{img_md}"

@register_tool('boll_detection')
class BollDetectionTool(BaseTool):
    description = '使用布林带(20日周期+2σ)检测股票的超买和超卖日期'
    parameters = [{
        'name': 'ts_code',
        'type': 'string',
        'description': '股票代码，如600519.SH',
        'required': True
    }, {
        'name': 'start_date',
        'type': 'string',
        'description': '检测起始日期，格式YYYY-MM-DD，默认为今天前1年',
        'required': False
    }, {
        'name': 'end_date',
        'type': 'string',
        'description': '检测结束日期，格式YYYY-MM-DD，默认为今天',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        args = json.loads(params)
        ts_code = args['ts_code']
        today = datetime.now().date()
        start_date = args.get('start_date', (today - timedelta(days=365)).isoformat())
        end_date = args.get('end_date', today.isoformat())

        conn = sqlite3.connect(SQLITE_DB_PATH)
        boll_start = (pd.to_datetime(start_date) - timedelta(days=60)).strftime('%Y-%m-%d')
        query = """
            SELECT trade_date, open, high, low, close, vol, stock_name
            FROM stock_history
            WHERE ts_code = ? AND trade_date >= ? AND trade_date <= ?
            ORDER BY trade_date ASC
        """
        df = pd.read_sql_query(query, conn, params=(ts_code, boll_start, end_date))
        conn.close()

        if df.empty:
            return f"未找到股票 {ts_code} 在指定时间范围内的历史数据，请检查股票代码和日期范围。"

        stock_name = df['stock_name'].iloc[0] if 'stock_name' in df.columns else ts_code
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df.sort_values('trade_date').reset_index(drop=True)
        df['close'] = df['close'].astype(float)

        window = 20
        num_std = 2
        df['ma'] = df['close'].rolling(window=window).mean()
        df['std'] = df['close'].rolling(window=window).std()
        df['upper'] = df['ma'] + num_std * df['std']
        df['lower'] = df['ma'] - num_std * df['std']

        df_detect = df[df['trade_date'] >= pd.to_datetime(start_date)].copy()
        df_detect = df_detect.dropna(subset=['ma', 'upper', 'lower']).reset_index(drop=True)

        if df_detect.empty:
            return f"股票 {ts_code} 在 {start_date} ~ {end_date} 范围内数据不足以计算布林带（需要至少{window}个交易日）。"

        overbought = df_detect[df_detect['close'] > df_detect['upper']].copy()
        oversold = df_detect[df_detect['close'] < df_detect['lower']].copy()

        md = f"**{stock_name}({ts_code}) 布林带异常检测报告**\n\n"
        md += f"- 检测周期: {start_date} ~ {end_date}\n"
        md += f"- 布林带参数: {window}日均线 ± {num_std}σ\n"
        md += f"- 分析数据量: {len(df_detect)}个交易日\n\n"

        md += f"### 超买日期（收盘价突破上轨，共{len(overbought)}天）\n\n"
        if overbought.empty:
            md += "无超买信号\n\n"
        else:
            ob_df = overbought[['trade_date', 'close', 'upper']].copy()
            ob_df.columns = ['日期', '收盘价', '上轨']
            ob_df['日期'] = ob_df['日期'].dt.strftime('%Y-%m-%d')
            ob_df['收盘价'] = ob_df['收盘价'].round(2)
            ob_df['上轨'] = ob_df['上轨'].round(2)
            md += ob_df.to_markdown(index=False)
            md += '\n\n'

        md += f"### 超卖日期（收盘价跌破下轨，共{len(oversold)}天）\n\n"
        if oversold.empty:
            md += "无超卖信号\n\n"
        else:
            os_df = oversold[['trade_date', 'close', 'lower']].copy()
            os_df.columns = ['日期', '收盘价', '下轨']
            os_df['日期'] = os_df['日期'].dt.strftime('%Y-%m-%d')
            os_df['收盘价'] = os_df['收盘价'].round(2)
            os_df['下轨'] = os_df['下轨'].round(2)
            md += os_df.to_markdown(index=False)
            md += '\n\n'

        latest = df_detect.iloc[-1]
        md += f"### 最新状态（{latest['trade_date'].strftime('%Y-%m-%d')}）\n\n"
        md += f"- 收盘价: {latest['close']:.2f}\n"
        md += f"- 中轨(MA{window}): {latest['ma']:.2f}\n"
        md += f"- 上轨: {latest['upper']:.2f}\n"
        md += f"- 下轨: {latest['lower']:.2f}\n"
        pct_in_band = (latest['close'] - latest['lower']) / (latest['upper'] - latest['lower']) * 100
        md += f"- 布林带位置: {pct_in_band:.1f}%\n\n"
        md += '> 注意: 布林带超买/超卖仅表示价格偏离均值程度，不构成买卖建议。'

        save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
        os.makedirs(save_dir, exist_ok=True)
        filename = f'boll_{int(time.time() * 1000)}.png'
        save_path = os.path.join(save_dir, filename)
        _plot_bollinger(df_detect, overbought, oversold, stock_name, ts_code, window, num_std, save_path)
        img_path = os.path.join('image_show', filename)
        img_md = f'![布林带图表]({img_path})'

        return f"{md}\n\n{img_md}"

def _plot_bollinger(df, overbought, oversold, stock_name, ts_code, window, num_std, save_path):
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df['trade_date'], df['close'], color='#2196F3', linewidth=1.2, label='收盘价')
    ax.plot(df['trade_date'], df['ma'], color='#FF9800', linewidth=1.2, label=f'MA{window}')
    ax.plot(df['trade_date'], df['upper'], color='#F44336', linewidth=1, linestyle='--', label=f'上轨(+{num_std}σ)')
    ax.plot(df['trade_date'], df['lower'], color='#4CAF50', linewidth=1, linestyle='--', label=f'下轨(-{num_std}σ)')

    ax.fill_between(df['trade_date'], df['upper'], df['lower'], alpha=0.08, color='#FF9800')

    if not overbought.empty:
        ax.scatter(overbought['trade_date'], overbought['close'],
                   color='#F44336', s=50, zorder=5, marker='v', label=f'超买({len(overbought)}天)')
    if not oversold.empty:
        ax.scatter(oversold['trade_date'], oversold['close'],
                   color='#4CAF50', s=50, zorder=5, marker='^', label=f'超卖({len(oversold)}天)')

    ax.set_title(f'{stock_name}({ts_code}) 布林带({window}日±{num_std}σ) 异常检测', fontsize=14)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('价格', fontsize=12)
    ax.legend(loc='best', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    set_sparse_xticks(ax, df['trade_date'].reset_index(drop=True))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def _plot_arima_forecast(hist_df, forecast_dates, forecast_values, stock_name, ts_code, save_path):
    fig, ax = plt.subplots(figsize=(14, 6))

    recent = hist_df.tail(60).copy()
    ax.plot(recent['trade_date'], recent['close'], color='#2196F3', linewidth=1.5, label='历史收盘价')

    ax.plot(forecast_dates, forecast_values, color='#FF5722', linewidth=2, marker='o', markersize=5, label='ARIMA预测')

    last_hist_date = recent['trade_date'].iloc[-1]
    last_hist_close = recent['close'].iloc[-1]
    ax.plot([last_hist_date, forecast_dates[0]], [last_hist_close, forecast_values[0]],
            color='#FF5722', linewidth=2, linestyle='--')

    ax.axvline(x=last_hist_date, color='gray', linestyle=':', alpha=0.7, label='预测起点')

    ax.fill_between(forecast_dates, forecast_values * 0.97, forecast_values * 1.03,
                    color='#FF5722', alpha=0.15, label='预测区间(±3%)')

    ax.set_title(f'{stock_name}({ts_code}) ARIMA(5,1,5) 收盘价预测', fontsize=14)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('收盘价', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    set_sparse_xticks(ax, pd.concat([recent['trade_date'].reset_index(drop=True), pd.Series(forecast_dates)]).reset_index(drop=True))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def generate_stock_chart(df_sql, save_path):
    columns = df_sql.columns.tolist()
    num_columns = df_sql.select_dtypes(include='number').columns.tolist()
    date_columns = []
    for col in columns:
        if df_sql[col].dtype == 'object':
            try:
                pd.to_datetime(df_sql[col])
                date_columns.append(col)
            except (ValueError, TypeError):
                pass
        elif pd.api.types.is_datetime64_any_dtype(df_sql[col]):
            date_columns.append(col)

    x_col = None
    if date_columns:
        x_col = date_columns[0]
    elif columns:
        x_col = columns[0]

    has_stock_name = 'stock_name' in columns or 'stock_name' in [c.lower() for c in columns]
    stock_name_col = None
    for c in columns:
        if c.lower() == 'stock_name':
            stock_name_col = c
            break

    if has_stock_name and x_col and len(num_columns) > 0:
        _plot_multi_series(df_sql, x_col, stock_name_col, num_columns, save_path)
    elif x_col and len(num_columns) > 0:
        _plot_single_series(df_sql, x_col, num_columns, save_path)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, '数据无法自动生成图表', ha='center', va='center', fontsize=16)
        ax.set_title("提示")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def sample_x_axis(df, x_col, max_points=10):
    n = len(df)
    if n <= max_points:
        return df
    indices = np.linspace(0, n-1, max_points, dtype=int)
    return df.iloc[indices].copy()

def set_sparse_xticks(ax, x_values, max_ticks=10):
    n = len(x_values)
    if n <= max_ticks:
        ax.set_xticks(x_values)
        return
    tick_indices = np.linspace(0, n-1, max_ticks, dtype=int)
    ax.set_xticks(x_values.iloc[tick_indices] if hasattr(x_values, 'iloc') else [x_values[i] for i in tick_indices])

def _plot_multi_series(df_sql, x_col, stock_name_col, num_columns, save_path):
    df_plot = df_sql.copy()
    try:
        df_plot[x_col] = pd.to_datetime(df_plot[x_col])
    except (ValueError, TypeError):
        pass

    stock_names = df_plot[stock_name_col].unique()
    plot_cols = [c for c in num_columns if c not in [stock_name_col]]

    if len(plot_cols) == 0:
        return

    primary_col = plot_cols[0]
    is_price = any(kw in primary_col.lower() for kw in ['open', 'high', 'low', 'close', 'pre_close', 'price'])
    
    n = len(df_plot)
    use_line = n > 10 or is_price

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if use_line:
        for name in stock_names:
            subset = df_plot[df_plot[stock_name_col] == name].sort_values(x_col)
            safe_label = str(name).replace('%', '%%').replace('{', '{{').replace('}', '}}')
            ax1.plot(subset[x_col], subset[primary_col], marker='o', markersize=2, label=safe_label)
        ax1.set_ylabel(primary_col)
        ax1.set_title(f"{' / '.join([str(n) for n in stock_names])} - {primary_col}走势")
        if pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        set_sparse_xticks(ax1, df_plot.sort_values(x_col)[x_col].reset_index(drop=True))
    else:
        for name in stock_names:
            subset = df_plot[df_plot[stock_name_col] == name].sort_values(x_col)
            subset_sampled = sample_x_axis(subset, x_col)
            safe_label = str(name).replace('%', '%%').replace('{', '{{').replace('}', '}}')
            ax1.bar(subset_sampled[x_col], subset_sampled[primary_col], label=safe_label, alpha=0.7)
        ax1.set_ylabel(primary_col)
        ax1.set_title(f"{' / '.join([str(n) for n in stock_names])} - {primary_col}对比")

    xlabel_str = str(x_col).replace('%', '%%').replace('{', '{{').replace('}', '}}')
    ax1.set_xlabel(xlabel_str)
    ax1.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def _plot_single_series(df_sql, x_col, num_columns, save_path):
    df_plot = df_sql.copy()
    try:
        df_plot[x_col] = pd.to_datetime(df_plot[x_col])
    except (ValueError, TypeError):
        pass

    n = len(df_plot)
    use_line = n > 10

    price_cols = [c for c in num_columns if any(kw in c.lower() for kw in ['open', 'high', 'low', 'close', 'pre_close', 'price'])]
    vol_cols = [c for c in num_columns if 'vol' in c.lower()]
    other_cols = [c for c in num_columns if c not in price_cols and c not in vol_cols]

    has_dual_axis = len(price_cols) > 0 and len(vol_cols) > 0

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if use_line:
        for col in price_cols:
            safe_label = str(col).replace('%', '%%').replace('{', '{{').replace('}', '}}')
            ax1.plot(df_plot[x_col], df_plot[col], marker='o', markersize=2, label=safe_label)
        ax1.set_ylabel('价格')
        ax1.tick_params(axis='y')

        if has_dual_axis:
            ax2 = ax1.twinx()
            for col in vol_cols:
                safe_label = str(col).replace('%', '%%').replace('{', '{{').replace('}', '}}')
                ax2.bar(df_plot[x_col], df_plot[col], alpha=0.3, label=safe_label, color='gray')
            ax2.set_ylabel('成交量')
            ax2.tick_params(axis='y')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            for col in other_cols:
                safe_label = str(col).replace('%', '%%').replace('{', '{{').replace('}', '}}')
                ax1.plot(df_plot[x_col], df_plot[col], marker='o', markersize=2, label=safe_label)
            ax1.legend()

        set_sparse_xticks(ax1, df_plot.sort_values(x_col)[x_col].reset_index(drop=True))
    else:
        df_sampled = sample_x_axis(df_plot, x_col)
        for col in price_cols:
            safe_label = str(col).replace('%', '%%').replace('{', '{{').replace('}', '}}')
            ax1.bar(df_sampled[x_col], df_sampled[col], alpha=0.7, label=safe_label)
        ax1.set_ylabel('价格')
        ax1.tick_params(axis='y')

        if has_dual_axis:
            ax2 = ax1.twinx()
            for col in vol_cols:
                safe_label = str(col).replace('%', '%%').replace('{', '{{').replace('}', '}}')
                ax2.bar(df_sampled[x_col], df_sampled[col], alpha=0.3, label=safe_label, color='gray')
            ax2.set_ylabel('成交量')
            ax2.tick_params(axis='y')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            for col in other_cols:
                safe_label = str(col).replace('%', '%%').replace('{', '{{').replace('}', '}}')
                ax1.bar(df_sampled[x_col], df_sampled[col], alpha=0.7, label=safe_label)
            ax1.legend()

    xlabel_str = str(x_col).replace('%', '%%').replace('{', '{{').replace('}', '}}')
    ax1.set_xlabel(xlabel_str)
    ax1.set_title("股票行情走势")
    if pd.api.types.is_datetime64_any_dtype(df_plot[x_col]):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def init_agent_service():
    llm_cfg = {
        'model': 'qwen-max',
        'timeout': 30,
        'retry_count': 3,
    }

    function_list = [
        'exc_sql',
        'arima_stock',
        'boll_detection',
        {
            "mcpServers": {
                "tavily-mcp": {
                "args": [
                    "-y",
                    "tavily-mcp@0.1.4"
                ],
                "autoApprove": [],
                "command": "npx",
                "env": {
                    "TAVILY_API_KEY": os.getenv('TAVILY_API_KEY')
                }
                }
            }
        }
    ]

    try:
        bot = Assistant(
            llm=llm_cfg,
            name='股票查询助手',
            description='股票行情查询与分析',
            system_message=system_prompt,
            function_list=function_list,
            files=['./QA.txt']
        )
        print("股票查询助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_tui():
    try:
        bot = init_agent_service()
        messages = []
        while True:
            try:
                query = input('user question: ')
                file = input('file url (press enter if no file): ').strip()
                if not query:
                    print('问题不能为空！')
                    continue
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})
                print("正在处理您的请求...")
                response = []
                for response in bot.run(messages):
                    print('bot response:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")

def app_gui():
    try:
        print("正在启动股票查询助手 Web 界面...")
        bot = init_agent_service()
        chatbot_config = {
            'prompt.suggestions': [
                '查询2025年全年贵州茅台的收盘价走势',
                '统计2025年4月广发证券的日均成交量',
                '对比2025年中芯国际和贵州茅台的涨跌幅',
                '预测贵州茅台未来5天的收盘价',
                '检测贵州茅台最近的超买超卖信号',
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")

if __name__ == '__main__':
    app_gui()
