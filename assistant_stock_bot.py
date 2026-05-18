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

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30

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

        db_user = "root"
        db_password = "cute1nan"
        db_host = "localhost:3306"
        db_name = "ai"
        engine = create_engine(
            f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}',
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        df = pd.read_sql(text(sql_input), engine)
        print('df=', df)

        if session_id:
            _last_df_dict[session_id] = df

        md = df.head(10).to_markdown(index=False)
        save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
        os.makedirs(save_dir, exist_ok=True)
        filename = f'chart_{int(time.time() * 1000)}.png'
        save_path = os.path.join(save_dir, filename)
        generate_stock_chart(df, save_path)
        img_path = os.path.join('image_show', filename)
        img_md = f'![图表]({img_path})'
        return f"{md}\n\n{img_md}"

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

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if is_price:
        for name in stock_names:
            subset = df_plot[df_plot[stock_name_col] == name].sort_values(x_col)
            safe_label = str(name).replace('%', '%%').replace('{', '{{').replace('}', '}}')
            ax1.plot(subset[x_col], subset[primary_col], marker='o', markersize=2, label=safe_label)
        ax1.set_ylabel(primary_col)
        ax1.set_title(f"{' / '.join([str(n) for n in stock_names])} - {primary_col}走势")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    else:
        for name in stock_names:
            subset = df_plot[df_plot[stock_name_col] == name].sort_values(x_col)
            safe_label = str(name).replace('%', '%%').replace('{', '{{').replace('}', '}}')
            ax1.bar(subset[x_col], subset[primary_col], label=safe_label, alpha=0.7)
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

    fig, ax1 = plt.subplots(figsize=(12, 6))

    price_cols = [c for c in num_columns if any(kw in c.lower() for kw in ['open', 'high', 'low', 'close', 'pre_close', 'price'])]
    vol_cols = [c for c in num_columns if 'vol' in c.lower()]
    other_cols = [c for c in num_columns if c not in price_cols and c not in vol_cols]

    has_dual_axis = len(price_cols) > 0 and len(vol_cols) > 0

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
        'model': 'qwen-turbo',
        'timeout': 30,
        'retry_count': 3,
    }
    try:
        bot = Assistant(
            llm=llm_cfg,
            name='股票查询助手',
            description='股票行情查询与分析',
            system_message=system_prompt,
            function_list=['exc_sql'], # 'code_interpreter'
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
