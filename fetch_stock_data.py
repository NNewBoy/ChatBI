import os
import tushare as ts
import pandas as pd
from datetime import datetime

ts.set_token(os.environ.get('TUSHARE_TOKEN'))
pro = ts.pro_api()

stocks = {
    "600519": "贵州茅台",
    "000858": "五粮液",
    "000776": "广发证券",
    "688981": "中芯国际",
}

end_date = datetime.now().strftime("%Y%m%d")

frames = []
for code, name in stocks.items():
    df = pro.daily(ts_code=f"{code}.SH" if code.startswith("6") else f"{code}.SZ",
                   start_date="20200101", end_date=end_date)
    df["股票名称"] = name
    frames.append(df)

result = pd.concat(frames, ignore_index=True)
result["trade_date"] = pd.to_datetime(result["trade_date"], format="%Y%m%d")
result.sort_values("trade_date", ascending=True, inplace=True)
result.reset_index(drop=True, inplace=True)

result.to_excel("stock_history.xlsx", index=False, sheet_name="历史价格")
print(f"已保存 {len(result)} 条记录到 stock_history.xlsx")
