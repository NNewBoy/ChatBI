import pandas as pd

df = pd.read_excel("e:/AICodeProgram/ChaiBI/stock_history.xlsx", sheet_name="历史价格")

df.sort_values("trade_date", ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

print("=== 前5行数据 ===")
print(df.head().to_string())
print()
print("=== 列名 ===")
print(list(df.columns))
print()
print("=== 各列数据类型 ===")
print(df.dtypes)
print()
print("=== 总行数 ===")
print(len(df))
