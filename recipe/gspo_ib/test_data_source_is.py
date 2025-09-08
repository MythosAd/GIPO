import pandas as pd  
df = pd.read_parquet('/model/data/gsm8k/train.parquet')  
print(df.columns)  
print("-----------------------------------------------------")  
print(df['data_source'].unique() if 'data_source' in df.columns else "No data_source column")