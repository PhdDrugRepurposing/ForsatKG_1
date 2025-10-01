import urllib

import pandas as pd
import pyodbc
from sqlalchemy import create_engine

table_name = "Forsat_Chembl_DTI_Raw_Protein_embedding"
join_column = 'protein_sequence'

csv_path = r'models\\local_esm2\\esm2_embeddings_20250929_083525.csv'
query = f"SELECT * FROM dbo.{table_name}"
conn = pyodbc.connect("DRIVER={SQL Server};SERVER=.;DATABASE=ForsatKG;Trusted_Connection=yes;")
conn_str = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=.;DATABASE=ForsatKG;Trusted_Connection=yes;'
params = urllib.parse.quote_plus(conn_str)
engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}', fast_executemany=True)

print(fr'read embedding from: {csv_path}')
df_embedding = pd.read_csv(csv_path, low_memory=False, index_col=None)

df_embedding = df_embedding.rename(columns={'text': join_column})
df_embedding = df_embedding.rename(columns={'embedding': 'esm2'})

print('read sql table')
sql_table_df = pd.read_sql(query, conn)
merged_df = pd.merge(df_embedding, sql_table_df, on=join_column, how='inner')

print('write to Sql...')

merged_df.to_sql(
    name=table_name,
    con=engine,
    if_exists='replace',  # 'replace' اگر بخوای جدول رو حذف و جدید بسازی
    index=False,
    chunksize=10000,
)
print("ذخیره‌سازی با موفقیت انجام شد!")
