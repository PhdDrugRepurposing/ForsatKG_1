"""
==========================================
Forsat DDI Dataset - Cleaned HSA Values
==========================================

توضیح فایل:
- این فایل از جدول dbo.Forsat_DDI پایگاه داده ForsatKG استخراج شده است.
- ستون HSA (Highest Single Agent) برای سنجش اثر ترکیبی داروها استفاده می‌شود.
- داده‌های پرت HSA با استفاده از روش IQR شناسایی و حذف شده‌اند.
- تمام ردیف‌ها شامل مقادیر HSA معتبر (float) هستند.
- فایل خروجی 'ddi.csv' شامل تمام ستون‌های اصلی جدول است، بدون داده‌های پرت.
- گزارش آماری شامل تعداد داده‌ها، تعداد و درصد داده‌های حذف شده، حداقل، حداکثر، میانگین و میانه پس از حذف پرت‌ها تهیه شده است.
- نمودار هیستوگرام HSA پس از حذف پرت‌ها رسم شده است.

Author: Your Name
Date: YYYY-MM-DD
"""


import pandas as pd
import pyodbc
import networkx as nx
from matplotlib import pyplot as plt
from pyvis.network import Network

# =========================
# اتصال به پایگاه داده
# =========================
conn = pyodbc.connect(
    "DRIVER={SQL Server};SERVER=.;DATABASE=ForsatKG;Trusted_Connection=yes;"
)

query = ("SELECT * FROM dbo.Forsat_DDI")

df = pd.read_sql(query, conn)
conn.close()





df['HSA'] = pd.to_numeric(df['HSA'], errors='coerce')  # مقادیر غیرقابل تبدیل به NaN می‌شوند

# بررسی نوع داده
print(df['HSA'].dtype)


hsa_values = df['HSA'].dropna()

# بررسی نوع داده
print("نوع داده ستون HSA:", hsa_values.dtype)
hsa_values.std()
# بررسی وجود مقادیر مثبت و منفی
has_positive = (hsa_values > 0).any()
has_negative = (hsa_values < 0).any()

print("آیا مقادیر مثبت وجود دارد؟", has_positive)
print("آیا مقادیر منفی وجود دارد؟", has_negative)
plt.figure(figsize=(8,5))
plt.hist(df['HSA'].dropna(), bins=100, color='skyblue', edgecolor='black')

plt.title('Histogram of HSA')
plt.xlabel('HSA value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
#########################################################

import pandas as pd
import pyodbc


# =========================
# تبدیل ستون HSA به float
# =========================
df['HSA'] = pd.to_numeric(df['HSA'], errors='coerce')

# حذف مقادیر NaN برای محاسبه IQR
hsa_values = df['HSA'].dropna()

# =========================
# شناسایی داده‌های پرت با IQR
# =========================
Q1 = hsa_values.quantile(0.25)
Q3 = hsa_values.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = hsa_values[(hsa_values < lower_bound) | (hsa_values > upper_bound)]

# =========================
# حذف داده‌های پرت از DataFrame اصلی
# =========================
df_clean = df[(df['HSA'] >= lower_bound) & (df['HSA'] <= upper_bound)]
hsa_clean = df_clean['HSA']

# =========================
# رسم هیستوگرام داده‌های پاک شده
# =========================
plt.figure(figsize=(10,6))
plt.hist(hsa_clean, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of HSA (Outliers Removed)')
plt.xlabel('HSA value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# =========================
# گزارش کامل پس از حذف پرت‌ها
# =========================
report = {
    "Total_data_points": len(hsa_values),
    "Number_of_outliers_removed": len(outliers),
    "Percentage_of_outliers_removed": len(outliers)/len(hsa_values)*100,
    "Min_value_after_removal": hsa_clean.min(),
    "Max_value_after_removal": hsa_clean.max(),
    "Mean_after_removal": hsa_clean.mean(),
    "Median_after_removal": hsa_clean.median()
}

print("=== Outlier Removal Report ===")
for k, v in report.items():
    print(f"{k}: {v}")

# =========================
# ذخیره DataFrame پاک شده به CSV
# =========================
df_clean.to_csv("ddi.csv", index=False)
print(f"\nDataFrame بدون داده‌های پرت در فایل 'ddi.csv' ذخیره شد.")
print(f"تعداد ردیف‌ها قبل از حذف پرت‌ها: {len(df)}")
print(f"تعداد ردیف‌ها بعد از حذف پرت‌ها: {len(df_clean)}")