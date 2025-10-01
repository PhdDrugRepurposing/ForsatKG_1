import pandas as pd

# =========================
# خواندن فایل ddi.csv
# =========================
from matplotlib import pyplot as plt

df = pd.read_csv("Data/ddi.csv")

# تبدیل ستون HSA به float
df['HSA'] = pd.to_numeric(df['HSA'], errors='coerce')

# =========================
# به‌روزرسانی ستون DDI_Relation بر اساس مقدار HSA
# =========================
def update_ddi_relation(hsa):
    if pd.isna(hsa):
        return None
    elif hsa >= 0:
        return 'synergy'
    elif hsa < 0:
        return 'antagonism'


df['DDI_Relation'] = df['HSA'].apply(update_ddi_relation)

# =========================
# نمایش نمونه داده‌ها
# =========================
print(df[['HSA', 'DDI_Relation']].head())

# =========================
# ذخیره مجدد فایل
# =========================
df.to_csv("ddi.csv", index=False)
print("ستون DDI_Relation بر اساس HSA به‌روزرسانی شد و در 'ddi_updated.csv' ذخیره شد.")






# =========================
# شمارش تعداد هر دسته
# =========================
counts = df['DDI_Relation'].value_counts()
print("تعداد هر DDI_Relation:")
for relation, count in counts.items():
    print(f"{relation}: {count:,}")

# =========================
# رسم هیستوگرام
# =========================
plt.figure(figsize=(8,5))

colors = ['orange', 'skyblue', 'green']

# رسم هر دسته با برچسب
bars = plt.bar(counts.index, counts.values, color=colors,
               label=counts.index)

# اضافه کردن تعداد روی هر ستون
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{int(height):,}',
             ha='center', va='bottom', fontsize=12)

plt.xlabel('DDI_Relation')
plt.ylabel('#')
plt.title('DDI_Relation')
plt.legend(title="Relations")
plt.show()