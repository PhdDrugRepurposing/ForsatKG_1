import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc  # یا sqlalchemy برای اتصال به SQL Server


df = pd.read_csv(r"Data\Forsat_DTI.csv")
# --- 3. گزارش‌ها ---

# تعداد دارو و تارگت یکتا
unique_counts = {
    "unique_drugs": df["drug_chembl_id"].nunique(),
    "unique_targets": df["uniprot_id"].nunique()
}
print(unique_counts)

# توزیع interaction_class
interaction_dist = df["interaction_class"].value_counts()
interaction_dist.plot(kind="bar", figsize=(6,4), title="Interaction Class Distribution")
plt.ylabel("Count")
plt.show()


# داروهای پر تکرار
top_drugs = df["drug_name"].value_counts().head(10)
top_drugs.plot(kind="barh", figsize=(6,4), title="Top 10 Drugs by Interaction Count")
plt.xlabel("Interactions")
plt.show()

# تارگت‌های پر تکرار
top_targets = df["target_name"].value_counts().head(10)
top_targets.plot(kind="barh", figsize=(6,4), title="Top 10 Targets by Interaction Count", color="green")
plt.xlabel("Interactions")
plt.show()

# توزیع assay_readout
assay_dist = df["assay_readout"].value_counts()
assay_dist.plot(kind="pie", figsize=(6,6), autopct="%1.1f%%", title="Assay Readout Distribution")
plt.ylabel("")
plt.show()

# Heatmap دارو–تارگت (Top 20x20)
drug_target_matrix = (
    df.groupby(["drug_name", "target_name"]).size().unstack(fill_value=0)
)
subset = drug_target_matrix.iloc[:20, :20]  # محدود به ۲۰ دارو و ۲۰ تارگت اول
plt.figure(figsize=(12,8))
sns.heatmap(subset, cmap="Blues")
plt.title("Drug-Target Interaction Heatmap (Top 20x20)")
plt.show()
