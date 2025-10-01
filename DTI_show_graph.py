import pyodbc
import networkx as nx
from pyvis.network import Network

# =========================
# اتصال به پایگاه داده
# =========================
conn = pyodbc.connect(
    "DRIVER={SQL Server};SERVER=.;DATABASE=ForsatKG;Trusted_Connection=yes;"
)
cursor = conn.cursor()

# =========================
# ساخت گراف
# =========================
G = nx.Graph()


# --- Drug-Drug interactions (DDI) ---
cursor.execute("SELECT top(1000) ChEMBL_id1, ChEMBL_id2 FROM dbo.Forsat_DDI")
for row in cursor.fetchall():
    G.add_node(row[0], type="drug")
    G.add_node(row[1], type="drug")
    G.add_edge(row[0], row[1], relation="DDI")

# --- Drug-Target interactions (DTI) ---
cursor.execute("SELECT top(10000)  drug_chembl_id, uniprot_id FROM dbo.Forsat_DTI")
for row in cursor.fetchall():
    G.add_node(row[0], type="drug")
    G.add_node(row[1], type="protein")
    G.add_edge(row[0], row[1], relation="DTI")

# --- Protein-Protein interactions (PPI) ---
cursor.execute("SELECT top(1000) protein_A, protein_B FROM dbo.Forsat_PPI")
for row in cursor.fetchall():
    G.add_node(row[0], type="protein")
    G.add_node(row[1], type="protein")
    G.add_edge(row[0], row[1], relation="PPI")

# =========================
# نمایش گراف با PyVis
# =========================
net = Network(notebook=False, directed=False, cdn_resources="in_line")

for n, d in G.nodes(data=True):
    if d["type"] == "drug":
        color = "skyblue"
        group = 1
    else:
        color = "lightgreen"
        group = 2
    net.add_node(n, label=str(n), color=color, group=group)

for u, v, d in G.edges(data=True):
    if d["relation"] == "DDI":
        color = "red"
    elif d["relation"] == "DTI":
        color = "gray"
    elif d["relation"] == "PPI":
        color = "purple"
    else:
        color = "black"
    net.add_edge(u, v, title=d["relation"], color=color)

# ذخیره و باز کردن در مرورگر
#net.show("forsat_graph.html")
# تولید HTML به صورت رشته
html_content = net.generate_html(notebook=False)

# نوشتن فایل با UTF-8
with open("forsat_graph.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ فایل گراف ساخته شد: forsat_graph.html (برای دیدن، در مرورگر باز کنید)")
