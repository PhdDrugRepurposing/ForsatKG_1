import pandas as pd
import networkx as nx
from pyvis.network import Network

# خواندن فایل
df = pd.read_csv("Data/ddi.csv")
df = df.head(5000)

# ایجاد گراف networkx
G = nx.Graph()

# تعریف رنگ‌ها برای روابط
relation_colors = {
    'synergy': 'green',
    'antagonism': 'red',
    'additive': 'gray'
}

# اضافه کردن یال‌ها
for _, row in df.iterrows():
    node1 = row['ChEMBL_id1']
    node2 = row['ChEMBL_id2']
    hsa = row['HSA']
    relation = row['DDI_Relation']

    color = relation_colors.get(relation, 'black')
    width = abs(hsa) / 2 if pd.notna(hsa) else 1

    G.add_edge(node1, node2, color=color, weight=width)

# ساخت pyvis network
net = Network(notebook=False, height="600px", width="100%", bgcolor="white", font_color="black")

# فعال‌سازی فیزیک
net.barnes_hut()

# اضافه کردن گره‌ها بدون نام و کوچک
for node in G.nodes():
    net.add_node(node, label="", size=5, color="lightblue")

# اضافه کردن یال‌ها
for u, v, data in G.edges(data=True):
    net.add_edge(u, v, color=data['color'], width=data['weight'])

#####################
for n, d in G.nodes(data=True):
    node_type = d.get("type", "unknown")
    if node_type == "drug":
        color = "skyblue"
        group = "Drug"
    elif node_type == "protein":
        color = "lightgreen"
        group = "Protein"
    else:
        color = "lightgray"
        group = "Other"

    net.add_node(n, label=str(n), color=color, group=group)

######################

for u, v, d in G.edges(data=True):
    if d["relation"] == "DDI":
        net.add_edge(u, v, color="red", group="DDI")
    elif d["relation"] == "DTI":
        net.add_edge(u, v, color="gray", group="DTI")
    elif d["relation"] == "PPI":
        net.add_edge(u, v, color="purple", group="PPI")


#####################
try:
    net.show("forsat_graph.html")
except Exception as e:
    print("pyvis.show failed:", e)
    html_content = net.generate_html(notebook=False)
    with open("forsat_graph.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("Fell back to generate_html -> forsat_graph.html")
