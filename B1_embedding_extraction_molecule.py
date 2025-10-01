import os
import pandas as pd
from sqlalchemy import create_engine
import urllib  # برای encode کردن connection string
import pyodbc

import pandas as pd
import pyodbc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conn = pyodbc.connect("DRIVER={SQL Server};SERVER=.;DATABASE=ForsatKG;Trusted_Connection=yes;")
print(f"device: {device}")
# --------------------------
# 1️⃣ تعریف مدل‌ها و تنظیم فعال/غیرفعال بودن آن‌ها
# --------------------------
models = {
    # پروتئین
    "protbert": {
        "active": False,
        "model_name": "Rostlab/prot_bert",
        "local_dir": "models/local_protbert"
    },
    "esm2": {
        "active": False,
        "model_name": "facebook/esm2_t33_650M_UR50D",
        "local_dir": "models/local_esm2"
    },
    # مولکول (SMILES)
    "chemberta": {
        "active": True,
        "model_name": "seyonec/ChemBERTa-zinc-base-v1",
        "local_dir": "models/local_chemberta"
    },
    # "molbert": {
    #     "active": True,
    #     "model_name": "huggingface/molbert",
    #     "local_dir": "models/local_molbert"
    # },

    "pubmedbert": {
        "active": False,
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "local_dir": "models/local_pubmedbert"
    },
    "biobert": {
        "active": False,
        "model_name": "dmis-lab/biobert-base-cased-v1.2",
        "local_dir": "models/local_biobert"
    },
    "scibert": {
        "active": False,
        "model_name": "allenai/scibert_scivocab_uncased",
        "local_dir": "models/local_scibert"
    },

}


# --------------------------
# 3️⃣ تابع استخراج امبدینگ
# --------------------------


def extract_embeddings(texts, tokenizer, model, device):
    model.to(device)
    model.eval()
    batch_size = 32
    records = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            try:
                batch_texts = texts[i:i + batch_size]
                inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)

                # mean pooling
                last_hidden = outputs.last_hidden_state
                embeddings = last_hidden.mean(dim=1).cpu().tolist()

                for t, e in zip(batch_texts, embeddings):
                    records.append({"text": t, "embedding": e})
            except Exception as ex:
                print(f'ERROR but we continue: {ex}')

    return pd.DataFrame(records)


def run_model(text):
    # --------------------------
    # 4️⃣ اجرای pipeline
    # --------------------------
    for model_key, model_info in models.items():
        if model_info["active"]:
            print(f"\n=== Processing model: {model_key} ===")
            local_dir = model_info["local_dir"]
            model_name = model_info["model_name"]

            # دانلود یا بارگذاری محلی
            if not os.path.exists(local_dir):
                print(f"Downloading {model_key}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                model = AutoModel.from_pretrained(model_name)
                tokenizer.save_pretrained(local_dir)
                model.save_pretrained(local_dir)
            else:
                print(f"Using cached model: {model_key}")
                tokenizer = AutoTokenizer.from_pretrained(local_dir)
                model = AutoModel.from_pretrained(local_dir)

            # استخراج امبدینگ
            embeddings = extract_embeddings(texts, tokenizer, model, device)

            break  # بعد از اولین استخراج امبدینگ از حلقه خارج می شود و برنامه تمام می شود
    return embeddings, local_dir, model_key


def load_molecules():
    conn = pyodbc.connect("DRIVER={SQL Server};SERVER=.;DATABASE=ForsatKG;Trusted_Connection=yes;")

    query = ("SELECT distinct drug_chembl_id, [canonical_smiles]  FROM [ForsatKG].[dbo].["
             "Forsat_Chembl_DTI_Raw_Drug_embedding]")

    df_prot = pd.read_sql(query, conn)
    conn.close()
    return df_prot


if __name__ == '__main__':
    from datetime import datetime

    df = load_molecules()

    texts = list(df['canonical_smiles'].values)

    embeddings, local_dir, model_key = run_model(texts)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(local_dir, f"{model_key}_embeddings_{timestamp}.csv")

    embeddings.to_csv(csv_path, index=False)
    print(f"Embeddings saved: {csv_path},\r\nshape: {embeddings.shape}")
    ###############################################################
