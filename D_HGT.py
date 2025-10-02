# D_HGT_full_with_visuals.py
# Put OpenMP workaround before heavy imports
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

# Standard imports
import random
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto
import itertools
import pandas as pd  # for printing param table nicely

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_scatter import scatter_add, scatter_softmax

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# deterministic seeds (optional)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# -----------------------
# Task enum
# -----------------------
class TaskType(Enum):
    LINK_PREDICTION = auto()
    NODE_PREDICTION = auto()
    WEIGHT_PREDICTION = auto()


# -----------------------
# NegativeSampler class
# -----------------------
class NegativeSampler:
    """Negative sampling strategies."""

    @staticmethod
    def uniform_random(num_samples, num_nodes, device):
        """Uniform random negative sampling."""
        return torch.randint(0, num_nodes, (num_samples,), device=device)

    @staticmethod
    def degree_based(edge_index, num_samples, num_nodes, device):
        """Degree-based negative sampling: probability proportional to node degree."""
        if edge_index.numel() == 0:
            return NegativeSampler.uniform_random(num_samples, num_nodes, device)
        # compute degree on same device
        src = edge_index[0]
        dst = edge_index[1]
        idxs = torch.cat([src, dst], dim=0)
        deg = torch.bincount(idxs, minlength=num_nodes).float()
        prob = deg + 1e-6
        prob = prob / prob.sum()
        return torch.multinomial(prob, num_samples, replacement=True)

    @staticmethod
    def fixed_set(num_samples, fixed_tensor):
        """Return elements from a precomputed fixed negative set."""
        if fixed_tensor.numel() == 0:
            raise ValueError("fixed_tensor is empty")
        idx = torch.randint(0, fixed_tensor.size(0), (num_samples,), device=fixed_tensor.device)
        return fixed_tensor[idx]


# -----------------------
# GraphProcessor
# -----------------------
class GraphProcessor:
    """
    Build and preprocess HeteroData. Provide splits and move everything to device.
    """

    def __init__(self, hetero_data: HeteroData = None):
        self.data = hetero_data

    @classmethod
    def from_synthetic(cls, num_papers=800, num_authors=200, num_venues=20, feat_dim=64, seed=0):
        """Create a synthetic heterogeneous graph."""
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        data = HeteroData()
        data['paper'].x = torch.randn(num_papers, feat_dim)
        data['author'].x = torch.randn(num_authors, feat_dim)
        data['venue'].x = torch.randn(num_venues, feat_dim)

        # author -> paper (writes)
        src = []; dst = []
        for p in range(num_papers):
            k = random.randint(1, 3)
            authors = np.random.choice(num_authors, k, replace=False)
            for a in authors:
                src.append(int(a)); dst.append(int(p))
        edge_index = np.vstack([np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)])
        data['author','writes','paper'].edge_index = torch.tensor(edge_index, dtype=torch.long)

        # paper -> venue (published_in)
        src = np.arange(num_papers, dtype=np.int64)
        dst = np.random.choice(num_venues, num_papers).astype(np.int64)
        edge_index = np.vstack([src, dst])
        data['paper','published_in','venue'].edge_index = torch.tensor(edge_index, dtype=torch.long)

        # paper -> paper (cites)
        num_cites = num_papers * 2
        src = np.random.choice(num_papers, num_cites).astype(np.int64)
        dst = np.random.choice(num_papers, num_cites).astype(np.int64)
        edge_index = np.vstack([src, dst])
        data['paper','cites','paper'].edge_index = torch.tensor(edge_index, dtype=torch.long)

        return cls(hetero_data=data)

    @classmethod
    def from_edge_lists(cls, node_features: dict, edge_lists: dict):
        """Build HeteroData from provided node features and edge lists."""
        data = HeteroData()
        for ntype, feat in node_features.items():
            if isinstance(feat, np.ndarray):
                data[ntype].x = torch.tensor(feat, dtype=torch.float)
            elif isinstance(feat, torch.Tensor):
                data[ntype].x = feat.float()
            else:
                raise ValueError("node_features values must be numpy.ndarray or torch.Tensor")
        for (src_type, rel, dst_type), (src_arr, dst_arr) in edge_lists.items():
            src_np = np.array(src_arr, dtype=np.int64)
            dst_np = np.array(dst_arr, dtype=np.int64)
            edge_index = np.vstack([src_np, dst_np])
            data[src_type, rel, dst_type].edge_index = torch.tensor(edge_index, dtype=torch.long)
        return cls(hetero_data=data)

    def to_device(self, device):
        """
        Move all tensors to device including train/val/test indices.
        """
        for ntype in self.data.node_types:
            if hasattr(self.data[ntype], 'x'):
                self.data[ntype].x = self.data[ntype].x.to(device)
        for et in self.data.edge_types:
            if hasattr(self.data[et], 'edge_index'):
                self.data[et].edge_index = self.data[et].edge_index.to(device)
            rel_data = self.data[et]
            for attr in ['train_idx', 'val_idx', 'test_idx', 'train_nodes', 'val_nodes', 'test_nodes']:
                if hasattr(rel_data, attr):
                    setattr(rel_data, attr, getattr(rel_data, attr).to(device))

    def get_x_dict(self):
        return {nt: self.data[nt].x for nt in self.data.node_types}

    def get_edge_index_dict(self):
        return {et: self.data[et].edge_index for et in self.data.edge_types}

    def get_pos_edges(self, relation):
        if relation not in self.data.edge_types:
            raise KeyError(f"{relation} not in graph")
        ei = self.data[relation].edge_index
        return ei[0], ei[1]

    def num_nodes(self, ntype):
        return self.data[ntype].x.size(0)

    # Splitting methods
    def split_edges_random(self, relation=('paper','cites','paper'), val_ratio=0.1, test_ratio=0.1, seed=0):
        """Edge-based random split (with leakage)."""
        src, dst = self.get_pos_edges(relation)
        num = src.size(0)
        perm = torch.randperm(num, generator=torch.Generator().manual_seed(seed))
        n_val = int(num * val_ratio)
        n_test = int(num * test_ratio)
        n_train = num - n_val - n_test
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]
        self.data[relation].train_idx = train_idx
        self.data[relation].val_idx = val_idx
        self.data[relation].test_idx = test_idx
        return train_idx, val_idx, test_idx

    def split_edges_node_inductive(self, relation=('paper','cites','paper'), node_val_ratio=0.1, node_test_ratio=0.1, seed=0):
        """Node-based inductive split (no leakage)."""
        src, dst = self.get_pos_edges(relation)
        num_nodes = self.num_nodes(relation[0])
        perm_nodes = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed))
        n_val_nodes = int(num_nodes * node_val_ratio)
        n_test_nodes = int(num_nodes * node_test_ratio)
        n_train_nodes = num_nodes - n_val_nodes - n_test_nodes
        train_nodes = perm_nodes[:n_train_nodes]
        val_nodes = perm_nodes[n_train_nodes:n_train_nodes + n_val_nodes]
        test_nodes = perm_nodes[n_train_nodes + n_val_nodes:]

        device = src.device
        train_mask_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask_nodes[train_nodes] = True
        val_mask_nodes[val_nodes] = True
        test_mask_nodes[test_nodes] = True

        train_edge_mask = train_mask_nodes[src] & train_mask_nodes[dst]
        val_edge_mask = val_mask_nodes[src] & val_mask_nodes[dst]
        test_edge_mask = test_mask_nodes[src] & test_mask_nodes[dst]

        train_idx = train_edge_mask.nonzero(as_tuple=False).view(-1)
        val_idx = val_edge_mask.nonzero(as_tuple=False).view(-1)
        test_idx = test_edge_mask.nonzero(as_tuple=False).view(-1)

        self.data[relation].train_idx = train_idx
        self.data[relation].val_idx = val_idx
        self.data[relation].test_idx = test_idx

        self.data[relation].train_nodes = train_nodes
        self.data[relation].val_nodes = val_nodes
        self.data[relation].test_nodes = test_nodes

        return train_idx, val_idx, test_idx, (train_nodes, val_nodes, test_nodes)


# -----------------------
# RelationAttention
# -----------------------
class RelationAttention(nn.Module):
    """
    Relation-specific projection and attention.
    Keywords (as comments) used:
    - Relation-specific projection
    - Relation-specific Attention
    - Normalized Relation-specific Attention
    - Intra-Type Aggregation
    """
    def __init__(self, out_dim, n_heads=4, relation_name="rel"):
        super().__init__()
        self.out_dim = out_dim
        self.n_heads = n_heads
        assert out_dim % n_heads == 0
        self.d_k = out_dim // n_heads
        self.relation_name = relation_name

        # Relation-specific projection for K and V
        self.rel_k = nn.Linear(out_dim, out_dim, bias=False)
        self.rel_v = nn.Linear(out_dim, out_dim, bias=False)

        # Optional weight prediction head (for weight-prediction downstream)
        self.weight_predictor = nn.Sequential(nn.Linear(out_dim*2, out_dim), nn.ReLU(), nn.Linear(out_dim,1))

    def forward(self, h_src, Q_dst, src_idx, dst_idx, num_dst_nodes):
        # h_src: [E, out_dim]  (source node embeddings per edge)
        # Q_dst: [E, heads, d_k]  (queries of destination nodes per edge)

        E = h_src.size(0)

        # Relation-specific projection -> K, V
        K = self.rel_k(h_src).view(-1, self.n_heads, self.d_k)
        V = self.rel_v(h_src).view(-1, self.n_heads, self.d_k)

        # Relation-specific Attention: dot(Q, K) / sqrt(d_k)
        attn_scores = (Q_dst * K).sum(dim=-1) / (self.d_k ** 0.5)  # [E, heads]

        # Normalized Relation-specific Attention: softmax per destination node
        heads = attn_scores.size(1)
        dst_idx_repeat = dst_idx.unsqueeze(1).repeat(1, heads).view(-1)
        attn_flat = attn_scores.view(-1)
        attn_norm_flat = scatter_softmax(attn_flat, dst_idx_repeat)
        attn_norm = attn_norm_flat.view(E, heads)

        # Intra-Type Aggregation: weight V by attn and aggregate per destination node
        weighted_V = V * attn_norm.unsqueeze(-1)
        weighted_V = weighted_V.view(E, self.out_dim)
        aggregated = scatter_add(weighted_V, dst_idx, dim=0, dim_size=num_dst_nodes)

        # Weight prediction (optional)
        dst_rep = Q_dst.view(E, self.out_dim).detach()
        weight_preds = self.weight_predictor(torch.cat([h_src, dst_rep], dim=1)).squeeze(-1)

        return aggregated, attn_norm, weight_preds


# -----------------------
# HGTLayer
# -----------------------
class HGTLayer(nn.Module):
    """
    HGTLayer: performs
    - Initial embedding (Type-specific projection)
    - calls RelationAttention per relation
    - Inter-Type aggregation
    - Update (skip + LayerNorm + ReLU)
    """
    def __init__(self, node_types, edge_types, in_dim, out_dim, n_heads=4, dropout=0.2):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.out_dim = out_dim
        self.n_heads = n_heads
        assert out_dim % n_heads == 0
        self.d_k = out_dim // n_heads

        # Type-specific projection (Initial embedding)
        self.type_proj = nn.ModuleDict({ntype: nn.Linear(in_dim, out_dim) for ntype in node_types})

        # Relation-specific modules (RelationAttention)
        self.rel_attention = nn.ModuleDict()
        for (src, rel, dst) in edge_types:
            name = f"{src}__{rel}__{dst}"
            self.rel_attention[name] = RelationAttention(out_dim=out_dim, n_heads=n_heads, relation_name=name)

        # Update layers
        self.update_lin = nn.ModuleDict({ntype: nn.Linear(out_dim, out_dim) for ntype in node_types})
        self.layer_norm = nn.ModuleDict({ntype: nn.LayerNorm(out_dim) for ntype in node_types})
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # x_dict: raw node features per type
        # Type-specific projection -> common hidden space
        h = {}
        for ntype, x in x_dict.items():
            h[ntype] = self.type_proj[ntype](x)

        # prepare inter-type aggregator
        inter_agg = {ntype: h[ntype].new_zeros(h[ntype].size(0), self.out_dim) for ntype in h}

        # Precompute Q for each dst type
        Q = {ntype: h[ntype].view(-1, self.n_heads, self.d_k) for ntype in h}

        # For each relation perform RelationAttention and accumulate (Inter-Type aggregation)
        for (src_type, rel, dst_type), edge_index in edge_index_dict.items():
            name = f"{src_type}__{rel}__{dst_type}"
            if name not in self.rel_attention:
                continue
            src_idx = edge_index[0]
            dst_idx = edge_index[1]
            if src_idx.numel() == 0:
                continue

            # source embeddings per edge
            h_src = h[src_type][src_idx]
            Q_dst = Q[dst_type][dst_idx]
            num_dst_nodes = h[dst_type].size(0)

            aggregated, attn_norm, weight_preds = self.rel_attention[name](h_src, Q_dst, src_idx, dst_idx, num_dst_nodes)

            # Inter-Type aggregation: sum messages from different relations
            inter_agg[dst_type] = inter_agg[dst_type] + aggregated

        # Update representations per node type
        out = {}
        for ntype in h:
            updated = self.update_lin[ntype](inter_agg[ntype])
            res = self.layer_norm[ntype](h[ntype] + self.dropout(updated))
            out[ntype] = F.relu(res)
        return out


# -----------------------
# HGTForLinkPred (Encoder + Scorer)
# -----------------------
class HGTForLinkPred(nn.Module):
    """
    Encoder: stack of HGTLayer(s)
    Scorer: downstream MLP for relation/weight prediction
    """
    def __init__(self, node_types, edge_types, in_dim=64, hid_dim=128, n_heads=4, n_layers=2, downstream_task=TaskType.LINK_PREDICTION):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.downstream_task = downstream_task
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(HGTLayer(node_types, edge_types, in_dim if i == 0 else hid_dim, hid_dim, n_heads=n_heads))
        if downstream_task == TaskType.LINK_PREDICTION:
            # Scorer (MLP) for relation-prediction
            self.scorer = nn.Sequential(nn.Linear(hid_dim*2, hid_dim), nn.ReLU(), nn.Linear(hid_dim,1))
        else:
            # Scorer for regression/weight-prediction
            self.scorer = nn.Sequential(nn.Linear(hid_dim*2, hid_dim), nn.ReLU(), nn.Linear(hid_dim,1))

    def forward(self, x_dict, edge_index_dict):
        # Encoder: stack HGT layers -> final embeddings
        h = x_dict
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        return h

    def score_edges(self, h_dict, edge_list):
        # Scorer combines hs and hd (concat) -> MLP -> logit/regression output
        scores = []
        for (src_type, dst_type, src_idx, dst_idx) in edge_list:
            hs = h_dict[src_type][src_idx]
            hd = h_dict[dst_type][dst_idx]
            inp = torch.cat([hs, hd], dim=1)
            s = self.scorer(inp).squeeze(-1)
            scores.append(s)
        return torch.cat(scores, dim=0)


# -----------------------
# HGTTrainer
# -----------------------
class HGTTrainer:
    """
    Trainer class: holds model, optimizer, train loop, and snapshot mechanism.
    """
    def __init__(self, data_processor: GraphProcessor,
                 device=None,
                 hid_dim=128, n_heads=4, n_layers=2,
                 lr=5e-3, weight_decay=1e-4,
                 downstream_task=TaskType.LINK_PREDICTION):
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        print(f"[Device] {self.device}")
        self.dp = data_processor
        self.dp.to_device(self.device)
        self.node_types = list(self.dp.data.node_types)
        self.edge_types = list(self.dp.data.edge_types)
        feat_dim = self.dp.get_x_dict()[self.node_types[0]].size(1)

        # Model (Encoder + Scorer)
        self.model = HGTForLinkPred(self.node_types, self.edge_types, in_dim=feat_dim,
                                   hid_dim=hid_dim, n_heads=n_heads, n_layers=n_layers,
                                   downstream_task=downstream_task).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.downstream_task = downstream_task
        if downstream_task == TaskType.LINK_PREDICTION:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()

        # positive edges for link-prediction
        self.relation = ('paper','cites','paper')
        self.pos_src, self.pos_dst = self.dp.get_pos_edges(self.relation)
        self.num_pos = self.pos_src.size(0)

        # bookkeeping
        self.losses = []
        # snapshots: to store embeddings before/after training
        self.embedding_snapshots = {}

        # print model summary and param table
        self.print_model_summary()
        self.print_param_table()

    # Model summary: ASCII + counts + small bar chart
    def print_model_summary(self):
        n_layers = len(self.model.layers)
        hid = self.model.layers[0].out_dim
        heads = self.model.layers[0].n_heads
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for layer in self.model.layers for p in layer.parameters() if p.requires_grad)
        scorer_params = sum(p.numel() for p in self.model.scorer.parameters() if p.requires_grad)

        print("===== Model Architecture Summary (Encoder) =====")
        print(f"Encoder: {n_layers:,} HGTLayer(s)")
        print(f"Hidden dim (out_dim): {hid:,}")
        print(f"Heads per layer: {heads:,}")
        print(f"Trainable parameters (total): {total_params:,}")
        print("Per-module trainable params:")
        print(f"  - Encoder layers total: {encoder_params:,}")
        print(f"  - Scorer head: {scorer_params:,}")
        print("Architecture diagram (ASCII):")
        for i, layer in enumerate(self.model.layers, 1):
            print(f" Layer {i:02d} [HGTLayer]  -> heads: {layer.n_heads:,}  out_dim: {layer.out_dim:,}")
        print("===============================================")

        # small bar chart
        modules = ['Encoder', 'Scorer']
        values = [encoder_params, scorer_params]
        plt.figure(figsize=(5,3))
        bars = plt.bar(modules, values)
        for i, v in enumerate(values):
            plt.text(i, v + max(values)*0.01, f"{v:,}", ha='center')
        plt.title("Trainable params per module")
        plt.tight_layout()
        plt.show()

    def print_param_table(self):
        """Print a table with parameter counts per submodule."""
        rows = []
        for name, module in self.model.named_children():
            # top-level modules: layers (ModuleList) and scorer
            if isinstance(module, nn.ModuleList):
                # enumerate layers
                for i, layer in enumerate(module):
                    cnt = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    rows.append({'module': f'layer_{i+1}', 'params': cnt})
            else:
                cnt = sum(p.numel() for p in module.parameters() if p.requires_grad)
                rows.append({'module': name, 'params': cnt})
        df = pd.DataFrame(rows)
        df['params_str'] = df['params'].apply(lambda x: f"{x:,}")
        print("Parameter counts per top-level submodule:")
        print(df[['module', 'params_str']].to_string(index=False))

    def plot_model_diagram(self, save_path=None):
        """Plot a block diagram of the model showing layers, heads, hid_dim, and params."""
        n_layers = len(self.model.layers)
        hid = self.model.layers[0].out_dim
        heads = self.model.layers[0].n_heads
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        fig, ax = plt.subplots(figsize=(8, 2 + n_layers*0.5))
        ax.axis('off')
        x = 0.1
        y = 0.8
        width = 0.6
        height = 0.12
        for i in range(n_layers):
            rect = plt.Rectangle((x, y - i*(height+0.05)), width, height, fill=True, edgecolor='black', facecolor='#cce5ff')
            ax.add_patch(rect)
            ax.text(x + 0.02, y - i*(height+0.05) + height/2, f"HGTLayer {i+1}: heads={heads}, out_dim={hid:,}", va='center', fontsize=9)
        # scorer box
        rect_s = plt.Rectangle((x + 0.75, y - (n_layers//2)*(height+0.05)), width*0.6, height, fill=True, edgecolor='black', facecolor='#ffdfba')
        ax.add_patch(rect_s)
        ax.text(x + 0.76, y - (n_layers//2)*(height+0.05) + height/2, f"Scorer\nparams={sum(p.numel() for p in self.model.scorer.parameters() if p.requires_grad):,}", va='center', fontsize=9)
        ax.text(0.02, 0.95, f"Total trainable params: {total_params:,}", fontsize=10)
        ax.set_xlim(0, 1.4)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    # snapshot embeddings helper
    def snapshot_embeddings(self, label='before'):
        """
        Compute and store node embeddings (Encoder outputs) under a label.
        Use later for t-SNE comparison.
        """
        self.model.eval()
        with torch.no_grad():
            x_dict = self.dp.get_x_dict()
            edge_index_dict = self.dp.get_edge_index_dict()
            h = self.model(x_dict, edge_index_dict)
            # store CPU numpy arrays
            snapshot = {nt: h[nt].cpu().numpy() for nt in h}
            self.embedding_snapshots[label] = snapshot
        print(f"Saved embedding snapshot '{label}'.")

    def get_embeddings(self):
        """Return current embeddings dict (tensors on device)."""
        self.model.eval()
        with torch.no_grad():
            x_dict = self.dp.get_x_dict()
            edge_index_dict = self.dp.get_edge_index_dict()
            h = self.model(x_dict, edge_index_dict)
        return h

    def sample_negative(self, num_samples, strategy='uniform', **kwargs):
        num_nodes = self.dp.num_nodes('paper')
        if strategy == 'uniform':
            return NegativeSampler.uniform_random(num_samples, num_nodes, self.device)
        elif strategy == 'degree':
            edge_index = self.dp.data[self.relation].edge_index
            return NegativeSampler.degree_based(edge_index, num_samples, num_nodes, self.device)
        elif strategy == 'fixed':
            if 'fixed_tensor' not in kwargs:
                raise ValueError("fixed_tensor required")
            return NegativeSampler.fixed_set(num_samples, kwargs['fixed_tensor'])
        else:
            raise ValueError("Unknown negative sampling strategy")

    def train_one_epoch(self, batch_size=1024, neg_strategy='uniform'):
        """One training epoch: compute logits via scorer, compute loss, update weights."""
        # Encoder -> embeddings
        self.model.train()
        self.opt.zero_grad()
        x_dict = self.dp.get_x_dict()
        edge_index_dict = self.dp.get_edge_index_dict()
        h = self.model(x_dict, edge_index_dict)

        # sample positives and negatives
        bs = min(batch_size, self.num_pos)
        perm = torch.randperm(self.num_pos, device=self.device)[:bs]
        pos_s = self.pos_src[perm]; pos_d = self.pos_dst[perm]
        neg_d = self.sample_negative(bs, strategy=neg_strategy)

        # scorer -> logits
        s_pos = self.model.score_edges(h, [('paper','paper', pos_s, pos_d)])
        s_neg = self.model.score_edges(h, [('paper','paper', pos_s, neg_d)])

        if self.downstream_task == TaskType.LINK_PREDICTION:
            labels = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)], dim=0)
            logits = torch.cat([s_pos, s_neg], dim=0)
            loss = self.loss_fn(logits, labels)
        else:
            pos_targets = torch.rand_like(s_pos)
            neg_targets = torch.zeros_like(s_neg)
            preds = torch.cat([s_pos, s_neg], dim=0)
            targets = torch.cat([pos_targets, neg_targets], dim=0)
            loss = self.loss_fn(preds, targets)

        # Update weights
        loss.backward()
        self.opt.step()

        self.losses.append(loss.item())
        return loss.item(), s_pos.mean().item()

    def validate_proxy(self, val_n=4096):
        """Simple proxy validation: compare pos vs neg mean."""
        self.model.eval()
        with torch.no_grad():
            val_n = min(val_n, self.num_pos)
            perm = torch.randperm(self.num_pos, device=self.device)[:val_n]
            v_s = self.pos_src[perm]; v_d = self.pos_dst[perm]
            v_neg_d = self.sample_negative(val_n, strategy='uniform')
            x_dict = self.dp.get_x_dict(); edge_index_dict = self.dp.get_edge_index_dict()
            h = self.model(x_dict, edge_index_dict)
            v_pos = torch.sigmoid(self.model.score_edges(h, [('paper','paper', v_s, v_d)]))
            v_neg = torch.sigmoid(self.model.score_edges(h, [('paper','paper', v_s, v_neg_d)]))
            acc = (v_pos > v_neg).float().mean().item()
        return acc

    def train(self, epochs=100, batch_size=1024, log_interval=5, neg_strategy='uniform'):
        """Full training loop."""
        for epoch in range(1, epochs+1):
            loss_val, pos_mean = self.train_one_epoch(batch_size=batch_size, neg_strategy=neg_strategy)
            if epoch % log_interval == 0 or epoch == 1:
                acc = self.validate_proxy()
                print(f"Epoch {epoch:04d} Loss: {loss_val:.4f} PosMean:{pos_mean:.4f} AccProxy:{acc:.4f}")

    def plot_losses(self, save_path='loss_plot.png', ma_window=7):
        plt.figure(figsize=(8,4.5))
        plt.plot(range(1, len(self.losses)+1), self.losses, label='Loss (raw)')
        if len(self.losses) >= 2:
            def moving_average(a, window=7):
                if window <= 1:
                    return a
                import numpy as np
                return np.convolve(a, np.ones(window)/window, mode='valid')
            ma = moving_average(self.losses, window=ma_window)
            plt.plot(range(len(self.losses)-len(ma)+1, len(self.losses)+1), ma, label=f'MA (w={ma_window})')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss curve')
        plt.legend(); plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight'); plt.show()
        print(f"Saved loss plot to: {save_path}")

    def plot_model_diagram_detailed(self, save_path=None, figsize=(10, 6)):
        """
        Detailed model diagram with per-layer param breakdown.
        Place this method inside HGTTrainer class.
        Call: trainer.plot_model_diagram_detailed('model_diagram.png')
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        # gather info
        layers = list(self.model.layers)
        n_layers = len(layers)
        scorer_params = count_params(self.model.scorer)
        total_params = count_params(self.model)

        # collect per-layer breakdown
        layer_infos = []
        for i, layer in enumerate(layers, 1):
            # type_proj params
            tp_params = 0
            for m in getattr(layer, 'type_proj').values():
                tp_params += count_params(m)
            # rel_attention params (sum over relations)
            ra_params = 0
            rel_counts = []
            for name, ra in getattr(layer, 'rel_attention').items():
                p = count_params(ra)
                ra_params += p
                rel_counts.append((name, p))
            # update_lin params
            upd_params = 0
            for m in getattr(layer, 'update_lin').values():
                upd_params += count_params(m)
            # layer_norm params
            ln_params = 0
            for m in getattr(layer, 'layer_norm').values():
                ln_params += count_params(m)
            # total layer params
            layer_total = tp_params + ra_params + upd_params + ln_params
            layer_infos.append({
                'layer_idx': i,
                'type_proj': tp_params,
                'rel_attention': ra_params,
                'rel_details': rel_counts,
                'update_lin': upd_params,
                'layer_norm': ln_params,
                'total': layer_total
            })

        # Start plotting
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        left = 0.05
        box_w = 0.6
        box_h = 0.12
        spacing = 0.04
        start_y = 0.85

        # Title and totals
        ax.text(0.02, 0.95, f"Model Diagram (detailed) — total params: {total_params:,}", fontsize=12, fontweight='bold')

        # draw layer boxes stacked vertically
        for idx, info in enumerate(layer_infos):
            y = start_y - idx * (box_h + spacing)
            # rounded rectangle
            box = FancyBboxPatch((left, y - box_h), box_w, box_h,
                                 boxstyle="round,pad=0.02", linewidth=1, facecolor="#e6f2ff", edgecolor="k")
            ax.add_patch(box)
            # header
            ax.text(left + 0.01, y - 0.02, f"Layer {info['layer_idx']:02d}  [HGTLayer]  total: {info['total']:,}",
                    fontsize=9, fontweight='bold', va='top')
            # breakdown lines (smaller font)
            bx = left + 0.01
            by = y - 0.045
            ax.text(bx, by, f"Type-specific projection: {info['type_proj']:,}", fontsize=8, va='top')
            ax.text(bx + 0.33, by, f"Update linear: {info['update_lin']:,}", fontsize=8, va='top')
            ax.text(bx, by - 0.02, f"Rel-attention (all relations): {info['rel_attention']:,}", fontsize=8, va='top')
            ax.text(bx + 0.33, by - 0.02, f"LayerNorm: {info['layer_norm']:,}", fontsize=8, va='top')

            # optionally list top few relations inside smaller text if too many relations
            rels = info['rel_details']
            if rels:
                # show up to 4 rel names + params
                short = rels if len(rels) <= 4 else rels[:4]
                rtext = "Relations: " + ", ".join([f"{name.split('__')[1]}({p:,})" for name, p in short])
                if len(rels) > 4:
                    rtext += f", ... (+{len(rels)-4} more)"
                ax.text(bx, by - 0.04, rtext, fontsize=7, va='top')

            # arrow from this layer to next (except last)
            if idx < len(layer_infos) - 1:
                ay = y - box_h - 0.01
                ax.annotate("", xy=(left + box_w/2, ay - spacing/2), xytext=(left + box_w/2, y - 0.01),
                            arrowprops=dict(arrowstyle="->", lw=1.0))

        # draw scorer box to the right
        sx = left + box_w + 0.15
        sy = start_y - (len(layer_infos)-1) * (box_h + spacing) + (box_h/2)
        sbox = FancyBboxPatch((sx, sy - box_h/2), box_w*0.6, box_h,
                             boxstyle="round,pad=0.02", linewidth=1, facecolor="#ffe6cc", edgecolor="k")
        ax.add_patch(sbox)
        ax.text(sx + 0.02, sy - 0.02, f"Scorer (MLP)\nparams: {scorer_params:,}", fontsize=9, va='center')

        # arrows from middle layer to scorer
        mid_idx = len(layer_infos)//2
        mid_y = start_y - mid_idx*(box_h + spacing) - box_h/2
        ax.annotate("", xy=(left + box_w, mid_y), xytext=(sx, mid_y),
                    arrowprops=dict(arrowstyle="->", lw=1.2))

        ax.set_xlim(0, 1.4)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        if save_path:
            # choose png or svg by file extension
            fig.savefig(save_path, bbox_inches='tight')
        plt.show()


# -----------------------
# Evaluator: metrics and visualizations
# -----------------------
class Evaluator:
    """
    Evaluation utilities:
    - evaluate_link_prediction_metrics (accuracy, f1, confusion matrix, ROC/AUC)
    - plot_confusion_matrix
    - plot_roc_curve
    - visualize_embeddings_tsne_all
    - compare_embeddings_tsne_all: compare snapshots 'before' and 'after'
    """
    def __init__(self, trainer: HGTTrainer):
        self.trainer = trainer
        self.dp = trainer.dp
        self.device = trainer.device
        self.model = trainer.model

    def evaluate_link_prediction_metrics(self, n_val=4096, threshold=0.5, seed=0, return_probs=False):
        """Compute accuracy, f1, confusion matrix, ROC/AUC for link-prediction."""
        torch.manual_seed(seed)
        self.model.eval()
        with torch.no_grad():
            rel = self.trainer.relation
            data_rel = self.dp.data[rel]
            if hasattr(data_rel, 'val_idx'):
                pos_idx = data_rel.val_idx.to(self.device)
                num_pos_available = pos_idx.size(0)
                n = min(n_val, num_pos_available)
                perm = torch.randperm(num_pos_available, device=self.device)[:n]
                sel = pos_idx[perm]
                pos_s = data_rel.edge_index[0][sel]
                pos_d = data_rel.edge_index[1][sel]
            else:
                total_pos = self.trainer.num_pos
                n = min(n_val, total_pos)
                perm = torch.randperm(total_pos, device=self.device)[:n]
                pos_s = self.trainer.pos_src[perm]; pos_d = self.trainer.pos_dst[perm]

            neg_d = torch.randint(0, self.dp.num_nodes('paper'), (n,), device=self.device)
            x_dict = self.dp.get_x_dict(); edge_index_dict = self.dp.get_edge_index_dict()
            h = self.model(x_dict, edge_index_dict)
            s_pos = torch.sigmoid(self.model.score_edges(h, [('paper','paper', pos_s, pos_d)])).cpu().numpy()
            s_neg = torch.sigmoid(self.model.score_edges(h, [('paper','paper', pos_s, neg_d)])).cpu().numpy()

            y_true = np.concatenate([np.ones_like(s_pos), np.zeros_like(s_neg)])
            y_prob = np.concatenate([s_pos, s_neg])
            y_pred = (y_prob >= threshold).astype(int)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            confmat = confusion_matrix(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_prob)
                fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            except Exception:
                auc = float('nan'); fpr, tpr, thresholds = None, None, None

        result = {
            'accuracy': acc,
            'f1_score': f1,
            'confusion_matrix': confmat,
            'y_true': y_true,
            'y_prob': y_prob,
            'y_pred': y_pred,
            'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
            'auc': auc
        }
        if return_probs:
            return result
        return result

    def plot_confusion_matrix(self, confmat, classes=['neg','pos'], cmap=plt.cm.Blues, figsize=(4,4), save_path=None):
        plt.figure(figsize=figsize)
        plt.imshow(confmat, interpolation='nearest', cmap=cmap)
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        thresh = confmat.max() / 2.
        for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
            plt.text(j, i, format(confmat[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confmat[i, j] > thresh else "black")
        plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_roc_curve(self, roc_dict, auc, figsize=(6,5), save_path=None):
        fpr = roc_dict.get('fpr'); tpr = roc_dict.get('tpr')
        if fpr is None:
            print("ROC data not available."); return
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0,1], [0,1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC curve'); plt.legend(loc='lower right'); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def visualize_embeddings_tsne_all(self, embeddings_dict, sample_n=2000, perplexity=30, random_state=0, save_dir=None):
        """
        Visualize given embeddings_dict: {ntype: np.array([N, hid])}
        This is a helper to visualize already-computed embeddings.
        """
        types = list(embeddings_dict.keys())
        num_types = len(types)
        cols = min(3, num_types)
        rows = (num_types + cols - 1) // cols
        plt.figure(figsize=(5*cols, 4*rows))
        for idx, ntype in enumerate(types):
            emb = embeddings_dict[ntype]
            N = emb.shape[0]
            if sample_n is not None and N > sample_n:
                rng = np.random.RandomState(random_state + idx)
                idx_sample = rng.choice(N, sample_n, replace=False)
                emb_s = emb[idx_sample]
            else:
                emb_s = emb
            scaler = StandardScaler()
            emb_s_scaled = scaler.fit_transform(emb_s)
            tsne = TSNE(n_components=2, perplexity=min(perplexity, max(5, emb_s_scaled.shape[0]//3)), random_state=random_state, init='pca')
            emb2d = tsne.fit_transform(emb_s_scaled)
            ax = plt.subplot(rows, cols, idx+1)
            ax.scatter(emb2d[:,0], emb2d[:,1], s=6, alpha=0.7)
            ax.set_title(f"t-SNE: {ntype} (n={emb2d.shape[0]:,})")
            ax.set_xlabel('TSNE-1'); ax.set_ylabel('TSNE-2')
            if save_dir:
                plt.savefig(f"{save_dir}/tsne_{ntype}.png", bbox_inches='tight')
        plt.tight_layout(); plt.show()

    def compare_embeddings_tsne_all(self, label_before='before', label_after='after', sample_n=2000, perplexity=30, random_state=0, save_dir=None):
        """
        Compare embeddings snapshots (must exist in trainer.embedding_snapshots) for all node types.
        Display before/after t-SNE side-by-side per node type.
        """
        snapshots = self.trainer.embedding_snapshots
        if label_before not in snapshots or label_after not in snapshots:
            print("Embedding snapshots missing. Call trainer.snapshot_embeddings('before') and trainer.snapshot_embeddings('after')")
            return
        emb_before = snapshots[label_before]
        emb_after = snapshots[label_after]
        types = list(emb_before.keys())
        num_types = len(types)
        cols = 2  # before and after
        rows = num_types
        plt.figure(figsize=(6*cols, 4*rows))
        for i, ntype in enumerate(types):
            # before
            emb_b = emb_before[ntype]
            N = emb_b.shape[0]
            if sample_n is not None and N > sample_n:
                rng = np.random.RandomState(random_state + i)
                idx_sample = rng.choice(N, sample_n, replace=False)
                emb_b_s = emb_b[idx_sample]
                emb_a_s = emb_after[ntype][idx_sample]
            else:
                emb_b_s = emb_b
                emb_a_s = emb_after[ntype]
            scaler = StandardScaler()
            emb_b_scaled = scaler.fit_transform(emb_b_s)
            emb_a_scaled = scaler.transform(emb_a_s) if emb_a_s.shape[0] == emb_b_s.shape[0] else StandardScaler().fit_transform(emb_a_s)
            tsne_b = TSNE(n_components=2, perplexity=min(perplexity, max(5, emb_b_scaled.shape[0]//3)), random_state=random_state, init='pca')
            emb2d_b = tsne_b.fit_transform(emb_b_scaled)
            tsne_a = TSNE(n_components=2, perplexity=min(perplexity, max(5, emb_a_scaled.shape[0]//3)), random_state=random_state, init='pca')
            emb2d_a = tsne_a.fit_transform(emb_a_scaled)
            ax1 = plt.subplot(rows, cols, i*2 + 1)
            ax1.scatter(emb2d_b[:,0], emb2d_b[:,1], s=6, alpha=0.7)
            ax1.set_title(f"{ntype} - before (n={emb2d_b.shape[0]:,})")
            ax2 = plt.subplot(rows, cols, i*2 + 2)
            ax2.scatter(emb2d_a[:,0], emb2d_a[:,1], s=6, alpha=0.7)
            ax2.set_title(f"{ntype} - after (n={emb2d_a.shape[0]:,})")
            if save_dir:
                plt.savefig(f"{save_dir}/tsne_compare_{ntype}.png", bbox_inches='tight')
        plt.tight_layout(); plt.show()


# -----------------------
# Example run (main)
# -----------------------
if __name__ == '__main__':
    # 1) build synthetic graph
    gp = GraphProcessor.from_synthetic(num_papers=800, num_authors=200, num_venues=20, feat_dim=64, seed=42)

    # 2) choose split (either random edge split OR node-inductive split)
    gp.split_edges_random(relation=('paper','cites','paper'), val_ratio=0.1, test_ratio=0.1, seed=1)
    # gp.split_edges_node_inductive(relation=('paper','cites','paper'), node_val_ratio=0.1, node_test_ratio=0.1, seed=1)

    # 3) trainer
    trainer = HGTTrainer(data_processor=gp, hid_dim=128, n_heads=4, n_layers=3,
                         lr=5e-3, weight_decay=1e-4, downstream_task=TaskType.LINK_PREDICTION)

    #-----------------------------------------
    # run this with your trainer.model available
    print('-----------------------------------------')
    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            print(f"{name:60s} {param.numel():10,d}  shape={tuple(param.shape)}")
    #------------------------------------------
    print('-----------------------------------')
    trainer.plot_model_diagram_detailed('model_diagram.png')
    # 4) snapshot embeddings BEFORE training
    trainer.snapshot_embeddings('before')

    # 5) optionally plot model diagram
    trainer.plot_model_diagram(save_path=None)

    # 6) train
    trainer.train(epochs=1000, batch_size=1024, log_interval=10, neg_strategy='degree')

    # 7) snapshot embeddings AFTER training
    trainer.snapshot_embeddings('after')

    # 8) loss curve
    trainer.plot_losses(save_path='loss_plot.png', ma_window=7)

    # 9) evaluate and visualize
    evaluator = Evaluator(trainer)
    metrics = evaluator.evaluate_link_prediction_metrics(n_val=4096, threshold=0.5)
    print("Accuracy:", metrics['accuracy'])
    print("F1-score:", metrics['f1_score'])
    print("AUC:", metrics['auc'])
    print("Confusion matrix:\n", metrics['confusion_matrix'])
    evaluator.plot_confusion_matrix(metrics['confusion_matrix'], classes=['neg','pos'], save_path='confusion_matrix.png')
    evaluator.plot_roc_curve(metrics['roc'], metrics['auc'], save_path='roc_curve.png')

    # 10) Visualize t-SNE for all node types BEFORE and AFTER (side-by-side)
    evaluator.compare_embeddings_tsne_all(label_before='before', label_after='after', sample_n=2000, perplexity=30, random_state=0, save_dir=None)

    print("Done.")
