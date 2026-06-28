import pandas as pd
import numpy as np
import random
import networkx as nx
import os
import sys

# Append T-EDGE to sys.path
sys.path.append(os.path.join(os.getcwd(), 'T-EDGE'))

from tGraph import tGraph
from tGraphNE import tGraphNE
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, f1_score

def get_edge_embeddings(edge_list, node_vectors, mode="append"):
    embs = []
    for u, v in edge_list:
        u, v = str(u), str(v)
        if u in node_vectors and v in node_vectors:
            vec_u = node_vectors[u]
            vec_v = node_vectors[v]
            if mode == "append":
                embs.append(np.concatenate([vec_u, vec_v]))
            else:
                embs.append(vec_u * vec_v)
        else:
            # If node not in train graph, just zeros
            embs.append(np.zeros(256 if mode=="append" else 128))
    return np.array(embs)

print("1. Loading Data and Sorting by Timestamp...")
data_file = 'xblock-network_analysis/Temporal Link Prediction on Ethereum/EthereumG1/LPsubG1_0.5_TransEdgelist.txt'
df = pd.read_csv(data_file, sep=',', names=['source', 'target', 'amount', 'timestamp'])
df = df.sort_values(by='timestamp').reset_index(drop=True)

print("2. Train/Test Split (50%)...")
split_idx = int(len(df) * 0.5)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

print("3. Creating Training Graph...")
train_G = nx.MultiDiGraph()
for _, row in train_df.iterrows():
    u, v, t, w = str(int(row['source'])), str(int(row['target'])), int(row['timestamp']), float(row['amount'])
    if train_G.has_edge(u, v, key=t):
        train_G[u][v][t]['weight'] += w
    else:
        train_G.add_edge(u, v, key=t, weight=w)

class WrapperG:
    def __init__(self, G):
        self.G = G
        times = [t for u, v, t in G.edges(keys=True)]
        if times:
            self.min_time = min(times)
            self.max_time = max(times)
        else:
            self.min_time = 0
            self.max_time = 0

tg = WrapperG(train_G)

print("4. TWMDG Embedding Generation (Biased Random Walk)...")
time_biased_type = "time_close_exp" # Strategy WS3/WS7
first_biased_type = "time_uniform"
amount_biased = "amount_exp"        # Strategy WS7 (combined)
alpha = 0.5
dimensions = 128
num_walks = 10
walk_length = 80
window_size = 10

ne = tGraphNE(tg, time_biased_type, first_biased_type, amount_biased, alpha,
              dimensions, num_walks, walk_length, "emb_G1.txt", window_size=window_size)
node_vectors = ne.vectors

print("5. Link Prediction Preparation...")
train_edges_pos = [(str(int(u)), str(int(v))) for u, v in zip(train_df['source'], train_df['target'])]
test_edges_pos = [(str(int(u)), str(int(v))) for u, v in zip(test_df['source'], test_df['target'])]

def generate_negative_samples(G, num_samples):
    nodes = list(G.nodes())
    neg_edges = set()
    while len(neg_edges) < num_samples:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v and not G.has_edge(u, v):
            neg_edges.add((u, v))
    return list(neg_edges)

print("Generating Negative Samples...")
train_edges_false = generate_negative_samples(train_G, len(train_edges_pos))
test_edges_false = generate_negative_samples(train_G, len(test_edges_pos))

print("Concatenating node vectors [Phi(u), Phi(v)]...")
edge_score_mode = "append"
train_pos_emb = get_edge_embeddings(train_edges_pos, node_vectors, edge_score_mode)
train_neg_emb = get_edge_embeddings(train_edges_false, node_vectors, edge_score_mode)
X_train = np.concatenate([train_pos_emb, train_neg_emb])
y_train = np.concatenate([np.ones(len(train_pos_emb)), np.zeros(len(train_neg_emb))])

test_pos_emb = get_edge_embeddings(test_edges_pos, node_vectors, edge_score_mode)
test_neg_emb = get_edge_embeddings(test_edges_false, node_vectors, edge_score_mode)
X_test = np.concatenate([test_pos_emb, test_neg_emb])
y_test = np.concatenate([np.ones(len(test_pos_emb)), np.zeros(len(test_neg_emb))])

print("6. Training SVM classifier...")
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

print("7. Evaluating Metrics...")
y_pred = svm.predict(X_test)
y_scores = svm.decision_function(X_test)

auc = roc_auc_score(y_test, y_scores)
f1 = f1_score(y_test, y_pred)

print(f"Metrics for EthereumG1 -> AUC: {auc:.4f}, F1-score: {f1:.4f}")
