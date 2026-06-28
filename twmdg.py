import networkx as nx
import pandas as pd
import json
from pathlib import Path

class TWMDG:
    """
    Temporal Weighted Multi-digraph (TWMDG) model.
    """
    def __init__(self):
        self.G = nx.MultiDiGraph()

    def build_from_dataframe(self, df, source_col='source', target_col='target', time_col='timestamp', weight_col='amount'):
        """
        Build the TWMDG from a pandas DataFrame.
        """
        for _, row in df.iterrows():
            u = row[source_col]
            v = row[target_col]
            t = row[time_col]
            w = row[weight_col]
            
            # Use timestamp as the edge key. If an edge with the same timestamp exists,
            # we can either update it or add a new one. Since it's a MultiDiGraph, 
            # we can just add it (NetworkX generates a new key if t exists).
            # But according to T-EDGE, they usually add weights for same (u, v, t).
            if self.G.has_edge(u, v, key=t):
                self.G[u][v][t]['weight'] += w
            else:
                self.G.add_edge(u, v, key=t, timestamp=t, weight=w)
                
    def load_data(self, filepath, source_col='source', target_col='target', time_col='timestamp', weight_col='amount'):
        """
        Convert raw .csv or .json transaction logs into the MultiDiGraph.
        Each edge preserves the Timestamp (t) and Amount (w).
        """
        file_ext = Path(filepath).suffix.lower()
        
        print(f"Loading data from {filepath}...")
        if file_ext == '.csv':
            df = pd.read_csv(filepath)
        elif file_ext == '.json':
            df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported file format '{file_ext}'. Only .csv and .json are supported.")
            
        self.build_from_dataframe(df, source_col, target_col, time_col, weight_col)
        print(f"Graph loaded with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        
    def get_graph(self):
        return self.G

def k_order_sampling(G, target_nodes, k_in=1, k_out=3):
    """
    K-order sampling method to extract subgraphs as described in Section 3.2.
    For each target node, traverse K_in steps backward (incoming edges) and 
    K_out steps forward (outgoing edges).
    
    Args:
        G (nx.MultiDiGraph): The Temporal Weighted Multi-digraph.
        target_nodes (list): List of initial target nodes (e.g., phishing nodes).
        k_in (int): Number of steps for incoming transactions.
        k_out (int): Number of steps for outgoing transactions.
        
    Returns:
        nx.MultiDiGraph: The induced subgraph.
    """
    sampled_nodes = set(target_nodes)
    
    # Reverse graph for incoming traversal
    G_rev = G.reverse(copy=False)
    
    for node in target_nodes:
        # K_in steps (Backward / Incoming edges)
        if k_in > 0 and node in G:
            queue = [(node, 0)]
            visited = {node}
            while queue:
                curr, depth = queue.pop(0)
                if depth < k_in:
                    for neighbor in G_rev.neighbors(curr):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            sampled_nodes.add(neighbor)
                            queue.append((neighbor, depth + 1))
                            
        # K_out steps (Forward / Outgoing edges)
        if k_out > 0 and node in G:
            queue = [(node, 0)]
            visited = {node}
            while queue:
                curr, depth = queue.pop(0)
                if depth < k_out:
                    for neighbor in G.neighbors(curr):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            sampled_nodes.add(neighbor)
                            queue.append((neighbor, depth + 1))
                            
    # Return the induced subgraph containing all sampled nodes
    return G.subgraph(sampled_nodes).copy()

if __name__ == "__main__":
    import pandas as pd
    import os
    
    # 1. Define paths (Adjust the txt_path if your folder structure differs)
    # Based on your screenshot, xblock-network_analysis is parallel to T-EDGE
    txt_path = r"C:\Users\dani9\.gemini\antigravity\scratch\blockchain\xblock-network_analysis\Temporal Link Prediction on Ethereum\EthereumG1\LPsubG1_0.5_TransEdgelist.txt"
    pickle_out_path = os.path.join("T-EDGE", "LPsubG1_df_train_0.5.pickle")
    
    print(f"Reading raw edgelist from: {txt_path}")
    
    # 2. Read the .txt file into a Pandas DataFrame
    # Edgelists are typically space or tab-separated. 
    # We use sep='\s+' to handle any amount of whitespace.
    try:
        # Assuming the txt file doesn't have a header row, we define the column names:
        # main.py expects From, To, TimeStamp, Value
        df = pd.read_csv(txt_path, sep=',', header=None, names=['From', 'To', 'Value', 'TimeStamp'])
        
        print("\nData Preview:")
        print(df.head())
        
        # 3. Save the DataFrame as a .pickle file for main.py to consume
        df.to_pickle(pickle_out_path)
        print(f"\nSuccess! Saved {len(df)} transactions to {pickle_out_path}")
        
        # 4. Verify it works with your TWMDG graph logic
        twmdg = TWMDG()
        twmdg.build_from_dataframe(df, source_col='From', target_col='To', time_col='TimeStamp', weight_col='Value')
        print(f"Graph successfully built with {twmdg.G.number_of_nodes()} nodes and {twmdg.G.number_of_edges()} edges.")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {txt_path}. Please check your path.")