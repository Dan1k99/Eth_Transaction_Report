# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:11:02 2019

@author: adm
"""

import sys
import networkx as nx

import pickle as pickle

class tGraph(object):
    def __init__( self, file_, filetype, output_Gpkl=False, exclude_failed=False, graph_mode="baseline", delta_t=120 ):
        self.G = nx.MultiDiGraph() 
        self.min_time = sys.maxsize
        self.max_time = 0
        self.min_amount = sys.maxsize
        self.max_amount = 0
        self.graph_mode = graph_mode
        self.delta_t = delta_t
        
        print("Loading file", file_, "...")
        
        edge_key = 0
        
        if filetype == "txt_raw":
            with open(file_) as f:
                for l in f:                    
                    x, y = l.strip().split(' ')
                    self.G.add_edge(x,y,key=edge_key)                    
                    edge_key = edge_key + 1
        else:            
            if filetype == "txt_f":
                with open(file_) as f:
                    for l in f:                    
                        x, y, a, t = l.strip().split(',')
                        a = float(a)
                        t = int(t)                       
                        if self.G.has_edge(x,y,t):
                            if self.G[x][y][t]['weight'] != a:
                                self.G[x][y][t]['weight'] += a
                        else:    
                            self.G.add_edge(x,y,key=t, weight=a)     
                        edge_key = edge_key + 1 
                                              
                        if t < self.min_time:
                            self.min_time = t
                        elif t > self.max_time:
                            self.max_time = t                            

            elif filetype == "pkl_f":
                import pandas as pd
                import numpy as np
                with open( file_,"rb") as f: 
                    df_in = pickle.load(f)            
                
                # Check graph_mode and perform Bipartite Folding / Router Bypass if requested
                if self.graph_mode == "router_bypass":
                    df_sorted = df_in.sort_values(by='TimeStamp').reset_index(drop=True)
                    
                    prune_indices = set()
                    bypassed_txs = []
                    
                    has_gas = 'Gas' in df_sorted.columns
                    has_relative_gas = 'RelativeGas' in df_sorted.columns
                    has_block_number = 'BlockNumber' in df_sorted.columns
                    has_friction = 'Friction' in df_sorted.columns
                    has_gas_limit = 'GasLimit' in df_sorted.columns
                    has_is_failed = 'IsFailed' in df_sorted.columns
                    has_from_contract = 'FromIsContract' in df_sorted.columns
                    has_to_contract = 'ToIsContract' in df_sorted.columns
                    
                    # Target is smart contract: ToIsContract == 1
                    contract_txs = df_sorted[df_sorted['ToIsContract'] == 1]
                    grouped_contracts = contract_txs.groupby('To')
                    
                    for contract, group1 in grouped_contracts:
                        # Find potential T2 outgoing transactions: From == contract, ToIsContract == 0 (to EOA)
                        group2 = df_sorted[(df_sorted['From'] == contract) & (df_sorted['ToIsContract'] == 0)]
                        if group2.empty:
                            continue
                        
                        t2_times = group2['TimeStamp'].values
                        t2_indices = group2.index.values
                        
                        for idx1, row1 in group1.iterrows():
                            t1 = row1['TimeStamp']
                            A = row1['From']
                            
                            # Binary search for transactions in [t1, t1 + delta_t]
                            pos_start = np.searchsorted(t2_times, t1)
                            pos_end = np.searchsorted(t2_times, t1 + self.delta_t, side='right')
                            
                            for pos in range(pos_start, pos_end):
                                idx2 = t2_indices[pos]
                                row2 = group2.loc[idx2]
                                B = row2['To']
                                
                                # Create direct edge from A to B with values from T2
                                bypassed_txs.append({
                                    'From': A,
                                    'To': B,
                                    'TimeStamp': row2['TimeStamp'],
                                    'Value': row2['Value'],
                                    'Gas': row2['Gas'] if has_gas else 1.0,
                                    'RelativeGas': row2['RelativeGas'] if has_relative_gas else 1.0,
                                    'BlockNumber': row2['BlockNumber'] if has_block_number else 0,
                                    'Friction': row2['Friction'] if has_friction else 1.0,
                                    'GasLimit': row2['GasLimit'] if has_gas_limit else 21000.0,
                                    'IsFailed': row2['IsFailed'] if has_is_failed else 0,
                                    'FromIsContract': row1['FromIsContract'] if has_from_contract else 0,
                                    'ToIsContract': row2['ToIsContract'] if has_to_contract else 0,
                                    'Router': contract
                                })
                                
                                prune_indices.add(idx1)
                                prune_indices.add(idx2)
                                
                    # Drop original intermediate transactions
                    df_clean = df_sorted.drop(index=list(prune_indices)).reset_index(drop=True)
                    if bypassed_txs:
                        df_bypassed = pd.DataFrame(bypassed_txs)
                        df_in = pd.concat([df_clean, df_bypassed], ignore_index=True)
                    else:
                        df_in = df_clean
                    
                    print(f"Bipartite Folding completed: bypassed {len(bypassed_txs)} transactions, pruned {len(prune_indices)} original transactions.")

                has_gas = 'Gas' in df_in.columns
                has_relative_gas = 'RelativeGas' in df_in.columns
                has_friction = 'Friction' in df_in.columns
                has_gas_limit = 'GasLimit' in df_in.columns
                has_is_failed = 'IsFailed' in df_in.columns
                has_from_contract = 'FromIsContract' in df_in.columns
                has_to_contract = 'ToIsContract' in df_in.columns
                
                for i in df_in.index:
                    x = str(int(df_in.From[i]))
                    y = str(int(df_in.To[i]))
                    t = int(df_in.TimeStamp[i])
                    a = df_in.Value[i]
                    
                    gas = df_in.Gas[i] if has_gas else 1.0
                    relative_gas = df_in.RelativeGas[i] if has_relative_gas else 1.0
                    fric = df_in.Friction[i] if has_friction else 1.0
                    gas_limit = df_in.GasLimit[i] if has_gas_limit else 21000.0
                    is_failed = int(df_in.IsFailed[i]) if has_is_failed else 0
                    
                    from_is_contract = int(df_in.FromIsContract[i]) if has_from_contract else 0
                    to_is_contract = int(df_in.ToIsContract[i]) if has_to_contract else 0
                    
                    if exclude_failed and is_failed == 1:
                        continue
                    
                    # Graph mode specific logic
                    if self.graph_mode == "shadow_nodes":
                        # Block outbound edges from any revert node
                        if x.endswith("_revert"):
                            continue
                        if to_is_contract == 1 and is_failed == 1:
                            y = f"{y}_revert"
                            to_is_contract = 1
                    
                    router = df_in.Router[i] if 'Router' in df_in.columns and not pd.isna(df_in.Router[i]) else None
                    
                    if x not in self.G.nodes:
                        self.G.add_node(x, is_contract=from_is_contract)
                    else:
                        if from_is_contract == 1:
                            self.G.nodes[x]['is_contract'] = 1
                        elif 'is_contract' not in self.G.nodes[x]:
                            self.G.nodes[x]['is_contract'] = from_is_contract
 
                    if y not in self.G.nodes:
                        self.G.add_node(y, is_contract=to_is_contract)
                    else:
                        if to_is_contract == 1:
                            self.G.nodes[y]['is_contract'] = 1
                        elif 'is_contract' not in self.G.nodes[y]:
                            self.G.nodes[y]['is_contract'] = to_is_contract
                    
                    if self.G.has_edge(x,y,t):
                        self.G[x][y][t]['weight'] += a                        
                        self.G[x][y][t]['gas'] = gas
                        self.G[x][y][t]['relative_gas'] = relative_gas
                        self.G[x][y][t]['friction'] = fric
                        self.G[x][y][t]['gas_limit'] = gas_limit
                        self.G[x][y][t]['is_failed'] = is_failed
                        if router is not None:
                            self.G[x][y][t]['router'] = router
                    else:    
                        self.G.add_edge(x,y,key=t, weight=a, gas=gas, relative_gas=relative_gas, friction=fric, gas_limit=gas_limit, is_failed=is_failed, router=router)     
                        
                    edge_key = edge_key + 1                     
                    if t < self.min_time:
                        self.min_time = t
                    elif t > self.max_time:
                        self.max_time = t    

            if output_Gpkl == True:            
                pklfile_G = "tGraph.pickle"
                with open(pklfile_G, "wb") as f:
                    print("Writing", pklfile_G, "...")
                    pickle.dump( self.G, pklfile_G, pickle.HIGHEST_PROTOCOL )

        self.number_of_nodes = self.G.number_of_nodes()
        self.number_of_edges = self.G.number_of_edges()
        print("Summary of graph:")
        print("Number of nodes: ", self.number_of_nodes)
        print("Number of edges: ", self.number_of_edges)
        print("Number of edge_key: ", edge_key)
        print("Min time: ", self.min_time) 
        print("Max time: ", self.max_time) 
