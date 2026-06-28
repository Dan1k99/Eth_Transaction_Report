# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 20:10:58 2018

@author: adm
"""

import time
from gensim.models import Word2Vec
import random
import numpy as np

from utils import sigmoid, softmax,  tanh
from weight_choice import weight_choice

class tGraphNE(object):
    
    def __init__(self, tG,  time_biased_type, first_biased_type, amount_biased, alpha,  dimensions, num_walks, walk_length, output, output_pklG = False, window_size=10, workers=8, hs=1, gas_biased=None, gamma=0.0, state_penalty=1.0, typology_weights=None, reconnect_revert=False):
        self.G = tG.G
        self.min_time = tG.min_time
        self.max_time = tG.max_time
        
        self.state_penalty = state_penalty
        self.reconnect_revert = reconnect_revert
        self.typology_weights = typology_weights if typology_weights is not None else {
            'eoa_to_eoa': 1.0,
            'eoa_to_contract': 1.0,
            'contract_to_eoa': 1.0,
            'contract_to_contract': 1.0
        }
        
        self.time_biased_type = time_biased_type # choice = "unbiased", "amount-weighted" "linear", "exp"
        self.first_biased_type = first_biased_type    
        self.amount_biased = amount_biased
        self.gas_biased = gas_biased
        self.alpha = alpha
        self.gamma = gamma

        # Node Failure Rate Calculation
        self.contract_fail_rates = {}
        for node in self.G.nodes():
            if self.G.nodes[node].get('is_contract', 0) == 1:
                in_edges = self.G.in_edges(node, data=True)
                if len(in_edges) > 0:
                    failed_count = sum(1 for u, v, d in in_edges if d.get('is_failed', 0) == 1)
                    self.contract_fail_rates[node] = failed_count / len(in_edges)
                else:
                    self.contract_fail_rates[node] = 0.0

        print("Walking...")
        t1 = time.time()
        walks = self.simulate_walks(num_walks, walk_length)#随机游走
        t2 = time.time()
        print("  Walking time:", t2-t1)
        
        print("Learn embeddings...")   
        
        #walks = [map(str, walk) for walk in walks]        
        word2vec_model = Word2Vec(sentences = walks, vector_size= dimensions, window= window_size, min_count=0, sg=1, hs=1, workers= workers)
        t3 = time.time()
        print("Learn embeddings time:", t3-t2) 
        
        self.vectors = {}
        for word in list(self.G.nodes()):
            self.vectors[str(word)] = word2vec_model.wv[str(word)]            

        print("  Embeddings are saved in ", output)
        word2vec_model.wv.save_word2vec_format(output)
        
        del word2vec_model
        
        return

###################################################################

    
    def simulate_walks(self, num_walks, walk_length):
        """
        Repeatedly simulate random walks from each node.
        对每个结点，根据num_walks得出其多条随机游走路径
        
        """
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print("Walk iteration:")
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.temporal_walk( walk_length = walk_length, start_node = node ))      
        return walks
            
    def temporal_walk(self, walk_length, start_node):        
        """
        功能： 从一个初始结点计算一个随机游走
        输入：
        walk_length: 随机游走序列长度
        start_node: 初始结点
        返回：
        列表，随机游走序列
        """
       
        walk = [start_node] #类型：list
        walk_edge = []
        walk_time = [] ##类型：list, 大小比walk的小1
        # walk_key = []
        
        cur = start_node
        next_node, next_time, next_key = self.get_first_step(cur)
        if next_node != None:
            is_failed = self.G[cur][next_node][next_key].get('is_failed', 0)
            if self.reconnect_revert and is_failed == 1:
                walk.append(next_node)
                walk_time.append(next_time)
                walk_edge.append(next_key)
                if len(walk) < walk_length:
                    walk.append(cur)
                    walk_time.append(next_time)
                    walk_edge.append(next_key)
            else:
                walk.append(next_node)
                walk_time.append(next_time)
                walk_edge.append(next_key)
        else:
            return walk
        
        while len(walk) < walk_length:
            prevtime = walk_time[-1]
            cur = walk[-1]  #名为walk的list的最后一个元素，当前游走到的结点
            next_node, next_time, next_key = self.get_next_step(cur, prevtime)
            if next_node != None:
                is_failed = self.G[cur][next_node][next_key].get('is_failed', 0)
                if self.reconnect_revert and is_failed == 1:
                    walk.append(next_node)
                    walk_time.append(next_time)
                    walk_edge.append(next_key)
                    if len(walk) < walk_length:
                        walk.append(cur)
                        walk_time.append(next_time)
                        walk_edge.append(next_key)
                else:
                    walk.append(next_node)
                    walk_time.append(next_time)
                    walk_edge.append(next_key)
            else:
                break
        return walk
    
    def get_first_step(self, cur):
        G = self.G
        tmp_key = []
        tmp_node = []
        tmp_time = []
        unnormalized_probs_t = []
        
        if cur not in G:
            return None, None, None
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) == 0:
            return None, None, None
        if self.time_biased_type == "simple_graph": #DeepWalk
            for nbr in cur_nbrs:
                tmp_node.append(nbr)
                unnormalized_probs_t.append(1)                
                
            if len(unnormalized_probs_t) > 0:
                idx = weight_choice(unnormalized_probs_t)
                next_node = tmp_node[idx]
                next_time = 0 
                next_key = 0
                return next_node, next_time, next_key
            else:
                return None, None, None  #没有符合条件的   

        else:
            for nbr in cur_nbrs:
                nbr_key = list(G.get_edge_data(cur,nbr))    #cur领边的key数组       
                for k in nbr_key:
                    t = k             
                    if self.first_biased_type == "time_uniform":
                        unnormalized_probs_t.append(1)
                    elif self.first_biased_type == "time_freq":
                        unnormalized_probs_t.append(self.max_time-t+1)
                    elif self.first_biased_type == "time_close_linear":
                        unnormalized_probs_t.append(self.max_time-t+1)
                    elif self.first_biased_type == "time_far":
                        unnormalized_probs_t.append(t-self.min_time+1) 
                    elif self.first_biased_type == "time_far_linear":
                        unnormalized_probs_t.append(t)
                    
                    tmp_node.append(nbr)
                    tmp_time.append(t)
                    tmp_key.append(k)
                
            if self.first_biased_type == "time_close_linear" :
                unnormalized_probs_t = linear_rank_mapping( unnormalized_probs_t, order='descending' )
            elif self.first_biased_type == "time_far_linear":
                unnormalized_probs_t = linear_rank_mapping( unnormalized_probs_t )
            
    
            if len(unnormalized_probs_t) > 0: #有符合条件的下一个点
                scaled_probs = []
                for i in range(len(unnormalized_probs_t)):
                    prob = float(unnormalized_probs_t[i])
                    nbr = tmp_node[i]
                    k = tmp_key[i]
                    
                    # 1. State Penalty
                    is_failed = G[cur][nbr][k].get('is_failed', 0)
                    penalty = self.state_penalty if is_failed == 1 else 1.0
                    
                    # 2. Node Typology Weight
                    from_is_contract = G.nodes[cur].get('is_contract', 0)
                    to_is_contract = G.nodes[nbr].get('is_contract', 0)
                    
                    if from_is_contract == 0 and to_is_contract == 0:
                        w_typo = self.typology_weights.get('eoa_to_eoa', 1.0)
                    elif from_is_contract == 0 and to_is_contract == 1:
                        w_typo = self.typology_weights.get('eoa_to_contract', 1.0)
                    elif from_is_contract == 1 and to_is_contract == 0:
                        w_typo = self.typology_weights.get('contract_to_eoa', 1.0)
                    else:
                        w_typo = self.typology_weights.get('contract_to_contract', 1.0)
                        
                    scaled_probs.append(prob * penalty * w_typo)
                    
                unnormalized_probs_t = scaled_probs
                
                selected = weight_choice(unnormalized_probs_t)               
                next_node = tmp_node[selected]
                next_time = tmp_time[selected]        
                next_key = tmp_key[selected]   
                return next_node, next_time, next_key
            
            else:
                return None, None, None  #没有符合条件的
        
        
    def get_next_step(self, cur, prevtime=0):
        return self.GetNextEdgeWithStrategies(cur, prevtime)
        
    def GetNextEdgeWithStrategies(self, cur, prevtime=0, gas_price_biased="gas_uniform"):
        """
        Modularized function to select the next edge using various walking strategies.
        This allows easy injection of the Gas Price variable later without refactoring
        the entire graph traversal logic.
        """
        G = self.G
        
        tmp_key = []
        tmp_node = []
        tmp_time = []
        unnormalized_probs_t = []
        unnormalized_probs_a = []
        unnormalized_probs_g = [] # Placeholder for Gas Price probabilities

        if cur not in G:
            return None, None, None
        cur_nbrs = list(G.neighbors(cur))
        if len(cur_nbrs) == 0:
            return None, None, None
        if self.time_biased_type == "simple_graph": #DeepWalk
            for nbr in cur_nbrs:
                tmp_node.append(nbr)
                unnormalized_probs_t.append(1)                
                
            if len(unnormalized_probs_t) > 0:
                idx = weight_choice(unnormalized_probs_t)
                next_node = tmp_node[idx]
                next_time = 0 
                next_key = 0
                return next_node, next_time, next_key
            else:
                return None, None, None  #没有符合条件的
        else:    
            for nbr in cur_nbrs:
                nbr_key = list(G.get_edge_data(cur,nbr))    #cur领边的key数组        
                for k in nbr_key:
                    t = k
                    a = G[cur][nbr][k]['weight']
                    if self.time_biased_type == "no_time_limit":
                        unnormalized_probs_t.append(1)
                    
                    elif t >= prevtime:
                        unnormalized_probs_a.append(a)
                        if self.gas_biased == "friction_inverse":
                            unnormalized_probs_g.append(G[cur][nbr][k].get('friction', 1.0))
                        elif self.gas_biased == "premium":
                            from_is_contract = G.nodes[cur].get('is_contract', 0)
                            to_is_contract = G.nodes[nbr].get('is_contract', 0)
                            if from_is_contract == 0 and to_is_contract == 0:
                                unnormalized_probs_g.append(G[cur][nbr][k].get('gas', 1e-18))
                            else:
                                unnormalized_probs_g.append(1e-18)
                        elif self.gas_biased == "relative_premium":
                            from_is_contract = G.nodes[cur].get('is_contract', 0)
                            to_is_contract = G.nodes[nbr].get('is_contract', 0)
                            if from_is_contract == 0 and to_is_contract == 0:
                                unnormalized_probs_g.append(G[cur][nbr][k].get('relative_gas', 1.0))
                            else:
                                unnormalized_probs_g.append(1.0)
                        elif self.gas_biased == "fail_rate":
                            from_is_contract = G.nodes[cur].get('is_contract', 0)
                            to_is_contract = G.nodes[nbr].get('is_contract', 0)
                            if from_is_contract == 0:
                                if to_is_contract == 1:
                                    unnormalized_probs_g.append(self.contract_fail_rates.get(nbr, 0.0))
                                else:
                                    unnormalized_probs_g.append(1e-6)
                            else:
                                unnormalized_probs_g.append(1.0)
                        else:
                            unnormalized_probs_g.append(1) 
                        
                        if self.time_biased_type == "time_uniform"  :
                             unnormalized_probs_t.append(1)
                        elif self.time_biased_type == "time_close_raw"  :
                             unnormalized_probs_t.append( self.max_time - t + 1 )
                        elif self.time_biased_type == "time_close_exp"  :
                             unnormalized_probs_t.append( t - prevtime )
                        else:
                             unnormalized_probs_t.append( t - prevtime + 1 )
                        tmp_time.append(t)
                        tmp_node.append(nbr)
                        tmp_key.append(k)
                        

            if len(unnormalized_probs_t) > 0:
                if self.time_biased_type == "time_close_linear" :
                    unnormalized_probs_t = linear_rank_mapping( unnormalized_probs_t, order='descending' )
                elif self.time_biased_type == "time_far_linear" :
                    unnormalized_probs_t = linear_rank_mapping( unnormalized_probs_t)   
                elif self.time_biased_type == "time_freq_tanh":
                    unnormalized_probs_t = tanh(unnormalized_probs_t)
                elif self.time_biased_type == "time_close_exp":
                    unnormalized_probs_t = softmax(unnormalized_probs_t)

                if self.amount_biased == "amount_linear":
                    unnormalized_probs_a = linear_rank_mapping(unnormalized_probs_a)
                elif self.amount_biased == "amount_tanh":
                    unnormalized_probs_a = tanh(unnormalized_probs_a)
                elif self.amount_biased == "amount_exp":
                    unnormalized_probs_a = softmax(unnormalized_probs_a)
                
                if self.gas_biased == "friction_inverse" and len(unnormalized_probs_g) > 0:
                    max_f = max(unnormalized_probs_g)
                    if max_f == 0: max_f = 1.0
                    unnormalized_probs_g = [max(1e-6, 1.0 - (f / max_f)) for f in unnormalized_probs_g]
                elif self.gas_biased == "premium" and len(unnormalized_probs_g) > 0:
                    max_g = max(unnormalized_probs_g)
                    if max_g == 0: max_g = 1e-18
                    unnormalized_probs_g = [max(0.01, g / max_g) for g in unnormalized_probs_g]
                elif self.gas_biased == "relative_premium" and len(unnormalized_probs_g) > 0:
                    max_g = max(unnormalized_probs_g)
                    if max_g == 0: max_g = 1.0
                    unnormalized_probs_g = [max(1e-6, g / max_g) for g in unnormalized_probs_g]
                elif self.gas_biased == "fail_rate" and len(unnormalized_probs_g) > 0:
                    max_g = max(unnormalized_probs_g)
                    if max_g == 0: max_g = 1.0
                    unnormalized_probs_g = [max(1e-6, g / max_g) for g in unnormalized_probs_g]

            
            if len(unnormalized_probs_t) > 0: #有符合条件的下一个点
                if self.gas_biased in ["friction_inverse", "premium", "relative_premium", "fail_rate"]:
                    unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, self.alpha, unnormalized_probs_g, self.gamma)
                elif self.amount_biased != "amount_uniform":
                    unnormalized_probs = combine_probs(unnormalized_probs_t, unnormalized_probs_a, self.alpha)        
                else:
                    unnormalized_probs = unnormalized_probs_t
                    
                # Apply State-Penalty and Node Typology Isolation scaling
                scaled_probs = []
                for i in range(len(unnormalized_probs)):
                    prob = float(unnormalized_probs[i])
                    nbr = tmp_node[i]
                    k = tmp_key[i]
                    
                    # 1. State Penalty
                    is_failed = G[cur][nbr][k].get('is_failed', 0)
                    penalty = self.state_penalty if is_failed == 1 else 1.0
                    
                    # 2. Node Typology Weight
                    from_is_contract = G.nodes[cur].get('is_contract', 0)
                    to_is_contract = G.nodes[nbr].get('is_contract', 0)
                    
                    if from_is_contract == 0 and to_is_contract == 0:
                        w_typo = self.typology_weights.get('eoa_to_eoa', 1.0)
                    elif from_is_contract == 0 and to_is_contract == 1:
                        w_typo = self.typology_weights.get('eoa_to_contract', 1.0)
                    elif from_is_contract == 1 and to_is_contract == 0:
                        w_typo = self.typology_weights.get('contract_to_eoa', 1.0)
                    else:
                        w_typo = self.typology_weights.get('contract_to_contract', 1.0)
                        
                    scaled_probs.append(prob * penalty * w_typo)
                    
                unnormalized_probs = scaled_probs
                    
                self.last_probs = unnormalized_probs
                self.last_nbrs = tmp_node
                self.last_amounts = unnormalized_probs_a
                self.last_times = unnormalized_probs_t
                
                selected = weight_choice(unnormalized_probs)               
                next_node = tmp_node[selected]  
                next_time = tmp_time[selected]        
                next_key = tmp_key[selected]   
                return next_node, next_time, next_key  
            
            else:
                return None, None, None  #没有符合条件的

    
def linear_rank_mapping( original_array, order='ascending' ):
    x = np.array(original_array)
    if order == 'ascending':
        return (np.argsort(x) + 1)
    elif order == 'descending':  
        return (np.argsort(-x) + 1)  
        # return (x.argsort() + 1)
        
def normalized_probs(unnormalized_probs):
    if len(unnormalized_probs) > 0: #有符合条件的下一个点
        norm_const = sum(unnormalized_probs)
        if norm_const == 0: norm_const = 1.0
        normalized_probs = [ u_prob / norm_const for u_prob in unnormalized_probs] #归一化  
    else:
        normalized_probs = []

    return normalized_probs    
    
def combine_probs(p1, p2, alpha, p3=None, gamma=None) :
    probs1 = normalized_probs(p1)
    probs2 = normalized_probs(p2)

    if len(probs1) != len(probs2):
        print("ERROR", "len(probs1) != len(probs2)" )

    if p3 is not None and gamma is not None:
        probs3 = normalized_probs(p3)
        beta = 1.0 - alpha - gamma
        combine_probs = np.multiply( np.multiply(np.power(probs1, alpha), np.power(probs2, beta)), np.power(probs3, gamma) )
    else:                          
        combine_probs = np.multiply( np.power(probs1, alpha), np.power(probs2, 1-alpha) )        
    
    return combine_probs

   
    








    
    