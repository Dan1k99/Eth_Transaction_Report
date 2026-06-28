import os
import pandas as pd
import numpy as np
import zipfile
import urllib.request
import time
import random
import pickle

def download_file(url, local_path):
    if os.path.exists(local_path):
        print(f"File {local_path} already exists. Skipping download.")
        return
    
    print(f"Downloading {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(local_path, 'wb') as out_file:
        file_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        chunk_size = 8192
        start_time = time.time()
        
        while True:
            buffer = response.read(chunk_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            
            if file_size and time.time() - start_time > 2:
                print(f"Downloaded {downloaded / 1024 / 1024:.2f} MB / {file_size / 1024 / 1024:.2f} MB", end='\r')
                start_time = time.time()
    print(f"\nSuccessfully downloaded {local_path}")

def generate_negative_edges(nodes, positive_edges, num_edges):
    negative_edges = set()
    nodes_list = list(nodes)
    while len(negative_edges) < num_edges:
        u = random.choice(nodes_list)
        v = random.choice(nodes_list)
        if u != v and (u, v) not in positive_edges and (u, v) not in negative_edges:
            negative_edges.add((u, v))
    return negative_edges

def prepare_pipeline():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    datasets_to_process = [
        {"name": "Eth_0_999k", "file": "0to999999_BlockTransaction"},
        {"name": "Eth_1M_1_99M", "file": "1000000to1999999_BlockTransaction"}
    ]
    
    for ds in datasets_to_process:
        dataset_name = ds["name"]
        file_prefix = ds["file"]
        
        print(f"\n================ Processing Dataset: {dataset_name} ================")
        zip_url = f"https://zhengpeilin.com/download.php?file={file_prefix}.zip"
        local_zip = os.path.join(data_dir, f"{file_prefix}.zip")
        
        download_file(zip_url, local_zip)
        
        print("Extracting transactions...")
        records = []
        addr2Idx = {}
        idx_counter = 1
        num_txs_to_read = 50000
        
        with zipfile.ZipFile(local_zip, 'r') as z:
            with z.open(f"{file_prefix}.csv") as f:
                header = f.readline()
                for _ in range(num_txs_to_read):
                    line = f.readline().decode('utf-8').strip()
                    if not line:
                        break
                    parts = line.split(",")
                    if len(parts) < 14:
                        continue
                    
                    block_num = int(parts[0])
                    ts = int(parts[1])
                    from_addr = parts[3].lower()
                    to_addr = parts[4].lower()
                    from_is_contract = int(parts[6])
                    to_is_contract = int(parts[7])
                    val_wei = float(parts[8])
                    gas_limit = float(parts[9])
                    gas_price_wei = float(parts[10])
                    is_error_str = parts[13]
                    is_failed = 1 if is_error_str != 'None' else 0
                    
                    val_eth = val_wei / 1e18
                    gas_eth = gas_price_wei / 1e18
                    
                    if from_addr not in addr2Idx:
                        addr2Idx[from_addr] = str(idx_counter)
                        idx_counter += 1
                    if to_addr not in addr2Idx:
                        addr2Idx[to_addr] = str(idx_counter)
                        idx_counter += 1
                        
                    records.append({
                        "From": addr2Idx[from_addr],
                        "To": addr2Idx[to_addr],
                        "Value": val_eth,
                        "TimeStamp": ts,
                        "BlockNumber": block_num,
                        "Gas": gas_eth,
                        "Friction": gas_eth / (val_eth + 1e-18),
                        "GasLimit": gas_limit,
                        "IsFailed": is_failed,
                        "FromIsContract": from_is_contract,
                        "ToIsContract": to_is_contract
                    })

        df = pd.DataFrame(records)
        print("Calculating block-relative gas prices...")
        if not df.empty:
            block_medians = df.groupby('BlockNumber')['Gas'].transform('median')
            block_medians = block_medians.replace(0, 1e-18)
            df['RelativeGas'] = df['Gas'] / block_medians
        else:
            df['RelativeGas'] = 1.0
        print(f"Extracted {len(df)} transactions.")
        
        print("Sorting by TimeStamp...")
        df = df.sort_values(by="TimeStamp").reset_index(drop=True)
        
        print("Splitting into train/test...")
        split_idx = int(len(df) * 0.5)
        df_train = df.iloc[:split_idx]
        df_test = df.iloc[split_idx:]
        
        train_nodes = set(df_train['From']).union(set(df_train['To']))
        train_edges_pos = set(zip(df_train['From'].astype(int), df_train['To'].astype(int)))
        
        test_edges_pos = set()
        for _, row in df_test.iterrows():
            u, v = int(row['From']), int(row['To'])
            if str(u) in train_nodes and str(v) in train_nodes and (u, v) not in train_edges_pos:
                test_edges_pos.add((u, v))
                
        print(f"Train positive edges: {len(train_edges_pos)}")
        print(f"Test positive edges: {len(test_edges_pos)}")
        
        print("Generating negative edges...")
        all_positive_edges = train_edges_pos.union(test_edges_pos)
        
        train_edges_false = generate_negative_edges(train_nodes, train_edges_pos, len(train_edges_pos))
        test_edges_false = generate_negative_edges(train_nodes, all_positive_edges, len(test_edges_pos))
        
        split_data = {
            'train_edges_pos': train_edges_pos,
            'train_edges_false': train_edges_false,
            'test_edges_pos': test_edges_pos,
            'test_edges_false': test_edges_false
        }
        
        out_dir = "T-EDGE"
        os.makedirs(out_dir, exist_ok=True)
        
        df_train_path = os.path.join(out_dir, f"{dataset_name}_df_train_0.5.pickle")
        split_path = os.path.join(out_dir, f"{dataset_name}_train_test_split_0.5.pickle")
        
        df_train.to_pickle(df_train_path)
        with open(split_path, 'wb') as f:
            pickle.dump(split_data, f)
            
        print(f"Pipeline complete for {dataset_name}! Saved to {df_train_path} and {split_path}.")

if __name__ == "__main__":
    prepare_pipeline()
