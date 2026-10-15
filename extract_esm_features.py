import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import EsmModel, AutoTokenizer

# esm2_t30_150M_UR50D: hidden_size=640
ESM_HIDDEN_SIZE = 640


def extract_and_save_features(csv_path, esm_model_path, save_path, device='cuda', max_length=1024):
    print("1. 正在加载 ESM 模型与分词器...")
    tokenizer = AutoTokenizer.from_pretrained(esm_model_path)
    model = EsmModel.from_pretrained(esm_model_path).to(device)
    model.eval()

    print(f"2. 正在读取数据集: {csv_path}")
    df = pd.read_csv(csv_path)
    unique_proteins = df['Protein'].unique().tolist()
    print(f"-> 发现 {len(unique_proteins)} 条不重复的蛋白质序列。")

    protein_features_dict = {}

    print("3. 开始提取 ESM 残基级特征并保存 (dim=640)...")
    with torch.no_grad():
        for seq in tqdm(unique_proteins, desc="Extracting ESM"):
            inputs = tokenizer(
                seq, return_tensors="pt", truncation=True,
                max_length=max_length, padding=False
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            hidden = outputs.last_hidden_state[0]  # (tokens, 640)
            # 去掉 <cls> 与 <eos>，保留残基级 token 特征
            token_len = inputs['input_ids'].shape[1]
            residue_features = hidden[1:token_len - 1, :].cpu()

            if residue_features.numel() == 0:
                residue_features = hidden.mean(dim=0, keepdim=True).cpu()

            protein_features_dict[seq] = residue_features

    print(f"4. 提取完成！正在保存至 {save_path} ...")
    torch.save(protein_features_dict, save_path)
    sample = next(iter(protein_features_dict.values()))
    print(f"-> 保存成功！示例 shape={sample.shape}, hidden_size={sample.shape[-1]}")


if __name__ == "__main__":
    DATASET_CSV = "datasets/Kinase_filtered/Kinase_filtered_full.csv"
    LOCAL_ESM_DIR = "./local_esm2"
    SAVE_FILE = "protein_esm_features-K.pt"

    extract_and_save_features(DATASET_CSV, LOCAL_ESM_DIR, SAVE_FILE)
