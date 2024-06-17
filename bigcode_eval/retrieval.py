import os
from typing import List, Literal, Optional, Tuple, Union, Dict
from tqdm import tqdm
from functools import wraps, partial
import time

import torch
import torch.nn.functional as F
import numpy as np

import faiss
import faiss.contrib.torch_utils

from datasets import load_dataset, concatenate_datasets
import transformers 

def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        start_time = time.perf_counter()
        result = f(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        wrap.total_time += elapsed_time
        wrap.runs += 1
        return result
    wrap.total_time = 0.0
    wrap.runs = 0
    return wrap

def unwind_index_ivf(index):
    if isinstance(index, faiss.IndexPreTransform):
        assert index.chain.size() == 1
        vt = index.chain.at(0)
        index_ivf, vt2 = unwind_index_ivf(faiss.downcast_index(index.index))
        assert vt2 is None
        if vt is None:
            vt = lambda x: x
        else:
            vt = faiss.downcast_VectorTransform(vt)
        return index_ivf, vt
    if hasattr(faiss, "IndexRefine") and isinstance(index, faiss.IndexRefine):
        return unwind_index_ivf(faiss.downcast_index(index.base_index))
    if isinstance(index, faiss.IndexIVF):
        return index, None
    else:
        return None, None
    
class DataStore:
    def __init__(self, embedding_model_checkpoint, index_path, dataset_dir_or_name, text_col, cont_col,
                 device, nprobe=32, distance_threshold=10., key_embs_path=None, enc_pool_strategy="cls", doc_split_len=200):
        
        self.index = faiss.read_index(index_path)
        print("ntotal", self.index.ntotal)
        index_ivf, vt = unwind_index_ivf(self.index)
        if index_ivf is not None:
            index_ivf.nprobe = nprobe     
        
        self.distance_threshold = distance_threshold
        
        self.doc_split_len = doc_split_len
        dataset_name, dataset_version = dataset_dir_or_name.split(":")
        self.dataset = load_dataset(dataset_name, dataset_version, split='train')
        self.text_col = text_col
        self.cont_col = cont_col
        assert self.text_col in self.dataset.column_names and self.cont_col in self.dataset.column_names
        
        print("number of documents", len(self.dataset))

        self.model = AutoEmbeddingModel(embedding_model_checkpoint, device, pooling_strategy=enc_pool_strategy)

        # TODO: send index to device
        # self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), device, self.index)
        # co = faiss.GpuMultipleClonerOptions()
        # co.shard = True
        # co.useFloat16 = True
        # self.index = faiss.index_cpu_to_all_gpus(self.index, co)

    @timing
    def retrieve(self, texts: Union[List[str], str], k):
        single_query = False
        if isinstance(texts, str):
            texts = [texts]
            single_query = True
        embs = self.model(texts).cpu().float().numpy()
        scores, ids = self.index.search(embs, k)
        if self.index.metric_type == 1:
            scores = -1. * scores
        examples = self.dataset.select(ids.flatten())
        problems = examples[self.text_col]
        solutions = examples[self.cont_col]
        retrieved_texts = []
        for problem, solution in zip(problems, solutions):
            retrieved_texts.append(f"Instruction: {problem}\nSolution: {solution}")
        retrieved_texts = [retrieved_texts[i : i + k] for i in range(0, len(retrieved_texts), k)]
        if single_query:
            return retrieved_texts[0]
        return retrieved_texts
    
    @staticmethod
    def split_documents(documents: dict, n=200) -> dict:
        """Split documents into passages"""

        def split_text(text: str, n=200, character=" ") -> List[str]:
            """Split the text every ``n``-th occurrence of ``character``"""
            text = text.split(character)
            return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]
        
        texts = []
        for text in documents["text"]:
            if text is not None:
                for passage in split_text(text, n=n):
                    if len(passage) > 10:
                        texts.append(passage)
        return {"text": texts}
        

class AutoEmbeddingModel:
    def __init__(self, embedding_model_checkpoint, device, pooling_strategy='mean'):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_model_checkpoint)
        self.model = transformers.AutoModel.from_pretrained(embedding_model_checkpoint).to(device)
        self.device = device
        self.pool_strategy = pooling_strategy

    def __call__(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            output = self.model(input_ids, attention_mask=attention_mask)
            if self.pool_strategy.lower() == 'mean':
                embs = output['last_hidden_state'] * attention_mask.unsqueeze(-1) # [batch_size, seq_len, emb_dim]
                embs = embs.sum(1) / attention_mask.sum(dim=-1, keepdim=True) # [batch_size, emb_dim]
            else:
                embs = output['last_hidden_state'][:, 0, :]
        return embs
    

if __name__ == '__main__':
    """
        Checking in-distribution performance
    """
    embedding_model_checkpoint = "facebook/dragon-plus-query-encoder"
    index_path = "/scratch/indexes/vault/flatindexIP.index"
    dataset_dir_or_name = "Fsoft-AIC/the-vault-function:train_full-python"
    text_col = "docstring"
    topk = 5
    device = torch.device("cuda:0")
    enc_pool_strategy = "cls"

    dstore = DataStore(embedding_model_checkpoint, index_path, dataset_dir_or_name, text_col, enc_pool_strategy=enc_pool_strategy, device=device)

    dataset = load_dataset("Fsoft-AIC/the-vault-function", "train_full-python", split='train')
    print("number of documents", len(dataset))
    print("ntotal", dstore.index.ntotal)

    queries = dataset[0]['docstring']
    scores, texts = dstore.retrieve(queries, topk)
    print(queries)
    print(texts)