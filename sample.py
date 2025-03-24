#python d:/编程/LLM/实践/FireFly/sample.py --save_dir D:/编程/LLM/实践/FireFly/models/policy --temperature 1.0 --batch_size 4 --nsamples 10
# encoding: utf-8
#!/usr/bin/env python3

import os
import argparse
from functools import partial
from mpi4py import MPI
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Policy(torch.nn.Module):
    def __init__(self, model, embed_queries, temperature):
        super(Policy, self).__init__()
        self.model = model
        self.embed_queries = embed_queries
        self.temperature = temperature

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def main_process():
    return MPI.COMM_WORLD.Get_rank() == 0

def run_sample(args):
    save_dir = args.save_dir
    temperature = args.temperature
    batch_size = args.batch_size
    nsamples = args.nsamples

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and tokenizer with local_files_only=True and corrected path
    model_path = os.path.normpath(save_dir)  # Normalize the path
    print(f"Loading model from {model_path}")  # Print the path for verification

    try:
        model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
        tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=True)
    except EnvironmentError as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure that the path is correct and contains the model files.")
        return

    # Initialize communication
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Check for sufficient samples
    if nsamples > 0:
        nsamples_per_rank = nsamples // size
        if nsamples % size != 0:
            if main_process():
                print("Warning: Number of samples not divisible by the number of processes.")
    else:
        nsamples_per_rank = -1  # Infinite samples

    # Define the query sampling function
    def sample_queries():
        queries = []
        for _ in range(batch_size):
            query = "Sample query"  # 可以尝试使用更有意义的查询
            encoded = tokenizer(query, max_length=64, truncation=True)
            queries.append(encoded['input_ids'])
        return queries

    # Sample continuously
    generated = 0
    while nsamples_per_rank == -1 or generated < nsamples_per_rank:
        # Sample queries
        queries = sample_queries()
        all_queries = np.asarray(queries, dtype=np.int64)

        # Generate responses
        input_ids = torch.tensor(all_queries).to(device)
        attention_mask = torch.tensor(all_queries != tokenizer.eos_token_id).float().to(device)

        model.to(device)
        model.eval()

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1] + 50,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # 显式设置 pad_token_id
        )

        # Decode responses
        for q, r in zip(all_queries, outputs):
            print('=' * 80)
            print(tokenizer.decode(q).replace("\n", "\\n"))
            print(tokenizer.decode(r).replace("\n", "\\n"))

        generated += batch_size * size

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample Policy')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory containing the policy model')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for sampling')
    parser.add_argument('--nsamples', type=int, default=0, help='Total number of samples to generate')

    args = parser.parse_args()

    if not main_process():
        MPI.COMM_WORLD.Abort()

    # Initialize MPI
    mpi_rank = MPI.COMM_WORLD.Get_rank()

    # Start sampling
    run_sample(args)