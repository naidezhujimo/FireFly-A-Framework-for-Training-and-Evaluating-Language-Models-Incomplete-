import os
from functools import partial
from mpi4py import MPI
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import requests
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModel
import torch.optim as optim

class PolicyModel(nn.Module):
    def __init__(self, model_name):
        super(PolicyModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Output logits for text generation
        return outputs.logits

class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, labels=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # Prepare input tensors
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

def train_policy(model, dataloader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            # Forward pass
            outputs = model(input_ids, attention_mask)
            # Placeholder loss function; replace with actual policy loss
            loss = outputs.mean()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

class RewardModel(nn.Module):
    def __init__(self, model_name):
        super(RewardModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Use pretrained model to get hidden states
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Use [CLS] token representation or last hidden state
        if hasattr(outputs, 'last_hidden_state'):
            pooled_output = outputs.last_hidden_state[:, 0, :]
        else:
            pooled_output = outputs.hidden_states[-1][:, 0, :]
        # Predict reward
        return self.fc(pooled_output)

def train_reward(model, dataloader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            # Forward pass
            outputs = model(input_ids, attention_mask)
            # MSE loss against human-provided rewards
            loss = nn.MSELoss()(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    data = ["Sample text 1", "Sample text 2", "Sample text 3"]
    labels = [1.0, 0.5, 0.8]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Add padding token

    # Create dataset and dataloader
    dataset = TextDataset(data, tokenizer, max_length=128, labels=labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize models
    policy_model = PolicyModel('gpt2').to(device)
    reward_model = RewardModel('gpt2').to(device)

    # Initialize optimizers
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=1e-5)
    reward_optimizer = optim.Adam(reward_model.parameters(), lr=2.5e-5)

    # Train models
    train_policy(policy_model, dataloader, policy_optimizer, device)
    train_reward(reward_model, dataloader, reward_optimizer, device)

if __name__ == '__main__':
    main()