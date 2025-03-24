# python download_model.py gpt2 D:/models/policy

from transformers import AutoTokenizer, AutoModel
import argparse
import os

def download_and_save_model(model_name, save_dir):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Load and save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)
    
    # Load and save the model
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a pretrained model and save it to a directory.")
    parser.add_argument("model_name", type=str, help="Name of the pretrained model to download.")
    parser.add_argument("save_dir", type=str, help="Directory to save the downloaded model.")
    args = parser.parse_args()

    model_name = args.model_name
    save_dir = args.save_dir

    download_and_save_model(model_name, save_dir)