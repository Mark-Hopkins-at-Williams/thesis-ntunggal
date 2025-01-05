"""
Contains useful utility functions:
- write_results: write a results dict to a csv file
- clean_checkpoint_folders: delete checkpoint folders in a directory
- CustomLineByLineTextDataset
"""
import time
from pathlib import Path
import os
import csv
import shutil
from torch.utils.data import DataLoader, Dataset

def acquire_lock(filename, check_interval=1):
    lock_file = filename + ".lock"
    print(f"waiting to acquire lock for {filename}...")
    while os.path.exists(lock_file):
        time.sleep(check_interval)
    print('...acquired!')
    file_path = Path(lock_file)
    file_path.touch()
    

def release_lock(filename):
    lock_file = filename + ".lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)
    print(f'lock released on: {filename}')
    

def write_results(filename: str, results: dict) -> None:    
    """
    Writes results to a csv file. filename should include .csv
    """
    data = [results[key] for key in sorted(results.keys())]  
    headers = sorted(results.keys())
    
    try:
        acquire_lock(filename)
        # If log file doesn't exist, write the headers
        if not os.path.isfile(filename):
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
        # Write results to csv
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        release_lock(filename)

    except Exception:
        release_lock(filename)

def clean_checkpoint_folders(dir) -> None:
    """
    Removes checkpoint folders in given directory
    """
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isdir(file_path) and filename.startswith("checkpoint-"):
            shutil.rmtree(file_path)


class CustomLineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        with open(file_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        encoding = self.tokenizer(line, truncation=True, padding='max_length', max_length=self.block_size, return_tensors="pt")

        input_ids = encoding['input_ids'].squeeze(0)  # Shape: (block_size,)
        attention_mask = encoding['attention_mask'].squeeze(0)  # Shape: (block_size,)

        return {'input_ids': input_ids, 'attention_mask': attention_mask}