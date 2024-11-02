"""
Contains useful functions. Currently not used by anything
"""
import time
from pathlib import Path
import os
import csv

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
    

def write_results(filename, results):    
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

def clean_checkpoint_folders():
    pass