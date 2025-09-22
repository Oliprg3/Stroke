"""Placeholder: parse RSNA ICH dataset CSV and generate train/val/test JSON meta.
Each row should be:
{ "study_id": "...", "nifti_path": "...", "label": 0 or 1 }
"""

#Update kaggle API key""

import json, random, os, csv

def main():
    samples = []  
    random.shuffle(samples)
    n = len(samples)
    train = samples[:int(0.7*n)]
    val = samples[int(0.7*n):int(0.85*n)]
    test = samples[int(0.85*n):]
    os.makedirs("data", exist_ok=True)
    json.dump(train, open("data/train_meta.json","w"))
    json.dump(val, open("data/val_meta.json","w"))
    json.dump(test, open("data/test_meta.json","w"))
    print("Wrote data splits.")

if __name__ == "__main__":
    main()
