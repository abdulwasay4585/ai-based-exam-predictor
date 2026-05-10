import os
from datasets import load_dataset
import pandas as pd

def download_sciq(output_dir="ml_pipeline/data"):
    print("Fetching SciQ dataset from Hugging Face...")
    dataset = load_dataset("allenai/sciq")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save splits to JSON
    for split in dataset.keys():
        file_path = os.path.join(output_dir, f"sciq_{split}.json")
        print(f"Saving {split} split to {file_path}...")
        dataset[split].to_json(file_path)
    
    print("Download and conversion complete!")

def download_classification_dataset(output_dir="ml_pipeline/data"):
    print("Fetching AG News dataset for topic classification...")
    try:
        dataset = load_dataset("ag_news")
        # AG News labels: 0:World, 1:Sports, 2:Business, 3:Sci/Tech
        # Mapping to academic-ish names
        label_map = {0: "Global Studies", 1: "Physical Education", 2: "Economics", 3: "Natural Sciences"}
        
        for split in dataset.keys():
            df = dataset[split].to_pandas()
            df['topic'] = df['label'].map(label_map)
            
            file_path = os.path.join(output_dir, f"topic_data_{split}.json")
            print(f"Saving {split} split to {file_path}...")
            df[['text', 'topic']].to_json(file_path, orient='records', lines=True)
    except Exception as e:
        print(f"Failed to download classification dataset: {e}")

if __name__ == "__main__":
    download_sciq()
    download_classification_dataset()
