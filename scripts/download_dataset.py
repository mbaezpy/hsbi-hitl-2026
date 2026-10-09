import sys
from datasets import load_dataset

def main():
    if len(sys.argv) not in [3, 4]:
        print("Usage: python download_dataset.py <dataset_name> [<config_name>] <output_path>")
        print("Example 1: python download_dataset.py dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset /tmp/spam")
        print("Example 2: python download_dataset.py Salesforce/wikitext wikitext-2-raw-v1 /tmp/wiki")
        sys.exit(1)

    # handle optional config
    if len(sys.argv) == 4:
        dataset_name = sys.argv[1]
        config_name = sys.argv[2]
        output_path = sys.argv[3]
    else:
        dataset_name = sys.argv[1]
        config_name = None
        output_path = sys.argv[2]

    print(f"Loading dataset: {dataset_name}" + (f" ({config_name})" if config_name else ""))
    
    if config_name:
        ds = load_dataset(dataset_name, config_name)
    else:
        ds = load_dataset(dataset_name)

    print(f"Saving dataset to: {output_path}")
    ds.save_to_disk(output_path)

    print("Done.")

if __name__ == "__main__":
    main()
