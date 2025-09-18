from datasets import load_dataset
from pprint import pprint

# Load the dataset
ds = load_dataset("/mnt/vast/data/speech/libriheavy")
pprint(ds)
print()
pprint(ds['train'][0])