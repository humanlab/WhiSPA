import os
from datasets import load_dataset
from pprint import pprint
from dotenv import load_dotenv

load_dotenv()

# Load the dataset
ds = load_dataset(os.getenv('LIBRISPEECH_DIR'))
pprint(ds)
print()
pprint(ds['train'][0])