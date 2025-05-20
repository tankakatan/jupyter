import os
import spacy
import spacy.cli

from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

spacy_cache = "./models/spacy"
spacy_model = "ru_core_news_lg"
spacy_path = os.path.join(spacy_cache, spacy_model)

try:
  print(f"Accessing spacy `{spacy_model}` model")
  nlp = spacy.load(spacy_path)
  print(f"Successfully loaded `{spacy_model}`")
except OSError as e:
  print(f"Model `{spacy_model}` not found, downloading")
  spacy.cli.download(spacy_model)
  print(f"Successfully downloaded `{spacy_model}`")
  nlp = spacy.load(spacy_model)
  nlp.to_disk(spacy_path)
  print(f"Dumped `{spacy_model}` on disk")

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

print(f"Obtained hugging face token: `{hf_token[0:4]}")

custom_hf_dir = "./models/huggingface/mistralai/Mistral-7B-Instruct-v0.1"
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

try:
  tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=custom_hf_dir, token=hf_token)
  print(f"Successfully loaded `{model_id}` from memory")
except Exception as e:
  print(f"Downloading model `{model_id}` from Hugging Face Hub")
  model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=custom_hf_dir, token=hf_token)
  print(f"Successfully downloaded model `{model_id}`")

