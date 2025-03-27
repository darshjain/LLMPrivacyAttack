from transformers import AutoTokenizer
from config import MODEL_NAME

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return tokenizer

def tokenize_texts(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")