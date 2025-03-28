import torch
from data_tokenizer.data_tokenizer import tokenize_texts
from config import BATCH_SIZE

def run_inference(model, tokenizer, texts, device):
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = tokenize_texts(batch, tokenizer)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            results.extend(predictions.cpu().numpy().tolist())
    return results
