from transformers import AutoModelForSequenceClassification
from config import MODEL_NAME
import torch

def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return model

def move_to_device(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device
