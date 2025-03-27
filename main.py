from datasets.data_loader import load_and_prepare_dataset
from tokenizer.tokenizer import load_tokenizer
from models.models import load_model, move_to_device
from attack.inference import run_inference
from attack.attack import simulate_privacy_attack

def main():
    train_texts, test_texts = load_and_prepare_dataset()
    tokenizer = load_tokenizer()
    model = load_model()
    model, device = move_to_device(model)
    predictions = run_inference(model, tokenizer, test_texts, device)
    attack_results = simulate_privacy_attack(test_texts, predictions)
    for text, label in attack_results:
        print("Potential Leakage:")
        print("Text:", text)
        print("Predicted Label:", label)

if __name__ == "__main__":
    main()
