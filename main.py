from data.data_loader import load_and_prepare_dataset
from data_tokenizer.data_tokenizer import load_tokenizer
from models.models import load_model, move_to_device
from attack.inference import run_inference
from attack.attack import simulate_privacy_attack
import pandas as pd

def main():
    train_texts, test_texts = load_and_prepare_dataset()
    tokenizer = load_tokenizer()
    model = load_model()
    model, device = move_to_device(model)
    predictions = run_inference(model, tokenizer, test_texts, device)
    attack_results = simulate_privacy_attack(test_texts, predictions)
    
    results = []
    for text, label in attack_results:
        print("Potential Leakage:")
        print("Text:", text)
        print("Predicted Label:", label)
        results.append({"text": text, "predicted_label": label})

    df = pd.DataFrame(results)
    df.to_csv("attack_results.csv", index=False)
    with open("attack_results.txt", "w") as f:
        for row in results:
            f.write(f"Text: {row['text']}\nPredicted Label: {row['predicted_label']}\n\n")

if __name__ == "__main__":
    main()