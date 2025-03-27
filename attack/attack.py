def simulate_privacy_attack(texts, predictions):
    leakage = []
    for text, pred in zip(texts, predictions):
        if "Private" in text or "Confidential" in text:
            leakage.append((text, pred))
    return leakage
