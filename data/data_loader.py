from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from config import DATASET_NAME, TEST_SIZE, RANDOM_STATE

def load_and_prepare_dataset():
    dataset = load_dataset(DATASET_NAME)
    df = pd.DataFrame(dataset['train'])
    df.dropna(inplace=True)
    df = df.astype(str)
    df['text'] = df.apply(lambda row: " | ".join(row.values), axis=1)
    train_texts, test_texts = train_test_split(df['text'].tolist(), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return train_texts, test_texts