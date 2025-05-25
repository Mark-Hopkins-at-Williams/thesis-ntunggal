from datasets import load_dataset, Dataset
from entropy import learn_unigram_model, evaluate_unigram_model


def find_final_period(line, max_length):
    period_index = -1
    while period_index < max_length:
        if line.find('。', period_index + 1) >= max_length:
            break
        elif line.find('。', period_index + 1) == -1:
            break
        period_index = line.find('。', period_index + 1)
    return period_index


def load_baai_data():
    train_dataset = load_dataset("BAAI/CCI3-HQ", split="train", streaming=True)
    subset = []
    for i, row in enumerate(train_dataset):
        if "text" in row and len(row["text"].strip()) > 0:  # Filter invalid data
            text = row["text"]
            period_index = find_final_period(text, 256)
            if 100 <= period_index < 256: 
                trimmed = text[:period_index+1]
                subset.append(trimmed)
        if len(subset) >= 5000:  # Limit the subset size
            break
    unigram_model = learn_unigram_model(subset)
    validation_entropy = evaluate_unigram_model(unigram_model, subset)
    validation_dataset = Dataset.from_dict({"text": [item for item in subset]})
    return train_dataset, validation_dataset, validation_entropy

if __name__ == "__main__":
    load_baai_data()