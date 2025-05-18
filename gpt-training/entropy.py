import math

LANG1 = ['abacabad', 'adababac', 'abababac', 'adacadab']  # structured
LANG2 = ['abdabadd', 'adbbaaac', 'aadabdaa', 'abcbbaca']  # unstructured


def normalize_counts(counts):
    sum = 0
    for key in counts:
        sum += counts[key]
    return {key: counts[key] / sum for key in counts}

def learn_unigram_model(sequences):
    counts = {'</s>': 0}
    for seq in sequences:
        for char in seq:
            if char not in counts:
                counts[char] = 0
            counts[char] += 1
        counts['</s>'] += 1
    # Returns dict of character counts, normalized
    return normalize_counts(counts)

def evaluate_unigram_model(model, sequences):
    entropy = 0.0
    num_chars = 0
    for seq in sequences:
        for char in seq:
            entropy += -math.log2(model[char])
            num_chars += 1
        entropy += -math.log2(model['</s>'])
        num_chars += 1
    #print(f"num chars validation set: {num_chars}", flush=True)
    return entropy

def unigram_model_baseline(text_file):
    with open(text_file) as reader:
        model = learn_unigram_model(reader)
    with open(text_file) as reader:
        entropy = evaluate_unigram_model(model, reader)
    return entropy


if __name__ == "__main__":
    print(unigram_model_baseline('/mnt/storage/hopkins/data/chinese-monolingual/simple/dev.txt'))