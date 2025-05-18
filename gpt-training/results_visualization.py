import pandas as pd
import matplotlib.pyplot as plt

tokenizer_categories = {
    'character': 'Basic (No BPE)',
    'byte': 'Basic (No BPE)',
    'character-bpe': 'Basic (BPE)',
    'byte-bpe': 'Basic (BPE)',
    'pinyin-above-bpe': 'Chinese BPE',
    'pinyin-after-bpe': 'Chinese BPE',
    'zhuyin-bpe': 'Chinese BPE',
    'cangjie-bpe': 'Chinese BPE',
    'cangjie-component': 'Chinese BPE',
    'wubi-bpe': 'Chinese BPE',
    'wubi-component': 'Chinese BPE',
    'repacked-character-bpe': 'Repacked',
    'repacked-cangjie': 'Repacked',
    'repacked-cangjie-component': 'Repacked',
    'repacked-pinyin-above': 'Repacked',
    'repacked-pinyin-after': 'Repacked',
    'repacked-wubi': 'Repacked',
    'repacked-wubi-component': 'Repacked',
    'repacked-zhuyin': 'Repacked',
}

category_colors = {
    'Basic (No BPE)': '#D94B4B',
    'Chinese BPE': '#009AC0',
    'Basic (BPE)': '#B580E0',
    'Repacked': '#D6C800',
}

def visualize_results(path):
    # Load the CSV file
    df = pd.read_csv(path)

    df.set_index('step', inplace=True)
    df.columns = df.columns.str.replace('experiments/', '', regex=False)
    df.columns = df.columns.str.replace('subword-bpe', 'character-bpe', regex=False)
    df.columns = df.columns.str.replace('repacked-subword', 'repacked-character-bpe', regex=False)
    
    # Bar graph at 20,000 steps
    step_20000 = df.loc[20000].sort_values(ascending=False)
    bar_colors = [category_colors[tokenizer_categories[tok]] for tok in step_20000.index]
    plt.figure(figsize=(12, 6))
    step_20000.plot(kind='bar', color=bar_colors)
    plt.title('Unigram-Normalized Entropy at Step 20,000 (Sorted)')
    plt.ylabel('UNE')
    plt.xlabel('Tokenizer')
    plt.xticks(rotation=30, ha='right')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=category)
        for category, color in category_colors.items()
    ]
    plt.legend(handles=legend_elements, title='Tokenizer Type')

    plt.tight_layout()
    plt.savefig("final_comparison.png", dpi=300)
    plt.show()

    # Time series chart
    plt.figure(figsize=(14, 7))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.title('Unigram-Normalized Entropy Over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel('UNE')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig("over_time.png", dpi=300)
    plt.show()

    print("Saved figure: final_comparison.png")
    print("Saved figure: over_time.png")

def visualize_per_token_entropy(path):
    # Load the CSV file
    df = pd.read_csv(path)

    # Bar graph of number of val. tokens
    df = df.T
    df.columns = ['Num_Tokens']
    df.index.name = 'Tokenizer'
    df_sorted = df.sort_values(by='Num_Tokens', ascending=False)
    df_sorted.index = df_sorted.index.str.replace('experiments/', '', regex=False)
    df_sorted.index = df_sorted.index.str.replace('subword-bpe', 'character-bpe', regex=False)
    df_sorted.index = df_sorted.index.str.replace('repacked-subword', 'repacked-character-bpe', regex=False)
    plt.figure(figsize=(12, 6))
    bar_colors = [category_colors[tokenizer_categories[tok]] for tok in df_sorted.index]
    plt.bar(df_sorted.index, df_sorted['Num_Tokens'], width=0.5, color=bar_colors)
    plt.title("Per-Token Entropy Ratios by Tokenizer (Sorted)")
    plt.xlabel("Tokenizer")
    plt.ylabel("Per-Token Entropy Ratios")
    plt.xticks(rotation=30, ha='right')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=category)
        for category, color in category_colors.items()
    ]
    plt.legend(handles=legend_elements, title='Tokenizer Type')

    plt.tight_layout()
    plt.savefig("per_token_entropy_ratios.png", dpi=300)
    plt.show()

    print("Saved figure: per_token_entropy_ratios.png")

def visualize_num_tokens(path):
    # Load the CSV file
    df = pd.read_csv(path)

    # Bar graph of number of val. tokens
    df = df.T
    df.columns = ['Num_Tokens']
    df.index.name = 'Tokenizer'
    df_sorted = df.sort_values(by='Num_Tokens', ascending=False)
    df_sorted.index = df_sorted.index.str.replace('experiments/', '', regex=False)
    df_sorted.index = df_sorted.index.str.replace('subword-bpe', 'character-bpe', regex=False)
    df_sorted.index = df_sorted.index.str.replace('repacked-subword', 'repacked-character-bpe', regex=False)
    plt.figure(figsize=(12, 6))
    bar_colors = [category_colors[tokenizer_categories[tok]] for tok in df_sorted.index]
    plt.bar(df_sorted.index, df_sorted['Num_Tokens'], width=0.5, color=bar_colors)
    plt.title("Number of Validation Tokens (Sorted)")
    plt.xlabel("Tokenizer")
    plt.ylabel("Number of Tokens")
    plt.xticks(rotation=30, ha='right')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=category)
        for category, color in category_colors.items()
    ]
    plt.legend(handles=legend_elements, title='Tokenizer Type')

    plt.tight_layout()
    plt.savefig("num_tokens.png", dpi=300)
    plt.show()

    print("Saved figure: num_tokens.png")

if __name__ == "__main__":
    # Generate per-token entropy ratios bar graph
    # visualize_per_token_entropy("/mnt/storage/ntunggal/thesis-ntunggal/gpt-training/per_token_entropy_ratios.csv")

    # Generate main results
    # visualize_results("/mnt/storage/ntunggal/thesis-ntunggal/gpt-training/compiled_entropy_ratios.csv")

    # View num tokens
    visualize_num_tokens("/mnt/storage/ntunggal/thesis-ntunggal/gpt-training/num_tokens.csv")