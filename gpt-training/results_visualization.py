import pandas as pd
import matplotlib.pyplot as plt

def visualize_results(path):
    # Load the CSV file
    df = pd.read_csv(path)

    #df.set_index('step', inplace=True)
    df.columns = df.columns.str.replace('experiments/', '', regex=False)

    # Bar graph of number of val. tokens
    df = df.T
    df.columns = ['Num_Tokens']
    df.index.name = 'Tokenizer'
    df_sorted = df.sort_values(by='Num_Tokens', ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(df_sorted.index, df_sorted['Num_Tokens'], width=0.5)
    plt.title("Number of Validation Tokens (Sorted)")
    plt.xlabel("Tokenizer")
    plt.ylabel("Number of Tokens")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("num_tokens.png", dpi=300)
    plt.show()
    return
    # Bar graph at 20,000 steps
    step_20000 = df.loc[20000].sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    step_20000.plot(kind='bar')
    plt.title('Unigram-Normalized Entropy at Step 20,000 (Sorted)')
    plt.ylabel('UNE')
    plt.xlabel('Tokenizer')
    plt.xticks(rotation=30, ha='right')
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

if __name__ == "__main__":
    # Set to generate num tokens bar graph
    visualize_results("/mnt/storage/ntunggal/thesis-ntunggal/gpt-training/num_tokens.csv")