"""
File containing evaluation metrics.
"""
import torch
import numpy as np
from torch.nn.functional import cross_entropy

def compute_bits_per_byte(model, tokenizer, eval_loader, device):
    
    model.eval()
    total_bpb = 0.0
    total_batches = 0

    for batch in eval_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = (batch['input_ids'] != tokenizer.pad_token_id).to(device)

        with torch.no_grad():
            
            # Get unnormalized log probs for each token in vocab, for each position in sequence
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits # shape: (batch_size, sequence_length, vocab_size)

            # Get log probs (cross entropy) for each token in sequence
            log_probs = cross_entropy(
                logits.view(-1, model.config.vocab_size),
                input_ids.view(-1),
                reduction='none'
            ).view(input_ids.size())  # Reshape to (batch_size, sequence_length)

            # Divide by ln(2) to get bits per token, since cross_entropy uses ln
            # Shape: (batch_size,)
            bits_per_token = (log_probs.sum(dim=1) / input_ids.size(1)) / np.log(2)

            # Calculate tokens per byte
            input_lengths_bytes = torch.tensor(
                [len(tokenizer.decode(ids, skip_special_tokens=True).encode('utf-8')) for ids in input_ids],
                dtype=torch.float32,
                device=device
            )
            tokens_per_byte = input_ids.size(1) / input_lengths_bytes

            # Add avg BPB calculation for batch
            total_bpb += (bits_per_token * tokens_per_byte).mean().item()
            total_batches += 1

    # Return average BPB over all batches
    return total_bpb / total_batches

