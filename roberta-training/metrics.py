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
                logits.view(-1, len(tokenizer)),
                input_ids.view(-1),
                reduction='none'
            ).view(input_ids.size())  # Reshape to (batch_size, sequence_length)

            

            # Divide by ln(2) to get bits per token, since cross_entropy uses ln
            # Shape: (batch_size,)
            bits_per_token = (log_probs.sum(dim=1) / input_ids.size(1)) / np.log(2)

            print(bits_per_token)

            # Calculate tokens per byte
            toks = []
            for ids in input_ids:
                toks.extend([tokenizer.decode(torch.tensor([token_id]), skip_special_tokens=True) for token_id in ids])
            print(toks)
            
            utf8_toks = [len(tok.encode('utf-8')) for tok in toks]
            tokens_per_byte = len(utf8_toks) / sum(utf8_toks)
            print(tokens_per_byte)
            
                
            
            #utf8_toks = [len(tok.encode('utf-8')) for tok in toks]
            #print(utf8_toks)
            
            # input_lengths_bytes = torch.tensor(
            #     [len(tokenizer.decode(ids, skip_special_tokens=True).encode('utf-8')) for ids in input_ids],
            #     dtype=torch.float32,
            #     device=device
            # )
            # tokens_per_byte = input_ids.size(1) / input_lengths_bytes
            # print(tokens_per_byte)

            # Add avg BPB calculation for batch
            total_bpb += (bits_per_token * tokens_per_byte).mean().item()
            total_batches += 1

    # Return average BPB over all batches
    return total_bpb / total_batches


class SimpleTokenizer:
    def __init__(self):
        self.token_map = '你好cd'
        self.pad_token_id = 4
    
    def __len__(self):
        return 4
    
    def _decode_token(self, id):
        return self.token_map[id]
    
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join([self._decode_token(tok) for tok in ids.tolist()])
        

class UnigramLMOutput:
    
    def __init__(self, logits):
        self.logits = logits
    
    
class UnigramLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
            
    def forward(self, input_ids, attention_mask, labels):
        
        def get_probs(token):
            return [0.8664,0.1732,-0.5199,-0.5199]  
        
        def map_probs(token_seq):
            return [get_probs(tok) for tok in token_seq]
        
        logits = torch.tensor([map_probs(tok_seq) for tok_seq in input_ids.tolist()])
        return UnigramLMOutput(logits)

EXAMPLE = torch.tensor([[0, 0, 1, 2]])
EXAMPLE1 = torch.tensor([[0, 1, 0, 1, 2, 0, 0, 3],
                                      [0, 1, 0, 1, 2, 0, 3, 0]])

def my_eval_loader():
    yield {'input_ids': EXAMPLE}

if __name__ == "__main__":
    input_ids = EXAMPLE
    labels = input_ids.clone()
    attention_mask = torch.ones((2, 8))
    model = UnigramLM()
    output = model(input_ids, attention_mask, labels)
    tokenizer = SimpleTokenizer()
    #print(tokenizer.decode(input_ids[0]))
    eval_loader = my_eval_loader()
    result = compute_bits_per_byte(model, tokenizer, eval_loader, device='cpu')
    print(result)
