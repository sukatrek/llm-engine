import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM



wiki_text = "In deep learning, the transformer is a family of artificial neural network architectures based on the multi-head attention mechanism, in which text is converted to numerical representations called tokens, and each token is converted into a vector via lookup from a word embedding table.[1] At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism, allowing the signal for key tokens to be amplified and less important tokens to be diminished."


wiki_text *= 100


# small_wiki = wiki_text[:len(wiki_text) // 4]

# TENSOR CONCATENATION REFERENCE
# Adding a new token to each sequence in a batch
#
# input_ids shape: [B, T] = [3, 4]
# [[1,  2,  3,  4],
#  [5,  6,  7,  8],
#  [9, 10, 11, 12]]
#
# next_tokens shape: [B] = [3]
# [100, 200, 300]
#
# Step 1: unsqueeze(1) → [B, 1] = [3, 1]
# [[100],
#  [200],
#  [300]]
#
# Step 2: torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1) → [B, T+1] = [3, 5]
# [[1,   2,  3,  4, 100],
#  [5,   6,  7,  8, 200],
#  [9,  10, 11, 12, 300]]
#
# dim=1 means "concatenate along columns" (T grows, B stays fixed)


def forward_pass_no_kv(model, tokenizer, batch_tensor, max_new_tokens=10):
    input_ids = batch_tensor
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            #shape is [b]
            #original input ids would have the shape [b, t]
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
    decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return decoded


def forward_pass_with_kv(model, tokenizer, batch_tensor, max_new_tokens=10):
    input_ids = batch_tensor
    full_ids = batch_tensor
    past_kv = None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            #this is where we compute the kv cache first - prefill step since the past_kv will start at None
            outputs = model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            #shape is [b] for next_tokens - we extract the highest prob token for each sequence in the batch
            #original input ids would have the shape [b, t]
            input_ids = next_tokens.unsqueeze(1)
            full_ids = torch.cat([full_ids, next_tokens.unsqueeze(1)], dim=1)

    decoded = tokenizer.batch_decode(full_ids, skip_special_tokens=True)
    return decoded


def test_speed(model, tokenizer, batch_tensor, batch, func):
    print("TESTING SPEED FOR func ", func.__name__)
    torch.cuda.synchronize()
    start = time.perf_counter()
    N = 20
    for _ in range(N):
        if torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.9:
            print("Warning: GPU memory > 90%, stopping early")
            break
        with torch.no_grad():
            func(model, tokenizer, batch_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Avg latency: {elapsed/N*1000:.2f}ms per batch")
    print(f"Throughput: {len(batch)*N/elapsed:.1f} sequences/sec")


def main():
    # print(len(wiki_text))
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m").to("cuda")

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    #creating one batch of size 4
    seq_len = 32
    batch = []

    #no masking needed here - pretraining style.

    full_wiki_ids = tokenizer.encode(wiki_text, return_tensors="pt")
    for i in range(0, full_wiki_ids.size(1) - seq_len, seq_len):
        chunk = full_wiki_ids[0, i : i + seq_len]
        batch.append(chunk)

    #stack then move to cuda
    batch_tensor = torch.stack(batch).to("cuda")
    # 1. Verify correctness

    # decoded = forward_pass_no_kv(model, tokenizer, batch_tensor)
    #
    # prompts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in batch]
    # for prompt, token in zip(prompts, decoded):
    #     print(f"{prompt!r} -> {token!r}")


    test_speed(model, tokenizer, batch_tensor, batch, forward_pass_no_kv)
    test_speed(model, tokenizer, batch_tensor, batch, forward_pass_with_kv)




if __name__ == "__main__":
    main()
