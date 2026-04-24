import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM



wiki_text = "In deep learning, the transformer is a family of artificial neural network architectures based on the multi-head attention mechanism, in which text is converted to numerical representations called tokens, and each token is converted into a vector via lookup from a word embedding table.[1] At each layer, each token is then contextualized within the scope of the context window with other (unmasked) tokens via a parallel multi-head attention mechanism, allowing the signal for key tokens to be amplified and less important tokens to be diminished."


wiki_text *= 100


# small_wiki = wiki_text[:len(wiki_text) // 4]


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

    input_ids = tokenizer.encode(wiki_text, return_tensors="pt")
    for i in range(0, input_ids.size(1) - seq_len, seq_len):
        chunk = input_ids[0, i : i + seq_len]
        batch.append(chunk)

    #stack then move to cuda
    batch_tensor = torch.stack(batch).to("cuda")
    # 1. Verify correctness
    with torch.no_grad():
        outputs = model(input_ids=batch_tensor)
    logits = outputs.logits
    next_token_logits = logits[:, -1, :]
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    decoded = tokenizer.batch_decode(next_tokens.unsqueeze(-1), skip_special_tokens=True)
    prompts = [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in batch]

    for prompt, token in zip(prompts, decoded):
        print(f"{prompt!r} -> {token!r}")

    torch.cuda.synchronize()
    start = time.perf_counter()
    N = 500
    for _ in range(N):
        with torch.no_grad():
            model(input_ids=batch_tensor)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Avg latency: {elapsed/N*1000:.2f}ms per batch")
    print(f"Throughput: {len(batch)*N/elapsed:.1f} sequences/sec")


if __name__ == "__main__":
    main()
