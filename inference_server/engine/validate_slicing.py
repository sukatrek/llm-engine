"""
Validation for per-sequence KV cache slicing (Week 1 deliverable).

Proves: extracting one sequence's KV cache mid-generation, then continuing
decode from ONLY that extracted cache, produces the same next token as
continuing that sequence inside the full batch.

If slicing is correct, the two must match exactly (same argmax token) —
this isn't a "close enough" check, it's a "must be identical" check, since
both paths are doing the same math on the same underlying data.
"""

import copy
import torch
from engine import Engine


def run():
    engine = Engine("Qwen/Qwen3-4B", "cuda")

    # Different lengths on purpose — forces real left-padding, so the mask
    # slicing is actually being exercised, not just the cache slicing.
    questions = [
        "What is the capital of France?",
        "Explain the water cycle in simple terms, covering evaporation, condensation, and precipitation.",
        "Why is the sky blue?",
    ]
    target_index = 1  # sequence we'll extract mid-generation

    messages_batch = [[{"role": "user", "content": q}] for q in questions]
    batch_dict = engine.tokenizer.apply_chat_template(
        messages_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict=True,
        enable_think=False,
    ).to(engine.device)

    attention_mask = batch_dict["attention_mask"]
    past_kv = None
    new_token = None
    steps_before_split = 3  # run a few normal decode steps first

    # --- run the batch normally for a few steps ---
    for i in range(steps_before_split):
        with torch.no_grad():
            if i == 0:
                outputs = engine.model(**batch_dict, past_key_values=past_kv, use_cache=True)
            else:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), device=engine.device, dtype=attention_mask.dtype)],
                    dim=1,
                )
                outputs = engine.model(
                    input_ids=new_token,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                )
        past_kv = outputs.past_key_values
        new_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)

    # --- PATH A: continue sequence `target_index` inside the full batch ---
    attention_mask_a = torch.cat(
        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=engine.device, dtype=attention_mask.dtype)],
        dim=1,
    )
    with torch.no_grad():
        outputs_a = engine.model(
            input_ids=new_token,
            attention_mask=attention_mask_a,
            past_key_values=copy.deepcopy(past_kv),
            use_cache=True,
        )
    token_a = torch.argmax(outputs_a.logits[target_index, -1, :], dim=-1)

    # --- PATH B: extract target_index's cache + mask, decode standalone ---
    extracted_cache = copy.deepcopy(past_kv)
    extracted_cache.batch_select_indices(torch.tensor([target_index]))
    extracted_mask = attention_mask[target_index : target_index + 1]
    extracted_mask = torch.cat(
        [extracted_mask, torch.ones((1, 1), device=engine.device, dtype=extracted_mask.dtype)],
        dim=1,
    )
    extracted_new_token = new_token[target_index : target_index + 1]

    with torch.no_grad():
        outputs_b = engine.model(
            input_ids=extracted_new_token,
            attention_mask=extracted_mask,
            past_key_values=extracted_cache,
            use_cache=True,
        )
    token_b = torch.argmax(outputs_b.logits[0, -1, :], dim=-1)

    # --- compare ---
    print(f"Path A (full batch)   token id: {token_a.item()}  -> {engine.tokenizer.decode([token_a.item()])!r}")
    print(f"Path B (extracted)    token id: {token_b.item()}  -> {engine.tokenizer.decode([token_b.item()])!r}")
    print("MATCH" if token_a.item() == token_b.item() else "MISMATCH — slicing is not equivalent, investigate")


if __name__ == "__main__":
    run()
