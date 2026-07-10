# Engineering Decisions + Hard Problems

## Slicing per-sequence KV cache from batched forward pass
**Problem:** Running prefill as a batched forward pass returns a single
past_key_values tensor covering all sequences. Need to split it back
into per-sequence caches for continuous batching. 
Each Sequence object should have its own past_key_values tensor

**Solution:** Slice on batch dim with i:i+1 to preserve batch dimension,
store each slice on the Sequence object.

**Why it's hard:** Shapes are non-obvious, easy to lose the batch dim
with a scalar index instead of a slice.

---

## num_processed_tokens missing from initial Sequence design
**Problem:** State enum alone isn't enough to track chunked prefill
progress — need to know how far through the prompt we are so we know when to update the states.

**Solution:** Added num_processed_tokens: int = 0 to Sequence dataclass.
Transitions to DECODE when num_processed_tokens == len(prompt_tokens).
