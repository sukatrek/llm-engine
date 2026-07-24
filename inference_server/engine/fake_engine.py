import torch
from scheduler.sequence import SequenceState

class FakeEngine:
    def __init__(self, hf_model_name, device):
        self.device = device
        self.tokenizer = None
        self.model = hf_model_name
        # self.tokenizer.padding_side = "left"
        # self.tokenizer.pad_token = self.tokenizer.eos_token

    #simulate predicting the next token for a batch
    def step(self, sequences) -> None:
        # decode_seqs = [s for s in sequences if s.state == SequenceState.DECODE]
        # prefill_seqs = [s for s in sequences if s.state == SequenceState.PREFILL]
        #assume both prefill and decode sequences are done
        #assuming no errors
        for s in sequences:
            s.num_processed_tokens += 1
            if s.num_processed_tokens == s.max_gen_tokens:
                s.state = SequenceState.FINISHED
            elif s.num_processed_tokens >= len(s.prompt_tokens):
                s.state = SequenceState.DECODE
        return

    def evict_finished(self, sequences, cache=None):
        """
        Remove finished sequences from the live cache (in place) and return
        the remaining sequences in the same relative order.

        Indices are recomputed fresh from `sequences` every call — never
        cached across steps, since eviction shifts every later position.
        """
        remaining_indices = [
            i for i, s in enumerate(sequences)
            if s.state != SequenceState.FINISHED
        ]
        remaining_sequences = [sequences[i] for i in remaining_indices]

        if cache is not None and len(remaining_indices) < len(sequences):
            cache.batch_select_indices(torch.tensor(remaining_indices))

        return remaining_sequences

    def admit_from_queue(self, waiting_queue, num_open_slots):
        """
        Pure selection: pull the next `num_open_slots` WAITING sequences off
        the front of the queue.

        Assumes `waiting_queue` is a collections.deque, already in
        arrival order by construction (requests are appended as they
        arrive) — no sort needed for plain FCFS. Does NOT prefill or merge
        anything, so priority/tier logic can replace just this function
        later without touching prefill_batch or merge_into_batch.

        Returns (admitted, remaining_queue) as a list and a deque —
        does not mutate the caller's waiting_queue in place.
        """
        if num_open_slots <= 0:
            return [], waiting_queue

        remaining_queue = waiting_queue.copy()
        admitted = [remaining_queue.popleft() for _ in range(min(num_open_slots, len(remaining_queue)))]

        return admitted, remaining_queue

    def prefill_batch(self, sequences, model, tokenizer, device):
        """
        Run one forward pass over the newly-admitted sequences together as
        their own mini-batch (own left-padding, own attention mask — this
        batch has nothing to do with the main running batch yet).

        Mutates each sequence's `state` and `num_processed_tokens` in
        place, same pattern as FakeEngine.step(). Does NOT populate
        `past_key_values` on the sequences — the resulting cache only
        exists as a combined batch-of-N tensor, returned separately, since
        nothing downstream ever needs a single sequence's cache in
        isolation (confirmed: eviction only needs `state`, not the cache).

        Returns: (combined_cache, combined_mask) for this mini-batch of N —
        the caller (merge_into_batch) is responsible for padding this
        against the main batch and concatenating.
        """
        prompts = [s.prompt for s in sequences]
        batch_dict = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**batch_dict, use_cache=True)

        for i, s in enumerate(sequences):
            s.state = SequenceState.PREFILL  # brief, resolves immediately below — kept for observability/logging, not correctness
            s.num_processed_tokens = len(s.prompt_tokens)  # prefill consumed the whole prompt in one pass
            s.state = SequenceState.DECODE

        return outputs.past_key_values, batch_dict["attention_mask"]
