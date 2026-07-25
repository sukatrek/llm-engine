import torch
import torch.nn.functional as F
from collections import deque
from scheduler.sequence import SequenceState

class FakeEngine:
    def __init__(self, model, tokenizer, device):
        self.device = device
        self.tokenizer = tokenizer
        self.model = model

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
            elif s.num_processed_tokens >= len(s.prompt_tokens) and s.state == SequenceState.PREFILL:
                s.state = SequenceState.DECODE
                s.num_processed_tokens = 0
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

    def prefill_batch(self, sequences):
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
        batch_dict = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict, use_cache=True)

        for i, s in enumerate(sequences):
            s.state = SequenceState.PREFILL  # brief, resolves immediately below — kept for observability/logging, not correctness
            s.num_processed_tokens = len(s.prompt_tokens)  # prefill consumed the whole prompt in one pass
            s.state = SequenceState.DECODE

        return outputs.past_key_values, batch_dict["attention_mask"]

    def merge_into_batch(self, batch_cache, batch_mask, running_max, new_cache, new_mask):
        """
        Pad whichever side (existing batch vs. newly-prefilled admits) has
        the shorter seq_len up to match the other, left-padding both cache
        and mask together (never one without the other — a padded cache
        position with no matching mask entry is silently wrong, not just
        differently shaped). Then concatenate along the batch axis.

        batch_cache/batch_mask may be None on the very first admission
        (nothing running yet) — in that case the new admits simply become
        the batch outright, no padding or concat needed.

        Returns: (merged_cache, merged_mask, new_running_max)
        """
        new_seq_len = new_mask.shape[1]

        if batch_cache is None:
            return new_cache, new_mask, new_seq_len

        if new_seq_len == running_max:
            pad_batch, pad_new = 0, 0
        elif new_seq_len > running_max:
            pad_batch, pad_new = new_seq_len - running_max, 0
        else:
            pad_batch, pad_new = 0, running_max - new_seq_len

        new_running_max = max(running_max, new_seq_len)

        if pad_batch > 0:
            for layer in batch_cache.layers:
                layer.keys = F.pad(layer.keys, (0, 0, pad_batch, 0), value=0.0)
                layer.values = F.pad(layer.values, (0, 0, pad_batch, 0), value=0.0)
            batch_mask = F.pad(batch_mask, (pad_batch, 0), value=0)

        if pad_new > 0:
            for layer in new_cache.layers:
                layer.keys = F.pad(layer.keys, (0, 0, pad_new, 0), value=0.0)
                layer.values = F.pad(layer.values, (0, 0, pad_new, 0), value=0.0)
            new_mask = F.pad(new_mask, (pad_new, 0), value=0)

        for batch_layer, new_layer in zip(batch_cache.layers, new_cache.layers):
            batch_layer.keys = torch.cat([batch_layer.keys, new_layer.keys], dim=0)
            batch_layer.values = torch.cat([batch_layer.values, new_layer.values], dim=0)
        merged_mask = torch.cat([batch_mask, new_mask], dim=0)

        return batch_cache, merged_mask, new_running_max


class Scheduler:
    """
    Owns the persistent batch state (running_max, batch_cache, batch_mask,
    the currently-running sequences, and the waiting queue) across steps.
    One instance = one independent batch. FakeEngine's functions stay
    stateless/reusable; this is where the state actually lives.
    """

    def __init__(self, engine: FakeEngine, max_batch_size):
        self.engine = engine
        self.max_batch_size = max_batch_size

        self.sequences = []          # currently running (PREFILL/DECODE)
        self.waiting_queue = deque() # WAITING, not yet admitted
        self.batch_cache = None
        self.batch_mask = None
        self.running_max = 0

    def add_request(self, sequence):
        self.waiting_queue.append(sequence)

    def step(self):
        # 1. evict finished sequences from the live batch + cache
        self.sequences = self.engine.evict_finished(self.sequences, self.batch_cache)

        # 2. figure out open slots, admit waiting sequences to fill them
        num_open_slots = self.max_batch_size - len(self.sequences)
        admitted, self.waiting_queue = self.engine.admit_from_queue(self.waiting_queue, num_open_slots)

        if admitted:
            # 3. prefill the newly-admitted sequences as their own mini-batch
            new_cache, new_mask = self.engine.prefill_batch(admitted)

            # 4. pad + merge them into the main running batch
            self.batch_cache, self.batch_mask, self.running_max = self.engine.merge_into_batch(
                self.batch_cache, self.batch_mask, self.running_max, new_cache, new_mask
            )
            self.sequences.extend(admitted)

        if not self.sequences:
            return None  # nothing running — nothing to decode this step

        # 5. run the normal decode step on the now-updated batch
        # NOTE: FakeEngine.step() only simulates state advancement — no real
        # model call, no batch_cache update. Real integration (porting into
        # Engine) needs a genuine batched decode forward pass here. See
        # concepts_log.md CHECKPOINT note.
        self.engine.step(self.sequences)
        self.running_max += 1

        return self.sequences
