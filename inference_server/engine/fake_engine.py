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
