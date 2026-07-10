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
            elif s.state == SequenceState.PREFILL:
                s.state = SequenceState.DECODE

        return
