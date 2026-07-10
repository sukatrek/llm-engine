from enum import Enum
import torch
from datetime import datetime
from dataclasses import dataclass, field

class SequenceState(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    CANCELLED = "cancelled"



class SequenceTier(Enum):
    FREE = "free"
    PRO = "pro"
    MAX = "max"



@dataclass
class Sequence:
    seq_id: int
    state: SequenceState
    prompt: str
    prompt_tokens: torch.Tensor
    arrival_time: datetime
    tier: SequenceTier
    max_gen_tokens: int
    num_processed_tokens: int = 0
    past_key_values: any = None
    output_tokens: list[int] = field(default_factory=list)



if __name__ == "__main__":
    seq = Sequence(
        seq_id=1,
        state=SequenceState.WAITING,
        prompt="What is the capital of France?",
        prompt_tokens=torch.tensor([1, 2, 3, 4]),
        arrival_time=datetime.now(),
        tier=SequenceTier.PRO,
        max_gen_tokens=50
    )
    print(seq)
    print(seq.state)
    seq.state = SequenceState.PREFILL
    print(seq.state)
