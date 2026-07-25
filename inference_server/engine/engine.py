from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class Engine:
    def __init__(self, hf_model_name, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(hf_model_name).to(self.device)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    #return a list of decoded model responses in order
    def forward_pass(self, batch_dict, max_gen_toks=50) -> list:
        prompt_ids = batch_dict["input_ids"]
        generated_ids = prompt_ids
        attention_mask = batch_dict["attention_mask"]
        past_kv = None
        new_token = None

        for i in range(max_gen_toks):
            with torch.no_grad():
                if i == 0:
                    outputs = self.model(**batch_dict, past_key_values=past_kv, use_cache=True)
                else:
                    # extend the mask by one column of 1s for the new token —
                    # every prior column (real tokens + original padding) keeps its value
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.shape[0], 1), device=self.device, dtype=attention_mask.dtype)],
                        dim=1,
                    )
                    outputs = self.model(
                        input_ids=new_token,
                        attention_mask=attention_mask,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                past_kv = outputs.past_key_values
                print(type(past_kv.layers[0]))
                print([a for a in dir(past_kv.layers[0]) if not a.startswith('_')])
                new_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
                generated_ids = torch.cat([generated_ids, new_token], dim=1)

        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return decoded



def main():
    engine = Engine("Qwen/Qwen3-4B", "cuda")
    #cpu work here
    questions = [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is the difference between machine learning and deep learning?",
        "Explain the water cycle in simple terms.",
        "Who wrote the theory of relativity?",
        "What causes earthquakes?",
        "How does the human immune system fight viruses?",
        "What is quantum entanglement?",
        "Why is the sky blue?",
        "What is the difference between a virus and a bacteria?",
    ]
    messages_batch = [[{"role": "user", "content": q}] for q in questions]

    #use chat template for qwen models

    batch_dict = engine.tokenizer.apply_chat_template(
        messages_batch,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict=True,
        enable_think=False
    ).to(engine.device)

    # batch_dict = engine.tokenizer(questions, padding=True, truncation=True, return_tensors="pt").to(engine.device)

    decoded = engine.forward_pass(batch_dict)

    for i, q_and_a in enumerate(decoded):
        print(f"{i}: {q_and_a}")



if __name__ == "__main__":
    main()
