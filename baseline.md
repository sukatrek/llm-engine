## Week 1 — Static Batching (Prefill Only)
- Model: Pythia-160m
- Batch size: 6 chunks, seq_len=32
- Avg latency: 44.39ms
- Throughput: 7230 seq/sec
- GPU util: 99% (during compute), ~25% average (including idle gaps)
- Power: 211W / 300W


## Week 2 — KV Cache
- No KV cache: 473.85ms/batch, 677 seq/sec
- With KV cache: 98.30ms/batch, 3265 seq/sec
- Speedup: 4.8x
- Model: Pythia-160m, batch ~2500 chunks, seq_len=32, max_new_tokens=10
