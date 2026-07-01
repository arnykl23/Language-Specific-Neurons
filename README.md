# Language-Specific-Neurons (HuggingFace transformers)

The original scripts above are pinned to `vllm==0.2.7`, which does not run on newer GPUs (e.g. NVIDIA Blackwell / RTX 50-series). This fork adds vLLM-free reimplementations built on plain HuggingFace `transformers` that produce the same outputs, so the full pipeline runs on current PyTorch/CUDA.

- `tokenizer.py` — tokenize `wikimedia/wikipedia` into the `data/id.{lang}.{split}.{tag}` token-id tensors the scripts consume.
- `activation_hf.py` — drop-in replacement for `activation.py` (records per-neuron activation counts via forward hooks).
- `ppl_hf.py` — drop-in replacement for `ppl.py` (perplexity under neuron deactivation).
- `generation_hf.py` — drop-in replacement for `generation.py` (batched open-ended generation under neuron deactivation).

```bash
# 1. Tokenize wikipedia (per language/dataset split)
python tokenizer.py -m meta-llama/Llama-2-7b-hf -l en -s train

# 2. Record activations, then identify neurons
CUDA_VISIBLE_DEVICES=0 python activation_hf.py -m meta-llama/Llama-2-7b-hf -l en
python identify.py

# 3. PPL and generation when deactivating neurons
CUDA_VISIBLE_DEVICES=0 python ppl_hf.py        -m meta-llama/Llama-2-7b-hf -a LLaMA-2-7B.neuron.pth
CUDA_VISIBLE_DEVICES=0 python generation_hf.py -m meta-llama/Llama-2-7b-hf -a LLaMA-2-7B.neuron.pth
```

Requires a recent `transformers` (>= 4.40 for `generation_hf.py`'s stop-string support) and a PyTorch build that matches your GPU/CUDA.
