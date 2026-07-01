import argparse

import torch
from transformers import AutoModel


def resolve_mlps(model):
    """Return (list_of_mlp_modules, hook_submodule_name, is_llama).

    hook_submodule_name is the attribute on each MLP whose output we count.
    """
    # LLaMA / Mistral / most SwiGLU decoders: model.layers[i].mlp.gate_proj
    if hasattr(model, "layers"):
        mlps = [layer.mlp for layer in model.layers]
        if hasattr(mlps[0], "gate_proj"):
            return mlps, "gate_proj", True
    # BLOOM: model.h[i].mlp.gelu_impl
    if hasattr(model, "h"):
        mlps = [block.mlp for block in model.h]
        if hasattr(mlps[0], "gelu_impl"):
            return mlps, "gelu_impl", False
    raise ValueError(
        "Unrecognized architecture; expected a LLaMA-like (.layers[i].mlp.gate_proj) "
        "or BLOOM-like (.h[i].mlp.gelu_impl) model."
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-m",
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HF model name.",
    )
    p.add_argument("-l", "--lang", type=str, default="zh", help="Language code.")
    p.add_argument(
        "-s",
        "--split",
        type=str,
        default="train",
        help="Split label in the input/output filenames.",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Output tag, e.g. 'llama-7b' or 'llama-70b'. Defaults to "
        "'llama-7b'/'bloom-7b' to match the original script.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Sequence block length. Defaults to the model's max "
        "position embeddings (capped at 4096).",
    )
    p.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Sequences per forward pass. Keep small on 16GB GPUs.",
    )
    p.add_argument(
        "--limit_tokens",
        type=int,
        default=99999744,
        help="Cap on total tokens processed (matches the original).",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--device", type=str, default="cuda")

    args = p.parse_args()
    dtype = getattr(torch, args.dtype)

    is_llama_name = args.model.lower().find("llama") >= 0
    default_tag = "llama-7b" if is_llama_name else "bloom-7b"
    tag = args.tag or default_tag
    model_tag = "llama" if is_llama_name else "bloom"

    print(f"Loading {args.model} ({args.dtype}) ...")
    # AutoModel (not ForCausalLM) skips the lm_head. We only need MLP activations,
    # which saves compute and a few GB of memory.
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device).eval()

    mlps, hook_attr, is_llama = resolve_mlps(model)
    num_layers = len(mlps)

    if is_llama:
        intermediate_size = mlps[0].gate_proj.out_features
    else:
        intermediate_size = mlps[0].dense_h_to_4h.out_features

    max_length = args.max_length or min(
        getattr(model.config, "max_position_embeddings", 4096), 4096
    )

    over_zero = torch.zeros(
        num_layers, intermediate_size, dtype=torch.int64, device=args.device
    )

    # Register forward hooks that accumulate positive-activation counts per neuron.
    def make_hook(idx):
        def hook(_module, _inp, out):
            # out: (batch, seq, intermediate). Count > 0 over batch+seq, per neuron
            over_zero[idx] += (out > 0).sum(dim=(0, 1)).to(torch.int64)
        return hook

    handles = [
        getattr(mlp, hook_attr).register_forward_hook(make_hook(i))
        for i, mlp in enumerate(mlps)
    ]

    # Load packed token ids (flat 1-D LongTensor produced by tokenizer.py).
    in_path = f"data/id.{args.lang}.{args.split}.{model_tag}"
    ids = torch.load(in_path)
    total = min(ids.size(0), args.limit_tokens) // max_length * max_length
    input_ids = ids[:total].reshape(-1, max_length)
    num_seqs = input_ids.size(0)
    print(
        f"{in_path}: {ids.size(0)} tokens -> {num_seqs} x {max_length} "
        f"({total} tokens used)"
    )

    with torch.no_grad():
        for start in range(0, num_seqs, args.batch_size):
            batch = input_ids[start : start + args.batch_size].to(args.device)
            model(input_ids=batch)
            done = min(start + args.batch_size, num_seqs)
            print(f"  {done}/{num_seqs} sequences", end="\r")
    print()

    for h in handles:
        h.remove()

    output = dict(n=total, over_zero=over_zero.to("cpu", torch.int32))
    out_path = f"data/activation.{args.lang}.{args.split}.{tag}"
    torch.save(output, out_path)
    print(f"Saved activations to {out_path}")


if __name__ == "__main__":
    main()
