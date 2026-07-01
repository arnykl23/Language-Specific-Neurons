import argparse

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("-a", "--activation_mask", type=str, default="")
    p.add_argument(
        "-s",
        "--split",
        type=str,
        default="valid",
        help="Split label in the input filename data/id.{lang}.{split}.{tag}.",
    )
    p.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated subset of languages to mask/evaluate "
        "(default: all for the model family). Handy for smoke tests, "
        "e.g. --languages en with only en data available.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Sequence block length (default: model max, capped at 4096).",
    )
    p.add_argument("-b", "--batch_size", type=int, default=1)
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load weights in 4-bit (nf4, bitsandbytes) to fit on limited "
        "VRAM; --dtype is used as the compute dtype. Requires "
        "bitsandbytes + accelerate.",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p


def load_causal_lm(args, dtype):
    """Load an AutoModelForCausalLM, optionally 4-bit-quantized, on args.device."""
    tag = "4bit" if args.load_in_4bit else args.dtype
    print(f"Loading {args.model} ({tag}) ...")
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        # device_map places the quantized model; do NOT call .to() afterwards.
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant_config, device_map={"": args.device}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model.to(args.device)
    return model.eval()


def resolve_down_projs(model):
    """Return (list_of_down_proj_modules, is_llama) for an AutoModelForCausalLM."""
    # LLaMA / Mistral / most SwiGLU decoders: model.model.layers[i].mlp.down_proj
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        mlps = [layer.mlp for layer in model.model.layers]
        if hasattr(mlps[0], "down_proj"):
            return [mlp.down_proj for mlp in mlps], True
    # BLOOM: model.transformer.h[i].mlp.dense_4h_to_h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        mlps = [block.mlp for block in model.transformer.h]
        if hasattr(mlps[0], "dense_4h_to_h"):
            return [mlp.dense_4h_to_h for mlp in mlps], False
    raise ValueError(
        "Unrecognized architecture; expected LLaMA-like (.model.layers[i].mlp.down_proj) "
        "or BLOOM-like (.transformer.h[i].mlp.dense_4h_to_h)."
    )


def make_mask_hook(mask):
    def hook(_module, inp):
        # inp is a tuple; inp[0] is (batch, seq, intermediate).
        inp[0].index_fill_(-1, mask, 0)
        return inp
    return hook


def main():
    args = build_parser().parse_args()
    dtype = getattr(torch, args.dtype)

    is_llama_name = args.model.lower().find("llama") >= 0
    model_tag = "llama" if is_llama_name else "bloom"

    model = load_causal_lm(args, dtype)

    down_projs, is_llama = resolve_down_projs(model)
    max_length = args.max_length or min(
        getattr(model.config, "max_position_embeddings", 4096), 4096
    )

    if args.activation_mask:
        activation_masks = torch.load(args.activation_mask)
    else:
        activation_masks = [None]

    if is_llama:
        languages = ["en", "zh", "fr", "es", "vi", "id", "ja"]
    else:
        languages = ["en", "zh", "fr", "es", "vi", "id"]
    if args.languages:
        languages = args.languages.split(",")

    final_output = []
    for activation_mask, _mask_lang in zip(activation_masks, languages):
        # (Re)apply hooks for this mask; clear any from the previous language first.
        handles = []
        if activation_mask is not None:
            for layer, layer_mask in zip(down_projs, activation_mask):
                handles.append(
                    layer.register_forward_pre_hook(
                        make_mask_hook(layer_mask.to(args.device))
                    )
                )

        ppls = []
        for lang in languages:
            ids = torch.load(f"data/id.{lang}.{args.split}.{model_tag}")
            l = ids.size(0)
            l = min(l, 2**20) // max_length * max_length
            input_ids = ids[:l].reshape(-1, max_length)

            seq_logprobs = []
            with torch.no_grad():
                for start in range(0, input_ids.size(0), args.batch_size):
                    batch = input_ids[start : start + args.batch_size].to(args.device)
                    logits = model(input_ids=batch).logits  # (B, L, V)
                    shift_logits = logits[:, :-1, :].float()
                    shift_targets = batch[:, 1:]
                    B, Lm1, V = shift_logits.shape
                    nll = F.cross_entropy(
                        shift_logits.reshape(-1, V),
                        shift_targets.reshape(-1),
                        reduction="none",
                    ).reshape(B, Lm1)
                    # per-sequence mean log-probability (logprob = -nll)
                    seq_logprobs.extend((-nll).mean(dim=1).cpu().tolist())
            ppls.append(float(np.mean(seq_logprobs)))
        final_output.append(ppls)

        for h in handles:
            h.remove()

    for ppls in final_output:
        print(" ".join([str(-ppl) for ppl in ppls]))


if __name__ == "__main__":
    main()
