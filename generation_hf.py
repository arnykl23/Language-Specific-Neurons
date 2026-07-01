import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

answer_lang = {
    "zh": "请用中文回答。",
    "en": " Answer in English.",
    "fr": " Veuillez répondre en français.",
    "es": " Por favor responda en español.",
    "id": " Tolong dijawab dalam bahasa Indonesia.",
    "ja": "日本語で答えてください。",
    "vi": " Hãy trả lời bằng tiếng Việt.",
}

STOP_STRINGS = ["\nQ:", "\nA:"]


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("-a", "--activation_mask", type=str, default="")
    p.add_argument("-b", "--batch_size", type=int, default=8)
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
    """Return the list of down-projection modules for an AutoModelForCausalLM."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        mlps = [layer.mlp for layer in model.model.layers]
        if hasattr(mlps[0], "down_proj"):
            return [mlp.down_proj for mlp in mlps]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        mlps = [block.mlp for block in model.transformer.h]
        if hasattr(mlps[0], "dense_4h_to_h"):
            return [mlp.dense_4h_to_h for mlp in mlps]
    raise ValueError(
        "Unrecognized architecture; expected LLaMA-like (.model.layers[i].mlp.down_proj) "
        "or BLOOM-like (.transformer.h[i].mlp.dense_4h_to_h)."
    )


def make_mask_hook(mask):
    def hook(_module, inp):
        inp[0].index_fill_(-1, mask, 0)
        return inp

    return hook


def load_prompts(lang):
    texts = [l.strip() for l in open(f"data/mvicuna/{lang}.txt")]
    texts = [t + answer_lang[lang] for t in texts]
    texts = [f"Q: {t}\nA:" for t in texts]
    return texts


def truncate_at_stop(text):
    """Mimic vllm, which excludes the stop string from the returned text."""
    cut = len(text)
    for s in STOP_STRINGS:
        idx = text.find(s)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def main():
    args = build_parser().parse_args()
    dtype = getattr(torch, args.dtype)
    is_llama = args.model.lower().find("llama") >= 0
    max_new_tokens = 2048 if is_llama else 1024

    print(f"Loading {args.model} ({args.dtype}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_lm(args, dtype)

    down_projs = resolve_down_projs(model)

    if args.activation_mask:
        activation_masks = torch.load(args.activation_mask)
        activation_mask_name = args.activation_mask.split("/")[-1].split(".")
        activation_mask_name = ".".join(activation_mask_name[1:])
    else:
        activation_masks = [None]

    output_folder = f"results/{args.model.split('/')[-1]}/mvicuna"
    os.makedirs(output_folder, exist_ok=True)

    for activation_mask, mask_lang in zip(
        activation_masks, ["en", "zh", "fr", "es", "vi", "id", "ja"]
    ):
        # (Re)apply hooks for this mask; clear the previous language's hooks first.
        handles = []
        if activation_mask is not None:
            for layer, layer_mask in zip(down_projs, activation_mask):
                handles.append(
                    layer.register_forward_pre_hook(
                        make_mask_hook(layer_mask.to(args.device))
                    )
                )

        for lang in ["zh", "en", "es", "fr", "id", "ja", "vi"]:
            texts = load_prompts(lang)
            outputs = []
            with torch.no_grad():
                for start in range(0, len(texts), args.batch_size):
                    batch_texts = texts[start : start + args.batch_size]
                    inputs = tokenizer(
                        batch_texts, return_tensors="pt", padding=True
                    ).to(args.device)
                    gen = model.generate(
                        **inputs,
                        do_sample=False,
                        repetition_penalty=1.1,
                        max_new_tokens=max_new_tokens,
                        stop_strings=STOP_STRINGS,
                        tokenizer=tokenizer,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    # Left-padding => all rows share the same prompt length.
                    new_tokens = gen[:, inputs["input_ids"].shape[1] :]
                    decoded = tokenizer.batch_decode(
                        new_tokens, skip_special_tokens=True
                    )
                    outputs.extend(truncate_at_stop(d).strip() for d in decoded)

            if activation_mask is not None:
                output_file = (
                    f"{output_folder}/{lang}.perturb.{mask_lang}."
                    f"{activation_mask_name}.jsonl"
                )
            else:
                output_file = f"{output_folder}/{lang}.jsonl"

            results = [{"input": t, "output": o} for t, o in zip(texts, outputs)]
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(results, indent=4, ensure_ascii=False) + "\n")
            print(f"wrote {output_file} ({len(results)} entries)")

        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
