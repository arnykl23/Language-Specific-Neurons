import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


# The consumer scripts (activation.py / ppl.py) load these files with
#   ids = torch.load(f'data/id.{lang}.train.llama')
#   l = ids.size(0)
#   input_ids = ids[:l].reshape(-1, max_length)
#   model.generate(prompt_token_ids=input_ids.tolist(), ...)
# so the saved object must be a flat 1-D LongTensor of concatenated token ids.

# wikimedia/wikipedia language configs keyed by the codes used in this repo.
LANG2CONFIG = {
    "en": "20231101.en",
    "zh": "20231101.zh",
    "fr": "20231101.fr",
    "es": "20231101.es",
    "vi": "20231101.vi",
    "id": "20231101.id",
    "ja": "20231101.ja",
}


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize wikimedia/wikipedia texts and save a flat LongTensor "
        "of concatenated token ids to data/id.{lang}.{split}.{model_tag}."
    )
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="HF model/tokenizer name.")
    parser.add_argument("-l", "--lang", type=str, required=True,
                        choices=list(LANG2CONFIG.keys()),
                        help="Language code.")
    parser.add_argument("-s", "--split", type=str, default="train",
                        help="Output split label used in the filename (e.g. train, valid).")
    parser.add_argument("--config", type=str, default=None,
                        help="Override the wikimedia/wikipedia config name "
                        "(defaults to a mapping keyed by --lang).")
    parser.add_argument("-n", "--num_docs", type=int, default=100000,
                        help="Number of wikipedia articles to tokenize.")
    parser.add_argument("--max_tokens", type=int, default=99999744,
                        help="Cap on the total number of tokens to keep.")
    parser.add_argument("-o", "--output_dir", type=str, default="data",
                        help="Directory to write the output file into.")
    args = parser.parse_args()

    is_llama = args.model.lower().find("llama") >= 0
    model_tag = "llama" if is_llama else "bloom"

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    config = args.config or LANG2CONFIG[args.lang]
    print(f"Loading wikimedia/wikipedia config '{config}' (streaming)...")
    dataset = load_dataset(
        "wikimedia/wikipedia", config, split="train", streaming=True
    )

    all_ids = []
    total = 0
    for i, example in enumerate(dataset):
        if i >= args.num_docs or total >= args.max_tokens:
            break
        text = example["text"]
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        all_ids.extend(ids)
        total += len(ids)
        if (i + 1) % 1000 == 0:
            print(f"  tokenized {i + 1} docs, {total} tokens")

    all_ids = all_ids[: args.max_tokens]
    ids_tensor = torch.tensor(all_ids, dtype=torch.long)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir, f"id.{args.lang}.{args.split}.{model_tag}"
    )
    torch.save(ids_tensor, out_path)
    print(f"Saved {ids_tensor.size(0)} tokens to {out_path}")


if __name__ == "__main__":
    main()
