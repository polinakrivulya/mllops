import argparse
import os

import torch
from transformers import AutoModelForSequenceClassification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model_dir", type=str, default="models/bert-tiny")
    parser.add_argument("--out_path", type=str, default="torchserve/model.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.hf_model_dir)
    state_dict = model.state_dict()
    torch.save(state_dict, args.out_path)
    print(f"Saved state_dict to: {args.out_path}")


if __name__ == "__main__":
    main()
