# main.py

import argparse
from train import train
from generate import generate_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma Model")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'generate'], help="Mode: train or generate")
    parser.add_argument('--prompt', type=str, help="Prompt text for generation")

    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'generate':
        if args.prompt:
            text = generate_text(args.prompt)
            print(text)
        else:
            print("Please provide a prompt using --prompt when in generate mode.")
