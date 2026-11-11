"""Train a character-level GPT on a plain-text file, then generate sample text.

Usage:
    python train.py                       # downloads Tiny Shakespeare if input.txt is missing
    python train.py --data my_corpus.txt --max-iters 3000
"""

import argparse
import os
import urllib.request

import torch

from model import GPT

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_text(path: str) -> str:
    if not os.path.exists(path):
        print(f"{path} not found, downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="input.txt")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--gen-tokens", type=int, default=500)
    parser.add_argument("--ckpt", default="ckpt.pt")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    # --- Character-level tokenizer -------------------------------------------
    text = load_text(args.data)
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)
    print(f"characters: {len(text):,}  vocab size: {vocab_size}")

    # --- Train / validation split --------------------------------------------
    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    def get_batch(split_name: str):
        source = train_data if split_name == "train" else val_data
        ix = torch.randint(len(source) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([source[i : i + args.block_size] for i in ix])
        y = torch.stack([source[i + 1 : i + 1 + args.block_size] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss() -> dict:
        model.eval()
        out = {}
        for name in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y = get_batch(name)
                _, loss = model(x, y)
                losses[k] = loss.item()
            out[name] = losses.mean().item()
        model.train()
        return out

    # --- Model and optimizer --------------------------------------------------
    model = GPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training loop --------------------------------------------------------
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss()
            print(f"step {it:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if it == args.max_iters:
            break

        x, y = get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # --- Save checkpoint (weights + tokenizer + config) -----------------------
    torch.save(
        {
            "model": model.state_dict(),
            "chars": chars,
            "config": {
                "vocab_size": vocab_size,
                "block_size": args.block_size,
                "n_layer": args.n_layer,
                "n_head": args.n_head,
                "n_embd": args.n_embd,
                "dropout": args.dropout,
            },
        },
        args.ckpt,
    )
    print(f"saved checkpoint to {args.ckpt}")

    # --- Generate a sample ----------------------------------------------------
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(context, max_new_tokens=args.gen_tokens, temperature=0.8, top_k=40)
    print("\n--- sample ---")
    print(decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
