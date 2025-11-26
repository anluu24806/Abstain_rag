import argparse
import json
import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModel


def load_corpus(corpus_path: str, max_passages: int | None = None) -> List[Dict]:
    """
    Đọc corpus JSONL, mỗi dòng: {id, title, text}
    Nếu max_passages != None thì chỉ lấy tối đa max_passages dòng đầu.
    """
    corpus = []
    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_passages is not None and i >= max_passages:
                break
            line = line.strip()
            if not line:
                continue
            corpus.append(json.loads(line))
    return corpus


def encode_passages(
    model_name: str,
    corpus: List[Dict],
    batch_size: int = 64,
    device: str = "cuda",
) -> torch.Tensor:
    """
    corpus: list {id, title, text}
    return: embeddings [N, D]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    N = len(corpus)
    print(f"[build_index] Encoding {N} passages with batch_size={batch_size} on {device} ...")

    all_embeds: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            chunk = corpus[start:end]
            texts = [c.get("title", "") + "\n" + c.get("text", "") for c in chunk]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(device)

            out = model(**enc)
            embeds = out.last_hidden_state.mean(dim=1)  # [B, D]
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.cpu())

            # progress: in every ~100 batches (tuỳ N), in ra 1 lần
            if (start // batch_size) % 100 == 0 or end == N:
                pct = end / N * 100
                print(f"[build_index] Encoded {end}/{N} passages ({pct:.2f}%)")

    return torch.cat(all_embeds, dim=0)  # [N, D]


def save_index(embeds: torch.Tensor, corpus: List[Dict], index_path: str):
    """
    Lưu embedding + doc_ids vào file .pt
    """
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    doc_ids = [c["id"] for c in corpus]
    obj = {
        "embeds": embeds,   # [N, D]
        "doc_ids": doc_ids  # list length N
    }
    torch.save(obj, index_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path", type=str, required=True,
                        help="JSONL corpus, mỗi dòng: {id,title,text}")
    parser.add_argument("--encoder-name", type=str,
                        default="facebook/contriever")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--index-path", type=str,
                        default="indexes/wiki_contriever.pt")
    parser.add_argument("--max-passages", type=int, default=None,
                        help="Nếu đặt, chỉ encode tối đa N passages đầu (debug / wiki_medium).")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus_path, max_passages=args.max_passages)
    print(f"Loaded {len(corpus)} passages from {args.corpus_path}")

    embeds = encode_passages(
        model_name=args.encoder_name,
        corpus=corpus,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("Embeddings shape:", embeds.shape)

    save_index(embeds, corpus, args.index_path)
    print(f"Index (embeds+doc_ids) saved to {args.index_path}")


if __name__ == "__main__":
    main()
