import argparse
import json
import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModel

def load_corpus(corpus_path: str) -> List[Dict]:
    corpus = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            corpus.append(json.loads(line))
    return corpus

def encode_passages(model_name: str,
                    corpus: List[Dict],
                    batch_size: int = 64,
                    device: str = "cuda") -> torch.Tensor:
    """
    corpus: list {id, title, text}
    return: embeddings [N, D]
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(corpus), batch_size):
            chunk = corpus[i:i+batch_size]
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
    args = parser.parse_args()

    corpus = load_corpus(args.corpus_path)
    print(f"Loaded {len(corpus)} passages from {args.corpus_path}")

    embeds = encode_passages(
        model_name=args.encoder_name,
        corpus=corpus,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("Embeddings:", embeds.shape)

    save_index(embeds, corpus, args.index_path)
    print(f"Index (embeds+doc_ids) saved to {args.index_path}")

if __name__ == "__main__":
    main()
