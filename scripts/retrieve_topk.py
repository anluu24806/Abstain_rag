import argparse
import json
import os
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModel

def load_queries(path: str) -> List[Dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def load_corpus(path: str) -> List[Dict]:
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            corpus.append(json.loads(line))
    return corpus

def encode_queries(model_name: str,
                   questions: List[str],
                   batch_size: int = 32,
                   device: str = "cuda") -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(questions), batch_size):
            chunk = questions[i:i+batch_size]
            enc = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            embeds = out.last_hidden_state.mean(dim=1)
            embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
            all_embeds.append(embeds.cpu())
    return torch.cat(all_embeds, dim=0)  # [N, D]

def retrieve_topk(
    q_embeds: torch.Tensor,
    index_obj: Dict,
    k: int,
    device: str = "cuda",
    batch_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q_embeds: [Nq, D]
    index_obj: {"embeds": [Nd, D], "doc_ids": list}
    return:
      scores: [Nq, k_eff]
      indices: [Nq, k_eff]  (k_eff = min(k, Nd))
    """
    doc_embeds = index_obj["embeds"].to(device)  # [Nd, D]
    q_embeds = q_embeds.to(device)
    Nd = doc_embeds.size(0)
    k_eff = min(k, Nd)
    if k_eff < k:
        print(f"[retrieve_topk] Requested k={k} > Nd={Nd}. Using k_eff={k_eff} instead.")

    all_scores = []
    all_indices = []

    with torch.no_grad():
        for i in range(0, q_embeds.size(0), batch_size):
            qs = q_embeds[i:i+batch_size]  # [B, D]
            scores = torch.matmul(qs, doc_embeds.t())  # [B, Nd]
            top_scores, top_idx = torch.topk(scores, k_eff, dim=-1)
            all_scores.append(top_scores.cpu())
            all_indices.append(top_idx.cpu())

    scores = torch.cat(all_scores, dim=0)   # [Nq, k_eff]
    indices = torch.cat(all_indices, dim=0) # [Nq, k_eff]
    return scores, indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-path", type=str, required=True,
                        help="JSONL NQ/TriviaQA/PopQA processed")
    parser.add_argument("--corpus-path", type=str, required=True,
                        help="JSONL corpus (id,title,text)")
    parser.add_argument("--index-path", type=str, required=True,
                        help="PT index path (embeds+doc_ids)")
    parser.add_argument("--encoder-name", type=str,
                        default="facebook/contriever")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()

    queries = load_queries(args.queries_path)
    corpus = load_corpus(args.corpus_path)
    corpus_dict = {c["id"]: c for c in corpus}

    print(f"Loaded {len(queries)} queries, {len(corpus)} passages")

    index_obj = torch.load(args.index_path, map_location="cpu")
    print("Index embeds shape:", index_obj["embeds"].shape)

    questions = [q["question"] for q in queries]
    q_embeds = encode_queries(
        model_name=args.encoder_name,
        questions=questions,
        batch_size=args.batch_size,
        device=args.device,
    )

    scores, indices = retrieve_topk(
        q_embeds,
        index_obj=index_obj,
        k=args.k,
        device=args.device,
        batch_size=args.batch_size,
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as fout:
        for ex, sc, idxs in zip(queries, scores, indices):
            retrieved = []
            for rank, (s, ix) in enumerate(zip(sc.tolist(), idxs.tolist()), start=1):
                doc_id = index_obj["doc_ids"][ix]
                doc = corpus_dict[doc_id]
                retrieved.append({
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": float(s),
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                })
            out_ex = {
                **ex,
                "retrieved": retrieved,
            }
            fout.write(json.dumps(out_ex, ensure_ascii=False) + "\n")

    print(f"Saved retrieval results to {args.out_path}")

if __name__ == "__main__":
    main()
