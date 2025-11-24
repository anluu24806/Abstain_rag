import argparse
import json
import os
from typing import Dict, List, Optional

def contains_answer(doc: Dict, answers: List[str]) -> bool:
    """
    Check xem doc (title + text) có chứa bất kỳ answer nào không (case-insensitive).
    """
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    full = (title + " " + text).lower()

    for ans in answers:
        if not ans:
            continue
        if str(ans).lower() in full:
            return True
    return False

def assign_scenario(example: Dict) -> str:
    """
    S1: ít nhất một doc chứa answer và doc tốt nhất ở rank 1 (gold@1)
    S2: không doc nào chứa answer (no-good-doc)
    S3: có doc chứa answer nhưng không ở rank 1 (gold@>1)
    (Bạn có thể mở rộng thêm S5 sau này.)
    """
    answers = example.get("answers", [])
    retrieved = example.get("retrieved", [])

    best_rank: Optional[int] = None
    for r in retrieved:
        if contains_answer(r, answers):
            rank = r.get("rank", None)
            if rank is None:
                continue
            if best_rank is None or rank < best_rank:
                best_rank = rank

    if best_rank is None:
        return "S2"   # no-good-doc
    elif best_rank == 1:
        return "S1"   # gold@1
    else:
        return "S3"   # gold@>1

def annotate_file(input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_total = 0
    counts = {}

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            scenario = assign_scenario(ex)
            ex["scenario"] = scenario
            counts[scenario] = counts.get(scenario, 0) + 1
            n_total += 1
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Annotated {n_total} examples from {input_path}")
    print("Scenario counts:", counts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True,
                        help="JSONL retrieval (output của retrieve_topk)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="JSONL output có thêm field 'scenario'")
    args = parser.parse_args()

    annotate_file(args.input_path, args.output_path)

if __name__ == "__main__":
    main()
