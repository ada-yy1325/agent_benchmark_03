import json

d = json.load(open("eval_gsm8k_base_correct_results.json"))
print("Total:", len(d["results"]))
correct = [r for r in d["results"] if r["correct"]]
wrong = [r for r in d["results"] if not r["correct"]]
print("Correct:", len(correct))
print("Wrong:", len(wrong))

# Show breakdown by q_idx ranges
for start in range(0, 1319, 100):
    end = min(start+100, 1319)
    subset = [r for r in d["results"] if start <= r["q_idx"] < end]
    c = sum(1 for r in subset if r["correct"])
    print(f"  q{start}-{end-1}: {c}/{len(subset)} = {c/len(subset)*100:.2f}%")