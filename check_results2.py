import json

with open('eval_gsm8k_base_correct_results.json') as f:
    data = json.load(f)
results = data['results']

print('=== WRONG ANSWERS with finish_reason ===')
for r in results:
    if not r['correct']:
        fr = r.get('finish_reason', 'N/A')
        snippet = r['output_snippet']
        print(f"Q{r['q_idx']}: true={r['true_answer']}, pred={r['pred_answer']}, finish={fr}, out_len={len(snippet)}")
        print(f"  end: ...{snippet[-60:]}")
        print()

print('=== CORRECT ANSWERS finish_reason ===')
for r in results:
    if r['correct']:
        fr = r.get('finish_reason', 'N/A')
        print(f"Q{r['q_idx']}: true={r['true_answer']}, pred={r['pred_answer']}, finish={fr}")
print()

print('=== FINISH_REASON distribution ===')
frs = {}
for r in results:
    fr = r.get('finish_reason', 'N/A')
    frs[fr] = frs.get(fr, 0) + 1
for k, v in sorted(frs.items()):
    print(f"  {k}: {v}")
