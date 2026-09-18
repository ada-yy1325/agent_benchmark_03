import json

with open('eval_gsm8k_base_correct_results.json') as f:
    data = json.load(f)
results = data['results']

print('=== Checking if outputs are truncated at 150 chars ===')
# The output_snippet is truncated at 150 chars in the eval script
# Let's check if there's a pattern
for r in results:
    if not r['correct']:
        snippet = r['output_snippet']
        # Check if it ends mid-sentence
        ends = snippet[-20:] if len(snippet) >= 20 else snippet
        print(f"Q{r['q_idx']}: true={r['true_answer']}, pred={r['pred_answer']}, len={len(snippet)}")
        print(f"  ends_with: ...{ends}")
        print(f"  full: {snippet}")
        print()
