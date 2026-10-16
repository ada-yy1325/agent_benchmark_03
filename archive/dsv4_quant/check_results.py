import json

with open('eval_gsm8k_base_correct_results.json') as f:
    data = json.load(f)
results = data['results']
wrong = [r for r in results if not r['correct']]
print(f'Wrong: {len(wrong)}/{len(results)}')
print()
print('=== WRONG ANSWERS ===')
for r in wrong[:10]:
    print(f"  Q{r['q_idx']}: true={r['true_answer']}, pred={r['pred_answer']}")
    print(f"    snippet: {r['output_snippet'][:300]}")
    print()
print('=== CORRECT ANSWERS (first 5) ===')
right = [r for r in results if r['correct']]
for r in right[:5]:
    print(f"  Q{r['q_idx']}: true={r['true_answer']}, pred={r['pred_answer']}")
    print(f"    snippet: {r['output_snippet'][:200]}")
    print()
