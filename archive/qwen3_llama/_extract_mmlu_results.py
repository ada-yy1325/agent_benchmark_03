"""提取 MMLU 评测结果"""
import json

ROOT = "/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test"

files = {
    "FP16": f"{ROOT}/outputs/mmlu_fp16/__inspire__sj-ssd3__project__project-public__s26068__agent_benchmark_test__models__Meta-Llama-3.1-8B-Instruct/results_2026-09-24T12-41-20.491752.json",
    "W8A8": f"{ROOT}/outputs/mmlu_w8a8/__inspire__sj-ssd3__project__project-public__s26068__agent_benchmark_test__models__Meta-Llama-3.1-8B-Instruct-W8A8-RedHat/results_2026-09-24T13-02-56.725246.json",
}

for label, path in files.items():
    with open(path) as f:
        d = json.load(f)
    
    r = d["results"]["mmlu"]
    acc = r["acc,none"]
    stderr = r.get("acc_stderr,none", 0)
    
    print(f"{'='*50}")
    print(f"{label} MMLU Results")
    print(f"{'='*50}")
    print(f"Overall: {acc*100:.2f}% ± {stderr*100:.2f}%")
    print()
    
    # Subgroups
    groups = d.get("results", {})
    subgroups = [
        ("humanities", "mmlu_humanities"),
        ("other", "mmlu_other"),
        ("social_sciences", "mmlu_social_sciences"),
        ("stem", "mmlu_stem"),
    ]
    for gname, gkey in subgroups:
        if gkey in groups:
            gacc = groups[gkey]["acc,none"]
            print(f"  {gname:20s}: {gacc*100:.2f}%")
    
    print()