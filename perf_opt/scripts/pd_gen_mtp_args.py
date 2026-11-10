import sys, json, os
sys.path.insert(0, '/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test')
from perf_opt.scripts.start_pd_mooncake import build_args, generate_shell_script

TMP = '/tmp/pd_test'
os.makedirs(TMP, exist_ok=True)

for pf, port, lb in [(True, 8000, 'prefill'), (False, 8001, 'decode')]:
    a = build_args(prefill=pf, port=port, mtp=True)
    out = f'{TMP}/pd_{lb}_mtp_args.json'
    json.dump(a, open(out, 'w'))
    print(f'Args[{lb}]: {len(a)} -> {out}')

for role, devs, log, af in [
    ('pd_decode', '8,9,10,11,12,13,14,15', 'decode_mtp.log', f'{TMP}/pd_decode_mtp_args.json'),
    ('pd_prefill', '0,1,2,3,4,5,6,7', 'prefill_mtp.log', f'{TMP}/pd_prefill_mtp_args.json'),
]:
    s = generate_shell_script(role, devs, af, f'{TMP}/{log}')
    out = f'{TMP}/_{role}_mtp_launch.sh'
    with open(out, 'w') as f:
        f.write(s)
    os.chmod(out, 0o755)
    print(f'Script: {out}')

print('DONE')
