"""Freeze a master campaign (sources, native libraries, atomic guesses) for the LCC GPUs.

usage: prepare_campaign.py NAME CASE_ID [CASE_ID ...]
The CPU references live in this master campaign; each GPU type then works in a fork of it (fork_campaign.py) whose
campaign.json is byte-identical, so every record keeps the same campaign hash and the audit applies unchanged.
"""
import os
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
os.environ.setdefault('QM_SUITE_NATIVE_ROOT', '/scratch/qsh226/g4p_lcc/unified/src')
os.environ.setdefault('QM_SUITE_GPU_TARGET', 'LCC: NVIDIA P100 (sm60), V100 (sm70), A100 (sm80); one unified build (60;70;80;90), LibXC 7.1.2 from source, CUDA 12.4')
sys.path.insert(0, str(SUITE))

from manifest import load_cases  # noqa: E402
from campaign import prepare  # noqa: E402
import schedule  # noqa: E402


def main():
    name, ids = sys.argv[1], sys.argv[2:]
    cases = {c['id']: c for g in (None,) for c in load_cases()}
    missing = [i for i in ids if i not in cases]
    if missing:
        raise SystemExit('unknown case IDs: %s' % missing)
    # Registry order: schedule.py filters --cases in registry order and refuses to resume a campaign frozen in another order.
    wanted = set(ids)
    chosen = [c for c in cases.values() if c['id'] in wanted]
    # Same policy keys main() records, so schedule.py resumes without a policy mismatch.
    reps = int(os.environ.get('QM_SUITE_REPETITIONS', '3'))
    policy = {'seed': 42, 'repetitions': reps, 'extension_ceiling': schedule.EXTENSION_CEILING}
    out = SUITE / 'results' / name
    prepare(out, chosen, policy)
    print(out)


if __name__ == '__main__':
    main()
