"""Fork a master campaign for one GPU type: identical campaign.json, shared snapshots/inputs, symlinked CPU references.

usage: fork_campaign.py MASTER_DIR FORK_DIR
"""
import shutil
import sys
from pathlib import Path


def main():
    master, fork = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    fork.mkdir(parents=True, exist_ok=False)
    for name in ('campaign.json', 'source_manifest.json'):
        shutil.copy2(master / name, fork / name)
    for name in ('snapshots', 'inputs'):
        (fork / name).symlink_to(master / name, target_is_directory=True)
    (fork / 'jobs').mkdir()
    count = 0
    for job in sorted((master / 'jobs').glob('*__CPU__reference__*')):
        if (job / 'result.json').exists() and '"reference_ready"' in (job / 'result.json').read_text():
            shutil.copytree(job, fork / 'jobs' / job.name)   # a copy, not a link: audits resolve paths relative to the fork
            count += 1
    print('forked %s with %d CPU references' % (fork, count))


if __name__ == '__main__':
    main()
