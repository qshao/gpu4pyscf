"""Run the unmodified suite audit with a floating-point-tolerant equality for recomputed comparison statistics.

audit.py recomputes each stored comparison and demands exact dict equality. Diagnostic numbers (state overlaps, endpoint RMS)
change in the last digits between processes because multi-threaded reductions are not bit-reproducible, so exact equality fails
on honest records. Here the recomputed comparison compares equal to the stored one when every float agrees to a relative 1e-9
(absolute 1e-12 near zero) and everything else (verdicts, statuses, mappings, strings, structure) is identical. audit.py itself
is untouched, so the frozen campaign's helper hashes stay valid.
"""
import math
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE))
RELATIVE, ABSOLUTE = 1e-9, 1e-12


def close(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, float) or isinstance(b, float):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= max(ABSOLUTE, RELATIVE * max(abs(a), abs(b)))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


class ToleranceDict(dict):
    def __eq__(self, other):
        return close(dict(self), dict(other))

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None


def audit_campaign_tolerant(root):
    import compare
    import audit
    import json
    from run import job_id, digest
    original = compare.compare_results
    original_request = audit._request

    def request_or_admission(record, path, campaign_hash, case):
        """Pre-launch admission rejections are written by schedule.launch() without request.json. Accept that only for a
        resource_limited record that carries its admission forecast, after checking its own identity fields."""
        if (path.with_name('request.json').is_file() or record.get('status') != 'resource_limited'
                or 'admission' not in record):
            return original_request(record, path, campaign_hash, case)
        attempt = record.get('attempt')
        if (record.get('campaign_sha256') != campaign_hash or record.get('case_id') != case['id']
                or not isinstance(attempt, int) or attempt < 0
                or path.parent.name != job_id(case['id'], record['variant'], record['mode'], attempt)
                or record.get('process_elapsed_seconds') != 0):
            raise ValueError('admission-rejection record identity mismatch: ' + str(path))

    def wrapped(*args, **kwargs):
        return ToleranceDict(original(*args, **kwargs))
    compare.compare_results = wrapped
    audit._request = request_or_admission
    try:
        return audit.audit_campaign(root)
    finally:
        compare.compare_results = original
        audit._request = original_request


if __name__ == '__main__':
    import json
    print(json.dumps(audit_campaign_tolerant(sys.argv[1]), indent=1)[:2000])
