"""Produce coverage and qualified timing tables from independently audited records."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics


def summarize(output):
    from audit import audit_campaign
    from run import write_once,digest
    from schedule import needs_more
    output=Path(output).resolve();audit=audit_campaign(output)
    campaign=json.loads((output/'campaign.json').read_text());cases=campaign['sources']['cases']
    records=[json.loads((output/r['path']).read_text()) for r in audit['records']]
    summaries=[]
    for case in cases:
        chosen=[d for d in records if d['case_id']==case['id']]
        entry={'case_id':case['id'],'task':case['task'],'statuses':dict(Counter(d['status'] for d in chosen)),
               'CPU_reference_ready':any(d['status']=='reference_ready' for d in chosen),
               'timing':{},'memory':{},'headline_eligible':False,'accuracy_qualified':False,
               'paired_ratios':[],'unmatched_timings':[]}
        qualified={v:{d['pair_index']:d for d in chosen if d['variant']==v and d['mode']=='timing' and d['status']=='qualified'} for v in ('G0','G1')}
        complete=sorted(set(qualified['G0']) & set(qualified['G1']))
        for v in ('G0','G1'):
            entry['unmatched_timings'].extend({'variant':v,'pair_index':i,'workflow_seconds':d['result']['workflow_seconds']}
                for i,d in sorted(qualified[v].items()) if i not in complete)
            times=[qualified[v][i]['result']['workflow_seconds'] for i in complete]
            if times:
                med=statistics.median(times)
                entry['timing'][v]={'n':len(times),'samples_seconds':times,'median_seconds':med,'min_seconds':min(times),'max_seconds':max(times),
                                      'mad_seconds':statistics.median(abs(x-med) for x in times)}
            memory=[d['phase_memory']['peak_live_bytes'] for d in chosen if d['variant']==v and d['mode']=='memory' and d['status']=='qualified']
            if memory:entry['memory'][v]={'tracked_cupy_peaks_bytes':memory,'maximum_tracked_cupy_bytes':max(memory)}
        cpu=[d['result']['workflow_seconds'] for d in sorted(chosen,key=lambda d:d.get('pair_index') or 0)
             if d['variant']=='CPU' and d['mode']=='timing' and d['status']=='cpu_timed']
        if cpu:
            med=statistics.median(cpu)
            entry['timing']['CPU']={'n':len(cpu),'samples_seconds':cpu,'median_seconds':med,'min_seconds':min(cpu),'max_seconds':max(cpu),
                                    'mad_seconds':statistics.median(abs(x-med) for x in cpu),
                                    'threads':sorted({str(d['environment'].get('threads')) for d in chosen if d['variant']=='CPU' and d['mode']=='timing'})}
        gpu_records=[d for d in chosen if d['variant'] in ('G0','G1')]
        entry['coverage_status']=('paired_timing' if complete else 'unmatched_timing' if entry['unmatched_timings']
            else 'gpu_no_qualified_timing' if gpu_records else 'gpu_not_measured')
        entry['paired_indices']=complete
        entry['paired_ratios']=[{'pair_index':i,'G0_over_G1':qualified['G0'][i]['result']['workflow_seconds']/qualified['G1'][i]['result']['workflow_seconds']} for i in complete]
        if all(v in entry['timing'] for v in ('G0','G1')):
            t=entry['timing'];entry['observed_speed_ratio']=t['G0']['median_seconds']/t['G1']['median_seconds']
            entry['accuracy_qualified']=True
            entry['sample_extension_needed']=needs_more(t['G0']['samples_seconds'],t['G1']['samples_seconds']) and min(t['G0']['n'],t['G1']['n'])<9
            entry['headline_eligible']=min(t['G0']['n'],t['G1']['n'])>=5 and not entry['sample_extension_needed']
        t=entry['timing']
        if 'CPU' in t:
            for v in ('G0','G1'):
                if v in t:entry['CPU_over_'+v]=t['CPU']['median_seconds']/t[v]['median_seconds']
        summaries.append(entry)
    pre=[]
    for p in sorted((output/'preflights').glob('*.json')):
        d=json.loads(p.read_text());pre.append({'path':str(p.relative_to(output)),'sha256':digest(p),'status':d['status']})
    return {'audit':audit,'preflights':pre,'cases':summaries,
       'note':'No GPU speed/memory claim without fresh qualified GPU records. CPU reference runtimes are not GPU or tuned CPU performance results.'}


def main():
    from run import write_once
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=summarize(a.output)
    reports=a.output/'reports';reports.mkdir(exist_ok=True)
    name=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    write_once(reports/(name+'.json'),result)
    lines=['# Broad QM benchmark coverage','',f"Configured cases: {result['audit']['case_count']}. Recorded attempts: {result['audit']['record_count']}.",'',
           '| Case | Task | CPU reference | Qualified GPU timing samples G0 / G1 |','|---|---|---|---|']
    for c in result['cases']:
        counts=[c['timing'].get(v,{}).get('n',0) for v in ['G0','G1']]
        lines.append(f"| {c['case_id']} | {c['task']} | {'ready' if c['CPU_reference_ready'] else 'pending'} | {counts[0]} / {counts[1]} |")
    lines+=['',result['note'],'','GPU preflights: '+', '.join(p['status'] for p in result['preflights'])]
    with (reports/(name+'.md')).open('x') as f:f.write('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
