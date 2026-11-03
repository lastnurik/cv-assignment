"""Build a limited within-setting candidate; never invent independent sessions."""
import json
from collections import defaultdict, Counter
from pathlib import Path
from gesture_cv.dataset import sha256, write_json, validate_manifest

SOURCE = Path('data/manifests/a1_clean_v1.json')
PAIRS = Path('artifacts/audit/a1_clean_v1_near_duplicates.json')
OUTPUT = Path('data/manifests/a1_limited_v1.json')

def build():
    manifest=json.loads(SOURCE.read_text())
    rows=[r for r in manifest['rows'] if r['review_status']=='accepted']
    parents={r['sample_id']:r['sample_id'] for r in rows}
    def root(s):
        while parents[s]!=s:
            parents[s]=parents[parents[s]];s=parents[s]
        return s
    pairs=json.loads(PAIRS.read_text())
    assert pairs['threshold']==4 and pairs['method']=='dHash64'
    for pair in pairs['pairs']:
        if pair['a'] in parents and pair['b'] in parents:
            a,b=sorted([root(pair['a']),root(pair['b'])]);parents[b]=a
    components=defaultdict(list)
    for r in rows:components[root(r['sample_id'])].append(r)
    assigned={};counts=Counter()
    # Keep mixed-label similarity components intact too; background similarity
    # can connect different gestures. The largest component covers all classes.
    largest=max(components,key=lambda g:len(components[g]))
    assigned[largest]='train'
    for r in components[largest]:counts[r['label'],'train']+=1
    remaining={g:v for g,v in components.items() if g!=largest}
    for group,items in remaining.items():
        assert len({r['label'] for r in items})==1, 'Additional mixed component needs allocation review'
    for label in manifest['class_names']:
        groups=sorted([(g,v) for g,v in remaining.items() if v[0]['label']==label],key=lambda gv:(-len(gv[1]),gv[0]))
        assert len(groups)>=2
        total=sum(r['label']==label for r in rows)
        for i,(group,items) in enumerate(groups):
            if i==0:split='val'
            elif i==1:split='test'
            else:
                targets={'train':.70*total,'val':.15*total,'test':.15*total}
                split=max(targets,key=lambda s:(targets[s]-counts[label,s])/targets[s])
            assigned[group]=split;counts[label,split]+=len(items)
    for r in manifest['rows']:
        r['participant_id']='teammate_p01'
        r['participant_consent_status']='private_use_confirmed'
        r['consent_evidence']='User confirmation in conversation 2026-11-04; teammate only; no public release'
        r['bystander_consent_status']='unknown'
        r['consent_status']='unknown' # Do not turn participant consent into permission for everyone.
        r['original_group_id']=r['group_id']
        if r['sample_id'] in parents:
            group=root(r['sample_id']);r['group_id']='dhash4_'+group;r['split']=assigned[group]
    manifest.update(version='a1_limited_v1',status='CANDIDATE_SPLIT_BYSTANDER_REVIEW_PENDING',
        protocol='Within-participant/setting exploratory evaluation; common-session correlation remains; no unseen-person/session/environment claim',
        notes='User-authorized limited protocol. dHash connected components are similarity groups, not sessions. Tiny evaluation support must be reported.',
        parent_manifest_sha256=sha256(SOURCE),similarity_graph_sha256=sha256(PAIRS))
    lookup={r['sample_id']:r for r in rows}
    assert len(rows)==453
    for pair in pairs['pairs']:
        if pair['a'] in lookup and pair['b'] in lookup:
            assert lookup[pair['a']]['split']==lookup[pair['b']]['split']
    for key in ['sha256','source_sha256','group_id']:
        seen=defaultdict(set)
        for r in rows:seen[r[key]].add(r['split'])
        assert all(len(v)==1 for v in seen.values())
    assert all(counts[c,s]>0 for c in manifest['class_names'] for s in ['train','val','test'])
    return manifest, {'counts':{f'{c}/{s}':n for (c,s),n in counts.items()},'components':len(components),
        'accepted':len(rows),'rejected':len(manifest['rows'])-len(rows),'cross_split_candidate_edges':0,
        'source_sha256':sha256(SOURCE),'similarity_graph_sha256':sha256(PAIRS),'structural_checks_passed':True}

def main():
    if OUTPUT.exists():raise FileExistsError('Candidate exists; use a new version')
    manifest,audit=build();assert manifest==build()[0]
    write_json(OUTPUT,manifest)
    audit['manifest_sha256']=sha256(OUTPUT)
    audit['readiness']=validate_manifest(manifest,Path('.'))
    write_json('artifacts/audit/a1_limited_v1_split.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k!='readiness'},indent=2))
if __name__=='__main__':main()
