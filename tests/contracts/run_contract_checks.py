#!/usr/bin/env python3
"""Run shared Python/TypeScript contract fixtures without scientific execution."""
import argparse
import os
import shutil
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from validate import run, equal

ROOT=Path(__file__).resolve().parents[2]
ARTIFACTS=ROOT/'contracts/preparation-v1'
def tooling(explicit):
    tsc=explicit or os.environ.get('CONTRACT_TSC') or shutil.which('tsc')
    sibling=ROOT.parent/'preparic/node_modules/.bin/tsc'
    if not tsc and sibling.is_file(): tsc=str(sibling)
    if not tsc or not Path(tsc).is_file():
        raise SystemExit('TypeScript compiler missing: use --tsc PATH, CONTRACT_TSC, PATH, or sibling preparic/node_modules/.bin/tsc')
    node=shutil.which('node')
    if not node: raise SystemExit('Node.js missing: add node to PATH')
    return tsc,node

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tsc',help='TypeScript compiler path (otherwise CONTRACT_TSC, PATH, sibling preparic)')
    args=parser.parse_args()
    tsc,node=tooling(args.tsc)
    schemas=json.loads((ARTIFACTS/'schemas.json').read_text())
    corpus=json.loads((ARTIFACTS/'fixtures.json').read_text())
    ids=[c['id'] for c in corpus['cases']]
    assert len(ids)==len(set(ids)), 'duplicate fixture IDs'
    python_results=run(corpus,schemas['$defs'])
    with tempfile.TemporaryDirectory(prefix='w0-contract-ts-') as output:
        subprocess.run([tsc,'--strict','--target','ES2020','--module','commonjs','--outDir',output,str(ROOT/'tests/contracts/validate.ts')],check=True)
        result=subprocess.run([node,str(Path(output)/'validate.js'),str(ARTIFACTS)],text=True,capture_output=True,check=True)
        ts_results=json.loads(result.stdout)
    assert equal(python_results,ts_results), 'Python/TypeScript fixture disagreement'
    # Round trip every actual JSON payload, including null and malformed shapes;
    # malformed file text itself stays text, never normalized behind the loader.
    for case in corpus['cases']:
        assert equal(case,json.loads(json.dumps(case,allow_nan=False))),case['id']
    refs=json.loads((ARTIFACTS/'golden-references.json').read_text())
    source=ROOT/refs['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==refs['source_sha256'],'model fixture data changed; review references'
    model=json.loads(source.read_text())
    assert model['schema']==1 and model['spec_digest']==refs['spec_digest']
    assert model['filter_set_version']==refs['filter_set_version']
    assert model['golden']['tolerance_rel']==refs['backend_obligations']['relative_score_tolerance']
    for name,pointer in refs['pointers'].items():
        data=model
        for part in pointer.strip('/').split('/'):
            data=data[part]
        if name in refs['counts']:
            assert len(data)==refs['counts'][name]
    for case in corpus['cases']:
        if case['operation']=='schema' and case['boundary']=='S25' and case['expected']=={'valid':True} and case['payload']:
            assert case['payload']['legal_filter_names']==list(model['thresholds'])
    inventory=json.loads((ARTIFACTS/'inventory.json').read_text())
    assert len(inventory['active_transformers'])==28 and 'T17' not in inventory['active_transformers']
    assert len(inventory['active_shares'])==24 and 'S16' not in inventory['active_shares']
    assert set(inventory['active_shares'])==set(k for k in inventory['schemas'] if k.startswith('S'))|set(inventory['non_enveloped_shares'])
    print(f"PASS: {len(ids)} shared cases in Python + TypeScript; exact agreement; JSON round trips; golden reference metadata; active inventory")
    print('No scientific calculations, caches, services, real application consumers or future step acceptance executed.')

if __name__=='__main__':
    main()
