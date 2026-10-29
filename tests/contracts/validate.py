"""Small W0-only validator. No application or scientific modules are imported."""
import copy
import json
import math
import re

class ContractError(ValueError):
    pass

def require(condition, code):
    if not condition:
        raise ContractError(code)

def equal(left, right):
    def canonical(value):
        if type(value) is float and math.isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value,list): return [canonical(x) for x in value]
        if isinstance(value,dict): return {k:canonical(v) for k,v in value.items()}
        return value
    return json.dumps(canonical(left),sort_keys=True,separators=(',',':')) == json.dumps(canonical(right),sort_keys=True,separators=(',',':'))

def schema_check(value, spec, definitions):
    """Only the explicitly used JSON Schema keywords; unknown ones fail closed."""
    supported={'$ref','type','const','enum','anyOf','properties','required','additionalProperties','patternProperties','items','minItems','maxItems','uniqueItems','minimum','maximum','minLength','pattern'}
    require(not set(spec)-supported,'validator-keyword')
    if '$ref' in spec:
        return schema_check(value,definitions[spec['$ref'].split('/')[-1]],definitions)
    if 'anyOf' in spec:
        errors=[]
        for option in spec['anyOf']:
            try:
                schema_check(value,option,definitions)
                return
            except ContractError as exc:
                errors.append(str(exc))
        # Nullable unions expose the useful non-null branch's diagnostic.
        raise ContractError(errors[-1])
    if 'const' in spec:
        require(equal(value,spec['const']),'const')
    if 'enum' in spec:
        require(any(equal(value,x) for x in spec['enum']),'enum')
    kind=spec.get('type')
    checks={'object':isinstance(value,dict),'array':isinstance(value,list),'string':isinstance(value,str),
            'boolean':type(value) is bool,'null':value is None,
            'integer':type(value) in (int,float) and math.isfinite(value) and int(value)==value,
            'number':type(value) in (int,float) and math.isfinite(value)}
    if kind:
        require(checks[kind],'type')
    if kind=='object':
        props=spec.get('properties',{})
        for key in spec.get('required',[]):
            require(key in value,'required')
        for key,item in value.items():
            if key in props:
                schema_check(item,props[key],definitions)
                continue
            patterns=[s for p,s in spec.get('patternProperties',{}).items() if re.search(p,key)]
            for pattern in patterns:
                schema_check(item,pattern,definitions)
            if not patterns:
                additional=spec.get('additionalProperties',True)
                require(additional is not False,'additional')
                if isinstance(additional,dict):
                    schema_check(item,additional,definitions)
    if kind=='array':
        require(len(value)>=spec.get('minItems',0),'minItems')
        require(len(value)<=spec.get('maxItems',float('inf')),'maxItems')
        if spec.get('uniqueItems'):
            require(not any(equal(value[i],value[j]) for i in range(len(value)) for j in range(i)),'unique')
        for item in value:
            schema_check(item,spec.get('items',{}),definitions)
    if kind in ('number','integer'):
        require(value>=spec.get('minimum',-float('inf')),'minimum')
        require(value<=spec.get('maximum',float('inf')),'maximum')
    if kind=='string':
        require(len(value)>=spec.get('minLength',0),'minLength')
        if 'pattern' in spec:
            require(re.search(spec['pattern'],value) is not None,'pattern')

def validate_boundary(value, boundary, definitions):
    # No field is consumed before the version check, including for malformed bases.
    if value is not None:
        require(isinstance(value,dict) and type(value.get('schema')) in (int,float) and value['schema']==1,'schema-version')
    schema_check(value,definitions[boundary],definitions)

def matrix_check(value, case):
    require(value['rows']==case['expected_rows'] and all(k in case['full_rows'] for k in value['rows']),'extent')
    require(value['ranges']==case['expected_ranges'],'columns')
    require(value['basis']==sorted({lock['key'] for lock in case['locks']}),'selection-basis')
    require(len(value['grid'])==len(value['rows']) and all(len(r)==len(value['ranges']) for r in value['grid']),'shape')
    require(all(type(c) is bool for r in value['grid'] for c in r),'type')
    def cell(address):
        return value['grid'][value['rows'].index(address['fragment'])][value['ranges'].index(address['range'])]
    for lock in case['locks']:
        require(lock['fragment'] in value['rows'] and lock['range'] in value['ranges'] and cell(lock),'lock')
        require(not any(e['fragment']==lock['fragment'] and e['range']==lock['range'] for e in case['errors']),'anchor-scoring-error')
    for error in case['errors']:
        if error['fragment'] in value['rows'] and error['range'] in value['ranges']:
            require(not cell(error),'scoring-error')
    for column,range_name in enumerate(value['ranges']):
        run=[]
        for index in range(len(value['rows'])+1):
            if index<len(value['rows']) and value['grid'][index][column]:
                run.append(value['rows'][index])
            elif run:
                require(any(lock['range']==range_name and lock['fragment'] in run for lock in case['locks']),'run-without-lock')
                run=[]

def fragment_check(value):
    residues=value['residues']
    # Gap detection in these canonical fixtures uses integer author IDs. Insertion
    # codes are stored explicitly; adjacency is never reconstructed by consumers.
    expected=[]
    for index,(first,second) in enumerate(zip(residues,residues[1:])):
        if first['chain']==second['chain'] and second['resid']==first['resid']+1:
            sequence=first['base']+second['base']
            expected.append(dict(number=index+1,key=f'frag-{index+1}-{sequence}',sequence=sequence,residues=[first,second]))
    require(equal(expected,value['fragments']),'fragment-numbering')
    expected_pairs=[dict(down=a['number'],up=b['number'],sequence=a['sequence']+b['sequence'][1])
                    for a,b in zip(expected,expected[1:]) if b['number']==a['number']+1]
    require(equal(expected_pairs,value['adjacent_pairs']),'adjacency')
    keys=[r['key'] for r in expected]
    require(all(c['fragment'] in keys and c['nucleotide'] in (1,2) for c in value['candidates']),'candidate-membership')
    require(len({(c['key'],c['fragment'],c['nucleotide']) for c in value['candidates']})==len(value['candidates']),'candidate-duplicate')

def type_energy(text):
    require(isinstance(text,str) and re.fullmatch(r'[+-]?(0|[1-9][0-9]*)(\.[0-9]{1,2})?',text) is not None,'decimal')
    text=text.lstrip('+')
    whole,_,places=text.partition('.')
    return whole+'.'+places.ljust(2,'0')

def execute(case, definitions):
    value=copy.deepcopy(case['payload'])
    op=case['operation']
    boundaries={'matrix':'S14','matrix-adapt':'S14','fragments':'S22','rmsd':'S11',
                'class-table':'S18','gaussian-consume':'S25','energy-copy':'S24','energy-reconcile':'S24','energy-complete':'S24','threshold-complete':'S11'}
    if op in boundaries:
        validate_boundary(value,boundaries[op],definitions)
    if op=='energy-complete':
        validate_boundary(case['matrix'],'S14',definitions)
    if op=='schema':
        validate_boundary(value,case['boundary'],definitions)
        return {'valid':True}
    if op=='load':
        # Caller retains original raw bytes and diagnostic before writing null.
        try:
            parsed=json.loads(value,parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError,TypeError):
            return dict(value=None,diagnostic='malformed')
        if parsed is None:
            return dict(value=None,diagnostic=None)
        if not isinstance(parsed,dict):
            return dict(value=None,diagnostic='legacy')
        if 'schema' in parsed and (type(parsed['schema']) not in (int,float) or parsed['schema']!=1):
            return dict(value=None,diagnostic='unsupported')
        if 'basis' not in parsed:
            return dict(value=None,diagnostic='legacy')
        if 'schema' not in parsed:
            return dict(value=None,diagnostic='unsupported')
        try:
            validate_boundary(parsed,'S11',definitions)
        except ContractError:
            return dict(value=None,diagnostic='malformed')
        return dict(value=parsed,diagnostic=None)
    if op=='matrix':
        matrix_check(value,case)
    elif op=='fragments':
        fragment_check(value)
    elif op=='clip-clamp':
        return dict(clip1=min(value['clip1'],value['anchor_min']-1),clip2=min(value['clip2'],value['n']-value['anchor_max']))
    elif op=='matrix-adapt':
        full=case['full_rows']; new_rows=case['new_rows']; old_rows=value['rows']
        grid=[copy.deepcopy(value['grid'][old_rows.index(k)]) if k in old_rows else [False]*len(value['ranges']) for k in new_rows]
        for col in range(len(value['ranges'])):
            for boundary,direction in [(0,-1),(-1,1)]:
                if not value['grid'][boundary][col]: continue
                index=full.index(old_rows[boundary])+direction
                while 0<=index<len(full) and full[index] in new_rows and full[index] not in old_rows:
                    key=full[index]
                    if not case['growable'][key][col]: break
                    grid[new_rows.index(key)][col]=True
                    index+=direction
        return {'value':dict(value,rows=new_rows,grid=grid)}
    elif op=='backend-expectation':
        require(value['expected']['filter_name'] in case['legal_names'],'filter-name')
        require(all(c['score_cutoff'] in case['legal_names'] for c in value['candidates']),'filter-name')
    elif op=='range-gate':
        return {'allowed':not value['has_unobserved']}
    elif op=='rmsd':
        for row in value['bases']:
            for key in ('nuc1_ovrmsd','nuc2_ovrmsd'):
                require(row[key] is None or float(row[key])>0,'positive')
    elif op=='type-energy':
        return {'value':type_energy(value)}
    elif op=='energy-copy':
        require(case['fresh'],'stale')
        require(case['confirmed'] or not any(v is not None for r in value['ranges'] for v in r['values'].values()),'confirmation')
        for row,range_name in zip(value['ranges'],case['range_ids']):
            for key,ranges in case['computed'].items():
                if range_name in ranges:
                    row['values'][key]=ranges[range_name]
        return {'value':value}
    elif op=='energy-reconcile':
        old=value['ranges'] if equal(value['basis'],case['basis']) else []
        value={'schema':1,'basis':case['basis'],'ranges':[]}
        for row in case['protein_rows']:
            match=next((r for r in old if equal(r['row'],row)),None)
            value['ranges'].append(copy.deepcopy(match) if match else dict(row=row,values={k:None for k in case['fragment_keys']}))
        return {'value':value}
    elif op=='energy-complete':
        matrix=case['matrix']
        for i,key in enumerate(matrix['rows']):
            for j,range_name in enumerate(matrix['ranges']):
                if not matrix['grid'][i][j]:
                    continue
                row=case['protein_rows'][int(range_name.split('-')[1])-2]
                entry=next((r for r in value['ranges'] if equal(r['row'],row)),None)
                if entry is None or entry['values'].get(key) is None:
                    return {'complete':False}
        return {'complete':True}
    elif op=='threshold-complete':
        kept=case['kept']
        wanted_bases=[b['number'] for b in case['fragment_map']['fragments'] if b['number'] in kept]
        wanted_pairs=[(p['down'],p['up']) for p in case['fragment_map']['adjacent_pairs'] if p['down'] in kept and p['up'] in kept]
        bases=[b for b in value['bases'] if b['fragment'] in kept]
        pairs=[p for p in value['pairs'] if p['down'] in kept and p['up'] in kept]
        complete=sorted(b['fragment'] for b in bases)==sorted(wanted_bases)
        complete=complete and sorted((p['down'],p['up']) for p in pairs)==sorted(wanted_pairs)
        complete=complete and all(b['nuc1_ovrmsd'] is not None and b['nuc2_ovrmsd'] is not None for b in bases)
        complete=complete and all(p['ovrmsd'] is not None and p['crmsd'] is not None for p in pairs)
        return {'complete':complete}
    elif op=='gate':
        return {'allowed':all(value['checks'].values()) and value['validator_status']!='failed'}
    elif op=='proceed':
        needs=value['state'] in ('red','amber') or value['energy']>=float(value['threshold'])
        return {'allowed':not needs or value['confirmed']}
    elif op=='class-table':
        require(value['basis']==case['selected_keys'],'selection-basis')
        keys=[r['key'] for r in value['rows']]
        require(len(set(keys))==len(keys),'class-duplicate')
        require(set(keys)==set(case['selected_keys']),'class-membership')
        for row in value['rows']:
            name='rrm'+str(int(row['rrm']))
            if row.get('tilt30'): name+='-tilt30'
            if row.get('disp45'): name+='-disp45'
            require(name in case['legal_names'],'filter-name')
    elif op=='gaussian-consume':
        require(set(value['legal_filter_names'])==set(case['legal_names']),'filter-name')
        keys=[r['key'] for r in value['classes']]
        require(len(set(keys))==len(keys),'class-duplicate')
        require(set(keys)==set(value['echo']['selected_keys']),'class-membership')
        addresses=[(c['key'],c['fragment'],c['nucleotide']) for c in value['candidates']]
        wanted=[(c['key'],c['fragment'],c['nucleotide']) for c in case['candidate_addresses']]
        require(len(set(addresses))==len(addresses),'candidate-duplicate')
        require(set(addresses)==set(wanted),'candidate-membership')
    elif op=='class-row':
        flags='tilt30' in value and 'disp45' in value
        require(flags if value['rrm'] in (90,95) else 'tilt30' not in value and 'disp45' not in value,'class-flags')
    else:
        raise AssertionError('Unknown operation '+op)
    return {'valid':True}

def run(corpus, definitions):
    results={}
    for case in corpus['cases']:
        try:
            result=execute(case,definitions)
        except ContractError as exc:
            result={'error':str(exc)}
        require(equal(result,case['expected']),'fixture '+case['id']+': '+repr(result)+' != '+repr(case['expected']))
        results[case['id']]=result
    return results
