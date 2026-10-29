"""Build only small W0 representation artifacts; never import scientific modules."""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def obj(properties, optional=(), additional=False):
    return dict(type="object", properties=properties,
                required=[k for k in properties if k not in optional], additionalProperties=additional)
def arr(items, **kw):
    return dict(type="array", items=items, **kw)
def ref(name):
    return {"$ref": "#/$defs/" + name}
def nullable(value):
    return {"anyOf": [{"type": "null"}, value]}
def envelope(properties, optional=()):
    return obj({"schema": {"const": 1}, **properties}, optional)
def marker(basis, properties):
    return nullable(envelope({"basis": basis, "active": {"type": "boolean"}, **properties}))

string = {"type": "string", "minLength": 1}
integer = {"type": "integer"}
positive_int = {"type": "integer", "minimum": 1}
number = {"type": "number"}
# Names are generated from the delivered authoritative JSON, never a second list.
filter_name = {"enum":list(json.loads((ROOT/"alaric/gaussian_stacking_filter.json").read_text())["thresholds"])}
boolean = {"type": "boolean"}
frag = {"type": "string", "pattern": "^frag-[1-9][0-9]*-[ACGU]{2}$"}
range_id = {"type": "string", "pattern": "^range-[1-9][0-9]*$"}
energy = {"type": "string", "pattern": "^-?(0|[1-9][0-9]*)\\.[0-9]{2}$"}
rmsd = {"type": "string", "pattern": "^([1-9][0-9]*(\\.[0-9]{1,2})?|0\\.([1-9][0-9]?|0[1-9]))$"}
row = obj(dict(chain=string, start=integer, end=integer))
project = obj(dict(pdbcode=string, biounit=positive_int))
basis = obj(dict(pdbcode=string, rna=row))
ranges = arr(row, minItems=1, uniqueItems=True)
residue = obj(dict(chain=string, resid=integer, insertion_code={"type":"string"}, base={"enum":list("ACGU")}))
fit_echo = obj(dict(pdb_code=string, rna=row, resids=arr(residue)))
energy_echo = obj(dict(best_fits=fit_echo, protein_ranges=arr(row, uniqueItems=True)))
clip = envelope(dict(basis=basis, clip1={"type":"integer","minimum":0}, clip2={"type":"integer","minimum":0}))
score_echo = obj(dict(bestfit_stackings=energy_echo, selected_keys=arr(string, uniqueItems=True), clip=clip))
class_row = {"anyOf":[obj(dict(key=string,rrm={"enum":[50,60,70,80]})),obj(dict(key=string,rrm={"enum":[90,95]},tilt30=boolean,disp45=boolean))]}
pair_fields = dict(down=positive_int, up=positive_int, sequence={"type":"string","pattern":"^[ACGU]{3}$"})
base_fields = dict(fragment=positive_int, sequence={"type":"string","pattern":"^[ACGU]{2}$"})
def thresholds(computed):
    value = rmsd if computed else nullable(rmsd)
    pair = dict(pair_fields, ovrmsd=value, crmsd=value)
    base = dict(base_fields, nuc1_ovrmsd=value, nuc2_ovrmsd=value)
    if computed:
        pair.update(ovrmsd_raw=number, crmsd_raw=number)
        base.update(nuc1_ovrmsd_raw=number, nuc2_ovrmsd_raw=number)
    return dict(pairs=arr(obj(pair)), bases=arr(obj(base)))

def keyed_values(value):
    return {"type":"object", "patternProperties":{frag['pattern']:value}, "additionalProperties":False}

candidate = obj(dict(key=string, fragment=frag, nucleotide={"enum":[1,2]}))
fragment_row = obj(dict(number=positive_int,key=frag,sequence={"type":"string","pattern":"^[ACGU]{2}$"},residues=arr(residue,minItems=2,maxItems=2)))
map_schema = envelope(dict(echo=basis,residues=arr(residue,minItems=2),fragments=arr(fragment_row,minItems=1),
    adjacent_pairs=arr(obj(pair_fields)),candidates=arr(candidate)))
descriptor = obj(dict(d=number,phi_deg=number,theta_deg=number))
stacking = obj(dict(key=string,range=range_id,protein_resid=integer,descriptor=descriptor,candidates=arr(candidate,minItems=1)))
fit_row = obj(dict(key=frag,fragment=positive_int,sequence={"type":"string","pattern":"^[ACGU]{2}$"},
                  conformer=integer,rotamer=integer,grid=arr(integer,minItems=3,maxItems=3),pdb=string))
energy_cell = obj(dict(fragment=frag,range=range_id,refe=nullable(number),bestfit=nullable(number),error=nullable(string)))
job = obj(dict(key=string,fragment=positive_int,nucleotide={"enum":[1,2]},sequence={"type":"string","pattern":"^[ACGU]{2}$"},
               conformer=integer,rotamer=integer,grid=arr(integer,minItems=3,maxItems=3),range=range_id,protein_resid=integer))
filter_result = {"anyOf":[obj(dict(emitted={"const":False},implied_by=filter_name)),obj(dict(count={"type":"integer","minimum":0},capacity=positive_int,fill={"type":"number","minimum":0,"maximum":1},emitted=boolean,pose_pass=boolean,score=number))]}
computed_class = obj(dict(key=string,class_score=number,rrm={"enum":[50,60,70,80,90,95]},auto_prefilters=obj(dict(tilt30=boolean,disp45=boolean)),filter_name=filter_name,reference_score=number,below_rrm95=boolean))
score_row = obj(dict(key=string,fragment=frag,nucleotide={"enum":[1,2]},score=number))

D = {
 "range_row":row, "project_basis":project, "rna_basis":basis, "fit_echo":fit_echo,"energy_echo":energy_echo,"score_echo":score_echo,
 "S4":marker(project,dict(rows=ranges)),
 "S5":nullable(envelope(dict(pdbcode=string,biounit=positive_int,rows=ranges))),
 "S7":nullable(envelope(dict(echo=obj(dict(structure_id=string,pdbcode=string,biounit=positive_int,rows=ranges)),rows=arr(stacking)))),
 "S8":marker(obj(dict(pdbcode=string,biounit=positive_int,rows=ranges)),dict(selected={"type":"object","additionalProperties":boolean})),
 "S9":nullable(envelope(dict(echo=fit_echo,**thresholds(True)))),
 "S10":nullable(envelope(dict(echo=fit_echo,rows=arr(fit_row)))),
 "S11":marker(basis,thresholds(False)), "S12":{"anyOf":[envelope(dict(basis={"type":"null"},clip1={"const":0},clip2={"const":0})),clip]},
 "S13":nullable(envelope(dict(echo=energy_echo,rows=arr(energy_cell)))),
 "S14":marker(arr(string,uniqueItems=True),dict(rows=arr(frag,uniqueItems=True),ranges=arr(range_id,uniqueItems=True),grid=arr(arr(boolean)))),
 "S15":nullable(envelope(dict(echo=energy_echo,values=keyed_values({"type":"object","patternProperties":{range_id['pattern']:energy},"additionalProperties":False})))),
 "S17":nullable(envelope(dict(echo=energy_echo,rows=arr(stacking)))),
 "S18":marker(arr(string,uniqueItems=True),dict(rows=arr(class_row))),
 "S21":nullable(envelope(dict(structure_id=string))),
 "S22":nullable(map_schema),
 "S23":nullable(envelope(dict(pdb_code=string,range_specs=ref('S5'),pdbs=arr(obj(dict(range=range_id,name=string,pdb=string)))))),
 "S24":nullable(envelope(dict(basis=basis,ranges=arr(obj(dict(row=row,values=keyed_values(nullable(energy)))))))),
 "S25":nullable(envelope(dict(echo=score_echo,filter_set_version=string,legal_filter_names=arr(filter_name,minItems=12,maxItems=12,uniqueItems=True),
                            candidates=arr(score_row),references=arr(obj(dict(key=string,score=number))),classes=arr(computed_class)))),
 "B17":envelope(dict(best_fit_keys=arr(frag,uniqueItems=True),jobs=arr(job))),
 "B18":envelope(dict(filter_set_version=string,min_cell_fraction={"type":"number","minimum":0,"maximum":1},jobs=arr(obj(dict(key=string,fragment=positive_int,nucleotide={"enum":[1,2]},error=string,filters=obj({name:filter_result for name in filter_name["enum"]},filter_name["enum"])),('error',))))),
 "energy-thresholds":envelope({},()),
 "interactions":envelope(dict(clip1={"type":"integer","minimum":0},clip2={"type":"integer","minimum":0},kept_residues=arr(residue),
    ranges=arr(obj(dict(index=positive_int,chain=string,start=integer,end=integer))),
    runs=arr(obj(dict(range=range_id,first_fragment=positive_int,last_fragment=positive_int,anchors=arr(obj(dict(fragment=positive_int,nucleotide={"enum":[1,2]})))))),
    interactions=keyed_values(arr(range_id,uniqueItems=True))))
}
D['energy-thresholds']['patternProperties']={frag['pattern']:{"type":"object","patternProperties":{range_id['pattern']:energy},"additionalProperties":False}}
# Versioned merged validation shares: state codes are established UI outcomes.
validation_row=obj(dict(key=string,state={"enum":["green","amber","red","blocked"]},messages=arr(string)))
D['S19']=nullable(envelope(dict(echo=obj(dict(classes=arr(class_row),clip=clip,selected_keys=arr(string,uniqueItems=True),best_fits=fit_echo)),rows=arr(validation_row))))
D['S20']=nullable(envelope(dict(echo=obj(dict(bases=arr(obj(dict(base_fields,nuc1_ovrmsd=nullable(rmsd),nuc2_ovrmsd=nullable(rmsd)))),clip=clip,selected_keys=arr(string,uniqueItems=True),best_fits=fit_echo)),rows=arr(validation_row))))
def exact_ends(value):
    # ECMA/Python $ also matches before a final newline. Boundary addresses and
    # decimal strings have no such whitespace allowance; require actual EOF.
    if isinstance(value,dict):
        if 'pattern' in value and value['pattern'].endswith('$'):
            value['pattern']=value['pattern'][:-1]+'(?![\\s\\S])'
        if 'patternProperties' in value:
            value['patternProperties']={(k[:-1]+'(?![\\s\\S])' if k.endswith('$') else k):v for k,v in value['patternProperties'].items()}
        for item in value.values(): exact_ends(item)
    elif isinstance(value,list):
        for item in value: exact_ends(item)
exact_ends(D)
SCHEMAS={"$schema":"https://json-schema.org/draft/2020-12/schema","$id":"urn:alaric:w0:contracts:1","$defs":D}

def write(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

if __name__ == '__main__':
    write('schemas.json',SCHEMAS)

# The corpus is generated deterministically; individual malformed payloads stay in JSON.
if __name__ == '__main__':
    import hashlib
    ROOT = HERE.parents[1]
    model_path = ROOT/'alaric/gaussian_stacking_filter.json'
    model = json.loads(model_path.read_text())
    names = list(model['thresholds'])
    rna={'chain':'R','start':101,'end':106}
    proteins=[{'chain':'A','start':10,'end':20},{'chain':'B','start':30,'end':40}]
    basis_value={'pdbcode':'1b7f','rna':rna}
    residues=[{'chain':'R','resid':101+i,'insertion_code':'','base':b} for i,b in enumerate('ACGUAC')]
    fragments=[{'number':i+1,'key':f'frag-{i+1}-'+''.join(x['base'] for x in residues[i:i+2]),'sequence':''.join(x['base'] for x in residues[i:i+2]),'residues':residues[i:i+2]} for i in range(5)]
    keys=[r['key'] for r in fragments]
    pairs=[dict(down=i,up=i+1,sequence='ACGUAC'[i-1:i+2]) for i in range(1,5)]
    fmap=dict(schema=1,echo=basis_value,residues=residues,fragments=fragments,adjacent_pairs=pairs,candidates=[dict(key='anchor-A',fragment=keys[2],nucleotide=1)])
    fit_echo_value=dict(pdb_code='1b7f',rna=rna,resids=residues)
    energy_echo_value=dict(best_fits=fit_echo_value,protein_ranges=proteins)
    clip_value=dict(schema=1,basis=basis_value,clip1=0,clip2=0)
    score_echo_value=dict(bestfit_stackings=energy_echo_value,selected_keys=['anchor-A'],clip=clip_value)
    matrix=dict(schema=1,basis=['anchor-A'],active=True,rows=keys,ranges=['range-2','range-3'],grid=[[False,False],[True,False],[True,False],[True,False],[False,False]])
    edits=dict(schema=1,basis=basis_value,ranges=[dict(row=p,values={k:None for k in keys}) for p in proteins])
    tedit=dict(schema=1,basis=basis_value,active=True,pairs=[dict(p,ovrmsd=None,crmsd=None) for p in pairs],bases=[dict(fragment=r['number'],sequence=r['sequence'],nuc1_ovrmsd=None,nuc2_ovrmsd=None) for r in fragments])
    computed=dict(schema=1,echo=fit_echo_value,pairs=[dict(p,ovrmsd='0.13',crmsd='0.13',ovrmsd_raw=0.123,crmsd_raw=0.123) for p in pairs],bases=[dict(fragment=r['number'],sequence=r['sequence'],nuc1_ovrmsd='0.13',nuc2_ovrmsd='0.13',nuc1_ovrmsd_raw=0.123,nuc2_ovrmsd_raw=0.123) for r in fragments])
    corpus=[]
    def case(name,operation,payload,expected,**context):
        corpus.append(dict(id=name,operation=operation,payload=copy.deepcopy(payload),expected=copy.deepcopy(expected),**copy.deepcopy(context)))
    def shape(name,schema,payload,error=None):
        case(name,'schema',payload,{'error':error} if error else {'valid':True},boundary=schema)
    for sid in ['S4','S5','S7','S8','S9','S10','S11','S13','S14','S15','S17','S18','S19','S20','S21','S22','S23','S24','S25']:
        shape(sid+'-undecided',sid,None)
    shape('threshold-empty-skeleton','S11',tedit)
    inactive=copy.deepcopy(tedit); inactive['active']=False
    shape('threshold-inactive','S11',inactive)
    shape('threshold-computed-raw','S9',computed)
    shape('schema-integral-float','S11',dict(tedit,schema=1.0))
    shape('schema-boolean-rejected','S11',dict(tedit,schema=True),'schema-version')
    shape('clip-integral-floats','S12',dict(schema=1.0,basis=basis_value,clip1=0.0,clip2=1.0))
    shape('clip-nonintegral-rejected','S12',dict(schema=1,basis=basis_value,clip1=.5,clip2=0),'type')
    shape('energy-empty-skeleton','S24',edits)
    shape('empty-selection-is-decided','S8',dict(schema=1,basis=dict(pdbcode='1b7f',biounit=1,rows=[rna]),active=True,selected={}))
    shape('empty-matrix-is-shaped-but-not-consumable','S14',dict(schema=1,basis=[],active=True,rows=[],ranges=[],grid=[]))
    for schema,payload in [('S11',tedit),('S14',matrix),('S22',fmap),('S24',edits)]:
        shape(schema+'-valid',schema,payload)
        missing=copy.deepcopy(payload); del missing['schema']
        shape(schema+'-missing-schema',schema,missing,'schema-version')
        bad=copy.deepcopy(payload); bad['schema']=2
        shape(schema+'-unsupported-schema',schema,bad,'schema-version')
    bad=copy.deepcopy(tedit); bad['active']=1
    shape('strict-boolean-active','S11',bad,'type')
    bad=copy.deepcopy(matrix); bad['grid'][0][0]=1
    shape('strict-boolean-grid','S14',bad,'type')
    bad=copy.deepcopy(tedit); bad['bases'][0]['nuc1_ovrmsd']='1.234'
    shape('threshold-too-many-decimals','S11',bad,'pattern')
    bad=copy.deepcopy(tedit); bad['bases'][0]['nuc1_ovrmsd']='0'
    case('threshold-zero-rejected','rmsd',bad,{'error':'pattern'})
    bad=copy.deepcopy(computed); del bad['pairs'][0]['ovrmsd_raw']
    shape('computed-raw-required','S9',bad,'required')
    for token in ['activation_token','parent_token','validation_receipt','model_identity']:
        bad=dict(tedit,**{token:'rejected-extension'})
        shape('reject-'+token,'S11',bad,'additional')
    for label,text in [('malformed','{"schema":1,'),('legacy','{"active":true,"pairs":[],"bases":[]}'),('unsupported','{"schema":2,"basis":{},"active":true}')]:
        case('load-'+label,'load',text,dict(value=None,diagnostic=label))
    case('load-undecided','load','null',dict(value=None,diagnostic=None))
    case('load-schema2-before-basis','load','{"schema":2}',dict(value=None,diagnostic='unsupported'))
    case('load-missing-schema-with-basis','load',json.dumps(dict(basis=basis_value,active=True)),dict(value=None,diagnostic='unsupported'))
    case('load-malformed-shape','load',json.dumps(dict(schema=1,basis=basis_value,active='yes')),dict(value=None,diagnostic='malformed'))
    case('load-valid-inactive','load',json.dumps(inactive),dict(value=inactive,diagnostic=None))
    case('fragment-ordinal-start101','fragments',fmap,{'valid':True})
    gap=copy.deepcopy(fmap); gap['residues']=[residues[i] for i in [0,1,3,4]]
    gap['residues'][2]['resid']=104;gap['residues'][3]['resid']=105
    gap['fragments']=[dict(number=n,key=f'frag-{n}-'+''.join(r['base'] for r in rs),sequence=''.join(r['base'] for r in rs),residues=rs) for n,rs in [(1,gap['residues'][:2]),(3,gap['residues'][2:])]]
    gap['adjacent_pairs']=[];gap['candidates']=[]
    case('fragment-gap-window2-omitted','fragments',gap,{'valid':True})
    bad=copy.deepcopy(gap);bad['fragments'][1]['number']=4
    case('reject-historical-offset-number4','fragments',bad,{'error':'fragment-numbering'})
    case('preparic-stage2-refuses-gap','range-gate',dict(has_unobserved=True),{'allowed':False})
    case('alaric-gap-is-representable','range-gate',dict(has_unobserved=False),{'allowed':True})
    context=dict(full_rows=keys,expected_rows=keys,expected_ranges=['range-2','range-3'],locks=[dict(fragment=keys[2],range='range-2',key='anchor-A')],errors=[])
    case('matrix-addressed-valid','matrix',matrix,{'valid':True},**context)
    for label,mut in [('reordered',list(reversed(keys))),('duplicate',[keys[0]]*5),('unknown',keys[:-1]+['frag-99-AC'])]:
        bad=dict(matrix,rows=mut);case('matrix-'+label,'matrix',bad,{'error':'unique'} if label=='duplicate' else {'error':'extent'},**context)
    for label,cols in [('reordered',['range-3','range-2']),('duplicate',['range-2','range-2'])]:
        case('columns-'+label,'matrix',dict(matrix,ranges=cols),{'error':'unique'} if label=='duplicate' else {'error':'columns'},**context)
    five=copy.deepcopy(matrix);five['rows']=keys[2:];five['grid']=matrix['grid'][2:]
    three=copy.deepcopy(matrix);three['rows']=keys[:3];three['grid']=matrix['grid'][:3]
    ctx=dict(context,expected_rows=keys[2:])
    case('equal-size-5prime','matrix',five,{'valid':True},**ctx)
    case('equal-size-3prime-not-5prime','matrix',three,{'error':'extent'},**ctx)
    bad=copy.deepcopy(matrix);bad['grid'][2][0]=False
    case('matrix-missing-lock','matrix',bad,{'error':'lock'},**context)
    bad=copy.deepcopy(matrix);bad['grid'][0][1]=True
    case('matrix-unlocked-run','matrix',bad,{'error':'run-without-lock'},**context)
    case('matrix-scoring-error','matrix',matrix,{'error':'scoring-error'},**dict(context,errors=[dict(fragment=keys[1],range='range-2')]))
    case('matrix-anchor-error-needs-deselection','matrix',matrix,{'error':'anchor-scoring-error'},**dict(context,errors=[dict(fragment=keys[2],range='range-2')]))
    case('matrix-empty-not-current-extent','matrix',dict(schema=1,basis=[],active=True,rows=[],ranges=[],grid=[]),{'error':'extent'},**context)
    for label,text,result in [('negative','-16.5','-16.50'),('integer','2','2.00'),('zero','0.00','0.00'),('plus','+1.2','1.20')]:
        case('typed-'+label,'type-energy',text,{'value':result})
    for text in ['NaN','Infinity','1.234','1e2','','-16.59\n',True]:
        case('typed-malformed-'+str(text),'type-energy',text,{'error':'decimal'})
    copied=copy.deepcopy(edits)
    for k in keys: copied['ranges'][0]['values'][k]='-16.59'
    copy_values={k:{'range-2':'-16.59'} for k in keys}
    case('explicit-copy','energy-copy',edits,{'value':copied},computed=copy_values,range_ids=['range-2','range-3'],fresh=True,confirmed=False)
    case('copy-stale-refused','energy-copy',edits,{'error':'stale'},computed=copy_values,range_ids=['range-2','range-3'],fresh=False,confirmed=False)
    case('copy-overwrite-needs-confirmation','energy-copy',copied,{'error':'confirmation'},computed=copy_values,range_ids=['range-2','range-3'],fresh=True,confirmed=False)
    case('copy-overwrite-confirmed','energy-copy',copied,{'value':copied},computed=copy_values,range_ids=['range-2','range-3'],fresh=True,confirmed=True)
    typed=copy.deepcopy(copied);typed['ranges'][0]['values'][keys[2]]='-17.00'
    shape('energy-saved-user-edit','S24',typed)
    case('bestfit-rejected-by-edited-threshold','proceed',dict(state='red',energy=-16.5913,threshold='-17.00',confirmed=False),{'allowed':False})
    case('bestfit-rejection-explicit-proceed','proceed',dict(state='red',energy=-16.5913,threshold='-17.00',confirmed=True),{'allowed':True})
    case('amber-needs-explicit-proceed','proceed',dict(state='amber',energy=-20,threshold='-17.00',confirmed=False),{'allowed':False})
    case('amber-explicit-proceed','proceed',dict(state='amber',energy=-20,threshold='-17.00',confirmed=True),{'allowed':True})
    changed=[proteins[0],dict(proteins[1],end=41)]
    reset_source=copy.deepcopy(copied)
    reset_source['ranges'][1]['values'][keys[2]]='-2.00'
    case('range-edit-only-changed-range-resets','energy-reconcile',reset_source,{'value':copied},basis=basis_value,protein_rows=changed,fragment_keys=keys)
    corpus[-1]['expected']['value']['ranges'][1]['row']=changed[1]
    case('range-reordering-preserves-row-edits','energy-reconcile',copied,{'value':dict(copied,ranges=list(reversed(copied['ranges'])))},basis=basis_value,protein_rows=list(reversed(proteins)),fragment_keys=keys)
    case('refit-same-cells-preserves-user-values','energy-reconcile',typed,{'value':typed},basis=basis_value,protein_rows=proteins,fragment_keys=keys)
    new_basis=dict(basis_value,pdbcode='2abc');new_table=copy.deepcopy(edits);new_table['basis']=new_basis
    case('project-change-all-empty','energy-reconcile',typed,{'value':new_table},basis=new_basis,protein_rows=proteins,fragment_keys=keys)
    for label,table,complete in [('empty',edits,False),('copied',copied,True),('typed',typed,True)]:
        case('energy-completeness-'+label,'energy-complete',table,{'complete':complete},matrix=matrix,protein_rows=proteins)
    shape('energy-export-no-final-newline','energy-thresholds',dict(schema=1,**{keys[2]:{'range-2':'-17.00\n'}}),'pattern')
    shape('energy-export-exact-top-level','energy-thresholds',dict(schema=1,**{keys[2]:{'range-2':'-17.00'}}))
    shape('energy-export-no-envelope','energy-thresholds',dict(schema=1,thresholds={}), 'additional')
    shape('energy-export-two-decimals','energy-thresholds',dict(schema=1,**{keys[2]:{'range-2':'-17.0'}}),'pattern')
    case('threshold-kept-only-complete','threshold-complete',tedit,{'complete':False},kept=[3],fragment_map=fmap)
    partial=copy.deepcopy(tedit)
    for b in partial['bases']:
        if b['fragment']==3: b.update(nuc1_ovrmsd='0.13',nuc2_ovrmsd='0.13')
    case('threshold-clipped-blanks-permitted','threshold-complete',partial,{'complete':True},kept=[3],fragment_map=fmap)
    for gate,checks,validator in [('4-5',['threshold_complete','clip_valid','energetics_fresh'],'T10'),('5-6',['matrix_valid','energetics_fresh','fit_fresh','clip_valid','geometry_fresh','scores_fresh','goldens_passed','energy_complete'],'T27')]:
        ready={k:True for k in checks}
        for status in ['failed','complete','pending','blocked']:
            case(gate+'-'+validator+'-'+status,'gate',dict(checks=ready,validator_status=status),{'allowed':status!='failed'})
        for check in checks:
            case(gate+'-missing-'+check,'gate',dict(checks=dict(ready,**{check:False}),validator_status='complete'),{'allowed':False})
    job_value=dict(key='anchor-A',fragment=3,nucleotide=1,sequence='GU',conformer=0,rotamer=1,grid=[0,1,-1],range='range-2',protein_resid=15)
    shape('gaussian-jobs','B17',dict(schema=1,best_fit_keys=keys,jobs=[job_value]))
    shape('gaussian-result-with-implied','B18',dict(schema=1,filter_set_version=model['filter_set_version'],min_cell_fraction=1/3,jobs=[dict(key='anchor-A',fragment=3,nucleotide=1,filters={'rrm95':dict(count=1,capacity=4,fill=.25,emitted=False,pose_pass=True,score=.1),'rrm50':dict(emitted=False,implied_by='rrm95')})]))
    # Synthetic transport example, not a scientific score or acceptance output.
    scores=dict(schema=1,echo=score_echo_value,filter_set_version=model['filter_set_version'],legal_filter_names=names,
        candidates=[dict(key='anchor-A',fragment=keys[2],nucleotide=1,score=.3)],references=[dict(key='anchor-A',score=4.0)],
        classes=[dict(key='anchor-A',class_score=.3,rrm=90,auto_prefilters=dict(tilt30=True,disp45=False),filter_name='rrm90-tilt30',reference_score=4.0,below_rrm95=False)])
    shape('backend-gaussian-scores-consumption','S25',scores)
    bad=copy.deepcopy(scores);bad['legal_filter_names'][0]='rrm-unknown'
    shape('backend-gaussian-unknown-name','S25',bad,'enum')
    bad=copy.deepcopy(scores);bad['legal_filter_names']=names[:-1]
    shape('backend-gaussian-names-missing','S25',bad,'minItems')
    shape('computed-versus-edited-classes','S18',dict(schema=1,basis=['anchor-A'],active=True,rows=[dict(key='anchor-A',rrm=95,tilt30=False,disp45=True)]))
    for rrm,flags,error in [(50,{},None),(90,dict(tilt30=True,disp45=False),None),(50,dict(tilt30=True), 'class-flags'),(95,{},'class-flags')]:
        case('class-flags-'+str(rrm)+'-'+str(len(flags)),'class-row',dict(key='anchor-A',rrm=rrm,**flags),{'error':error} if error else {'valid':True})
    bad=copy.deepcopy(fmap);bad['candidates'][0]['fragment']='frag-99-AC'
    case('fragment-unknown-candidate','fragments',bad,{'error':'candidate-membership'})
    bad=copy.deepcopy(fmap);bad['adjacent_pairs']=[]
    case('fragment-missing-adjacency','fragments',bad,{'error':'adjacency'})
    for label,c1,c2,lo,hi in [('clamp-selected',3,3,2,4),('unchanged',0,0,2,4)]:
        case('clip-'+label,'clip-clamp',dict(clip1=c1,clip2=c2,n=5,anchor_min=lo,anchor_max=hi),{'clip1':min(c1,lo-1),'clip2':min(c2,5-hi)})
    for label,new_rows,expected_grid in [('expanded',keys,[[True,False],[True,False],*matrix['grid'][2:]]),('reduced',keys[2:4],matrix['grid'][2:4])]:
        old=copy.deepcopy(five)
        case('matrix-adapt-'+label,'matrix-adapt',old,{'value':dict(old,rows=new_rows,grid=expected_grid)},full_rows=keys,new_rows=new_rows,growable={k:[True,True] for k in keys})
    stopped=copy.deepcopy(five);stopped['grid'][0]=[False,False];stopped['grid'][1]=[True,False]
    case('matrix-adapt-stopped-end-preserved','matrix-adapt',stopped,{'value':dict(stopped,rows=keys,grid=[[False,False],[False,False],*stopped['grid']])},full_rows=keys,new_rows=keys,growable={k:[True,True] for k in keys})
    bad=copy.deepcopy(matrix);bad['basis']=[]
    case('matrix-selection-basis-mismatch','matrix',bad,{'error':'selection-basis'},**context)
    shape('gaussian-job-error-no-invented-metrics','B18',dict(schema=1,filter_set_version=model['filter_set_version'],min_cell_fraction=1/3,jobs=[dict(key='anchor-A',fragment=3,nucleotide=1,error='incomplete ring',filters={})]))
    shape('gaussian-implied-no-fabricated-score','B18',dict(schema=1,filter_set_version=model['filter_set_version'],min_cell_fraction=1/3,jobs=[dict(key='anchor-A',fragment=3,nucleotide=1,filters={'rrm50':dict(emitted=False,implied_by='rrm95',score=.1)})]),'required')
    class_table=dict(schema=1,basis=['anchor-A'],active=True,rows=[dict(key='anchor-A',rrm=90,tilt30=True,disp45=False)])
    class_context=dict(selected_keys=['anchor-A'],legal_names=names)
    case('class-table-consume-valid','class-table',class_table,{'valid':True},**class_context)
    case('class-table-numeric-rrm90-float','class-table',dict(class_table,rows=[dict(key='anchor-A',rrm=90.0,tilt30=True,disp45=False)]),{'valid':True},**class_context)
    for label,rows_value,error in [('duplicate',class_table['rows']*2,'class-duplicate'),('missing',[],'class-membership'),('foreign',[dict(key='foreign',rrm=50)],'class-membership')]:
        case('class-table-'+label,'class-table',dict(class_table,rows=rows_value),{'error':error},**class_context)
    case('class-table-stale-selection','class-table',dict(class_table,basis=['old']),{'error':'selection-basis'},**class_context)
    score_context=dict(legal_names=names,candidate_addresses=fmap['candidates'])
    case('gaussian-score-consume-valid','gaussian-consume',scores,{'valid':True},**score_context)
    for label,bad,error in [('candidate-duplicate',dict(scores,candidates=scores['candidates']*2),'candidate-duplicate'),('candidate-missing',dict(scores,candidates=[]),'candidate-membership'),('class-duplicate',dict(scores,classes=scores['classes']*2),'class-duplicate'),('class-missing',dict(scores,classes=[]),'class-membership')]:
        case('gaussian-consume-'+label,'gaussian-consume',bad,{'error':error},**score_context)
    bad=copy.deepcopy(fmap);bad['candidates']*=2
    case('canonical-candidate-duplicate','fragments',bad,{'error':'candidate-duplicate'})
    # Declaration-only backend expected cases; TS consumes S25 and never derives classes.
    backend_cases=[
      dict(id='cutoff-equality',candidates=[dict(score_cutoff='rrm90',offset=0,theta_deg=29,phi_deg=44,clipped=False)],expected=dict(rrm=90,filter_name='rrm90-tilt30-disp45',below_rrm95=False)),
      dict(id='worst-unclipped',candidates=[dict(score_cutoff='rrm50',offset=1,theta_deg=0,phi_deg=0,clipped=False),dict(score_cutoff='rrm90',offset=0,theta_deg=30,phi_deg=45,clipped=False)],expected=dict(rrm=90,filter_name='rrm90',below_rrm95=False)),
      dict(id='clipped-worst-ignored',candidates=[dict(score_cutoff='rrm50',offset=0,theta_deg=0,phi_deg=0,clipped=False),dict(score_cutoff='rrm95',offset=-.01,theta_deg=90,phi_deg=90,clipped=True)],expected=dict(rrm=50,filter_name='rrm50',below_rrm95=False)),
      dict(id='below95-fallback-red',candidates=[dict(score_cutoff='rrm95',offset=-.01,theta_deg=30,phi_deg=45,clipped=False)],expected=dict(rrm=95,filter_name='rrm95',below_rrm95=True))]
    for expectation in backend_cases:
        case('backend-declared-'+expectation['id'],'backend-expectation',expectation,{'valid':True},legal_names=names)
    shape('unique-numeric-equivalent-range-rows','S5',dict(schema=1,pdbcode='1b7f',biounit=1,rows=[rna,dict(rna,start=101.0)]),'unique')
    shape('clip-undecided-startup','S12',dict(schema=1,basis=None,clip1=0,clip2=0))
    # One concrete valid payload for each active JSON boundary. Existing samples
    # above take precedence; these are transport examples, not scientific output.
    samples={
      'S4':dict(schema=1,basis=dict(pdbcode='1b7f',biounit=1),active=True,rows=[rna,*proteins]),
      'S5':dict(schema=1,pdbcode='1b7f',biounit=1,rows=[rna,*proteins]),
      'S7':dict(schema=1,echo=dict(structure_id='1b7f',pdbcode='1b7f',biounit=1,rows=[rna,*proteins]),rows=[]),
      'S10':dict(schema=1,echo=fit_echo_value,rows=[]),
      'S12':clip_value,
      'S13':dict(schema=1,echo=energy_echo_value,rows=[]),
      'S15':dict(schema=1,echo=energy_echo_value,values=copy_values),
      'S17':dict(schema=1,echo=energy_echo_value,rows=[]),
      'S19':dict(schema=1,echo=dict(classes=[],clip=clip_value,selected_keys=[],best_fits=fit_echo_value),rows=[]),
      'S20':dict(schema=1,echo=dict(bases=[],clip=clip_value,selected_keys=[],best_fits=fit_echo_value),rows=[]),
      'S21':dict(schema=1,structure_id='1b7f'),
      'S23':dict(schema=1,pdb_code='1b7f',range_specs=dict(schema=1,pdbcode='1b7f',biounit=1,rows=[rna,*proteins]),pdbs=[dict(range='range-1',name='range-1-aa.pdb',pdb='ATOM  example\n')]),
      'interactions':dict(schema=1,clip1=0,clip2=0,kept_residues=residues,ranges=[dict(index=2,**proteins[0])],runs=[],interactions={})}
    for c in list(corpus):
        if c['operation']=='schema' and c['expected']=={'valid':True} and c['payload'] is not None:
            samples.setdefault(c['boundary'],c['payload'])
    for sid,sample in samples.items():
        shape(sid+'-concrete-valid',sid,sample)
        for version in [None,2]:
            bad=copy.deepcopy(sample)
            if version is None: del bad['schema']
            else: bad['schema']=version
            shape(sid+'-all-version-'+str(version),sid,bad,'schema-version')
    missing_base=copy.deepcopy(partial);missing_base['bases']=[b for b in partial['bases'] if b['fragment']!=3]
    case('threshold-missing-kept-base','threshold-complete',missing_base,{'complete':False},kept=[3],fragment_map=fmap)
    all_filled=copy.deepcopy(tedit)
    for b in all_filled['bases']: b.update(nuc1_ovrmsd='0.13',nuc2_ovrmsd='0.13')
    for p in all_filled['pairs']: p.update(ovrmsd='0.13',crmsd='0.13')
    all_filled['pairs']=[]
    case('threshold-missing-kept-pair','threshold-complete',all_filled,{'complete':False},kept=[2,3],fragment_map=fmap)
    bad=copy.deepcopy(copied);bad['ranges'][0]['values'][keys[2]]='garbage'
    case('energy-complete-malformed-nonnull','energy-complete',bad,{'error':'pattern'},matrix=matrix,protein_rows=proteins)
    consumers={'class-table','gaussian-consume','matrix','matrix-adapt','fragments','rmsd','energy-copy','energy-reconcile','energy-complete','threshold-complete'}
    seen=set()
    for c in list(corpus):
        if c['operation'] not in consumers or c['operation'] in seen: continue
        seen.add(c['operation'])
        for version in [None,2]:
            mutation=copy.deepcopy(c);mutation['id']=c['operation']+'-semantic-version-'+str(version)
            if version is None: del mutation['payload']['schema']
            else: mutation['payload']['schema']=version
            mutation['expected']={'error':'schema-version'};corpus.append(mutation)
    # References, not copies of the delivered numerical corpus or a new scorer.
    goldens=dict(schema=1,source='alaric/gaussian_stacking_filter.json',source_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        filter_set_version=model['filter_set_version'],spec_digest=model['spec_digest'],
        pointers={k:'/golden/'+k for k in model['golden'] if k!='notes'},
        counts={k:len(v) for k,v in model['golden'].items() if isinstance(v,list)},
        backend_obligations=dict(relative_score_tolerance=1e-12,future_T28_A9_1b7f_tolerance=1e-6,
          classification=['worst unclipped candidate score','smallest reached RRM cutoff; below95 red fallback','strict tilt30 and disp45 across all unclipped candidates','mask d<5 phi<60; equality excluded','score equality reaches cutoff'],
          rounding_source='plans/final/contract-boundaries.md §1; future A4/A6 goldens',rounding_examples=[dict(raw='-16.5913',rounded='-16.59'),dict(raw='-16.59',rounded='-16.58'),dict(raw='0.123',rounded='0.13')]))
    write('fixtures.json',dict(schema=1,cases=corpus))
    write('golden-references.json',goldens)
    write('inventory.json',dict(schema=1,active_transformers=['T'+str(i) for i in range(29) if i!=17],active_shares=['S'+str(i) for i in range(1,26) if i!=16],
      transformer_names=dict(zip(['T'+str(i) for i in range(29) if i!=17],['parse_ranges','ranges_to_pdb','pdbs_to_folder','aareduce_ranges','detect_stacking','best_fit','refe_fragments','best_fit_tables','best_fit_thresholds','parse_best_fits','validate_threshold_edits','clip_fragments','energetics_ligands','energetics_range_<k>','merge_energetics','interaction_handoff','energy_thresholds','detect_stacking_bestfit','validate_gaussian_classes','gaussian_check_jobs','gaussian_check','merge_gaussian_validation','validate_refe_anchors','list_fragments','share_range_pdbs_aa','structure_id','validate_energy_threshold_edits','gaussian_scores'])),
      schemas={k:'schemas.json#/$defs/'+k for k in D if k.startswith(('S','B'))},
      non_enveloped_shares={'S1':'pdbcode text','S2':'biounit positive integer','S3':'CIF text','S6':'unshared list[str]; internal cell'},
      backend_boundaries={'B17':'gaussian_check_jobs','B18':'gaussian_check','interactions':'interaction_handoff','energy-thresholds':'energy-thresholds.json'}))
