/** W0 transport consumer checks only. No Gaussian scoring/classification. */
declare function require(name: string): any;
declare const process: any;
const fs = require('fs');
type Json = any;
class ContractError extends Error {}
function need(condition: boolean, code: string): void { if (!condition) throw new ContractError(code); }
function sorted(value: Json): Json {
  if (Array.isArray(value)) return value.map(sorted);
  if (value !== null && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(k=>[k,sorted(value[k])]));
  return value;
}
function equal(a: Json,b: Json): boolean { return JSON.stringify(sorted(a))===JSON.stringify(sorted(b)); }
function schemaCheck(value: Json,spec: Json,defs: Json): void {
  const supported=['$ref','type','const','enum','anyOf','properties','required','additionalProperties','patternProperties','items','minItems','maxItems','uniqueItems','minimum','maximum','minLength','pattern'];
  need(Object.keys(spec).every(k=>supported.includes(k)),'validator-keyword');
  if (spec.$ref) return schemaCheck(value,defs[spec.$ref.split('/').pop()],defs);
  if (spec.anyOf) {
    let error: any;
    for (const option of spec.anyOf) { try { schemaCheck(value,option,defs); return; } catch(e) { if (!(e instanceof ContractError)) throw e; error=e; } }
    throw error;
  }
  if ('const' in spec) need(equal(value,spec.const),'const');
  if (spec.enum) need(spec.enum.some((x: Json)=>equal(x,value)),'enum');
  const kind=spec.type;
  const checks: {[key: string]: boolean}={object:value!==null && !Array.isArray(value) && typeof value==='object',array:Array.isArray(value),string:typeof value==='string',boolean:typeof value==='boolean',null:value===null,integer:typeof value==='number' && Number.isInteger(value),number:typeof value==='number' && Number.isFinite(value)};
  if (kind) need(checks[kind],'type');
  if (kind==='object') {
    const props=spec.properties ?? {};
    for (const key of spec.required ?? []) need(key in value,'required');
    for (const [key,item] of Object.entries(value)) {
      if (key in props) { schemaCheck(item,props[key],defs); continue; }
      const patterns=Object.entries(spec.patternProperties ?? {}).filter(([p])=>new RegExp(p).test(key));
      for (const [,pattern] of patterns) schemaCheck(item,pattern,defs);
      if (!patterns.length) { const additional=spec.additionalProperties ?? true; need(additional!==false,'additional'); if (typeof additional==='object') schemaCheck(item,additional,defs); }
    }
  }
  if (kind==='array') {
    need(value.length >= (spec.minItems ?? 0),'minItems'); need(value.length <= (spec.maxItems ?? Infinity),'maxItems');
    if (spec.uniqueItems) need(new Set(value.map((x: Json)=>JSON.stringify(sorted(x)))).size===value.length,'unique');
    for (const item of value) schemaCheck(item,spec.items ?? {},defs);
  }
  if (kind==='number' || kind==='integer') { need(value >= (spec.minimum ?? -Infinity),'minimum'); need(value <= (spec.maximum ?? Infinity),'maximum'); }
  if (kind==='string') { need(value.length >= (spec.minLength ?? 0),'minLength'); if (spec.pattern) need(new RegExp(spec.pattern).test(value),'pattern'); }
}
function validateBoundary(value: Json,boundary: string,defs: Json): void {
  if (value!==null) need(typeof value==='object' && value.schema===1,'schema-version');
  schemaCheck(value,defs[boundary],defs);
}
function matrixCheck(value: Json,c: Json): void {
  need(equal(value.rows,c.expected_rows) && value.rows.every((k: string)=>c.full_rows.includes(k)),'extent');
  need(equal(value.ranges,c.expected_ranges),'columns');
  need(equal(value.basis,[...new Set(c.locks.map((l: Json)=>l.key))].sort()),'selection-basis');
  need(value.grid.length===value.rows.length && value.grid.every((r: Json)=>r.length===value.ranges.length),'shape');
  need(value.grid.every((r: Json)=>r.every((v: Json)=>typeof v==='boolean')),'type');
  const cell=(a: Json)=>value.grid[value.rows.indexOf(a.fragment)][value.ranges.indexOf(a.range)];
  for (const lock of c.locks) {
    need(value.rows.includes(lock.fragment) && value.ranges.includes(lock.range) && cell(lock),'lock');
    need(!c.errors.some((e: Json)=>e.fragment===lock.fragment && e.range===lock.range),'anchor-scoring-error');
  }
  for (const error of c.errors) if (value.rows.includes(error.fragment) && value.ranges.includes(error.range)) need(!cell(error),'scoring-error');
  for (let col=0;col<value.ranges.length;col++) {
    let run: string[]=[];
    for (let i=0;i<=value.rows.length;i++) {
      if (i<value.rows.length && value.grid[i][col]) run.push(value.rows[i]);
      else if (run.length) { need(c.locks.some((l: Json)=>l.range===value.ranges[col] && run.includes(l.fragment)),'run-without-lock'); run=[]; }
    }
  }
}
function fragmentCheck(value: Json): void {
  const expected: Json[]=[];
  for(let i=0;i<value.residues.length-1;i++) {
    const a=value.residues[i],b=value.residues[i+1];
    if(a.chain===b.chain && b.resid===a.resid+1) { const sequence=a.base+b.base; expected.push({number:i+1,key:`frag-${i+1}-${sequence}`,sequence,residues:[a,b]}); }
  }
  need(equal(expected,value.fragments),'fragment-numbering');
  const pairs: Json[]=[];
  for(let i=0;i<expected.length-1;i++) { const a=expected[i],b=expected[i+1]; if(b.number===a.number+1) pairs.push({down:a.number,up:b.number,sequence:a.sequence+b.sequence[1]}); }
  need(equal(pairs,value.adjacent_pairs),'adjacency');
  need(value.candidates.every((c: Json)=>expected.some(r=>r.key===c.fragment) && [1,2].includes(c.nucleotide)),'candidate-membership');
  need(new Set(value.candidates.map((c: Json)=>JSON.stringify([c.key,c.fragment,c.nucleotide]))).size===value.candidates.length,'candidate-duplicate');
}
function typeEnergy(text: Json): string {
  need(typeof text==='string' && /^[+-]?(0|[1-9][0-9]*)(\.[0-9]{1,2})?$(?![\s\S])/.test(text),'decimal');
  const [whole,places='']=text.replace(/^\+/,'').split('.'); return whole+'.'+places.padEnd(2,'0');
}
function execute(c: Json,defs: Json): Json {
  const value=JSON.parse(JSON.stringify(c.payload));
  const boundaries: {[key: string]: string}={matrix:'S14','matrix-adapt':'S14',fragments:'S22',rmsd:'S11','class-table':'S18','gaussian-consume':'S25','energy-copy':'S24','energy-reconcile':'S24','energy-complete':'S24','threshold-complete':'S11'};
  if(c.operation in boundaries) validateBoundary(value,boundaries[c.operation],defs);
  if(c.operation==='energy-complete') validateBoundary(c.matrix,'S14',defs);
  switch(c.operation) {
    case 'schema': validateBoundary(value,c.boundary,defs); return {valid:true};
    case 'load': {
      let parsed;
      try { parsed=JSON.parse(value); } catch { return {value:null,diagnostic:'malformed'}; }
      if(parsed===null) return {value:null,diagnostic:null};
      if(typeof parsed!=='object') return {value:null,diagnostic:'legacy'};
      if('schema' in parsed && parsed.schema!==1) return {value:null,diagnostic:'unsupported'};
      if(!('basis' in parsed)) return {value:null,diagnostic:'legacy'};
      if(!('schema' in parsed)) return {value:null,diagnostic:'unsupported'};
      try { validateBoundary(parsed,'S11',defs); } catch(e) { if (!(e instanceof ContractError)) throw e; return {value:null,diagnostic:'malformed'}; }
      return {value:parsed,diagnostic:null};
    }
    case 'matrix': matrixCheck(value,c); break;
    case 'fragments': fragmentCheck(value); break;
    case 'clip-clamp': return {clip1:Math.min(value.clip1,value.anchor_min-1),clip2:Math.min(value.clip2,value.n-value.anchor_max)};
    case 'matrix-adapt': {
      const grid=c.new_rows.map((key: string)=>value.rows.includes(key) ? [...value.grid[value.rows.indexOf(key)]] : value.ranges.map(()=>false));
      for(let col=0;col<value.ranges.length;col++) for(const [boundary,direction] of [[0,-1],[value.rows.length-1,1]]) {
        if(!value.grid[boundary][col]) continue;
        let i=c.full_rows.indexOf(value.rows[boundary])+direction;
        while(i>=0 && i<c.full_rows.length && c.new_rows.includes(c.full_rows[i]) && !value.rows.includes(c.full_rows[i])) {
          const key=c.full_rows[i]; if(!c.growable[key][col]) break;
          grid[c.new_rows.indexOf(key)][col]=true; i+=direction;
        }
      }
      return {value:{...value,rows:c.new_rows,grid}};
    }
    case 'backend-expectation':
      need(c.legal_names.includes(value.expected.filter_name),'filter-name');
      need(value.candidates.every((x: Json)=>c.legal_names.includes(x.score_cutoff)),'filter-name'); break;
    case 'range-gate': return {allowed:!value.has_unobserved};
    case 'rmsd': for(const row of value.bases) for(const key of ['nuc1_ovrmsd','nuc2_ovrmsd']) need(row[key]===null || Number(row[key])>0,'positive'); break;
    case 'type-energy': return {value:typeEnergy(value)};
    case 'energy-copy': {
      need(c.fresh,'stale'); need(c.confirmed || !value.ranges.some((r: Json)=>Object.values(r.values).some(v=>v!==null)),'confirmation');
      value.ranges.forEach((r: Json,i: number)=>{ for(const [key,ranges] of Object.entries(c.computed) as [string,Json][]) if(c.range_ids[i] in ranges) r.values[key]=ranges[c.range_ids[i]]; });
      return {value};
    }
    case 'energy-reconcile': {
      const old=equal(value.basis,c.basis) ? value.ranges : [];
      return {value:{schema:1,basis:c.basis,ranges:c.protein_rows.map((row: Json)=>old.find((r: Json)=>equal(r.row,row)) ?? {row,values:Object.fromEntries(c.fragment_keys.map((k: string)=>[k,null]))})}};
    }
    case 'energy-complete': {
      for(let i=0;i<c.matrix.rows.length;i++) for(let j=0;j<c.matrix.ranges.length;j++) if(c.matrix.grid[i][j]) {
        const row=c.protein_rows[Number(c.matrix.ranges[j].split('-')[1])-2];
        const entry=value.ranges.find((r: Json)=>equal(r.row,row));
        if(!entry || entry.values[c.matrix.rows[i]]==null) return {complete:false};
      }
      return {complete:true};
    }
    case 'threshold-complete': {
      const wantedBases=c.fragment_map.fragments.filter((b: Json)=>c.kept.includes(b.number)).map((b: Json)=>b.number).sort();
      const wantedPairs=c.fragment_map.adjacent_pairs.filter((p: Json)=>c.kept.includes(p.down) && c.kept.includes(p.up)).map((p: Json)=>[p.down,p.up]).sort();
      const bases=value.bases.filter((b: Json)=>c.kept.includes(b.fragment));
      const pairs=value.pairs.filter((p: Json)=>c.kept.includes(p.down) && c.kept.includes(p.up));
      return {complete:equal(bases.map((b: Json)=>b.fragment).sort(),wantedBases) && equal(pairs.map((p: Json)=>[p.down,p.up]).sort(),wantedPairs) && bases.every((b: Json)=>b.nuc1_ovrmsd!==null && b.nuc2_ovrmsd!==null) && pairs.every((p: Json)=>p.ovrmsd!==null && p.crmsd!==null)};
    }
    case 'gate': return {allowed:Object.values(value.checks).every(Boolean) && value.validator_status!=='failed'};
    case 'proceed': return {allowed:!(['red','amber'].includes(value.state) || value.energy>=Number(value.threshold)) || value.confirmed};
    case 'class-table': {
      need(equal(value.basis,c.selected_keys),'selection-basis');
      const keys=value.rows.map((r: Json)=>r.key);
      need(new Set(keys).size===keys.length,'class-duplicate');
      need(equal([...keys].sort(),[...c.selected_keys].sort()),'class-membership');
      for(const r of value.rows) need(c.legal_names.includes('rrm'+r.rrm+(r.tilt30 ? '-tilt30' : '')+(r.disp45 ? '-disp45' : '')),'filter-name');
      break;
    }
    case 'gaussian-consume': {
      need(equal([...value.legal_filter_names].sort(),[...c.legal_names].sort()),'filter-name');
      const keys=value.classes.map((r: Json)=>r.key);
      need(new Set(keys).size===keys.length,'class-duplicate');
      need(equal([...keys].sort(),[...value.echo.selected_keys].sort()),'class-membership');
      const address=(x: Json)=>JSON.stringify([x.key,x.fragment,x.nucleotide]);
      const addresses=value.candidates.map(address),wanted=c.candidate_addresses.map(address);
      need(new Set(addresses).size===addresses.length,'candidate-duplicate');
      need(equal([...addresses].sort(),[...wanted].sort()),'candidate-membership'); break;
    }
    case 'class-row': need([90,95].includes(value.rrm) ? 'tilt30' in value && 'disp45' in value : !('tilt30' in value) && !('disp45' in value),'class-flags'); break;
    default: throw new Error('Unknown operation '+c.operation);
  }
  return {valid:true};
}
const dir=process.argv[2];
const defs=JSON.parse(fs.readFileSync(dir+'/schemas.json','utf8')).$defs;
const corpus=JSON.parse(fs.readFileSync(dir+'/fixtures.json','utf8'));
const results: Json={};
for(const c of corpus.cases) {
  let result;
  try { result=execute(c,defs); } catch(e) { if(!(e instanceof ContractError)) throw e; result={error:e.message}; }
  need(equal(result,c.expected),'fixture '+c.id+': '+JSON.stringify(result)+' != '+JSON.stringify(c.expected));
  results[c.id]=result;
}
process.stdout.write(JSON.stringify(results));
