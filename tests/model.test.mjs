import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {forward,tokenize,softmax,position,trainingExample} from '../model.mjs';
const model=JSON.parse(readFileSync(new URL('../model.json',import.meta.url)));
const cases=JSON.parse(readFileSync(new URL('../parity.json',import.meta.url)));
const near=(a,b,tolerance=1e-10)=>assert.ok(Math.abs(a-b)<tolerance,`${a} vs ${b}`);
test('browser logits match the independent NumPy reference at every position',()=>{for(const example of cases){const trace=forward(model,tokenize(example.prompt,model));trace.logits.forEach((row,i)=>row.forEach((v,j)=>near(v,example.logits[i][j])));}});
test('attention normalizes over allowed keys and masks future entries exactly',()=>{
 const ids=tokenize(cases[0].prompt,model);
 for(const checkpoint of ['trained','untrained']){
  const trace=forward(model,ids,{checkpoint});
  for(const head of trace.heads){
   head.weights.forEach((row,i)=>{
    near(row.reduce((a,b)=>a+b),1);
    row.forEach((v,j)=>{assert.ok(v>=0&&Number.isFinite(v));if(j>i)assert.equal(v,0);});
   });
  }
 }
});
test('future input edits cannot change earlier causal predictions',()=>{const ids=tokenize('the cat sat on the',model),edited=ids.slice();edited[5]=model.vocab.indexOf('rug');const a=forward(model,ids),b=forward(model,edited);for(let i=0;i<5;i++)a.logits[i].forEach((v,j)=>near(v,b.logits[i][j]));});
test('unmasked attention without positions is permutation equivariant',()=>{const ids=tokenize('the cat sat on the',model),permutation=[3,0,5,2,1,4];const options={causal:false,positions:false};const a=forward(model,ids,options),b=forward(model,permutation.map(i=>ids[i]),options);b.logits.forEach((row,i)=>row.forEach((v,j)=>near(v,a.logits[permutation[i]][j])));});
test('training improves loss and the model retrieves the noun-dependent completion',()=>{let trained=0,untrained=0;for(const sentence of model.sentences){trained+=trainingExample(model,sentence).loss;untrained+=trainingExample(model,sentence,{checkpoint:'untrained'}).loss;}near(trained/model.sentences.length,model.history.at(-1).loss);near(untrained/model.sentences.length,model.history[0].loss);assert.ok(trained<untrained/4);for(const [prefix,target] of [['the cat sat on the','mat'],['the dog slept on the','rug'],['the bird stood on the','branch']]){const p=forward(model,tokenize(prefix,model)).probabilities.at(-1);assert.equal(model.vocab[p.indexOf(Math.max(...p))],target);assert.ok(p[model.vocab.indexOf(target)]>.95);}});
test('teacher forcing uses shifted labels, including the final END token',()=>{const {trace,targets}=trainingExample(model,model.sentences[0]);assert.deepEqual(trace.ids.slice(1),targets.slice(0,-1));assert.equal(trace.ids[0],0);assert.equal(targets.at(-1),1);});
test('embedding identity, positions, and block widths remain consistent',()=>{const trace=forward(model,tokenize(cases[0].prompt,model));assert.deepEqual(trace.embeddings[1],trace.embeddings[5]);assert.notDeepEqual(position(1,8),position(5,8));assert.equal(trace.heads.length,2);assert.equal(trace.heads[0].output[0].length,4);assert.equal(trace.merged[0].length,8);assert.equal(trace.pre[0].length,16);assert.equal(trace.norm2[0].length,8);});
test('tokenizer rejects unknown vocabulary and context overflow; softmax is stable',()=>{assert.throws(()=>tokenize('a wonderful new sentence',model),/vocabulary/);assert.throws(()=>tokenize('the cat sat on the mat the cat',model),/at most/);assert.deepEqual(softmax([10000,10000,-Infinity]),[.5,.5,0]);assert.throws(()=>softmax([0,1],0),/positive/);});
