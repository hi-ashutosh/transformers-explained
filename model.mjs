/** Numerical Transformer forward pass, with every intermediate exposed.
 * One causal post-LN decoder block; row vectors; exactly matches train.py.
 * No framework, server, pretrained language service, or scripted predictions.
 */
export function softmax(values,temperature=1){
 if(!(temperature>0))throw new Error('Temperature must be positive.');
 const finite=values.filter(Number.isFinite);
 if(!finite.length)throw new Error('An attention row must have at least one allowed key.');
 const max=Math.max(...finite),e=values.map(x=>Number.isFinite(x)?Math.exp((x-max)/temperature):0),sum=e.reduce((a,b)=>a+b,0);
 return e.map(x=>x/sum);
}
export const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
export const add=(a,b)=>a.map((v,i)=>v+b[i]);
export function linear(x,w,b=null){return w[0].map((_,j)=>x.reduce((s,v,i)=>s+v*w[i][j],b?b[j]:0));}
export function position(i,d){return Array.from({length:d},(_,j)=>{const angle=i/10000**(2*Math.floor(j/2)/d);return j%2?Math.cos(angle):Math.sin(angle);});}
export function normalize(x,g,b,epsilon=1e-5){const mean=x.reduce((a,b)=>a+b,0)/x.length,variance=x.reduce((s,v)=>s+(v-mean)**2,0)/x.length;return x.map((v,i)=>(v-mean)/Math.sqrt(variance+epsilon)*g[i]+b[i]);}
export function tokenize(text,model){
 const words=text.trim().toLowerCase().split(/\s+/).filter(Boolean);
 if(words.length+1>model.config.context)throw new Error(`Use at most ${model.config.context-1} words; this toy model has an ${model.config.context}-token context including <bos>.`);
 const unknown=words.filter(w=>!model.vocab.includes(w));
 if(unknown.length)throw new Error(`Outside the toy vocabulary: ${[...new Set(unknown)].join(', ')}. Choose an example or use words listed below.`);
 return [0,...words.map(w=>model.vocab.indexOf(w))];
}
export function forward(model,ids,{causal=true,positions=true,checkpoint='trained'}={}){
 const w=checkpoint==='untrained'?model.untrained:model.weights;
 const {dimension:d,heads:h,epsilon}=model.config,dh=d/h,n=ids.length;
 const embeddings=ids.map(id=>w.embedding[id].map(v=>v*Math.sqrt(d)));
 const positional=ids.map((_,i)=>positions?position(i,d):Array(d).fill(0));
 const x=embeddings.map((v,i)=>add(v,positional[i]));
 const q=x.map(v=>linear(v,w.wq)),k=x.map(v=>linear(v,w.wk)),v=x.map(v=>linear(v,w.wv));
 const heads=Array.from({length:h},(_,head)=>{
  const slice=row=>row.slice(head*dh,(head+1)*dh),qs=q.map(slice),ks=k.map(slice),vs=v.map(slice);
  const raw=qs.map(a=>ks.map(b=>dot(a,b)/Math.sqrt(dh)));
  const masked=raw.map((row,i)=>row.map((score,j)=>causal&&j>i?-Infinity:score));
  const weights=masked.map(row=>softmax(row));
  const output=weights.map(row=>Array.from({length:dh},(_,c)=>row.reduce((s,a,j)=>s+a*vs[j][c],0)));
  return {q:qs,k:ks,v:vs,raw,masked,weights,output};
 });
 const merged=ids.map((_,i)=>heads.flatMap(head=>head.output[i]));
 const attention=merged.map(row=>linear(row,w.wo));
 const sum1=x.map((row,i)=>add(row,attention[i]));
 const norm1=sum1.map(row=>normalize(row,w.g1,w.n1,epsilon));
 const pre=norm1.map(row=>linear(row,w.w1,w.b1));
 const activated=pre.map(row=>row.map(v=>Math.max(0,v)));
 const ffn=activated.map(row=>linear(row,w.w2,w.b2));
 const sum2=norm1.map((row,i)=>add(row,ffn[i]));
 const norm2=sum2.map(row=>normalize(row,w.g2,w.n2,epsilon));
 const logits=norm2.map(row=>linear(row,w.readout,w.bias));
 const probabilities=logits.map(row=>softmax(row));
 return {ids,embeddings,positional,x,q,k,v,heads,merged,attention,sum1,norm1,pre,activated,ffn,sum2,norm2,logits,probabilities};
}
export function trainingExample(model,sentence,options={}){
 const words=sentence.split(' '),ids=[0,...words.map(w=>model.vocab.indexOf(w))],targets=[...ids.slice(1),1];
 const trace=forward(model,ids,options);
 const losses=targets.map((target,i)=>-Math.log(Math.max(1e-300,trace.probabilities[i][target])));
 return {trace,targets,losses,loss:losses.reduce((a,b)=>a+b,0)/losses.length};
}
export function sample(probabilities,random=Math.random){const u=random();let c=0;for(let i=0;i<probabilities.length;i++){c+=probabilities[i];if(u<c)return i;}return probabilities.length-1;}
