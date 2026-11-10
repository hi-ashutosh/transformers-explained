import {forward,tokenize,softmax,dot,trainingExample,sample} from './model.mjs';
const $=s=>document.querySelector(s);
const model=await (await fetch('model.json')).json();
let text='the cat sat on the',ids=tokenize(text,model),selected=ids.length-1,key=2,head=0,stage=1,ended=false;
const options={causal:true,positions:true,checkpoint:'trained'};
const labels=['Represent the tokens','Compute attention','Inspect Q, K, V','Combine multiple heads','Complete the block','Predict the next token'];
const descriptions=[
 'Turn token identities into learned vectors, then add a positional signal. The same word starts with the same embedding at every position; its context changes later in the block.',
 'For the selected query token, compare its query with every allowed key. Scale the scores, apply the causal mask, and use softmax to get a distribution of attention weights.',
 'Q, K, and V are learned projections of the current token representations. Queries and keys determine the weights; the values supply the information that gets mixed.',
 'Each head has its own learned projections and produces a different mixture. Concatenate their outputs, then apply a learned output projection. A head is not assigned a grammatical role.',
 'Attention is one sublayer. Residual connections preserve a route for existing information, normalization stabilizes feature scales, and a feed-forward network transforms each token separately.',
 'The final representation at the last position becomes one score per vocabulary token. Softmax turns those scores into probabilities. Generation appends one token, then runs the model again.'
];
const word=id=>model.vocab[id]==='<bos>'?'START':model.vocab[id]==='<eos>'?'END':model.vocab[id];
function vector(label,values){const max=Math.max(1,...values.map(Math.abs));return `<div class="vector-row"><span>${label}</span><div class="vector">${values.map((v,i)=>`<span title="Feature ${i+1}: ${v.toFixed(5)}" style="background:${v>=0?`rgba(41,105,145,${.07+.65*Math.abs(v)/max})`:`rgba(199,101,54,${.07+.65*Math.abs(v)/max})`};color:${Math.abs(v)/max>.68?'white':'var(--ink)'}">${v.toFixed(2)}</span>`).join('')}</div></div>`;}
function attentionBars(trace,h,q){return trace.heads[h].weights[q].map((a,j)=>`<button class="weight-row ${j===key?'inspected':''}" data-key="${j}" aria-label="Inspect key ${j}: ${word(ids[j])}, attention ${(a*100).toFixed(2)} percent"><span>${j} · ${word(ids[j])}</span><span class="weight-track"><i style="width:${100*a}%"></i></span><strong>${options.causal&&j>q?'masked':(100*a).toFixed(1)+'%'}</strong></button>`).join('');}
function render(){
 const trace=forward(model,ids,options),h=trace.heads[head],d=model.config.dimension,dh=d/model.config.heads;
 selected=Math.min(selected,ids.length-1);key=Math.min(key,ids.length-1);
 $('#tokens').innerHTML=ids.map((id,i)=>`<button data-token="${i}" class="token ${selected===i?'selected':''}" aria-pressed="${selected===i}"><span>${word(id)}</span><small>position ${i}</small></button>`).join('');
 $('#stage-title').textContent=labels[stage];$('#stage-description').textContent=descriptions[stage];
 $('#stage-steps').innerHTML=labels.map((label,i)=>`<button data-stage="${i}" aria-pressed="${stage===i}"><span>0${i+1}</span> ${['Tokens','Attention','Q / K / V','Heads','Block','Prediction'][i]}</button>`).join('');
 $('#query-label').textContent=`${word(ids[selected])} · position ${selected}`;
 $('#checkpoint-tag').textContent=options.checkpoint==='untrained'?'BEFORE TRAINING':options.causal&&options.positions?'TRAINED TOY TRANSFORMER':'COUNTERFACTUAL SETTINGS';
 $('#head-0').setAttribute('aria-pressed',String(head===0));$('#head-1').setAttribute('aria-pressed',String(head===1));
 let content='';
 if(stage===0){content=`<p class="micro-note">Eight numerical features per token. Blue = positive; orange = negative. Feature coordinates are learned numbers, not named meanings.</p>${vector('Scaled embedding',trace.embeddings[selected])}${vector('Position signal',trace.positional[selected])}${vector('Their sum → x',trace.x[selected])}<p class="micro-note">Select either occurrence of “the” to compare its starting representation. The embedding is shared; its positional signal differs when enabled.</p>`;}
 if(stage===1){content=`<div class="attention-caption"><strong>What “${word(ids[selected])}” can read</strong><span>Head ${head+1} · row sums to 100%</span></div><div class="weights">${attentionBars(trace,head,selected)}</div><p class="micro-note">${options.causal?'Positions to the right are masked before softmax. A position may read itself because its output predicts the following token.':'The mask is disabled for this experiment. Earlier positions can now read future input. This is not a valid causal language-model training setup.'}</p>`;}
 if(stage===2){const q=h.q[selected],k=h.k[key],products=q.map((v,i)=>v*k[i]),a=h.weights[selected][key],masked=options.causal&&key>selected;content=`<div class="pair-caption"><strong>Query: ${word(ids[selected])} (${selected})</strong><strong>Key: ${word(ids[key])} (${key})</strong></div>${vector('q',q)}${vector('k',k)}${vector('q × k, coordinatewise',products)}<div class="score-calculation"><span>Sum of products <strong>${dot(q,k).toFixed(3)}</strong></span><span>Divide by √${dh} <strong>${h.raw[selected][key].toFixed(3)}</strong></span><span>After mask + softmax <strong>${masked?'0% · masked':(100*a).toFixed(2)+'%'}</strong></span></div>${vector('Value v at this key',h.v[key])}${vector('Weighted contribution a × v',h.v[key].map(v=>a*v))}<p class="micro-note">Select a cell in the matrix below to change the query–key pair. The raw score alone is not a probability: all allowed keys compete through softmax.</p>`;}
 if(stage===3){content=`<div class="two-heads">${trace.heads.map((hh,i)=>`<div><h4>Head ${i+1} · ${dh} features</h4>${attentionBars(trace,i,selected)}${vector('Output',hh.output[selected])}</div>`).join('')}</div>${vector('Concatenate both heads',trace.merged[selected])}${vector('After output projection',trace.attention[selected])}<p class="micro-note">Heads run in parallel. Their outputs are concatenated, then mixed by a learned matrix. Neither head is preassigned a “noun” or “position” function.</p>`;}
 if(stage===4){content=`<div class="block-path"><span>Attention</span> → <span>Add & normalize</span> → <span>Feed-forward</span> → <span>Add & normalize</span></div>${vector('Input x',trace.x[selected])}${vector('Attention output',trace.attention[selected])}${vector('Residual sum',trace.sum1[selected])}${vector('After first LayerNorm',trace.norm1[selected])}${vector('FFN output',trace.ffn[selected])}${vector('Final block output',trace.norm2[selected])}<details><summary>Inside the feed-forward network</summary>${vector('Expand: 8 → 16',trace.pre[selected])}${vector('ReLU: negative values → 0',trace.activated[selected])}<p class="micro-note">The second learned projection returns 16 features to 8. The same feed-forward weights act independently at every token position.</p></details>`;}
 if(stage===5){const last=ids.length-1;content=`<p class="micro-note">Generation reads the final position: ${word(ids[last])} (${last}). The inspection selection may be a different position.</p>${vector('Final representation',trace.norm2[last])}<div class="logit-list">${model.vocab.map((_,i)=>({id:i,score:trace.logits[last][i]})).sort((a,b)=>b.score-a.score).slice(0,6).map(x=>`<span>${word(x.id)}<strong>${x.score.toFixed(3)}</strong></span>`).join('')}</div><p class="micro-note">The learned readout gives one logit per vocabulary item. The probability panel applies softmax, with the generation temperature selected below.</p>`;}
 $('#stage-content').innerHTML=content;
 const temp=Number($('#temperature').value),prob=softmax(trace.logits.at(-1),temp),ranked=prob.map((p,id)=>({p,id})).sort((a,b)=>b.p-a.p);
 $('#prediction').textContent=word(ranked[0].id);$('#prediction-confidence').textContent=(100*ranked[0].p).toFixed(1)+'% at the current temperature';
 $('#probabilities').innerHTML=ranked.slice(0,6).map(x=>`<div class="prob-row"><span>${word(x.id)}</span><div><i style="width:${x.p*100}%"></i></div><strong>${(100*x.p).toFixed(1)}%</strong></div>`).join('');
 $('#temperature-value').textContent=temp.toFixed(1);$('#generate').disabled=ended||ids.length>=model.config.context;
 $('#shape-label').textContent=`${ids.length} tokens × 8 features · 2 heads × 4 features`;
 const tokens=ids.map((id,i)=>`<span>${i} · ${word(id)}</span>`).join('');
 $('#matrix').style.setProperty('--n',ids.length);
 $('#matrix').innerHTML='<span class="matrix-corner">query ↓ / key →</span>'+tokens+ids.map((id,i)=>`<span class="matrix-row-label">${i} · ${word(id)}</span>`+ids.map((_,j)=>{const value=h.weights[i][j],masked=options.causal&&j>i;return `<button data-row="${i}" data-col="${j}" class="matrix-cell ${masked?'masked':''} ${i===selected&&j===key?'selected':''}" style="background:${masked?'':'rgba(88,68,132,'+(.04+.86*value)+')'};color:${value>.55?'white':'var(--ink)'}" aria-label="Query ${i} ${word(id)}, key ${j} ${word(ids[j])}: ${masked?'masked':(100*value).toFixed(2)+' percent'}">${masked?'×':value.toFixed(2)}</button>`;}).join('')).join('');
 $('#positions-message').textContent=options.positions?'Position signals are enabled, as during training.':'Position signals are disabled. The causal mask still affects which context is visible; removing positional encodings alone does not make this causal model fully order-blind.';
 renderTraining();
}
function renderTraining(){
 const sentence=model.sentences.find(s=>s.startsWith(text.replace(/ <eos>.*/,'')))||model.sentences[0];
 const example=trainingExample(model,sentence,options);
 $('#training-sentence').textContent=sentence;
 $('#training-grid').innerHTML='<div class="training-row training-header"><span>Position</span><span>Input</span><span>Target next token</span><span>P(target)</span><span>Loss</span></div>'+example.targets.map((target,i)=>`<div class="training-row"><span>${i}</span><span>${word(example.trace.ids[i])}</span><span>${word(target)}</span><span>${(100*example.trace.probabilities[i][target]).toFixed(1)}%</span><span>${example.losses[i].toFixed(3)}</span></div>`).join('');
 $('#example-loss').textContent=example.loss.toFixed(3)+' nats/token';
 $('#training-settings').textContent=options.causal?'Each row predicts its next token using only its allowed prefix. All rows are evaluated in one masked forward pass.':'CAUTION IN THIS EXPERIMENT: the mask is off, so training positions can see future inputs, including the target token. This is information leakage.';
}
function apply(next){try{const nextIds=tokenize(next,model);text=next.trim().toLowerCase();ids=nextIds;selected=ids.length-1;key=Math.min(2,selected);ended=false;$('#prompt').value=text;$('#message').textContent='';render();}catch(error){$('#message').textContent=error.message;}}
$('#examples').innerHTML=model.sentences.map(s=>{const prefix=s.split(' ').slice(0,-1).join(' ');return `<option value="${prefix}">${prefix} …</option>`;}).filter((v,i,a)=>a.indexOf(v)===i).join('');
$('#examples').value=text;$('#examples').addEventListener('change',e=>apply(e.target.value));
$('#prompt-form').addEventListener('submit',e=>{e.preventDefault();apply($('#prompt').value);});
$('#tokens').addEventListener('click',e=>{const b=e.target.closest('[data-token]');if(b){selected=Number(b.dataset.token);render();}});
$('#stage-steps').addEventListener('click',e=>{const b=e.target.closest('[data-stage]');if(b){stage=Number(b.dataset.stage);render();}});
$('#stage-content').addEventListener('click',e=>{const b=e.target.closest('[data-key]');if(b){key=Number(b.dataset.key);stage=2;render();}});
$('#matrix').addEventListener('click',e=>{const b=e.target.closest('[data-row]');if(b){selected=Number(b.dataset.row);key=Number(b.dataset.col);stage=2;render();}});
$('#head-0').addEventListener('click',()=>{head=0;render();});$('#head-1').addEventListener('click',()=>{head=1;render();});
$('#causal').addEventListener('change',e=>{options.causal=e.target.checked;render();});$('#positions').addEventListener('change',e=>{options.positions=e.target.checked;render();});
$('#checkpoint').addEventListener('change',e=>{options.checkpoint=e.target.value;render();});
$('#temperature').addEventListener('input',render);
$('#generate').addEventListener('click',()=>{const trace=forward(model,ids,options),p=softmax(trace.logits.at(-1),Number($('#temperature').value));const id=$('#sampling').value==='sample'?sample(p):p.indexOf(Math.max(...p));if(id===1){ended=true;$('#message').textContent='END was selected. The generated sentence is complete.';render();return;}if(id===0){$('#message').textContent='The toy model selected START; try the trained checkpoint or another sample.';return;}apply((text+' '+model.vocab[id]).trim());});
$('#reset').addEventListener('click',()=>apply('the cat sat on the'));
$('#vocabulary').textContent=model.vocab.filter(w=>!w.startsWith('<')).join(' · ');
$('#model-size').textContent=`${model.vocab.length} tokens · 8 features · 2 heads · 1 block`;
const history=model.history,max=history[0].loss;
$('#loss-curve').innerHTML=`<line x1="30" y1="130" x2="470" y2="130" stroke="#cdd3cd"/><polyline fill="none" stroke="var(--accent)" stroke-width="2.5" points="${history.map(x=>`${30+x.step/history.at(-1).step*440},${130-x.loss/max*105}`).join(' ')}"/><text x="30" y="152" font-size="11" fill="#566269">0</text><text x="385" y="152" font-size="11" fill="#566269">${history.at(-1).step} training steps</text><text x="30" y="15" font-size="11" fill="#566269">Actual full-batch training loss</text>`;
$('#loss-summary').textContent=`${history[0].loss.toFixed(3)} → ${history.at(-1).loss.toFixed(3)} nats/token`;
$('#present').addEventListener('click',()=>{const on=document.body.classList.toggle('presentation');$('#present').textContent=on?'Exit presentation':'Presentation view';$('#present').setAttribute('aria-pressed',String(on));if(on)window.scrollTo({top:0});});
window.addEventListener('keydown',e=>{if(e.key==='Escape'&&document.body.classList.contains('presentation'))$('#present').click();});
apply(text);
