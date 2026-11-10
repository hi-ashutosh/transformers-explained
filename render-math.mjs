/** Rebuild pre-rendered math after editing math.json. No npm install needed.
 * Run node render-math.mjs; use --check to validate sources without writing.
 * KaTeX HTML + semantic MathML is generated at authoring time. Browsers load
 * only the committed CSS/fonts; no equation rendering JavaScript or CDN.
 */
import {readFile,writeFile} from 'node:fs/promises';
import katex from './vendor/katex/katex.mjs';
const root=new URL('.',import.meta.url);
const source=JSON.parse(await readFile(new URL('math.json',root),'utf8'));
let html=await readFile(new URL('index.html',root),'utf8');
const original=html;
for(const [id,{tex,display}] of Object.entries(source)){
 if(!/^(inline|display)-\d+$/.test(id))throw new Error(`Invalid math ID: ${id}`);
 const start=`<!--math:${id}:start-->`,end=`<!--math:${id}:end-->`;
 const a=html.indexOf(start),b=html.indexOf(end);
 if(a<0||b<a)throw new Error(`Missing math slot ${id}`);
 const rendered=katex.renderToString(tex,{displayMode:display,output:'htmlAndMathml',throwOnError:true,strict:'error',trust:false});
 const pattern=new RegExp(start+'[\\s\\S]*?'+end,'g');
 html=html.replace(pattern,()=>start+rendered+end);
}
const slots=[...html.matchAll(/<!--math:([^:]+):start-->/g)].map(m=>m[1]);
if(Object.keys(source).some(id=>!slots.includes(id))||slots.some(id=>!source[id]))throw new Error('Math slots and math.json differ.');
if(process.argv.includes('--check')){
 if(html!==original)throw new Error('Rendered math is stale. Run node render-math.mjs.');
}else await writeFile(new URL('index.html',root),html);
console.log(`${Object.keys(source).length} LaTeX expressions validated${process.argv.includes('--check')?' and current':' and typeset'}.`);
