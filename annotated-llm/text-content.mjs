// Authored wording has stable keys. Values in braces remain supplied by the model.
export const escapeText=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const placeholders=text=>[...text.matchAll(/\{([A-Za-z][A-Za-z0-9_]*)\}/g)].map(m=>m[1]).sort();
export const COMMON_PLACEHOLDERS=['positions','embeddingDim','expandedDim','vocabularySize','heads','kvHeads','headDim','blocks'];
let commonValueProvider=()=>({});
export const setCommonCopyValues=provider=>{commonValueProvider=provider;};
export const commonCopyValues=()=>commonValueProvider();
export const isEmptyCopy=value=>String(value).trim()==='EMPTY';
export function validateText(original,value){
  if(typeof value!=='string'||!value.trim())throw new Error('Enter some text, or use Restore original.');
  if(value.length>10000)throw new Error('Keep this passage under 10,000 characters.');
  if(isEmptyCopy(value))return;
  const required=new Set(placeholders(original)),used=new Set(placeholders(value)),allowed=new Set([...required,...COMMON_PLACEHOLDERS]);
  if([...required].some(key=>!used.has(key))||[...used].some(key=>!allowed.has(key)))throw new Error('Keep the original values in braces. You can reuse them and add any of the available model values.');
}
export function renderTemplate(template,values={}){
  const resolved={...commonCopyValues(),...values};
  return template.replace(/\{([A-Za-z][A-Za-z0-9_]*)\}/g,(match,key)=>Object.hasOwn(resolved,key)?String(resolved[key]):match);
}
export class CopyText extends String {
  constructor(id,template,values){super(renderTemplate(template,values));this.id=id;this.values=values;}
}
export class TextCatalog {
  constructor(defaults={},edits={}){this.defaults=defaults;this.edits={...edits};this.saved={...edits};}
  template(id){if(!Object.hasOwn(this.defaults,id))throw new Error('Unknown text: '+id);return this.edits[id]??this.defaults[id];}
  text(id,values={}){return new CopyText(id,this.template(id),values);}
  set(id,value){validateText(this.defaults[id],value);if(value===this.defaults[id])delete this.edits[id];else this.edits[id]=value;}
  reset(id){delete this.edits[id];}
  get dirty(){return JSON.stringify(Object.entries(this.edits).sort())!==JSON.stringify(Object.entries(this.saved).sort());}
  savedAs(edits){this.saved={...edits};}
}
export let textCatalog=new TextCatalog();
export function loadTextCatalog(defaults,edits){textCatalog=new TextCatalog(defaults,edits);}
export const copy=(id,values={})=>textCatalog.text(id,values);
export const refreshCopy=value=>value instanceof CopyText?copy(value.id,value.values):value;
export function copyAttributes(value){
  return value instanceof CopyText?`data-copy="${escapeText(value.id)}" data-copy-values="${escapeText(JSON.stringify(value.values))}"`:'';
}
function emphasis(source,svg){
  const pattern=/(__|\*\*|\*)(.+?)\1/g;let out='',end=0;
  const inline=value=>svg?escapeText(value).replaceAll('Kᵀ','<tspan class="diagram-math">K<tspan baseline-shift="super" font-size=".64em">T</tspan></tspan>').replace(/[Δ∆]/g,symbol=>`<tspan class="diagram-delta">${symbol}</tspan>`):escapeText(value);
  for(const match of source.matchAll(pattern)){
    out+=inline(source.slice(end,match.index));
    const style=match[1]==='__'?'text-decoration:underline':match[1]==='**'?'font-weight:600':'font-style:italic';
    const tag=svg?'tspan':'span',math=svg&&match[1]==='*'&&['q','k'].includes(match[2]);
    out+=`<${tag}${math?' class="diagram-math"':''} style="${style}">${inline(match[2])}</${tag}>`;end=match.index+match[0].length;
  }
  return out+inline(source.slice(end));
}

function markdownLink(source,start){
  const closing=source.indexOf(']',start+1);
  if(closing<0||source[closing+1]!=='(')return null;
  let depth=1,end=closing+2;
  for(;end<source.length;end++){
    if(source[end]==='\\'){end++;continue;}
    if(source[end]==='(')depth++;
    if(source[end]===')'&&--depth===0)break;
  }
  if(depth)return null;
  const url=source.slice(closing+2,end).replace(/\\([()\\])/g,'$1').trim();
  let safe=false;
  try{safe=['https:','http:','mailto:'].includes(new URL(url,'http://localhost/').protocol);}catch{}
  return {label:source.slice(start+1,closing),url,safe,end:end+1};
}

export function richText(value,svg=false){
  // Parse links before emphasis so punctuation in URLs stays intact. Only
  // allow ordinary link protocols; authored HTML always remains escaped text.
  const source=String(refreshCopy(value));if(isEmptyCopy(source))return '';let out='',plain=0,index=0;
  while(index<source.length){
    if(source[index]!=='['){index++;continue;}
    const link=markdownLink(source,index);
    if(link){
      out+=emphasis(source.slice(plain,index),svg);
      const noteKey=link.url.startsWith('note:')?link.url.slice(5):null;
      if(!svg&&noteKey&&Object.hasOwn(textCatalog.defaults,noteKey)){
        out+=`<button type="button" class="inline-note-trigger" data-inline-note="${escapeText(noteKey)}" aria-controls="inline-note" aria-expanded="false">${emphasis(link.label,false)}</button>`;
      }else out+=link.safe?`<a class="copy-link" data-copy-link href="${escapeText(link.url)}" target="_blank" rel="noopener noreferrer">${emphasis(link.label,svg)}</a>`:escapeText(source.slice(index,link.end));
      plain=index=link.end;continue;
    }
    if(source.startsWith('[code]',index)){
      out+=emphasis(source.slice(plain,index),svg);
      out+=svg?'<a class="copy-code" data-copy-code href="#code-drawer" aria-label="View Python code"><tspan>&lt;/&gt;</tspan></a>':'<button type="button" class="copy-code" data-copy-code aria-label="View Python code" title="View code">&lt;/&gt;</button>';
      plain=index=index+6;continue;
    }
    index++;
  }
  return out+emphasis(source.slice(plain),svg);
}
export function svgTextBody(value,x,lineHeight=26,lineGaps){
  const lines=String(refreshCopy(value)).split('\n');
  if(lines.length===1)return richText(lines[0],true);
  return lines.map((line,i)=>`<tspan x="${x}"${i?` dy="${lineGaps?.[i-1]??lineHeight}"`:''}>${i?' ':''}${richText(line||' ',true)}</tspan>`).join('');
}
// Sidebar prose supports paragraphs and flat numbered lists. Keep the same
// escaped inline formatting used elsewhere, including links and code icons.
export function richTextBlocks(value){
  const source=String(refreshCopy(value));if(isEmptyCopy(source))return '';
  const lines=source.replace(/\r\n?/g,'\n').split('\n'),blocks=[];
  const marker=line=>/^ {0,3}(\d{1,9})([.)])(?:[ \t]+(.*)|$)/.exec(line);
  let i=0;
  while(i<lines.length){
    if(!lines[i].trim()){i++;continue;}
    const first=marker(lines[i]);
    if(first){
      const items=[],start=Number(first[1]);let next;
      while(i<lines.length&&(next=marker(lines[i]))&&next[2]===first[2]){
        const content=next[3]||'',indent=next[0].length-content.length,item=[content];i++;
        while(i<lines.length&&!marker(lines[i])){
          if(!lines[i].trim()){
            let following=i+1;while(following<lines.length&&!lines[following].trim())following++;
            i=following;
            if(i>=lines.length||!lines[i].startsWith(' '.repeat(indent)))break;
            item.push('');
          }
          item.push(lines[i].startsWith(' '.repeat(indent))?lines[i].slice(indent):lines[i]);i++;
        }
        items.push(`<li>${richText(item.join('\n'))}</li>`);
      }
      blocks.push(`<ol${start===1?'':` start="${start}"`}>${items.join('')}</ol>`);
    }else{
      const paragraph=[lines[i++]];
      while(i<lines.length&&lines[i].trim()&&Number(marker(lines[i])?.[1])!==1)paragraph.push(lines[i++]);
      blocks.push(`<p>${richText(paragraph.join('\n'))}</p>`);
    }
  }
  return blocks.join('');
}
function htmlCopyBody(element,value){
  if(Object.hasOwn(element.dataset,'copyBlocks'))return richTextBlocks(value);
  if(!Object.hasOwn(element.dataset,'copyParagraphs'))return richText(value);
  return String(refreshCopy(value)).split(/\r?\n/).map(paragraph=>`<p>${paragraph.trim()?richText(paragraph):'<br>'}</p>`).join('\n');
}
export function setCopyText(element,value){
  if(!element)return;
  if(value instanceof CopyText){element.dataset.copy=value.id;element.dataset.copyValues=JSON.stringify(value.values);}
  else{delete element.dataset.copy;delete element.dataset.copyValues;}
  element.classList.toggle('copy-empty',isEmptyCopy(value));
  element.innerHTML=htmlCopyBody(element,refreshCopy(value));
}
export function updateCopyElement(element,template){
  const values=JSON.parse(element.dataset.copyValues||'{}');
  const value=renderTemplate(template,values);
  element.classList.toggle('copy-empty',isEmptyCopy(value));
  if(element.namespaceURI==='http://www.w3.org/2000/svg'){
    const x=element.getAttribute('x')||0,size=Number(element.getAttribute('font-size'))||22;
    const gap=Number(element.dataset.copyLineHeight)||size*1.15,lines=value.split('\n').length;
    if(element.dataset.copyBottom)element.setAttribute('y',Number(element.dataset.copyBottom)-(lines-1)*gap);
    if(element.dataset.copyCenter)element.setAttribute('y',Number(element.dataset.copyCenter)-(lines-1)*gap/2);
    element.innerHTML=svgTextBody(value,x,gap,element.dataset.copyLineGaps?.split(',').map(Number));
  }else element.innerHTML=htmlCopyBody(element,value);
}

export function refreshCopyElements(root=document){
  for(const el of root.querySelectorAll('[data-copy]'))updateCopyElement(el,textCatalog.template(el.dataset.copy));
}
