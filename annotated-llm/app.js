import {BLOCK,GQA,blockLayout,matrixProportions,matrixWindow,attentionLayout,foldableAttentionLayout,attentionAccordionFrame} from './geometry.mjs';
import {parameterAxes,axisDescription,parameterSummary,parameterBytes,formatBytes} from './matrix-info.mjs';
import {installZoomOutGestures,installSelectionClickGuard} from './navigation.mjs';
import {copy,CopyText,loadTextCatalog,copyAttributes,svgTextBody,setCopyText,richText,setCommonCopyValues,isEmptyCopy,refreshCopyElements} from './text-content.mjs';
import {createDataProvider,resolveModelId} from './data-provider.mjs';
import {createQwen38Diagrams} from './qwen38.mjs';
import {createMoEDiagrams} from './qwen3moe.mjs';
import {createKimiDiagrams,illustrativeKimiRouting} from './kimi.mjs';
import {createGLMDiagrams,illustrativeGLMRouting} from './glm53.mjs';
import {createGPT2Diagrams} from './gpt2.mjs';
import {createMamba2Diagrams,mambaHeadRows} from './mamba2.mjs';
import {createDeepSeekDiagrams,deepseekLayerKind,illustrativeDeepSeekRouting} from './deepseek41.mjs';
import {createGemma4Diagrams,gemmaHeadDim,gemmaKVSource} from './gemma4.mjs';
import {createQwen3Flow,prepareFlowTargets,syncFlowTargets,highlightFlow,clearFlowHighlight} from './qwen3-flow.mjs';
import {createMobileExplorer} from './mobile.mjs';
let textEditor=null, mobile=null;

const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (n) => n.toLocaleString('en-US');
const superscript = (n) => String(n).replace(/\d/g,digit=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[Number(digit)]);
const EXAMPLE_CONTEXT = 'In my beginning is my';
const siteConfig = JSON.parse(document.querySelector('#site-config')?.textContent || '{}');
const requestedModel = new URLSearchParams(location.search).get('model');
const modelId = resolveModelId(requestedModel,siteConfig);
const siteMode = siteConfig.mode || 'local';
if(siteMode==='static'&&requestedModel&&requestedModel!==modelId){
  const url=new URL(location.href);url.searchParams.delete('model');history.replaceState(null,'',url);
}
const provider = createDataProvider({modelId,mode:siteMode,baseURL:new URL('./',import.meta.url).href});
const isOriginalQwen = modelId === 'qwen-3-dense-no-kv';
const isHybrid = () => modelId === 'qwen-3.8-dense-no-cache';
const isMoE = () => modelId === 'qwen-3-moe-no-kv';
const isKimi = () => modelId === 'kimi-k3-no-cache';
const isGLM = () => modelId === 'glm-5.3-flash-no-cache';
const isGPT2 = () => modelId === 'gpt-2-small-no-kv';
const isGemma = () => modelId === 'gemma-4-e2b-no-cache';
const isDeepSeek = () => modelId === 'deepseek-v4.1-flash-no-cache';
const isMamba = () => modelId === 'mamba-2-130m-no-cache';
const hasCustomDiagrams = () => isKimi() || isGLM() || isGPT2() || isMamba() || isDeepSeek() || isGemma();
const custom = () => isGemma()?gemma:isDeepSeek()?deepseek:isMamba()?mamba:isGPT2()?gpt2:isGLM()?glm:kimi;
const blockGeometry = () => isGPT2()?{...BLOCK,step:54,closedWidth:34}:isKimi()?{...BLOCK,step:10,closedWidth:7}: isHybrid()?{...BLOCK,step:14,closedWidth:10}:(isMoE()||isGLM()||isDeepSeek()||isGemma())?{...BLOCK,step:20,closedWidth:14}:BLOCK;
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const instantMobileNavigation = ['mobile-animation','mobile-block-animation'].some(key=>new URLSearchParams(location.search).get(key)==='off');
// The older layout is available by a separate reference link in the README.
const twoStacksReference = new URLSearchParams(location.search).get('attention-layout')==='two-stacks';
const singleFeedforwardReference = new URLSearchParams(location.search).get('feedforward-layout')==='single';
const FF_TAIL_SHIFT = -55;
const exploreHintBlock=2;
const state = {model:null, level:'model', transitionLevel:null, block:exploreHintBlock, head:0, tokens:[], pieces:[], token:0, selected:null, camera:{x:0,y:0,k:1}, codeOpen:false, sourceKey:'model', source:null, sourceRequest:0, tokenRequest:0, weightRequest:0, patchRequest:0, weight:null, patch:null, overview:null, cell:[0,0]};
const nodes = new Map();
const viewPaths={model:['model'],embedding:['model','embedding'],block:['model','block'],output:['model','output'],attention:['model','block','attention'],feedforward:['model','block','feedforward'],head:['model','block','attention','head'],score:['model','block','attention','head','score'],expert:['model','block','feedforward','expert'],depth:['model','block','depth'],indexer:['model','block','attention','indexer']};
state.expert=0;
state.moePosition=0;
state.unfold=0;
state.blockHeight=BLOCK.height;
state.phase='idle';
state.attentionMode=twoStacksReference?'two-stacks':'foldable';
if(twoStacksReference)state.level='attention';
if(singleFeedforwardReference)state.level='feedforward';
state.activeHeadGroup=0;
state.attentionTransition=null;
state.scoreRow=0;state.scoreCol=0;state.scoreChosen=false;
let serial = 0, animation = 0, cameraMoving = false;
let navigation=0, animationResolve=null;
let transforms = {}, views = {}, layers = {};
const canvas = $('#canvas'), world = $('#world');
// Hide after opening a block, but let a page refresh show the invitation again.
let hasOpenedBlock=false;
const exploreHint=document.createElementNS('http://www.w3.org/2000/svg','g');
exploreHint.classList.add('explore-hint');exploreHint.dataset.exploreHint='';
exploreHint.setAttribute('role','note');exploreHint.style.display='none';
exploreHint.innerHTML='<path class="explore-hint-arrow"/><path class="explore-hint-arrowhead"/><text class="diagram-note" text-anchor="middle" font-size="23"></text>';
canvas.append(exploreHint);
let blockHintVisited=false,blockHintActive=false;
const blockHint=document.createElementNS('http://www.w3.org/2000/svg','g');
blockHint.classList.add('explore-hint');blockHint.dataset.blockHint='';
blockHint.setAttribute('role','note');blockHint.style.display='none';
blockHint.innerHTML='<text class="diagram-note" text-anchor="middle"></text>';
canvas.append(blockHint);
const C = () => state.model.config;
const N = () => state.tokens.length;
const d = () => C().token_embedding_dim;
const h = () => isGemma()?gemmaHeadDim(C(),state.block):C().head_dim;
const param = (suffix) => `${isMamba()?'blocks':'transformer_blocks'}.${state.block}.${suffix}`;
const ffParam = suffix => param(`feed_forward.${isMoE()?`experts.${state.expert}.`:''}${suffix}`);
const kvHead = () => Math.floor(state.head / (C().num_heads / C().num_kv_heads));
const headCount = () => hasCustomDiagrams()?C().num_heads:hybrid.isDelta()?C().linear_num_value_heads:C().num_heads;
const matrixShape = (name, head=false) => state.model.parameters[name].slice(head ? 1 : 0).map(fmt).join(' × ');

const {flow,boundary:flowBoundary}=createQwen3Flow({enabled:isOriginalQwen,context:()=>({config:C(),positions:N(),vocabularySize:state.vocabularySize}),escape:esc});

const hybridBase=createQwen38Diagrams({state,C,N,d,fmt,text,line,path,outline,grid,stream,operation,matrix,record,plus,dimensions,views,param});
const hybrid={...hybridBase,isDelta:()=>isHybrid()&&hybridBase.isDelta()};
const kimi=createKimiDiagrams({state,C,N,d,fmt,copy,text,line,path,outline,grid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,attentionMatrix,richText,copyAttributes});
const glm=createGLMDiagrams({state,C,N,d,fmt,copy,text,line,path,outline,grid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,attentionMatrix,richText,copyAttributes});
const gpt2=createGPT2Diagrams({state,C,N,d,fmt,copy,text,line,path,outline,grid,brokenGrid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,attentionMatrix,richText,copyAttributes});
const gemma=createGemma4Diagrams({state,C,N,d,fmt,copy,text,line,path,outline,grid,brokenGrid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,attentionMatrix,richText,copyAttributes});
const mamba=createMamba2Diagrams({state,C,N,d,fmt,copy,text,line,path,outline,grid,brokenGrid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,richText,copyAttributes});
const deepseek=createDeepSeekDiagrams({state,C,N,d,fmt,copy,text,line,path,outline,grid,brokenGrid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,richText,copyAttributes});
const moe=createMoEDiagrams({state,C,N,d,fmt,copy,text,line,path,outline,grid,stream,operation,matrix,record,plus,dimensions,views,param});


function text(x,y,value,size=22,kind='diagram-label',anchor='middle') {
  return `<text x="${x}" y="${y}" font-size="${size}" class="${kind}" text-anchor="${anchor}" ${copyAttributes(value)} data-copy-line-height="${size*1.15}">${svgTextBody(value,x,size*1.15)}</text>`;
}
function headHousing(x,y,width,head) {
  return outline(x,y-16,width,32,head===state.head?'#eef6f6':'#fff')
    +text(x+width/2,y+7,copy("label.head-head", {head:head}),23,'diagram-title');
}
const tokenAxes = (columns='residual coordinate') => ({rows:'token position',columns});
const positionConvention = () => N()?`Positions 1–${N()} correspond to Python sequence indices 0–${N()-1}.`:'N is the number of input tokens; Python sequence indices start at 0.';
function shapeLabel(x,y,value,description,size=17,anchor='middle') {
  const width=String(value).length*size*.61,left=anchor==='end'?x-width:anchor==='start'?x:x-width/2;
  return `<g class="dimension-label" data-tooltip="${esc(description)}" tabindex="0" role="img" aria-label="${esc(value+'. '+description)}"><rect x="${left-5}" y="${y-size-3}" width="${width+10}" height="${size+10}" fill="transparent"/>${text(x,y,value,size,'diagram-shape',anchor)}</g>`;
}
function dimensions(x,y,shape,axes,size=17,anchor='middle',note='',suffix='') {
  const [rows,columns]=shape.split(' × ');
  const positions=/position/.test(axes.rows+' '+axes.columns)?'\n'+positionConvention():'';
  return shapeLabel(x,y,shape+suffix,`Rows (${rows}): ${axes.rows}.\nColumns (${columns}): ${axes.columns}.${positions}${note?'\n'+note:''}`,size,anchor);
}
function projectionDimensions(x,y,input,output) {
  return shapeLabel(x,y,`${fmt(input)} → ${fmt(output)}`,`Vector width: ${fmt(input)} input coordinates → ${fmt(output)} output coordinates.\nThe same projection is applied independently to every token position.`,19);
}
function line(x1,y1,x2,y2,kind='diagram-arrow') {
  return `<path d="M${x1},${y1} L${x2},${y2}" class="${kind}"/>`;
}
function path(points,kind='diagram-arrow') { return `<path d="${points}" class="${kind}"/>`; }
function outlinePath(x,y,w,height){return `M${x},${y+1} L${x+w*.51},${y} L${x+w},${y+1.4} L${x+w-.6},${y+height} L${x+w*.41},${y+height-.7} L${x+.5},${y+height} Z`;}
function outline(x,y,w,height,fill='') {return `<path class="housing" d="${outlinePath(x,y,w,height)}" ${fill ? `style="fill:${fill}"` : ''}/>`;}
function grid(x,y,w,height,activation=false,rows=5,cols=7) {
  let out = `<rect x="${x}" y="${y}" width="${w}" height="${height}" class="diagram-grid${activation?' act':''}"/>`;
  let p='';
  for(let i=1;i<cols;i++)p+=`M${x+w*i/cols},${y}v${height}`;
  for(let i=1;i<rows;i++)p+=`M${x},${y+height*i/rows}h${w}`;
  return out+`<path d="${p}" fill="none" stroke="${activation?'#c3d9db':'#c8d0c9'}" stroke-width=".7"/>`;
}
function vector(x,y,w,height=22,activation=true) { return grid(x,y,w,height,activation,1,8); }
function brokenGrid(x,y,w,height){
  const top=height*.43, bottom=height*.43, gap=height-top-bottom;
  return grid(x,y,w,top,false,6,6)+grid(x,y+top+gap,w,bottom,false,6,6)
    +`<path d="M${x-4},${y+top+4} l${w*.3},-5 l${w*.4},10 l${w*.3+8},-5 M${x-4},${y+top+gap-4} l${w*.3},-5 l${w*.4},10 l${w*.3+8},-5" class="axis-break"/>`;
}
function stream(x1,x2,y){return `<g class="residual-ribbon"><path d="M${x1},${y} H${x2}" class="stream-band"/><path d="M${x1},${y} H${x2}" class="stream-line"/></g>`;}
function record(spec,inner) {
  const id='n'+serial++;
  nodes.set(id,spec);
  const code=spec.codePosition||(spec.box?{x:spec.box.x+spec.box.w,y:spec.box.y-10,size:16}:null);
  let actions='';
  if(code) {
    const {x,y,size}=code;
    actions+=`<g class="node-action node-code" data-code="${id}" data-tooltip="View code" transform="translate(${x},${y})" role="button" tabindex="0" aria-label="View Python for ${esc(spec.title)}"><rect x="${-size*2}" y="${-size*1.1}" width="${size*2.2}" height="${size*1.5}" fill="transparent"/><text text-anchor="end" font-size="${size}">&lt;/&gt;</text></g>`;
    if(spec.weight&&state.model.weight_inspection!==false)actions+=`<g class="node-action node-weight" data-weight="${id}" data-tooltip="Inspect weights" transform="translate(${x-size*3.5},${y-size*.85}) scale(${size/16})" role="button" tabindex="0" aria-label="Inspect weights for ${esc(spec.title)}"><rect x="-3" y="-3" width="23" height="23" fill="transparent"/><circle cx="6.5" cy="6.5" r="5"/><path d="M10.5,10.5 L16,16"/></g>`;
  }
  return `<g class="diagram-node${state.selected?.key===spec.key?' selected':''}" data-node="${id}" ${spec.enter?`data-enter="${spec.enter}"`:''} ${spec.tooltip?`data-tooltip="${esc(spec.tooltip)}"`:''} ${spec.weight?'data-has-weight="true"':''} role="${spec.cellGrid?'group':'button'}" ${spec.cellGrid?'':'tabindex="0"'} ${spec.expanded!==undefined?`aria-expanded="${spec.expanded}"`:''} aria-label="${esc(spec.title)}${spec.enter?', look inside':''}">${inner}${actions}</g>`;
}
function operation(x,y,w,height,title,spec={}) {
  const value=spec.labelText||title,labels=spec.labelLines||String(value).split('\n'),size=spec.fontSize||22,gap=size*1.1;
  const labelDrawing=spec.labelLines?labels.map((label,i)=>text(x+w/2,y+height/2+size*.3+(i-(labels.length-1)/2)*gap,label,size,spec.labelKinds?.[i]||'diagram-title')).join('')
    :`<text x="${x+w/2}" y="${y+height/2+size*.3-(labels.length-1)/2*gap}" font-size="${size}" class="diagram-title" text-anchor="middle" ${copyAttributes(value)} data-copy-line-height="${gap}" data-copy-center="${y+height/2+size*.3}">${svgTextBody(value,x+w/2,gap)}</text>`;
  return record({title,source:'model',...spec,box:{x,y,w,h:height}},outline(x,y,w,height)+labelDrawing);
}
function matrix(x,y,w,height,label,shape,spec={}) {
  const activation=spec.activation||false;
  const axes=spec.axes||(!activation&&spec.weight?parameterAxes(spec.weight,false,C()):tokenAxes());
  const labelValue=spec.labelText||label,labelLines=String(labelValue).split('\n'),labelHeight=(labelLines.length-1)*24;
  const labelDrawing=spec.transposeLabel?`<text x="${x+w/2}" y="${y-14}" font-size="23" text-anchor="middle" class="diagram-math">${esc(label)}<tspan dy="-9" font-size="15">T</tspan></text>`
    :`<text x="${x+w/2}" y="${y-14-labelHeight}" font-size="23" text-anchor="middle" class="${spec.labelKind||(spec.weight&&!activation?'diagram-math':'diagram-title')}" ${copyAttributes(labelValue)} data-copy-line-height="24" data-copy-bottom="${y-14}">${svgTextBody(labelValue,x+w/2,24)}</text>`;
  return record({title:label,source:'attention',inputAxes:activation&&spec.input===shape?axes:undefined,outputAxes:activation&&spec.output===shape?axes:undefined,...spec,box:{x,y:y-25-labelHeight,w,h:height+65+labelHeight}},grid(x,y,w,height,activation,spec.rows||5,spec.cols||7)+labelDrawing+dimensions(x+w/2,y+height+28,shape,axes,17,'middle','Grid cells and proportions are schematic.'));
}
// Show individual token pairs, rather than the decorative grid used for large tensors.
// Only the causal structure is known here. Dot sizes illustrate variation;
// they do not encode measured scores, signs, or attention probabilities.
function attentionMatrix(x,y,size,stage,spec) {
  const count=Math.min(N(),8),cell=size/Math.max(count,1);
  let inner=text(x+size/2,y-77,spec.title,24,'diagram-title')
    +dimensions(x+size/2,y-48,`${N()} × ${N()}`,{rows:'query position',columns:'key position'},17,'middle',N()>count?'Only the first 8 positions are drawn on each axis.':'Each cell corresponds to one query–key pair.',N()>count?' · first 8 shown':'');
  for(let row=0;row<count;row++) {
    inner+=text(x-13,y+(row+.5)*cell+5,row+1,14,'diagram-shape','end');
    inner+=text(x+(row+.5)*cell,y-15,row+1,14,'diagram-shape');
    for(let col=0;col<count;col++) {
      const future=col>row,blocked=future&&stage!=='scores';
      const value=blocked?(stage==='masked'?'−∞':'0'):stage==='weights'&&row===0?'1':null;
      const variation=(row*17+col*11+row*col*3+(stage==='weights'?4:0))%9;
      const cx=x+(col+.5)*cell,cy=y+(row+.5)*cell;
      const inspectable=stage==='scores'&&spec.inspectable!==false,selected=inspectable&&state.scoreChosen&&row===state.scoreRow&&col===state.scoreCol;
      const related=inspectable&&state.level==='score'&&(row===state.scoreRow||col===state.scoreCol);
      const tabstop=selected||(!state.scoreChosen&&row===0&&col===0)?0:-1;
      inner+=`<g class="attention-cell${blocked?' masked':''}${selected?' score-selected':''}${related?' score-related':''}" data-row="${row}" data-col="${col}" ${inspectable?`data-score-row="${row}" data-score-col="${col}" role="button" data-tabstop="${tabstop}" tabindex="${tabstop}" aria-label="Inspect score: query position ${row+1}, key position ${col+1}" aria-pressed="${selected}"`:''}><rect x="${x+col*cell}" y="${y+row*cell}" width="${cell}" height="${cell}"/>`
        +(value===null?`<circle cx="${cx}" cy="${cy}" r="${Math.min(6,cell*(.07+variation*.009))}" class="activation-dot"/>`:text(cx,cy+cell*.17,value,Math.min(22,cell*.53),'diagram-math'))
        +`<title>Query position ${row+1}, key position ${col+1} (Python indices ${row}, ${col}): ${blocked?(stage==='masked'?'future score replaced by negative infinity':'future position has zero attention weight'):stage==='weights'?(row===0?'the only allowed position has weight one':'input-dependent attention weight'):'input-dependent score'}</title></g>`;
    }
  }
  if(!count)inner+=grid(x,y,size,size,true,1,1)+text(x+size/2,y+size/2+6,copy("label.empty"),20,'diagram-note');
  return record({...spec,cellGrid:stage==='scores'&&spec.inspectable!==false,codePosition:{x:x+size+21,y:y-99,size:15}},`<g class="attention-matrix" data-stage="${stage}">${inner}</g>`);
}
function plus(x,y,spec={}) {
  return record({title:copy("title.residual-addition"),source:'residual',kind:copy("kind.operation"),description:copy("description.add-the-result-back-to-the-residual-stream-this-preserves-the-ori"),input:`${N()} × ${fmt(d())}`,output:`${N()} × ${fmt(d())}`,...spec},`<circle cx="${x}" cy="${y}" r="20" fill="white" stroke="#809d95"/>`+text(x,y+8,'+',28));
}
function transformPoint(t,box) { return {x:t.x+box.x*t.s,y:t.y+box.y*t.s,w:box.w*t.s,h:box.h*t.s}; }
function nested(parent,x,y,s) { return {x:parent.x+x*parent.s,y:parent.y+y*parent.s,s:parent.s*s}; }
function layer(name,content) {
  const t=transforms[name];
  return `<g id="layer-${name}" class="${name}-layer" transform="translate(${t.x},${t.y}) scale(${t.s})">${content}</g>`;
}

function tokenCard(x,y,word,id) {
  const shown=word.trim(),label=shown.length>12?shown.slice(0,10)+'…':shown;
  return `<g class="token-card"${id==null?'':` data-token-id="${id}"`}><title>${esc(shown)}${id==null?'':` · token ID ${id}`}</title><rect x="${x}" y="${y}" width="126" height="29" rx="5"/><text x="${x+63}" y="${y+20}" text-anchor="middle" class="diagram-title" font-size="20"><tspan>${esc(label)}</tspan>${id==null?'':`<tspan class="diagram-token-id"> (${id})</tspan>`}</text></g>`;
}

function blockTypeLabel(b) { return isGemma()?`${C().layer_types[b]==='full_attention'?'Global attention':'Sliding-window attention'}${b>=C().first_shared_block?' · K/V from block '+gemmaKVSource(C(),b):''}`:isDeepSeek()?deepseekLayerKind(C(),b):isGLM()?(C().layer_types[b]==='linear_attention'?'Kimi Delta Attention':'sparse latent attention'):isKimi()?(C().layer_types[b]==='linear_attention'?'Kimi Delta Attention':'latent full attention'):isHybrid()?(C().layer_types[b]==='linear_attention'?'Gated DeltaNet':'gated full attention'):null; }

function overviewDiagram() {
  const c=C(), blockStart=BLOCK.start, blockStep=blockGeometry().step, blockEnd=blockStart+c.num_transformers*blockStep;
  let out='<g id="overview-left">'+text(65,128,copy("label.context"),25,'diagram-title');
  const samples=state.pieces.slice(0,5).map(p=>p.replaceAll('Ġ',' ').replaceAll('Ċ','↵').replaceAll('▁',isGemma()?' ':'▁'));
  out+=record({key:'tokens',title:copy("title.text-becomes-tokens"),kind:copy("kind.the-input"),source:'tokenizer',description:copy("description.the-tokenizer-breaks-text-into-pieces-and-assigns-each-piece-an-i"),output:`${N()} token IDs`,note:copy("note.this-example-uses-plain-text-tokenization-the-chat-cli-also-adds")},
    samples.map((s,i)=>tokenCard(4,139+i*31,s,state.tokens[i])).join('')+text(66,315,copy("label.positions-tokens-value2", {positions:N(), value2:state.pieces.length>5?' · first 5 shown':''}),18,'diagram-shape'));
  out+=flow(line(145,235,171,235),'tokens','tokens','embedding-table');
  if(isGemma())out+=gemma.embeddingOverview();else if(isGPT2())out+=gpt2.embeddingOverview();else{
  out+=record({key:'embedding-table',title:copy("title.embedding-table"),codePosition:{x:247,y:52,size:16},kind:copy("kind.learned-lookup-table"),source:'embedding',description:copy("description.this-table-has-vocabularysize-token-rows-each-with-embeddingdim-l", {vocabularySize:fmt(c.vocabulary_size), embeddingDim:fmt(d())}),input:`${N()} token IDs`,output:`${N()} × ${fmt(d())}`,weight:'token_embedding_layer.weight',note:copy("note.the-break-shortens-the-token-axis-at-equal-scale-this-table-would", {value1:(c.vocabulary_size/d()).toFixed(1)})},brokenGrid(186,103,61,210)+text(216,77,copy("label.embedding-table"),22,'diagram-title')+dimensions(216,347,matrixShape('token_embedding_layer.weight'),parameterAxes('token_embedding_layer.weight'),17,'middle','The break shortens the token axis; grid cells are schematic.')+text(216,376,copy("label.token-axis-shortened"),18,'diagram-note'));
  out+=flow(line(259,235,295,235),'embeddings','embedding-table','embedding');
  const rows=Math.max(1,Math.min(N(),6));
  out+=record({key:'embedding',title:copy("title.token-embedding-vectors"),kind:copy("kind.the-initial-residual-stream"),source:'embedding',description:copy("description.the-selected-rows-form-the-initial-residual-stream-there-is-one-v"),input:`${N()} token IDs`,output:`${N()} × ${fmt(d())}`,note:copy("note.this-small-activation-matrix-is-distinct-from-the-full-learned-em")},grid(311,215,95,40,true,rows,10)+text(358,144,copy("label.token-embedding-vectors-lines"),22,'diagram-title')+dimensions(358,307,`${N()} × ${fmt(d())}`,tokenAxes('embedding coordinate'),18,'middle','Grid cells and proportions are schematic.'));
  }
  out+='</g>';
  out+='<g class="overview-annotation">'+text((blockStart+blockEnd)/2,115,copy("label.blocks-transformer-blocks", {blocks:c.num_transformers}),26,'diagram-title')+'</g>';
  for(let b=0;b<c.num_transformers;b++) {
    const blockType=blockTypeLabel(b);
    out+=`<g id="overview-block-${b}" transform="translate(${blockStart+b*blockStep},0)">`+record({key:`block-${b}`,title:copy("title.transformer-block-b", {b:b}),kind:copy("kind.transformer-block"),source:'block',enter:'block',block:b,tooltip:blockType?`Block ${b}: ${blockType}`:undefined,description:isHybrid()?`Block ${b}: ${blockType}, followed by feedforward. Both branches add their contributions to the residual stream.`:copy("description.attention-and-feedforward-read-from-the-residual-stream-then-add"),input:`${N()} × ${fmt(d())}`,output:`${N()} × ${fmt(d())}`},outline(0,BLOCK.top,blockGeometry().closedWidth,BLOCK.closedHeight,isDeepSeek()?(b<C().encoder_layers?'#f7efdf':'#d9eceb'):isGemma()?(C().layer_types[b]==='full_attention'?'#d9eceb':'#f7efdf'):(isHybrid()||isKimi()||isGLM())?(C().layer_types[b]==='linear_attention'?'#f7efdf':'#d9eceb'):b===state.block?'#edf5f4':'#fff')+text(blockGeometry().closedWidth/2,304,(isHybrid()||isMoE()||isKimi()||isGLM()||isDeepSeek()||isGemma())&&b%4!==0&&b!==C().num_transformers-1?'':b,12,'diagram-shape'))+'</g>';
  }
  out+='<g id="overview-flow">'+flow('<path id="overview-flow-in" class="diagram-arrow"/>','embeddings','embedding','block-0')+flow('<path id="overview-flow-out" class="diagram-arrow"/>','afterFeedforward',`block-${c.num_transformers-1}`,'output-stage')+'</g><g id="overview-stream" class="residual-ribbon"><path class="stream-band"/><path class="stream-line"/></g>';
  if(isDeepSeek())out+=`<g class="overview-annotation">${text((blockStart+blockEnd)/2,80,'20 encoder + 20 decoder blocks · Engram at 1 and 14',21,'diagram-note')}</g>`;
  if(isGLM())out+=`<g class="overview-annotation">${text((blockStart+blockEnd)/2,80,'34 KDA + 11 sparse attention · four residual streams',21,'diagram-note')}</g>`;
  if(isKimi())out+=`<g class="overview-annotation">${text((blockStart+blockEnd)/2,80,'69 KDA + 24 latent attention · AttnRes groups of twelve blocks',21,'diagram-note')}</g>`;
  if(isHybrid())out+=`<g class="overview-annotation">${text((blockStart+blockEnd)/2,80,'3 DeltaNet + 1 gated attention, repeated 16 times',21,'diagram-note')}</g>`;
  out+='<g id="overview-right">';
  let x=blockEnd+38;
  out+=operation(x,190,134,90,copy("label.output-projection"),{...finalOutputSpec(),enter:'output',labelText:copy("label.output-projection-lines")});
  out+=projectionDimensions(x+67,307,d(),c.vocabulary_size);
  out+=flow(line(x+146,235,x+193,235),'logits','output-stage','output');x+=205;
  out+=matrix(x,218,92,34,copy("label.logits"),`${N()} × ${fmt(c.vocabulary_size)}`,{key:'output',axes:tokenAxes('vocabulary entry'),title:copy("title.scores-for-every-token"),kind:copy("kind.activations"),source:'output',activation:true,description:copy("description.each-row-contains-the-vocabulary-scores-for-one-sequence-position"),output:`${N()} × ${fmt(c.vocabulary_size)}`});
  out+=flow(line(x+104,235,x+149,235),'selectedLogits','output','sampling');x+=160;
  out+=operation(x,204,120,63,copy("label.sample"), {key:'sampling',title:copy("title.choose-the-next-token"),kind:copy("kind.sampling"),source:'sampling',description:copy("description.use-the-final-position-s-logits-apply-temperature-and-the-top-k-a"),input:'Last position → vocabulary scores',output:'One token ID',note:copy("note.this-view-illustrates-the-computation-no-prompt-has-been-run-thro")});
  out+=record({key:'last-position',title:copy("title.keep-the-last-position"),kind:copy("kind.select-the-next-token-scores"),source:'last',description:copy("description.only-the-last-position-is-used-to-predict-the-next-token-the-gene"),input:`${N()} × ${fmt(c.vocabulary_size)}`,output:`${fmt(state.vocabularySize||c.vocabulary_size)} usable scores`},text(x+60,307,copy("label.last-position-only"),19,'diagram-note'))+flow(line(x+133,235,x+169,235),'next','sampling','next');x+=180;
  out+=record({key:'next',title:copy("title.append-and-repeat"),kind:copy("kind.the-next-token"),source:'repeat',description:copy("description.append-the-chosen-token-to-the-context-then-run-the-model-again-t"),input:`${N()} tokens`,output:`${N()+1} tokens`,note:state.exampleContext?copy("note.illustrative-continuation"):undefined},tokenCard(x,220.5,state.exampleContext?'end':'next',state.exampleNextId)+text(x+63,307,copy("label.next-token"),20,'diagram-note'));
  out+='</g>';
  out+='<g class="overview-annotation">'+flow(`<path id="overview-repeat" data-end="${x+63}" class="diagram-residual return-direction"/>`,'next','next','tokens')+'<g id="overview-repeat-directions" aria-hidden="true"><path class="diagram-residual return-direction"/><path class="diagram-residual return-direction"/><path class="diagram-residual return-direction"/></g>'+text((x+66)/2,436,copy("label.append-to-the-context-then-take-another-trip-through-the-model"),23,'diagram-note')+'</g>';
  views.modelBase={x:-20,y:30,w:x+150,h:435};
  return out;
}

function finalOutputSpec() {
  if(hasCustomDiagrams())return custom().defaultSpec('output');
  return {key:'output-stage',title:copy("title.output-projection"),kind:copy("kind.from-residual-vectors-to-logits"),source:'output_stage',parameterScope:'output',
    description:copy("description.after-the-final-transformer-block-rmsnorm-normalizes-each-positio", {embeddingDim:fmt(d()), vocabularySize:fmt(C().vocabulary_size)}),
    input:`${N()} × ${fmt(d())}`,output:`${N()} × ${fmt(C().vocabulary_size)}`,inputAxes:tokenAxes(),outputAxes:tokenAxes('vocabulary entry')};
}
function finalOutputDiagram() {
  if(hasCustomDiagrams())return custom().output();
  const shape=`${N()} × ${fmt(d())}`,logits=`${N()} × ${fmt(C().vocabulary_size)}`,rows=Math.max(1,Math.min(N(),8));
  let out=outline(0,0,1180,550)+text(28,38,copy("label.output-projection"),28,'diagram-title','start');
  out+=flow(stream(-160,50,285),'afterFeedforward','','output-input')+flowBoundary(-65,245,'residual','Input','middle',true);
  out+=matrix(50,260,110,50,copy("label.residual-vectors"),shape,{key:'output-input',title:copy("title.the-final-residual-vectors"),kind:copy("kind.activations"),source:'output_stage',activation:true,rows,axes:tokenAxes(),output:shape,outputAxes:tokenAxes(),
    description:copy("description.these-are-the-position-vectors-after-the-final-transformer-block")});
  out+=text(105,377,copy("label.from-block-lastblock", {lastBlock:C().num_transformers-1}),20,'diagram-note');
  out+=flow(line(172,285,228,285),'residual','output-input','final-norm');
  out+=operation(240,240,100,90,copy("label.rmsnorm"),{key:'final-norm',title:copy("title.final-normalization"),kind:copy("kind.normalization"),source:'norm',labelText:copy("label.rms-norm-lines"),
    description:copy("description.normalize-each-final-residual-vector-by-its-root-mean-square-then"),input:shape,output:shape,weight:'output_norm.weight'});
  out+=flow(line(352,285,428,285),'normalized','final-norm','output-normalized');
  out+=matrix(440,260,110,50,copy("label.normalized-vectors"),shape,{key:'output-normalized',title:copy("title.normalized-residual-vectors"),kind:copy("kind.activations"),source:'output_stage',activation:true,rows,axes:tokenAxes(),output:shape,outputAxes:tokenAxes(),
    description:copy("description.final-rmsnorm-preserves-the-number-of-positions-and-coordinates-t")});
  out+=flow(line(562,285,709,285),'normalized','output-normalized','output-projection');
  out+=record({key:'output-projection',title:copy("title.output-matrix"),kind:copy("kind.learned-projection"),source:'output',weight:'output_layer.weight',codePosition:{x:807,y:113,size:16},
    description:copy("description.each-of-this-matrix-s-vocabularysize-rows-contains-embeddingdim-l", {vocabularySize:fmt(C().vocabulary_size), embeddingDim:fmt(d())}),
    input:shape,output:logits,note:copy("note.w-is-output-layer-weight-pytorch-stores-it-as-vocabulary-entries")},
    text(757,146,'W',26,'diagram-math')+brokenGrid(725,176,64,220)
    +dimensions(757,430,matrixShape('output_layer.weight'),parameterAxes('output_layer.weight'),18,'middle','The break shortens the vocabulary axis; grid cells are schematic.')
    +text(757,460,copy("label.vocabulary-axis-shortened"),19,'diagram-note'));
  out+=flow(line(803,285,928,285),'logits','output-projection','output');
  out+=matrix(940,260,170,50,copy("label.logits"),logits,{key:'output',title:copy("title.scores-for-every-token"),kind:copy("kind.activations"),source:'output',activation:true,rows,axes:tokenAxes('vocabulary entry'),output:logits,
    description:copy("description.each-row-contains-vocabulary-scores-for-one-sequence-position-sam")});
  out+=flow(stream(1110,1340,285),'logits','output','')+flowBoundary(1240,245,'logits','Output','middle',true);
  out+=text(1025,377,copy("label.on-to-sampling"),20,'diagram-note');
  out+=`<text x="590" y="510" font-size="25" text-anchor="middle" class="diagram-math">logits = (normalized vectors) W<tspan dy="-10" font-size="16">T</tspan></text>`;
  return out;
}

function attentionPreview() {
  const heads=headCount(),count=Math.min(4,heads),abbreviated=heads>count,firstY=142,step=15,lastY=firstY+(count-1)*step+(abbreviated?16:0),middle=(firstY+lastY)/2;
  let out='<g class="attention-preview" aria-hidden="true">';
  out+=path(`M410,${firstY} V${lastY} M540,${firstY} V${lastY}`,'diagram-wire');
  out+=line(395,middle,410,middle)+line(540,middle,555,middle,'diagram-contribution');
  for(let head=0;head<count;head++) {
    const y=head===count-1?lastY:firstY+head*step;
    out+=`<g data-preview-head="${head}">`+line(410,y,430,y)+outline(438,y-5,74,10,hybrid.isDelta()?'#f8f3e9':'#e8f2f1')+line(520,y,540,y,'diagram-contribution')+'</g>';
  }
  if(abbreviated)out+=text(475,191,'…',23,'diagram-label');
  return out+'</g>';
}

function feedforwardPreview() {
  if(isMoE())return moe.preview();
  // One shared gated pipeline, in the same arrangement as the detail view.
  const box=(x,y,w,label)=>outline(x,y,w,18)+text(x+w/2,y+13,label,12,'diagram-title');
  return '<g class="feedforward-preview" aria-hidden="true">'
    +box(887,156,30,copy("label.rms"))
    +line(917,165,945,165,'diagram-wire')
    +path('M945,165 H955 M955,147 V187','diagram-wire')
    +path('M955,147 H965 M955,187 H965','diagram-wire')
    +box(965,138,30,copy("label.gate"))+box(965,178,30,copy("label.value"))
    +line(995,147,1002,147,'diagram-wire')+box(1002,138,30,copy("label.silu"))
    +path('M1032,147 H1041 Q1047,147 1047,153 V158','diagram-wire')
    +path('M995,187 H1041 Q1047,187 1047,181 V172','diagram-wire')
    +outline(1040,158,14,14)+text(1047,169,'⊙',13,'diagram-math')
    +line(1054,165,1067,165,'diagram-wire')+box(1067,156,33,copy("label.down"))
    +line(1100,165,1118,165,'diagram-wire')+'</g>';
}

function blockDiagram() {
  if(hasCustomDiagrams())return custom().block();
  const td=`${N()} × ${fmt(d())}`;
  let out=`<path id="block-frame" class="housing" d="${outlinePath(-15,BLOCK.frameTop,BLOCK.width,state.blockHeight)}"/>`+text(28,35,copy("label.transformer-block-block", {block:state.block}),27,'diagram-title','start')+dimensions(1180,35,td,tokenAxes(),19,'end');
  out+=stream(-38,1235,340);
  if(isOriginalQwen)out+=flow(path('M-38,340 H627','flow-guide'),'residual','','attention-add')+flow(path('M667,340 H1156','flow-guide'),'afterAttention','attention-add','ff-add')+flow(path('M1196,340 H1235','flow-guide'),'afterFeedforward','ff-add','');
  out+=flowBoundary(16,366,'residual','Input')+flowBoundary(1170,389,'afterFeedforward','Output','end');
  out+=text(36,310,'r',28,'diagram-title')+text(689,387,copy("label.r"),27,'diagram-title')+text(1202,387,copy("label.r-faf512"),27,'diagram-title');
  out+=text(95,397,state.block===0?copy("label.from-embeddings"):copy("label.from-previous-block",{block:state.block-1}),20,'diagram-note');
  out+=text(1115,422,state.block===C().num_transformers-1?copy("label.to-output"):copy("label.to-next-block",{block:state.block+1}),20,'diagram-note');
  out+=flow(path('M135,340 V165 H178'),'residual','','attention')+`<circle cx="135" cy="340" r="4" class="stream-tap"/>`;
  out+=record({key:'attention',title:hybrid.isDelta()?'Gated DeltaNet':copy("title.grouped-query-attention"),kind:copy("kind.an-attention-contribution"),source:'residual',enter:'attention',description:hybrid.isDelta()?'This branch normalizes the residual, applies Gated DeltaNet, and adds its result back. A state matrix in each value head summarizes earlier sequence positions.':copy("description.this-branch-normalizes-the-incoming-residual-vectors-with-rmsnorm"),input:td,output:td,note:hybrid.isDelta()?'48 value heads, 16 shared query/key heads; 128 coordinates per key and value.':copy("note.heads-query-heads-kvheads-key-value-heads-headdim-dimensions-per", {heads:C().num_heads, kvHeads:C().num_kv_heads, headDim:h()})},
    outline(190,92,405,149)+text(392.5,124,hybrid.isDelta()?'Gated DeltaNet':isHybrid()?'Gated attention':copy("label.attention"),28,'diagram-title')
    +line(190,165,312.5,165,'diagram-wire')+`<g transform="translate(-82.5,-7.5)">${attentionPreview()}</g>`+line(472.5,165,595,165,'diagram-wire')
    +text(392.5,227,hybrid.isDelta()?'48 value heads · 16 shared Q/K heads':copy("label.heads-heads", {heads:C().num_heads}),18,'diagram-shape'));
  out+=flow(path('M605,165 H647 V313','diagram-contribution'),'attention','attention','attention-add')+text(628,277,hybrid.isDelta()?'ΔDeltaNet':copy("label.attention-db7108"),20,'diagram-note','end');
  out+=plus(647,340,{key:'attention-add',title:copy("title.add-the-attention-contribution"),source:'residual',description:copy("description.add-the-attention-result-to-the-incoming-residual-stream-r-r-atte")});
  out+=flow(path('M731,340 V165 H774'),'afterAttention','attention-add','feedforward')+`<circle cx="731" cy="340" r="4" class="stream-tap"/>`;
  out+=record({key:'feedforward',title:copy("title.feedforward-network"),kind:copy("kind.a-feedforward-contribution"),source:'residual_ff',enter:'feedforward',description:copy("description.this-branch-reads-the-stream-after-attention-s-addition-rmsnorm-n"),input:td,output:td,note:copy("note.embeddingdim-expandeddim-embeddingdim", {embeddingDim:fmt(d()), expandedDim:fmt(C().expanded_dim)})},
    outline(785,92,333,149)+text(951.5,124,copy("label.feedforward"),25,'diagram-title')
    +line(785,165,807,165,'diagram-wire')+`<g transform="translate(-80,0)">${feedforwardPreview()}</g>`+line(1038,165,1118,165,'diagram-wire'));
  out+=flow(path('M1128,165 H1176 V313','diagram-contribution'),'feedforward','feedforward','ff-add')+text(1156,277,copy("label.feedforward-dce033"),20,'diagram-note','end');
  out+=plus(1176,340,{key:'ff-add',title:copy("title.add-the-feedforward-contribution"),source:'residual_ff',description:copy("description.add-the-feedforward-result-to-the-updated-stream-r-r-ff-rmsnorm-r")});
  out+=text(327,463,hybrid.isDelta()?'r₁ = r + DeltaNet(RMSNorm(r))':copy("label.r-r-attention-rmsnorm-r"),19,'diagram-shape')+text(929,463,copy("label.r-r-ff-rmsnorm-r"),19,'diagram-shape');
  const prose=copy("label.each-branch-reads-from-the-residual-stream-and-adds-its-result-ba");
  out+=`<foreignObject id="block-prose" x="30" y="486" width="1165" height="160"><div xmlns="http://www.w3.org/1999/xhtml" id="block-prose-text" class="block-prose" ${copyAttributes(prose)}>${richText(prose)}</div></foreignObject>`;
  return out;
}

function outputProjectionSpec(head) {
  const first=head*h(),last=first+h()-1;
  return {key:`output-head-${head}`,title:copy("title.output-projection-head-head", {head:head}),kind:copy("kind.this-head-s-learned-output-projection"),source:'combine',
    description:copy("description.map-this-head-s-headdim-coordinate-result-into-a-embeddingdim-coo", {headDim:h(), embeddingDim:fmt(d())}),
    input:`${N()} × ${h()}`,output:`${N()} × ${fmt(d())}`,weight:param('attention.out_proj.weight'),weightHead:head,outputHead:true,
    note:copy("note.columns-value1-value2-of-out-proj-weight-the-python-applies-all-h", {value1:fmt(first), value2:fmt(last)})};
}

function attentionDiagram() {
  if(hasCustomDiagrams())return custom().attention();
  if(hybrid.isDelta())return hybrid.attention();
  return state.attentionMode==='foldable'?foldableAttentionDiagram():attentionTwoStacksDiagram();
}
function currentAttentionLayout() {
  if(hybrid.isDelta())return foldableAttentionLayout(C().linear_num_value_heads,C().linear_num_key_heads,0);
  if(state.attentionTransition)return attentionAccordionFrame(state.attentionTransition.from,state.attentionTransition.to,state.attentionTransition.progress,state.attentionTransition.stage);
  return state.attentionMode==='foldable'?foldableAttentionLayout(C().num_heads,C().num_kv_heads,state.activeHeadGroup):attentionLayout(C().num_heads,C().num_kv_heads);
}
function attentionTwoStacksDiagram() {
  const c=C(),layout=attentionLayout(c.num_heads,c.num_kv_heads),td=`${N()} × ${fmt(d())}`;
  const firstRows=layout.rows.filter(row=>row.bank===0),inputY=(firstRows[0].y+firstRows.at(-1).y)/2;
  const tapX=-270,normX=-220,normWidth=150,splitX=-40,sharedY=108;
  let out=outline(-310,-10,layout.width+330,layout.box.h-10)
    +text(-280,30,copy("label.each-head-makes-a-contribution"),29,'diagram-title','start')
    +text(layout.width-5,30,copy("label.block-block-the-attention-branch", {block:state.block}),18,'diagram-shape','end')
    +text(-280,67,copy("label.read-the-residual-normalize-once-and-share-that-input-with-every"),22,'diagram-note','start');
  out+=path(`M${tapX},${layout.streamY} V${inputY} H${normX-10}`);
  out+=operation(normX,inputY-24,normWidth,48,copy("label.rmsnorm"),{key:'attention-input-norm',title:copy("title.normalize-the-residual-for-attention"),kind:copy("kind.read-from-the-residual-stream"),source:'norm',
    description:copy("description.read-the-incoming-residual-stream-and-normalize-it-once-the-same"),
    input:td,output:td,weight:param('attention_norm.weight'),
    note:copy("note.this-is-attention-norm-in-transformerblock-it-is-included-here-as")});
  out+=dimensions(normX+normWidth/2,inputY+53,td,tokenAxes());
  out+=line(normX+normWidth+10,inputY,76,inputY);
  if(layout.bankCount>1){
    const lastBranch=(layout.bankCount-1)*layout.bankWidth-80;
    out+=`<path id="attention-shared-input" d="M${splitX},${inputY} V${sharedY} H${lastBranch}" class="diagram-wire"/>`;
    out+=`<circle cx="${splitX}" cy="${inputY}" r="3" class="input-tap"/>`;
    out+=text((splitX+lastBranch)/2,sharedY-13,copy("label.the-same-normalized-input"),20,'diagram-note');
  }
  for(let bank=0;bank<layout.bankCount;bank++) {
    const base=bank*layout.bankWidth,rows=layout.rows.filter(row=>row.bank===bank),bottom=rows.at(-1).y;
    const bankInputY=(rows[0].y+bottom)/2;
    if(bank>0)out+=`<path id="attention-stack-input-${bank}" d="M${base-80},${sharedY} V${bankInputY} H${base+76}" class="diagram-arrow"/>`;
    out+=`<circle cx="${base+80}" cy="${bankInputY}" r="2.8" class="input-tap"/>`;
    out+=path(`M${base+80},${rows[0].y} V${bottom}`,'diagram-wire');
    out+=text(base+242,148,copy("label.attention-head"),19,'diagram-note')+text(base+414,148,copy("label.output-projection-6d3094"),19,'diagram-note')+dimensions(base+552,148,td,tokenAxes(),17,'middle','Each head contribution has this shape. Grid cells and proportions are schematic.');
    out+=path(`M${base+628},${rows[0].y} V${layout.sumY}`,'diagram-collection');
    for(const row of rows) {
      const {head,y}=row,group=row.group;
      out+=line(base+80,y,base+158,y)+`<circle cx="${base+80}" cy="${y}" r="2.2" class="input-tap"/>`;
      out+=record({key:`head-${head}`,title:copy("title.query-head-head", {head:head}),kind:copy("kind.one-parallel-attention-head"),source:'attention',enter:'head',head,
        description:copy("description.head-head-computes-its-own-attention-pattern-then-mixes-the-value", {head:head, group:group}),
        input:td,output:`${N()} × ${h()}`,note:copy("note.the-bracket-marks-the-heads-using-k-v-pair-group-they-share-keys", {group:group})},
        headHousing(base+170,y,144,head));
      out+=line(base+323,y,base+356,y);
      out+=record({...outputProjectionSpec(head),codePosition:{x:base+507,y:y-22,size:12}},
        outline(base+365,y-16,102,32)+grid(base+374,y-10,23,20,false,4,3)+text(base+430,y+7,`Wₒ⁽${superscript(head)}⁾`,21,'diagram-math'));
      out+=line(base+478,y,base+511,y);
      out+=record({key:`contribution-${head}`,title:copy("title.contribution-from-head-head", {head:head}),kind:copy("kind.one-head-s-contribution"),source:'combine',
        description:copy("description.after-its-output-projection-head-head-contributes-one-embeddingdi", {head:head, embeddingDim:fmt(d()), heads:c.num_heads}),
        input:`${N()} × ${h()}`,output:td,note:copy("note.the-grid-is-schematic-this-view-does-not-run-the-prompt-through-t")},
        vector(base+522,y-11,70,22,true)+text(base+557,y+7,`Δ${head}`,19,'diagram-title'));
      out+=line(base+602,y,base+624,y,'diagram-contribution');
    }
    const groups=[...new Set(rows.map(row=>row.group))];
    for(const group of groups) {
      const pair=rows.filter(row=>row.group===group),top=pair[0].y-16,end=pair.at(-1).y+16,mid=(top+end)/2;
      out+=record({key:`kv-${group}`,title:copy("title.shared-keys-values-group", {group:group}),codePosition:{x:base+28,y:top-8,size:12},kind:copy("kind.a-shared-k-v-pair"),source:'sharing',
        description:copy("description.heads-value1-use-the-same-key-and-value-projections-each-head-sti", {value1:pair.map(row=>row.head).join(' and ')}),input:td,output:`K, V: ${N()} × ${h()}`,
        parameterScope:'kv-pair',weight:param('attention.W_k'),weightHead:group,note:copy("note.inspect-the-key-projection-here-or-open-either-head-to-inspect-it")},
        `<rect x="${base-18}" y="${top}" width="77" height="${end-top}" fill="transparent"/>`+
        path(`M${base+51},${top} H${base+39} V${end} H${base+51}`,'diagram-bracket')+
        text(base+9,mid-3,'K/V',16,'diagram-shape')+text(base+9,mid+19,group,20,'diagram-title'));
    }
    const collector=base+628,edge=collector<layout.sumX?layout.sumX-81:layout.sumX+81;
    out+=line(collector,layout.sumY,edge,layout.sumY,'diagram-contribution');
  }
  out+=operation(layout.sumX-70,layout.sumY-22,140,44,copy("label.sum"),{key:'combine',title:copy("title.sum-the-head-contributions"),kind:copy("kind.add-the-projected-outputs"),source:'combine',
    description:copy("description.add-the-heads-projected-contributions-element-by-element-every-co", {heads:c.num_heads, td:td}),input:`${c.num_heads} contributions · each ${td}`,output:td,
    note:copy("note.concatenate-then-project-and-project-each-then-sum-are-mathematic")});
  const addX=layout.sumX;
  out+=path(`M${addX},${layout.sumY+23} V${layout.streamY-28}`,'diagram-contribution');
  out+=text(addX+20,layout.sumY+48,copy("label.one-attention-update"),21,'diagram-note','start');
  out+=stream(-325,layout.width+30,layout.streamY);
  out+=`<circle cx="${tapX}" cy="${layout.streamY}" r="4" class="stream-tap"/>`;
  out+=text(tapX+25,layout.streamY-18,copy("label.the-original-residual-r"),22,'diagram-note','start');
  out+=plus(addX,layout.streamY,{key:'attention-add',title:copy("title.add-the-total-attention-contribution"),description:copy("description.add-the-sum-of-all-head-contributions-to-the-original-residual-st"),source:'residual'});
  out+=text(layout.width+10,layout.streamY+39,copy("label.on-to-feedforward"),21,'diagram-note','end');
  out+=text(-280,layout.streamY+39,copy("label.each-w-slice-maps-headdim-embeddingdim-click-a-head-to-open-it-cl", {headDim:h(), embeddingDim:fmt(d())}),21,'diagram-note','start');
  views.attentionLocal=layout.box;
  return out;
}

function foldedHeadPaper(layer,expansion) {
  const folded=1-expansion,width=GQA.contributionX+GQA.contributionWidth-GQA.headX;
  return outlinePath(GQA.headX+(layer?4:9)*folded,2-(layer?3:6)*folded,width-(layer?6:14)*folded,44);
}

function foldableAttentionDiagram() {
  const c=C(),layout=currentAttentionLayout(),td=`${N()} × ${fmt(d())}`;
  const {inputY,sumX,sumY,streamY}=layout,tapX=-270;
  const {inputX,headX,headWidth,projectionX:projX,projectionWidth:projWidth,contributionX:deltaX,contributionWidth:deltaWidth}=GQA;
  const packetWidth=deltaX+deltaWidth-headX;
  const visible=layout.groups.find(group=>group.open).count;
  let out=outline(-310,-10,1090,layout.box.h-10+(isOriginalQwen?28:0))
    +text(-280,30,copy("label.each-head-makes-a-contribution"),29,'diagram-title','start')
    +text(750,30,copy("label.block-block", {block:state.block}),18,'diagram-shape','end')
    +text(-280,67,copy("label.all-heads-share-the-input-unfold-a-group-to-inspect-its-heads"),22,'diagram-note','start')
    +text(-280,99,copy("label.visible-of-heads-heads-unfolded-all-heads-still-contribute", {visible:visible, heads:c.num_heads}),18,'diagram-shape','start');
  out+=flow(`<path id="gqa-read-input" d="M${tapX},${streamY} V${inputY} H-230" class="diagram-arrow"/>`,'residual','','attention-input-norm');
  out+=`<g id="gqa-normalized-input" transform="translate(0,${inputY})">`;
  out+=operation(-220,-33,86,66,copy("label.rmsnorm"),{key:'attention-input-norm',title:copy("title.normalize-the-residual-for-attention"),kind:copy("kind.read-from-the-residual-stream"),source:'norm',labelText:copy("label.rms-norm-lines"),
    description:copy("description.read-the-incoming-residual-stream-and-normalize-it-once-this-same"),
    input:td,output:td,weight:param('attention_norm.weight'),note:copy("note.folding-changes-how-much-of-the-diagram-is-visible-it-does-not-sk")});
  out+=dimensions(-177,62,td,tokenAxes())+flow(line(-124,0,inputX-4,0),'normalized','attention-input-norm','')+`<circle cx="${inputX}" cy="0" r="3" class="input-tap"/></g>`;
  out+=flow(`<path id="gqa-input-bus" d="M${inputX},${Math.min(inputY,layout.connections[0])} V${Math.max(inputY,layout.connections.at(-1))}" class="diagram-wire"/>`,'normalized','attention-input-norm','');
  out+=text(headX+headWidth/2,139,copy("label.attention-heads"),19,'diagram-note')+text(projX+projWidth/2,139,copy("label.output-projection-6d3094"),19,'diagram-note')+dimensions(deltaX+deltaWidth/2,139,td,tokenAxes(),17,'middle','Each head contribution has this shape. Grid cells and proportions are schematic.');
  out+=flow(`<path id="gqa-output-bus" d="M${sumX},${layout.connections[0]} V${sumY-34}" class="diagram-contribution"/>`,'contributions','','combine');
  for(const group of layout.groups) {
    const range=`${group.first}–${group.last}`;
    const folded=1-group.expansion;
    out+=`<g data-attention-group="${group.index}" data-expanded="${group.open}" data-expansion="${group.expansion}" data-height="${group.height}" transform="translate(0,${group.top})">`;
    if(folded>0||state.attentionTransition) {
      out+=`<defs><clipPath id="gqa-summary-clip-${group.index}" clipPathUnits="userSpaceOnUse"><rect data-summary-clip x="${inputX-10}" y="-8" width="${sumX-inputX+20}" height="${56*folded}"/></clipPath></defs>`;
      out+=`<g data-group-summary clip-path="url(#gqa-summary-clip-${group.index})" aria-hidden="${group.expansion>0}">`;
      out+=flow(line(inputX,24,headX-10,24),'normalized','attention-input-norm',`unfold-head-group-${group.index}`)+`<circle cx="${inputX}" cy="24" r="2.2" class="input-tap"/>`+flow(line(deltaX+deltaWidth+10,24,sumX-4,24,'diagram-contribution'),'contributions',`unfold-head-group-${group.index}`,'combine',{count:group.count});
      const kvRange=group.firstKV===group.lastKV?`${group.firstKV}`:`${group.firstKV}–${group.lastKV}`;
      out+=record({key:`unfold-head-group-${group.index}`,title:copy("title.unfold-value1-heads-range", {value1:group.count, range:range}),action:'toggle',group:group.index,expanded:false,
        kind:copy("kind.one-group-of-heads"),source:'attention',description:copy("description.heads-range-each-contribute-to-the-attention-update-opening-anoth", {range:range})},
        `<path data-fold-paper="0" class="housing" style="fill:#f7faf9" d="${foldedHeadPaper(0,group.expansion)}"/>`+
        `<path data-fold-paper="1" class="housing" style="fill:#f7faf9" d="${foldedHeadPaper(1,group.expansion)}"/>`+
        outline(headX,2,packetWidth,44,'#f5f9f8')+text(headX+20,32,copy("label.value1-heads", {value1:group.count}),24,'diagram-title','start')+
        text(182,31,range,20,'diagram-shape')+text(340,31,copy("label.k-v-kvrange", {kvRange:kvRange}),18,'diagram-note')+text(deltaX+deltaWidth-18,31,copy("label.unfold"),20,'diagram-note','end'));
      out+='</g>';
    }
    const hasDetails=group.expansion>0||state.attentionTransition&&(state.attentionTransition.from.groups[group.index].open||state.attentionTransition.to.groups[group.index].open);
    if(hasDetails) {
      out+=`<defs><clipPath id="gqa-detail-clip-${group.index}" clipPathUnits="userSpaceOnUse"><rect data-detail-clip x="${inputX-10}" y="-16" width="${sumX-inputX+20}" height="${group.detailHeight+16*group.expansion}"/></clipPath></defs>`;
      out+=`<g data-group-details transform="translate(0,${group.headerHeight})" clip-path="url(#gqa-detail-clip-${group.index})" aria-hidden="${group.expansion===0}" style="pointer-events:${state.attentionTransition?'none':'inherit'}">`;
      const rows=layout.rows.filter(row=>row.disclosure===group.index);
      for(const pair of group.pairs) {
        out+=`<g data-kv-group="${pair.kv}">`+record({key:`kv-${pair.kv}`,title:copy("title.shared-keys-values-kvhead", {kvHead:pair.kv}),codePosition:{x:headX+headWidth+78,y:pair.top+2,size:12},kind:copy("kind.a-shared-k-v-pair"),source:'sharing',
          description:copy("description.heads-value1-share-key-and-value-projections-each-head-still-has", {value1:rows.filter(row=>row.group===pair.kv).map(row=>row.head).join(' and ')}),input:td,output:`K, V: ${N()} × ${h()}`,
          parameterScope:'kv-pair',weight:param('attention.W_k'),weightHead:pair.kv,note:copy("note.open-either-head-to-inspect-its-shared-key-and-value-matrices")},
          path(outlinePath(headX-16,pair.top-4,headWidth+32,pair.height+4),'gqa-kv-box')+
          text(headX+headWidth/2,pair.labelY,copy("label.shared-k-v"),18,'diagram-note'))+'</g>';
      }
      for(const row of rows) {
        const {head}=row,y=row.localY;
        out+=`<g data-visible-head="${head}">`;
        out+=flow(line(inputX,y,headX-10,y),'normalized','attention-input-norm',`head-${head}`)+`<circle cx="${inputX}" cy="${y}" r="2.2" class="input-tap"/>`;
        out+=record({key:`head-${head}`,title:copy("title.query-head-head", {head:head}),kind:copy("kind.one-parallel-attention-head"),source:'attention',enter:'head',head,
          description:copy("description.head-head-computes-its-own-attention-pattern-then-mixes-the-value-33038b", {head:head, kvHead:row.group}),
          input:td,output:`${N()} × ${h()}`,note:copy("note.the-dotted-box-marks-the-heads-using-k-v-pair-kvhead-folding-a-gr", {kvHead:row.group})},
          headHousing(headX,y,headWidth,head));
        out+=flow(line(headX+headWidth+10,y,projX-10,y),'head',`head-${head}`,`output-head-${head}`);
        out+=record({...outputProjectionSpec(head),codePosition:{x:deltaX-20,y:y-20,size:12}},
          outline(projX,y-16,projWidth,32)+grid(projX+9,y-10,23,20,false,4,3)+text(projX+68,y+7,`Wₒ⁽${superscript(head)}⁾`,21,'diagram-math'));
        out+=flow(line(projX+projWidth+10,y,deltaX-10,y),'projectedHead',`output-head-${head}`,`contribution-${head}`);
        out+=record({key:`contribution-${head}`,title:copy("title.contribution-from-head-head", {head:head}),kind:copy("kind.one-head-s-contribution"),source:'combine',
          description:copy("description.head-head-contributes-one-embeddingdim-coordinate-vector-per-toke", {head:head, embeddingDim:fmt(d()), heads:c.num_heads}),
          input:`${N()} × ${h()}`,output:td,note:copy("note.the-grid-is-schematic-this-view-does-not-run-the-prompt-through-t")},
          vector(deltaX,y-11,deltaWidth,22,true)+text(deltaX+deltaWidth/2,y+7,`Δ${head}`,19,'diagram-title'));
        out+=flow(line(deltaX+deltaWidth+10,y,sumX-4,y,'diagram-contribution'),'projectedHead',`contribution-${head}`,'combine');
        out+='</g>';
      }
      out+='</g>';
    }
    out+='</g>';
  }
  out+=operation(sumX-70,sumY-22,140,44,copy("label.sum"),{key:'combine',title:copy("title.sum-the-head-contributions"),kind:copy("kind.add-all-projected-outputs"),source:'combine',
    description:copy("description.add-all-heads-projected-contributions-including-the-heads-in-fold", {heads:c.num_heads, td:td}),input:`${c.num_heads} contributions · each ${td}`,output:td,
    note:copy("note.the-python-concatenates-the-head-outputs-and-projects-them-togeth")});
  out+=flow(path(`M${sumX},${sumY+34} V${streamY-28}`,'diagram-contribution'),'attention','combine','attention-add')+text(sumX-25,sumY+65,copy("label.one-attention-update"),21,'diagram-note','end');
  out+=stream(-325,790,streamY)
    +(isOriginalQwen?flow(path(`M-325,${streamY} H${sumX-20}`,'flow-guide'),'residual','','attention-add')+flow(path(`M${sumX+20},${streamY} H790`,'flow-guide'),'afterAttention','attention-add',''):'')
    +flowBoundary(-280,streamY-49,'residual','Input')+flowBoundary(765,streamY+65,'afterAttention','Output','end')+`<circle cx="${tapX}" cy="${streamY}" r="4" class="stream-tap"/>`;
  out+=text(tapX+25,streamY-18,copy("label.the-original-residual-r"),22,'diagram-note','start');
  out+=plus(sumX,streamY,{key:'attention-add',title:copy("title.add-the-total-attention-contribution"),description:copy("description.add-the-sum-of-all-head-contributions-to-the-original-residual-st-684194"),source:'residual'});
  out+=text(765,streamY+39,copy("label.on-to-feedforward"),21,'diagram-note','end');
  out+=text(-280,streamY+39,copy("label.each-w-slice-maps-headdim-embeddingdim", {headDim:h(), embeddingDim:fmt(d())}),21,'diagram-note','start');
  views.attentionLocal=isOriginalQwen?{...layout.box,h:layout.box.h+28}:layout.box;
  return out;
}

function renderAttentionAccordion(frame) {
  canvas.dataset.accordionProgress=state.attentionTransition.progress.toFixed(4);
  canvas.dataset.accordionStage=state.attentionTransition.stage;
  for(const group of frame.groups) {
    const el=world.querySelector(`[data-attention-group="${group.index}"]`);
    const folded=1-group.expansion;
    el.setAttribute('transform',`translate(0,${group.top})`);
    el.dataset.height=group.height;
    el.dataset.expansion=group.expansion;
    el.querySelector('[data-fold-paper="0"]').setAttribute('d',foldedHeadPaper(0,group.expansion));
    el.querySelector('[data-fold-paper="1"]').setAttribute('d',foldedHeadPaper(1,group.expansion));
    el.querySelector('[data-summary-clip]').setAttribute('height',56*folded);
    el.querySelector('[data-group-summary]').setAttribute('aria-hidden',String(group.expansion>0));
    el.querySelector('[data-detail-clip]')?.setAttribute('height',group.detailHeight+16*group.expansion);
    el.querySelector('[data-group-details]')?.setAttribute('transform',`translate(0,${group.headerHeight})`);
    el.querySelector('[data-group-details]')?.setAttribute('aria-hidden',String(group.expansion===0));
  }
  $('#gqa-input-bus').setAttribute('d',`M${GQA.inputX},${Math.min(frame.inputY,frame.connections[0])} V${Math.max(frame.inputY,frame.connections.at(-1))}`);
  $('#gqa-output-bus').setAttribute('d',`M${frame.sumX},${frame.connections[0]} V${frame.sumY-34}`);
  if(isOriginalQwen)syncFlowTargets(layers.attention);
}

async function changeAttentionView(action,group) {
  if(state.phase!=='idle'||state.level!=='attention')return;
  const request=++navigation,previous=currentAttentionLayout(),oldMode=state.attentionMode;
  if(action==='foldable'||action==='two-stacks') {
    if(state.attentionMode===action)return;
    state.attentionMode=action;
  } else if(action==='toggle') {
    if(state.activeHeadGroup===group)return;
    state.activeHeadGroup=group;
  }
  const exchangingGroups=oldMode==='foldable'&&state.attentionMode==='foldable';
  if(exchangingGroups)state.attentionTransition={from:previous,to:currentAttentionLayout(),stage:'closing',progress:0};
  setPhase('folding-heads');
  buildDiagram();selectDefault('attention');
  for(const control of layers.attention.querySelectorAll('[tabindex]'))control.setAttribute('tabindex','-1');
  let complete;
  if(exchangingGroups) {
    for(const [stage,duration] of [['closing',260],['pause',110],['opening',500]]) {
      state.attentionTransition.stage=stage;
      state.attentionTransition.progress=0;
      renderAttentionAccordion(currentAttentionLayout());
      complete=await tween(duration,progress=>{
        state.attentionTransition.progress=progress;
        renderAttentionAccordion(currentAttentionLayout());
      });
      if(!complete||request!==navigation)return;
    }
  } else complete=await moveCamera(frameCamera(views.attention));
  if(!complete||request!==navigation)return;
  state.attentionTransition=null;
  if(exchangingGroups)buildDiagram();
  delete canvas.dataset.accordionProgress;delete canvas.dataset.accordionStage;
  setPhase('idle');renderCamera();
  const control=group===undefined?null:world.querySelector(`[data-attention-group="${group}"] [data-visible-head] [data-node]`);
  control?.focus({preventScroll:true});
}

function headDiagram() {
  if(hasCustomDiagrams())return custom().head();
  if(hybrid.isDelta())return hybrid.head();
  const shape=`${N()} × ${h()}`, input=`${N()} × ${fmt(d())}`,square=`${N()} × ${N()}`;
  const rows=Math.max(1,Math.min(N(),8)),matrixY=227,matrixSize=180;
  // Enter on grid boundaries so arrows clear the token-position labels.
  const keyY=rows<3?335:matrixY+matrixSize*Math.round(rows*.6)/rows;
  const queryY=keyY-180,valueY=keyY+180;
  const queryPort=rows<2?263:matrixY+matrixSize*Math.max(1,Math.floor(rows*.25))/rows;
  const scoresFlowY=rows<2?281:matrixY+matrixSize*Math.floor(rows/2)/rows;
  const mixX=1610;
  let out=outline(-95,-30,isHybrid()?2500:2050,isHybrid()?930:710)+text(16,14,copy("label.head-head-how-attention-is-computed", {head:state.head}),28,'diagram-title','start')+text(1920,14,copy("label.block-block-k-v-pair-kvhead", {block:state.block, kvHead:kvHead()}),19,'diagram-shape','end');
  out+=flow(path(`M-143,${keyY} H0`),'normalized','','head-input')+flowBoundary(-67,keyY-43,'normalized','Input','middle',true)+(isOriginalQwen?'':text(-63,keyY-17,copy("label.input"),21,'diagram-note'));
  out+=matrix(12,keyY-66,100,132,'x',input,{key:'head-input',title:copy("title.the-full-input-to-this-head"),kind:copy("kind.activations"),source:'attention',activation:true,rows,description:copy("description.this-head-reads-every-coordinate-of-every-input-token-through-its"),input});
  out+=flow(path(`M124,${keyY} H149`,'diagram-wire')+path(`M149,${queryY} V${valueY}`,'diagram-wire'),'normalized','head-input','')+`<circle cx="149" cy="${keyY}" r="3" class="input-tap"/>`;
  const letters=['q','k','v'],centers=[queryY,keyY,valueY];
  letters.forEach((letter,i)=>{
    const center=centers[i],y=center-50,head=letter==='q'?state.head:kvHead(),weight=param(`attention.W_${letter}`);
    out+=flow(line(149,center,174,center),'normalized','head-input',`weight-${letter}`);
    out+=matrix(186,y,80,100,`W_${letter}`,matrixShape(weight,true),{key:`weight-${letter}`,title:copy("title.projection-projection-weights", {projection:letter.toUpperCase()}),kind:copy("kind.learned-matrix"),source:letter,description:copy("description.multiply-the-full-input-by-this-learned-matrix-to-produce-value1", {value1:letter==='q'?'queries':letter==='k'?'keys':'values', value2:letter==='q'?`This matrix belongs to query head ${head}.`:`This matrix belongs to shared key/value head ${head}.`}),input,output:shape,weight,weightHead:head});
    out+=text(292,center+9,'×',27)+flow(line(312,center,328,center),letter,`weight-${letter}`,`activation-${letter}`);
    out+=matrix(340,center-36,64,72,letter.toUpperCase(),shape,{key:`activation-${letter}`,axes:tokenAxes(({q:'query',k:'key',v:'value'})[letter]+' coordinate'),title:copy("title.projection-activations", {projection:letter.toUpperCase()}),kind:copy("kind.activations"),source:letter,activation:true,rows,labelKind:'diagram-math',description:copy("description.these-value1-depend-on-the-input-text-each-token-has-a-headdim-di", {value1:letter==='q'?'queries':letter==='k'?'keys':'values', headDim:h()}),input:shape,note:copy("note.schematic-grid-execution-playback-will-supply-activation-values")});
    if(letter!=='v') {
      out+=flow(line(416,center,438,center),letter,`activation-${letter}`,`${letter}-norm`);
      out+=operation(450,center-33,66,66,copy("label.rmsnorm"),{key:`${letter}-norm`,kind:copy("kind.normalization"),source:'qk_norm',description:copy("description.normalize-each-query-or-key-vector-along-its-head-dimension-value"),input:shape,output:shape,weight:param(`attention.${letter}_norm.weight`),fontSize:20,labelText:copy("label.rms-norm-lines")});
      out+=flow(line(528,center,543,center),`${letter}Norm`,`${letter}-norm`,`${letter}-rope`);
      out+=operation(555,center-27,70,54,copy("label.rope"),{key:`${letter}-rope`,kind:copy("kind.position-information"),source:'rope',description:copy("description.rotate-pairs-of-query-or-key-coordinates-by-an-angle-determined-b"),input:shape,output:shape,fontSize:22});
    }
  });
  out+=flow(path(`M637,${queryY} H649 Q661,${queryY} 661,${queryY+12} V${queryPort-12} Q661,${queryPort} 673,${queryPort} H704`),'qRoPE','q-rope','scores');
  out+=flow(line(637,keyY,704,keyY),'kRoPE','k-rope','scores');
  out+=text(655,queryY-16,'Q',21,'diagram-math')+text(661,keyY-16,'K',21,'diagram-math');
  out+=text(720,73,copy("label.rows-query-positions-columns-key-positions"),21,'diagram-note','start');
  out+=attentionMatrix(720,matrixY,matrixSize,'scores',{key:'scores',title:copy("title.attention-scores"),kind:copy("kind.raw-dot-products"),source:'scores',
    description:copy("description.compare-every-query-with-every-key-each-dot-stands-for-a-dot-prod"),
    input:`(${shape}) @ (${h()} × ${N()})`,output:square,note:copy("note.the-raised-t-means-transpose-dot-sizes-are-illustrative-they-do-n")});
  out+=`<text x="810" y="443" font-size="23" text-anchor="middle" class="diagram-math">QK<tspan dy="-9" font-size="15">T</tspan></text>`;
  out+=flow(line(912,scoresFlowY,938,scoresFlowY),'scores','scores','mask');
  out+=operation(950,scoresFlowY-28,80,56,copy("label.mask"),{key:'mask',title:copy("title.hide-future-tokens"),kind:copy("kind.causal-mask"),source:'mask',description:copy("description.replace-scores-above-the-diagonal-with-negative-infinity-column-r"),input:square,output:square});
  out+=flow(line(1042,scoresFlowY,1084,scoresFlowY),'masked','mask','masked-scores');
  out+=attentionMatrix(1100,matrixY,matrixSize,'masked',{key:'masked-scores',title:copy("title.masked-scores"),kind:copy("kind.future-positions-hidden"),source:'mask',
    description:copy("description.the-upper-triangle-contains-wherever-the-key-is-later-than-the-qu"),input:square,output:square,
    note:copy("note.the-mask-pattern-is-exact-for-this-context-length-dots-retain-the")});
  out+=text(1190,443,copy("label.future-scores"),21,'diagram-note');
  out+=flow(line(1292,scoresFlowY,1318,scoresFlowY),'masked','masked-scores','softmax');
  out+=operation(1330,scoresFlowY-39,128,78,copy("label.scale-and-softmax"),{key:'softmax',title:copy("title.scale-and-softmax"),kind:copy("kind.scale-then-normalize-each-row"),source:'softmax',
    description:copy("description.divide-the-masked-scores-by-headdim-then-apply-softmax-across-eac", {headDim:h()}),
    input:square,output:square,fontSize:23,labelLines:[`÷ √${h()}`,copy("label.softmax")],labelKinds:['diagram-math','diagram-title'],note:copy("note.moving-this-division-after-the-mask-gives-the-same-attention-weig")});
  out+=flow(line(1470,scoresFlowY,1504,scoresFlowY),'weights','softmax','attention-weights');
  out+=attentionMatrix(1520,matrixY,matrixSize,'weights',{key:'attention-weights',title:copy("title.attention-weights"),kind:copy("kind.input-dependent-mixing-weights"),source:'softmax',
    description:copy("description.each-row-tells-a-query-how-much-to-read-from-each-value-vector-ze"),
    input:square,output:square,note:copy("note.dot-sizes-are-illustrative-the-0-1-entries-follow-exactly-from-th")});
  out+=flow(line(mixX,matrixY+matrixSize+12,mixX,valueY-38,'diagram-contribution'),'weights','attention-weights','value-mix');
  out+=text(mixX-24,443,copy("label.each-row-sums-to-1"),21,'diagram-note','end')+text(mixX+22,(matrixY+matrixSize+valueY)/2,copy("label.weights"),20,'diagram-note','start');
  out+=flow(line(416,valueY,mixX-38,valueY,'diagram-contribution'),'v','activation-v','value-mix')+text(926,valueY+32,copy("label.v-carries-the-information-to-read-from-each-token"),24,'diagram-note');
  out+=text(mixX-62,valueY-15,'V',24,'diagram-math');
  const mix={source:'mix',kind:copy("kind.weighted-value-mixture"),description:copy("description.multiply-the-attention-weight-matrix-by-v-every-output-row-is-a-w"),input:`(${square}) @ (${shape})`,output:shape};
  out+=record({...mix,key:'value-mix',title:copy("title.mix-values-using-attention-weights"),codePosition:{x:mixX+39,y:valueY-23,size:15}},
    `<circle cx="${mixX}" cy="${valueY}" r="26" class="housing"/>`+text(mixX,valueY+10,'×',33,'diagram-math'));
  out+=text(mixX,valueY+61,copy("label.matrix-multiply"),21,'diagram-note')+flow(line(mixX+38,valueY,1718,valueY,'diagram-contribution'),'head','value-mix','head-output');
  out+=matrix(1730,valueY-40,144,80,isHybrid()?'Mixed values':copy("label.head-output"),shape,{...mix,key:'head-output',axes:tokenAxes('head coordinate'),title:isHybrid()?'Mixed values before the gate':copy("title.the-output-of-this-head"),activation:true,rows,
    note:copy("note.one-mixed-value-vector-per-token-the-next-step-shown-one-level-up")});
  if(isHybrid()){
    out+=line(1886,valueY,1930,valueY,'diagram-contribution');
    out+=record({key:'attention-output-gate',title:'Apply the sigmoid gate',kind:'ELEMENTWISE MULTIPLICATION',source:'attention_gate',description:'Multiply the mixed value vector by this position’s sigmoid gate, coordinate by coordinate. The gated result is what enters the output projection.',input:`Two ${shape} tensors`,output:shape},`<circle cx="1960" cy="${valueY}" r="23" class="housing"/>`+text(1960,valueY+9,'⊙',31,'diagram-math'));
    out+=line(1995,valueY,2048,valueY,'diagram-contribution');
    out+=matrix(2060,valueY-40,144,80,'Gated head output',shape,{key:'gated-head-output',title:'Gated head output',kind:'ACTIVATIONS',source:'attention_gate',activation:true,rows,output:shape,axes:tokenAxes('head coordinate'),description:'These gated vectors pass to this head’s output-projection slice, shown one level up.'});
    out+=line(2216,valueY,2425,valueY,'diagram-contribution')+text(2310,valueY-17,'output',21,'diagram-note');
    out+=path(`M149,${valueY} V760 H1008`,'diagram-wire');
    out+=matrix(1020,710,100,100,'W_gate',matrixShape(param('attention.W_gate'),true),{key:'gate-projection',title:'Output-gate projection',kind:'LEARNED MATRIX',source:'gate_projection',weight:param('attention.W_gate'),weightHead:state.head,input,output:shape,description:'Each attention head has a separate gate projection from the same normalized input. The checkpoint packs its rows beside that head’s query rows; the educational code separates them.'});
    out+=line(1132,760,1218,760);
    out+=operation(1230,728,140,64,'Sigmoid',{key:'gate-sigmoid',source:'gate_projection',description:'Map each gate coordinate to a number between zero and one.',input:shape,output:shape});
    out+=path(`M1382,760 H1960 V${valueY+34}`,'diagram-contribution');
    out+=text(630,793,'The same normalized input also determines the gate.',21,'diagram-note');
  }else{
    out+=flow(line(1886,valueY,1990,valueY,'diagram-contribution'),'head','head-output','')+flowBoundary(1938,valueY-43,'head','Output','middle',true)+(isOriginalQwen?'':text(1935,valueY-17,copy("label.output"),21,'diagram-note'));
  }
  out+=text(1802,valueY+104,copy("label.one-vector-per-token"),20,'diagram-note');
  out+=text(24,isHybrid()?865:644,copy("label.w-learned-matrices-q-k-v-input-dependent-activations"),21,'diagram-note','start');
  out+=text(720,isHybrid()?865:644,copy("label.dot-sizes-are-illustrative-the-causal-mask-is-exact"),20,'diagram-note','start');
  return out;
}

function scoreSpec() {
  const row=state.scoreRow,col=state.scoreCol;
  return {key:'score-detail',title:copy("title.score-row-column", {row:row+1, column:col+1}),kind:copy("kind.query-key-dot-product"),source:'scores',
    description:copy("description.use-the-query-at-position-row-from-head-head-and-the-key-at-posit", {row:row+1, head:state.head, column:col+1, kvHead:kvHead()}),
    input:`Two vectors of ${h()} coordinates`,inputAxes:'Coordinates within the attention head.',output:'One scalar',
    note:col>row?copy("note.future-score"):copy("note.allowed-score")};
}
function scoreDiagram() {
  if(!N())return '';
  const count=Math.min(N(),8),row=state.scoreRow,col=state.scoreCol,shape=`${N()} × ${h()}`;
  let out=outline(0,0,1140,570)+text(28,38,copy("label.a-single-attention-score"),28,'diagram-title','start')
    +text(1110,38,copy("label.block-block-head-head", {block:state.block, head:state.head}),19,'diagram-shape','end')
    +text(70,73,copy("label.q-and-k-here-are-after-normalization-and-rope"),21,'diagram-note','start');
  const normalized={source:isGPT2()?'q':'qk_norm',activation:true,note:copy("note.the-grid-shows-symbolic-activations-after-q-k-normalization-and-r")};
  out+=matrix(70,150,260,160,'Q',shape,{...normalized,key:'score-query',title:copy("title.query-at-position-row", {row:row+1}),axes:tokenAxes('query coordinate'),rows:count,
    description:copy("description.every-row-is-a-normalized-and-rotated-query-vector-from-head-head", {head:state.head, row:row+1}),output:shape,labelKind:'diagram-math'});
  out+=`<rect class="score-vector-highlight" data-query-row="${row}" x="70" y="${150+row*160/count}" width="260" height="${160/count}"/>`;
  out+=`<g class="score-vector-label" pointer-events="none">${text(200,150+(row+.5)*160/count+6,copy("label.q-vector"),18,'diagram-note')}</g>`;
  for(let i=0;i<count;i++)out+=text(57,150+(i+.5)*160/count+5,i+1,14,'diagram-shape','end');
  out+=text(385,245,'×',30,'diagram-math');
  out+=matrix(440,130,160,200,'K',`${h()} × ${N()}`,{...normalized,source:isGPT2()?'k':'qk_norm',key:'score-key',title:copy("title.key-at-position-column", {column:col+1}),axes:{rows:'key coordinate',columns:'key position'},rows:7,cols:count,transposeLabel:true,
    description:copy("description.transpose-turns-each-key-row-into-a-column-column-column-contains", {column:col+1, kvHead:kvHead()}),output:`${h()} × ${N()}`,labelKind:'diagram-math'});
  out+=`<rect class="score-vector-highlight" data-key-column="${col}" x="${440+col*160/count}" y="130" width="${160/count}" height="200"/>`;
  out+=`<g class="score-vector-label" pointer-events="none" transform="translate(${440+(col+.5)*160/count},230) rotate(-90)">${text(0,6,copy("label.k-vector"),18,'diagram-note')}</g>`;
  out+=text(690,245,'=',30,'diagram-math');
  out+=attentionMatrix(780,170,160,'scores',{...scoreSpec(),key:'score-matrix',title:copy("title.attention-scores"),input:`(${shape}) @ (${h()} × ${N()})`,output:`${N()} × ${N()}`,inputAxes:'Q rows are query positions; Kᵀ columns are key positions.',outputAxes:{rows:'query position',columns:'key position'}});
  out+=text(200,380,copy("label.row-row-q", {row:row+1}),22,'diagram-note')+text(520,380,copy("label.column-column-k", {column:col+1}),22,'diagram-note')
    +text(860,380,copy("label.score-row-column", {row:row+1, column:col+1}),22,'diagram-note');
  if(N()>count)out+=text(1075,300,copy("label.first-8"),17,'diagram-note')+text(1075,325,copy("label.positions-shown"),17,'diagram-note');
  const explanation=copy("caption.score-diagram",{row:row+1,column:col+1});
  out+=`<text x="70" y="434" font-size="22" class="diagram-note score-explanation" ${copyAttributes(explanation)} data-copy-line-height="32" data-copy-line-gaps="32,50">${svgTextBody(explanation,70,32,[32,50])}</text>`;
  return out;
}

function siluSketch(x,y,width,height) {
  // The curve uses SiLU itself; pen-style axes and rounded strokes match the diagram.
  const px=value=>x+(value+4)/7*width,py=value=>y+(3-value)/3.5*height;
  const points=Array.from({length:57},(_,i)=>{
    const value=-4+i/8;
    return `${i?'L':'M'}${px(value).toFixed(2)},${py(value/(1+Math.exp(-value))).toFixed(2)}`;
  }).join(' ');
  const zeroX=px(0),zeroY=py(0);
  return `<g class="silu-sketch" aria-hidden="true">`
    +path(`M${x},${zeroY+.3} Q${x+width/2},${zeroY-.4} ${x+width},${zeroY} M${zeroX-.3},${y+height} Q${zeroX+.3},${y+height/2} ${zeroX},${y}`,'silu-axis')
    +path(points,'silu-curve')+'</g>';
}

function feedforwardDiagram(asExpert=false) {
  if(hasCustomDiagrams())return asExpert?custom().expert():custom().feedforward();
  if(isMoE()&&!asExpert)return moe.diagram();
  const positions=asExpert?1:N();
  const input=`${positions} × ${fmt(d())}`, expanded=`${positions} × ${fmt(C().expanded_dim)}`;
  const annotate=!singleFeedforwardReference,rows=Math.min(positions,8);
  const tailShift=annotate?FF_TAIL_SHIFT:0,siluX=annotate?490:515,bendX=745+tailShift;
  const matrixHeight=annotate?Math.max(1,Math.min(positions,5))*25.6:128;
  const matrixY=273-matrixHeight/2;
  const rowInfo=annotate?{rows:Math.max(1,rows),axes:{rows:'sequence position',columns:'residual coordinate'}}:{};
  const offset=row=>(row-(rows-1)/2)*3.5;
  const rowY=row=>matrixY+(row+.5)*matrixHeight/Math.max(1,rows);
  const bundle=(draw,inside=false)=>`<g class="ff-bundle${inside?' inside-operation':''}" aria-hidden="true">`+Array.from({length:rows},(_,row)=>`<path d="${draw(offset(row),row)}"/>`).join('')+'</g>';
  const connection=(original,draw,kind,from,to)=>flow(annotate?bundle(draw):path(original),kind,from,to);
  const sharedOperation=(x,y,w,height,title,spec,flowY)=>{
    if(!annotate)return operation(x,y,w,height,title,spec);
    const top=y-12,boxHeight=height+12;
    return record({title,source:asExpert?'expert':'feedforward',...spec,box:{x,y:top,w,h:boxHeight}},outline(x,top,w,boxHeight)
      +bundle(o=>`M${x},${flowY+o} H${x+w}`,true)
      +text(x+w/2,flowY-23,title,spec.fontSize||22,'diagram-title')
      +(spec.key==='silu'?siluSketch(x+30,flowY-14,w-60,46):''));
  };
  let out=(asExpert?outline(-25,-20,1345,555):annotate?outline(-190,-20,1530+tailShift,720):outline(-20,-20,1320,550))
    +text(asExpert?0:annotate?-155:20,26,asExpert?copy('moe.expert-title',{expert:state.expert}):copy("label.inside-the-feedforward-network"),28,'diagram-title','start')+text(1300+tailShift,26,copy("label.block-block", {block:state.block}),20,'diagram-shape','end');
  if(annotate&&!asExpert){
    out+=flow(path('M-165,520 V273 H-132'),'afterAttention','','ff-input-norm');
    out+=operation(-120,240,86,66,copy("label.rmsnorm"),{key:'ff-input-norm',title:copy("title.normalize-the-residual-for-feedforward"),kind:copy("kind.read-the-updated-residual-stream"),source:'norm',labelText:copy("label.rms-norm-lines"),
      description:copy("description.read-the-residual-stream-after-attention-s-addition-and-normalize"),input,output:input,weight:param('feed_forward_norm.weight')});
    out+=flow(line(-22,273,-7,273),'normalized','ff-input-norm','ff-input');
  }
  out+=matrix(20,matrixY,115,matrixHeight,annotate?'Normalized embedding vectors':'x',input,{key:'ff-input',title:copy("title.input-to-the-feedforward-network"),labelText:asExpert?copy('moe.expert-input'):annotate?copy("label.normalized-embedding-vectors-lines"):undefined,kind:copy("kind.activations"),source:asExpert?'expert':'feedforward',activation:true,...rowInfo,description:copy("description.each-row-is-the-current-representation-at-a-sequence-position-aft"),input,note:annotate?copy("note.feedforward-rows"):undefined});
  out+=connection('M148,273 H191 V153 H239',(o,row)=>`M135,${rowY(row)} C163,${rowY(row)} 184,${153+o} 222,${153+o} H252`,'normalized','ff-input','gate')
    +connection('M191,273 V377 H239',(o,row)=>`M135,${rowY(row)} C163,${rowY(row)} 184,${377+o} 222,${377+o} H252`,'normalized','ff-input','value');
  out+=sharedOperation(252,110,193,85,copy("label.gate-projection"),{key:'gate',kind:copy("kind.learned-projection"),source:'gate',description:copy("description.expand-each-row-into-the-intermediate-dimension-using-the-same-le"),input,output:expanded,weight:ffParam('gate_proj.weight')},153);
  out+=sharedOperation(252,336,193,85,copy("label.value-projection"),{key:'value',kind:copy("kind.learned-projection"),source:'value',description:copy("description.a-second-learned-projection-expands-each-input-row-the-gate-modul"),input,output:expanded,weight:ffParam('value_proj.weight')},377);
  out+=projectionDimensions(348,228,d(),C().expanded_dim)+projectionDimensions(348,455,d(),C().expanded_dim);
  out+=connection('M454,153 L502,153',o=>`M445,${153+o} H${siluX}`,'gate','gate','silu')+sharedOperation(siluX,annotate?110:113,135,annotate?85:80,copy("label.silu"),{key:'silu',kind:copy("kind.nonlinear-activation"),source:'silu',description:copy("description.silu-multiplies-each-value-by-its-sigmoid-x-sigmoid-x-this-produc"),input:expanded,output:expanded},153);
  out+=connection('M661,153 H745 V244',o=>`M${siluX+135},${153+o} H${bendX-35-o} Q${bendX-o},${153+o} ${bendX-o},${188+o} V247`,'activatedGate','silu','product')
    +connection('M454,378 H745 V301',o=>`M445,${377+o} H${bendX-35+o} Q${bendX+o},${377+o} ${bendX+o},${342+o} V299`,'value','value','product');
  out+=`<g class="ff-downstream" transform="translate(${tailShift},0)">`;
  out+=operation(717,247,58,52,'⊙',{key:'product',title:copy("title.elementwise-gating"),kind:copy("kind.operation"),source:'silu',description:copy("description.multiply-the-silu-gate-and-the-value-path-coordinate-by-coordinat"),input:`Two ${expanded} tensors`,output:expanded,fontSize:30});
  out+=connection('M786,273 L832,273',o=>`M775,${273+o} H845`,'product','product','down');
  out+=sharedOperation(845,231,199,85,copy("label.down-projection"),{key:'down',kind:copy("kind.learned-projection"),source:'down',description:copy("description.project-each-gated-row-back-to-the-residual-stream-width-using-on"),input:expanded,output:input,weight:ffParam('output_proj.weight')},273);
  out+=projectionDimensions(945,350,C().expanded_dim,d())+connection('M1054,273 L1110,273',(o,row)=>`M1044,${273+o} H1070 C1094,${273+o} 1101,${rowY(row)} 1125,${rowY(row)}`,'feedforward','down','ff-output');
  out+=matrix(1125,annotate?matrixY:223,117,annotate?matrixHeight:100,asExpert?copy('moe.expert-output'):annotate?copy("label.contribution"):copy("label.output"),input,{key:'ff-output',title:asExpert?copy('moe.expert-output'):copy("title.a-contribution-to-the-residual-stream"),kind:copy("kind.activations"),source:asExpert?'expert':'feedforward',activation:true,...rowInfo,description:asExpert?copy('moe.expert-output-description'):copy("description.each-output-row-is-added-to-the-residual-stream-at-the-correspond"),output:input});
  out+='</g>';
  if(annotate){
    out+='<g class="ff-position-labels" aria-hidden="true">';
    for(let row=0;row<rows;row++){
      const y=matrixY+(row+.5)*matrixHeight/rows+4-(asExpert?20:0);
      const position=asExpert?state.moePosition+1:row+1;
      out+=text(8,y,position,13,'diagram-shape','end')+text(1254+tailShift,y,position,13,'diagram-shape','start');
    }
    out+='</g>';
    if(positions>rows)out+=text(78,matrixY+matrixHeight+52,copy("label.first-8-rows-shown"),16,'diagram-note')+text(1184+tailShift,matrixY+matrixHeight+52,copy("label.first-8-rows-shown"),16,'diagram-note');
  }
  if(annotate&&!asExpert){
    out+=flow(path(`M${1272+tailShift},273 H${1295+tailShift} V493`,'diagram-contribution'),'feedforward','ff-output','ff-add')+text(1270+tailShift,477,copy("label.feedforward-dce033"),21,'diagram-note','end');
    out+=stream(-215,1370+tailShift,520)
      +(isOriginalQwen?flow(path(`M-215,520 H${1275+tailShift}`,'flow-guide'),'afterAttention','','ff-add')+flow(path(`M${1315+tailShift},520 H${1370+tailShift}`,'flow-guide'),'afterFeedforward','ff-add',''):'')
      +flowBoundary(-155,476,'afterAttention','Input')+flowBoundary(1315+tailShift,595,'afterFeedforward','Output','end')+`<circle cx="-165" cy="520" r="4" class="stream-tap"/>`;
    out+=text(-142,500,copy("label.residual-r"),23,'diagram-note','start')+text(-155,561,copy("label.after-attention-s-addition"),21,'diagram-note','start');
    out+=plus(1295+tailShift,520,{key:'ff-add',title:copy("title.add-the-feedforward-contribution"),source:'residual_ff',description:copy("description.add-each-feedforward-output-row-to-the-corresponding-row-of-the-s"),input,output:input});
    out+=text(1315+tailShift,561,state.block===C().num_transformers-1?copy("label.stream-to-output"):copy("label.stream-to-next-block",{block:state.block+1}),21,'diagram-note','end');
    const captionX=590+tailShift/2;
    out+=`<text x="${captionX}" y="605" font-size="22" text-anchor="middle" class="diagram-note" ${copyAttributes(copy("caption.feedforward-diagram"))} data-copy-line-height="27">${svgTextBody(copy("caption.feedforward-diagram"),captionX,27)}</text>`;
  }
  else if(asExpert){
    out+=line(-72,273,8,273)+line(1254+tailShift,273,1390,273,'diagram-contribution')+text(1385,351,copy('moe.expert-return'),19,'diagram-note','end');
    out+=text(650,485,copy('moe.expert-caption'),22,'diagram-note');
  }
  else out+=text(650,504,copy("label.the-vector-width-expands-and-contracts-while-the-number-of-positi"),24,'diagram-note');
  return out;
}

let blockProseObserver=null;
function resizeBlockProse(){
  const prose=$('#block-prose-text'),height=isEmptyCopy(prose.textContent)?0:prose.scrollHeight;
  $('#block-prose').setAttribute('height',height);
  const blockHeight=Math.max(BLOCK.height,486+height+38),changed=blockHeight!==state.blockHeight;
  state.blockHeight=blockHeight;
  $('#block-frame').setAttribute('d',outlinePath(-15,BLOCK.frameTop,BLOCK.width,blockHeight));
  return changed;
}
function buildDiagram() {
  mobile?.invalidateMap();
  blockProseObserver?.disconnect();
  nodes.clear();serial=0;
  state.moePosition=Math.max(0,Math.min(state.moePosition,N()-1));
  const overview=overviewDiagram(), block=blockDiagram(), attention=attentionDiagram(), head=headDiagram(), ff=feedforwardDiagram(), score=(hybrid.isDelta()||isKimi()||isGLM()||isMamba()||isDeepSeek()||isGemma())?'':scoreDiagram(), output=finalOutputDiagram(), expert=(isMoE()||(hasCustomDiagrams()&&!custom().isDense()))?feedforwardDiagram(true):'', depth=hasCustomDiagrams()?custom().depth():'', indexer=isDeepSeek()?deepseek.indexer():isGLM()?glm.indexer():'', embedding=isGemma()?gemma.embedding():isGPT2()?gpt2.embedding():'';
  computeLayout();
  world.innerHTML=layer('model',overview)+layer('block',block)+layer('attention',attention)+layer('head',head)+layer('feedforward',ff)+layer('score',score)+layer('output',output)+layer('expert',expert)+layer('depth',depth)+layer('indexer',indexer)+layer('embedding',embedding);
  if(isOriginalQwen)prepareFlowTargets(world);
  for(const label of world.querySelectorAll('.token-card text')){
    if(label.getComputedTextLength()>114){label.setAttribute('textLength','114');label.setAttribute('lengthAdjust','spacingAndGlyphs');}
  }
  layers=Object.fromEntries(['model','block','attention','head','feedforward','score','output','expert','depth','indexer','embedding'].map(n=>[n,$('#layer-'+n)]));
  resizeBlockProse();
  updateLayout();renderCamera();updateChrome();
  blockProseObserver=new ResizeObserver(()=>{
    if(!resizeBlockProse())return;
    updateLayout();renderCamera();updateChrome();
    if(state.phase==='idle')fit(false).then(()=>textEditor?.reposition());
  });
  blockProseObserver.observe($('#block-prose-text'));
}
function computeLayout(){
  const layout=blockLayout(state.block,C().num_transformers,state.unfold,state.blockHeight,blockGeometry());
  transforms.model={x:0,y:0,s:1};
  const outputScale=134/1180;
  transforms.output={x:BLOCK.start+C().num_transformers*blockGeometry().step+38+layout.right,y:235-285*outputScale,s:outputScale};
  transforms.block=layout.transform;
  const attentionBox=hasCustomDiagrams()?views.kimiAttention:views.attentionLocal;
  transforms.attention=nested(transforms.block,(isGemma()?224:isMamba()?625:hasCustomDiagrams()?463:392.5)-(attentionBox.x+attentionBox.w/2)*.15,165-(attentionBox.y+attentionBox.h/2)*.15,.15);
  const headRow=hasCustomDiagrams()?{x:20,y:[165,220,275,395,555][state.head<3?state.head:state.head===headCount()-1?4:3]}:hybrid.isDelta()?{x:24,y:state.head<3?183+52*state.head:state.head>=45?483+52*(state.head-45):391}:currentAttentionLayout().rows[state.head];
  transforms.head=nested(transforms.attention,headRow.x+8,headRow.y-10,.075);
  if(isMamba()){const row=mambaHeadRows(headCount()).find(r=>r.head===state.head)||mambaHeadRows(headCount())[3];transforms.head=nested(transforms.attention,630-(views.kimiHead.x+views.kimiHead.w/2)*.075,row.y-(views.kimiHead.y+views.kimiHead.h/2)*.075,.075);}
  transforms.score=nested(transforms.head,720,227,.14);
  transforms.feedforward=nested(transforms.block,951.5-(575+(singleFeedforwardReference?0:FF_TAIL_SHIFT/2))*.18,165-340*.18,.18);
  views.model={...views.modelBase,x:views.modelBase.x+layout.left,w:views.modelBase.w+layout.growth};
  views.block=transformPoint(transforms.block,{x:-155,y:-38,w:1545,h:state.blockHeight+80});
  views.attention=transformPoint(transforms.attention,attentionBox);
  views.head=transformPoint(transforms.head,hasCustomDiagrams()?views.kimiHead:hybrid.isDelta()?views.headLocal:isHybrid()?{x:-165,y:-50,w:2630,h:1000}:{x:-165,y:-50,w:2210,h:750});
  views.score=transformPoint(transforms.score,{x:-25,y:-20,w:1190,h:610});
  views.feedforward=transformPoint(transforms.feedforward,singleFeedforwardReference?{x:-40,y:-40,w:1360,h:590}:{x:-235,y:-40,w:1630+FF_TAIL_SHIFT,h:760});
  if(isMoE())views.feedforward=transformPoint(transforms.feedforward,views.moeLocal);
  const expertRow=isMoE()?moe.route().selected.findIndex(item=>item.expert===state.expert):-1;
  transforms.expert=nested(transforms.feedforward,expertRow>=0?415:-230,expertRow>=0?350+52*expertRow:688,.10);
  views.expert=transformPoint(transforms.expert,{x:-90,y:-50,w:1510,h:625});
  views.output=transformPoint(transforms.output,{x:-170,y:-25,w:1520,h:600});
  transforms.embedding={x:185+layout.left,y:167,s:.13};
  views.embedding=transformPoint(transforms.embedding,views.gptEmbedding||{x:0,y:0,w:1,h:1});
  transforms.depth=nested(transforms.block,190,125,.13);
  views.depth=transformPoint(transforms.depth,{x:-65,y:-45,w:1760,h:785});
  transforms.indexer=nested(transforms.attention,95,655,.12);
  views.indexer=transformPoint(transforms.indexer,views.glmIndexer||{x:-145,y:-45,w:2080,h:990});
  if(hasCustomDiagrams()){
    const expertIndex=(isDeepSeek()?illustrativeDeepSeekRouting:isGLM()?illustrativeGLMRouting:illustrativeKimiRouting)(state.moePosition,state.block).findIndex(item=>item.expert===state.expert);
    const ffBox=views.kimiFeedforward,expertBox=views.kimiExpert||{x:-115,y:-45,w:1690,h:735};
    transforms.feedforward=nested(transforms.block,(isGemma()?639:isGLM()?1009:1036.5)-(ffBox.x+ffBox.w/2)*.09,165-(ffBox.y+ffBox.h/2)*.09,.09);
    transforms.expert=nested(transforms.feedforward,(state.expert===-1?-20:518)-(expertBox.x+expertBox.w/2)*.10,(state.expert===-1?722.5:310+40*Math.max(0,expertIndex))-(expertBox.y+expertBox.h/2)*.10,.10);
    transforms.depth=nested(transforms.block,(isGemma()?1001:state.depthBranch==='attention'?240:836)-(views.kimiDepth.x+views.kimiDepth.w/2)*.055,165-(views.kimiDepth.y+views.kimiDepth.h/2)*.055,.055);
    views.feedforward=transformPoint(transforms.feedforward,views.kimiFeedforward);
    views.expert=transformPoint(transforms.expert,expertBox);
    views.output=transformPoint(transforms.output,views.kimiOutput);
    views.depth=transformPoint(transforms.depth,views.kimiDepth);
  }
  return layout;
}
function updateLayout(){
  const layout=computeLayout();
  for(const[name,el]of Object.entries(layers)){
    const t=transforms[name];el.setAttribute('transform',`translate(${t.x},${t.y}) scale(${t.s})`);
  }
  $('#overview-left').setAttribute('transform',`translate(${layout.left},0)`);
  $('#overview-right').setAttribute('transform',`translate(${layout.right},0)`);
  layout.positions.forEach((x,i)=>{
    const el=$('#overview-block-'+i);el.setAttribute('transform',`translate(${x},${layout.neighborY})`);
    el.querySelector('text').style.fontSize=(12-7.8*state.unfold)+'px';
    el.querySelector('text').setAttribute('y',304-16*state.unfold);
    el.querySelector('.housing').style.strokeWidth=1.15*(1-state.unfold+state.unfold*130/state.blockHeight);
    if(i===state.block){el.querySelector('.housing').setAttribute('d',outlinePath(0,BLOCK.top,layout.width,BLOCK.closedHeight));el.style.opacity=1-state.unfold;}
  });
  // A subtle overview stream strengthens as the selected block unfolds.
  const flow=$('#overview-flow'),blockEnd=BLOCK.start+C().num_transformers*blockGeometry().step;
  $('#overview-flow-in').setAttribute('d',`M${416+layout.left},235 H${BLOCK.start-10+layout.left}`);
  $('#overview-flow-out').setAttribute('d',`M${blockEnd+5+layout.right},235 H${blockEnd+28+layout.right}`);
  flow.style.opacity=1-state.unfold;
  $('#overview-stream').style.opacity=.5+.5*state.unfold;
  for(const el of $('#overview-stream').children){
    el.setAttribute('d',((isGLM()||isDeepSeek())?[-21,-7,7,21]:[0]).map(offset=>`M${406+layout.left},${235+offset*transforms.block.s} H${BLOCK.start+C().num_transformers*blockGeometry().step+30+layout.right}`).join(' '));
    const band=el.classList.contains('stream-band'),closedWidth=band?8:1.5,openWidth=band?13:2.3;
    el.style.strokeWidth=closedWidth*(1-state.unfold)+openWidth*state.unfold*130/state.blockHeight;
  }
  const repeat=$('#overview-repeat'),right=Number(repeat.dataset.end)+layout.right,left=66+layout.left;
  repeat.setAttribute('d',`M${right},328 V405 H${left} V330`);
  // Short overlapping segments reuse the usual arrowhead without adding
  // markers at the corners of the return path.
  const first=left+(right-left)*2/3,second=left+(right-left)/3;
  const directions=[`M${right},358.5 V366.5`,`M${first+8},405 H${first}`,`M${second+8},405 H${second}`];
  $('#overview-repeat-directions').querySelectorAll('path').forEach((arrow,i)=>arrow.setAttribute('d',directions[i]));
  for(const el of world.querySelectorAll('.overview-annotation')){el.style.opacity=1-state.unfold;el.style.pointerEvents=state.unfold>.5?'none':'auto';el.setAttribute('aria-hidden',state.unfold>.5?'true':'false');}
  if(isOriginalQwen)syncFlowTargets(world);
  canvas.dataset.unfold=state.unfold.toFixed(4);
}

function mobileViewBounds(box){
  if(state.level!=='block'||!layers.block)return box;
  // Desktop leaves room for neighboring blocks. On a phone, begin at the
  // actual drawing, including the residual stream entering from the left.
  const transform=transforms.block,x=transform.x+layers.block.getBBox().x*transform.s;
  // Leave a little headroom for the first-visit note above the block border.
  const headroom=60*transform.s;
  return {...box,x,w:box.x+box.w-x,y:box.y-headroom,h:box.h+headroom};
}
function frameCamera(box) {
  if(!box)return {...state.camera};
  if(mobile?.active)return mobile.fitCamera(mobileViewBounds(box),transforms[state.level]);
  const {width,height}=canvas.getBoundingClientRect();
  const availableHeight=Math.max(180,height-155);
  const k=Math.min((width-45)/box.w,availableHeight/box.h);
  return {k,x:width/2-(box.x+box.w/2)*k,y:height/2+10-(box.y+box.h/2)*k};
}
function stopMotion(){
  cancelAnimationFrame(animation);
  if(animationResolve){animationResolve(false);animationResolve=null;}
  cameraMoving=false;
}
function setPhase(phase){hideNodeTooltip();state.phase=phase;canvas.dataset.phase=phase;if(phase==='idle')textEditor?.refresh();}
function tween(duration,step){
  stopMotion();
  if(reducedMotion||duration===0){step(1);return Promise.resolve(true);}
  const start=performance.now();
  return new Promise(resolve=>{
    animationResolve=resolve;
    function frame(now){
      const t=Math.min(1,(now-start)/duration),e=t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
      step(e);
      if(t<1)animation=requestAnimationFrame(frame);
      else{animationResolve=null;resolve(true);}
    }
    animation=requestAnimationFrame(frame);
  });
}
async function fit(animate=true) {
  if(state.phase!=='idle')return;
  await moveCamera(frameCamera(views[state.level]),animate);
}
async function moveCamera(target,animate=true,duration=620,onFrame=null) {
  const start={...state.camera};
  const rect=canvas.getBoundingClientRect(), cx=rect.width/2,cy=rect.height/2;
  const startCenter={x:(cx-start.x)/start.k,y:(cy-start.y)/start.k};
  const endCenter={x:(cx-target.x)/target.k,y:(cy-target.y)/target.k};
  const complete=await tween(animate?duration:0,e=>{
    cameraMoving=e<1;
    const k=start.k*Math.pow(target.k/start.k,e);
    state.camera={k,x:cx-(startCenter.x+(endCenter.x-startCenter.x)*e)*k,y:cy-(startCenter.y+(endCenter.y-startCenter.y)*e)*k};
    onFrame?.(e);
    renderCamera();
  });
  if(complete){state.transitionLevel=null;cameraMoving=false;renderCamera();}
  return complete;
}
async function unfoldTo(target,instant=false){
  const start=state.unfold;
  setPhase(target?'unfolding':'folding');
  return tween(instant?0:state.mobileTransition?(target?300:240):720,e=>{
    state.unfold=start+(target-start)*e;updateLayout();
    if(state.mobileTransition&&target===0)state.camera=mobile.transitionCamera(views.model,transforms.model);
    renderCamera();
  });
}
function renderCamera(){
  hideNodeTooltip();
  const {x,y,k}=state.camera;world.setAttribute('transform',`translate(${x} ${y}) scale(${k})`);
  const a=k*transforms.attention?.s;
  const hh=k*transforms.head?.s;
  const ss=k*transforms.score?.s;
  const ff=k*transforms.feedforward?.s;
  const fade=(v)=>Math.max(0,Math.min(1,(v-.16)/.14));
  const renderLevel=state.transitionLevel||state.level;
  const block=state.unfold;
  const attention=['attention','head','score','indexer'].includes(renderLevel)?fade(a):0;
  const head=['head','score'].includes(renderLevel)?fade(hh):0;
  const score=renderLevel==='score'?fade(ss):0;
  const feedforward=['feedforward','expert'].includes(renderLevel)?fade(ff):0;
  const expert=renderLevel==='expert'?fade(k*transforms.expert?.s):0;
  const output=renderLevel==='output'?fade(k*transforms.output?.s):0;
  const depth=renderLevel==='depth'?fade(k*transforms.depth?.s):0;
  const indexer=renderLevel==='indexer'?fade(k*transforms.indexer?.s):0;
  const embedding=renderLevel==='embedding'?fade(k*transforms.embedding?.s):0;
  let values={embedding,indexer,depth,model:1-Math.max(attention,feedforward,output,depth,embedding),block:block*(1-Math.max(attention,feedforward,output,depth,embedding)),attention:attention*(1-Math.max(head,indexer)),head:head*(1-score),feedforward:feedforward*(1-expert),score,output,expert};
  // Scale controls transition blending, never whether the selected view exists.
  if(state.phase==='idle'||mobile?.active){
    values=Object.fromEntries(Object.keys(layers).map(name=>[name,name===state.level?1:0]));
    if(state.level==='block'&&!mobile?.active)values.model=1;
  }
  if(state.mobileTransition){
    const {from,to,blend}=state.mobileTransition;
    values=Object.fromEntries(Object.keys(layers).map(name=>[name,0]));
    values[from]=1-blend;values[to]=blend;
    if(from==='model'&&viewPaths[to].includes('block'))values.block=state.unfold*(to==='block'?1:1-blend);
    if(to==='model'&&state.unfold)values.block=state.unfold*(from==='block'?1:blend);
  }
  for(const [name,el] of Object.entries(layers)){
    const value=values[name];el.style.opacity=value;el.style.pointerEvents=value>.45?'auto':'none';el.setAttribute('aria-hidden',value<=.45?'true':'false');
    for(const node of el.querySelectorAll('[tabindex]'))node.setAttribute('tabindex',value>.45&&!(state.unfold>.5&&node.closest('.overview-annotation'))?(node.dataset.tabstop||'0'):'-1');
  }
  // Overview stages have a different drawing scale from an unfolded block.
  // Fade them out as the camera approaches, retaining the neighboring blocks.
  const edgeOpacity=1-state.unfold*fade(k*transforms.block.s);
  for(const id of ['#overview-left','#overview-right']){
    const el=$(id),visible=values.model*edgeOpacity>.45;
    el.style.opacity=edgeOpacity;el.style.pointerEvents=visible?'auto':'none';el.setAttribute('aria-hidden',visible?'false':'true');
    for(const node of el.querySelectorAll('[tabindex]'))node.setAttribute('tabindex',visible?(node.dataset.tabstop||'0'):'-1');
  }
  if(mobile?.active)mobile.refreshMap(layers[state.level]);
  const selected=$('#overview-block-'+state.block);
  if(selected){selected.setAttribute('aria-hidden',state.unfold>.5?'true':'false');selected.style.pointerEvents=state.unfold>.5?'none':'auto';if(state.unfold>.5)selected.querySelector('[tabindex]').setAttribute('tabindex','-1');}
  renderExploreHint();
  renderBlockHint();
}
function renderBlockHint(){
  const visible=blockHintActive&&state.level==='block'&&state.phase==='idle';
  blockHint.style.display=visible?'':'none';blockHint.setAttribute('aria-hidden',String(!visible));
  const attention=layers.block?.querySelector('[data-enter="attention"] .housing');
  if(!visible||!attention)return;
  const bounds=attention.getBBox(),point=new DOMPoint(bounds.x+bounds.width/2,bounds.y).matrixTransform(attention.getCTM());
  const label=blockHint.querySelector('text'),width=$('.diagram-surface').clientWidth;
  const size=mobile?.active?20:23;
  label.textContent=`Keep ${mobile?.active?'tapping':'clicking'} to go deeper and deeper`;
  label.setAttribute('font-size',size);
  if(label.getComputedTextLength()>width-24)label.setAttribute('font-size',size*(width-24)/label.getComputedTextLength());
  const half=label.getComputedTextLength()/2;
  label.setAttribute('x',Math.max(half+12,Math.min(point.x,width-half-12)));
  label.setAttribute('y',point.y-14);
  // At smaller drawing scales, leave room for the transformer block's title.
  const title=layers.block.querySelector('.diagram-title'),frame=layers.block.querySelector('#block-frame');
  if(title&&frame){
    const note=label.getBoundingClientRect(),heading=title.getBoundingClientRect();
    if(note.left<heading.right+8&&note.right>heading.left-8&&note.top<heading.bottom+8){
      const top=new DOMPoint(0,frame.getBBox().y).matrixTransform(frame.getCTM()).y;
      label.setAttribute('y',Math.max(size+8,top-24));
    }
  }
}
function renderExploreHint(){
  const visible=!hasOpenedBlock&&state.level==='model'&&state.phase==='idle';
  exploreHint.style.display=visible?'':'none';
  exploreHint.setAttribute('aria-hidden',String(!visible));
  const block=$(`#overview-block-${exploreHintBlock} .housing`);if(!visible||!block)return;
  // Keep the lettering readable, and anchor the whole note to the drawing.
  // It scrolls away with the blocks instead of following the viewport.
  const bounds=block.getBBox(),point=new DOMPoint(bounds.x+bounds.width/2,bounds.y).matrixTransform(block.getCTM());
  const surface=$('.diagram-surface');
  const width=Math.min(244,surface.clientWidth-24),half=width/2;
  const compact=point.y<118,size=compact?20:23;
  const x=Math.max(half+12,Math.min(point.x+100,surface.clientWidth-half-12)),y=Math.max(compact?22:31,point.y-85);
  const caption=(document.body.classList.contains('mobile-layout')?'Tap':'Click')+' to look inside!';
  const label=exploreHint.querySelector('text');label.textContent=caption;label.setAttribute('x',x);label.setAttribute('y',y);label.setAttribute('font-size',size);
  exploreHint.setAttribute('aria-label',caption+` Open ${isMamba()?'Mamba':'transformer'} block ${exploreHintBlock}.`);
  const tail=x+(x>point.x?-60:70);
  const end={x:point.x,y:point.y-6},bend={x:point.x,y:point.y-35};
  exploreHint.querySelector('.explore-hint-arrow').setAttribute('d',`M${tail},${y+10} C${tail+(end.x-tail)*.2},${y+31} ${bend.x},${bend.y} ${end.x},${end.y}`);
  // An explicit open arrowhead also works on browsers without SVG context-stroke.
  const angle=Math.atan2(end.y-bend.y,end.x-bend.x),back=8,wing=3.5;
  const ax=end.x-back*Math.cos(angle),ay=end.y-back*Math.sin(angle);
  exploreHint.querySelector('.explore-hint-arrowhead').setAttribute('d',`M${ax-wing*Math.sin(angle)},${ay+wing*Math.cos(angle)} L${end.x},${end.y} L${ax+wing*Math.sin(angle)},${ay-wing*Math.cos(angle)}`);
}
async function go(level, options={}) {
  if(textEditor?.editing)return;
  if(viewPaths[level]?.includes('block')&&!hasOpenedBlock){
    hasOpenedBlock=true;exploreHint.style.display='none';
  }
  if(state.mobileTransition)finishMobileTransition();
  if(level==='score'&&(!N()||hybrid.isDelta()))level='head';
  const previousLevel=state.level, changedBlock=options.block!==undefined&&options.block!==state.block;
  if(level==='block'&&previousLevel!=='block'&&!blockHintVisited){
    blockHintVisited=true;blockHintActive=true;
  }else if(level!=='block'||changedBlock)blockHintActive=false;
  const animateMobile=mobile?.active&&!instantMobileNavigation&&!reducedMotion&&!options.instant&&previousLevel!==level&&(!changedBlock||previousLevel==='model');
  mobile?.prepareNavigation();
  const request=++navigation;stopMotion();
  state.attentionTransition=null;delete canvas.dataset.accordionProgress;delete canvas.dataset.accordionStage;
  const alreadyDrawn=!!layers[state.level];
  const instant=!!options.instant||!!mobile?.active&&!animateMobile;
  if(animateMobile){state.mobileTransition={from:previousLevel,to:level,blend:0};state.camera=mobile.beginTransition();}
  const inBlock=viewPaths[level].includes('block');
  let returnedToOverview=false;
  const lastScorePosition=Math.max(0,Math.min(N(),8)-1);
  state.scoreRow=Math.max(0,Math.min(options.row??state.scoreRow,lastScorePosition));
  state.scoreCol=Math.max(0,Math.min(options.col??state.scoreCol,lastScorePosition));
  if(level==='score')state.scoreChosen=true;
  if(!instant&&!animateMobile&&(((changedBlock||['output','embedding'].includes(level))&&state.unfold>0)||(['output','embedding'].includes(previousLevel)&&inBlock))){
    state.transitionLevel=state.level;state.level='model';setPhase('zooming');
    if(!await moveCamera(frameCamera(views.model))||request!==navigation)return;
    if(state.unfold>0&&(!await unfoldTo(0)||request!==navigation))return;
    returnedToOverview=true;
  }
  if(options.block!==undefined)state.block=options.block;
  state.head=Math.min(state.head,headCount()-1);
  if(options.head!==undefined)state.head=options.head;
  if(options.expert!==undefined)state.expert=Math.max(hasCustomDiagrams()?-1:0,Math.min(options.expert,C().num_experts-1));
  if(options.depthBranch)state.depthBranch=options.depthBranch;
  if(options.position!==undefined)state.moePosition=Math.max(0,Math.min(options.position,N()-1));
  if(!hasCustomDiagrams()&&['head','score'].includes(level)&&state.attentionMode==='foldable')state.activeHeadGroup=currentAttentionLayout().rows[state.head].disclosure;
  if(instant)state.unfold=inBlock?1:0;
  state.transitionLevel=!instant&&!returnedToOverview&&viewPaths[previousLevel].length>viewPaths[level].length?previousLevel:null;
  state.level=level;buildDiagram();selectDefault(level);
  if(animateMobile){await animateMobileNavigation(request);return;}
  if(inBlock&&state.unfold<1){
    if(!await unfoldTo(1,instant)||request!==navigation)return;
    // A short, stationary preview makes the two stages perceptually distinct.
    if(!await tween(instant?0:160,()=>{})||request!==navigation)return;
  }
  setPhase('zooming');
  if(!await moveCamera(frameCamera(views[level]),!instant)||request!==navigation)return;
  if(level==='model'&&state.unfold>0){
    if(!await unfoldTo(0,instant)||request!==navigation)return;
    if(!await moveCamera(frameCamera(views.model),!instant)||request!==navigation)return;
  }
  setPhase('idle');renderCamera();
  if(mobile?.active&&alreadyDrawn)mobile.returnToDiagram();
  if(level==='score'||(previousLevel==='score'&&level==='head')){
    layers[level].querySelector(`[data-score-row="${state.scoreRow}"][data-score-col="${state.scoreCol}"]`)?.focus({preventScroll:true});
  }
}
async function animateMobileNavigation(request){
  if(viewPaths[state.level].includes('block')&&state.unfold<1){
    if(!await unfoldTo(1)||request!==navigation)return;
    if(!await tween(60,()=>{})||request!==navigation)return;
  }
  setPhase('zooming');
  const target=mobile.transitionCamera(mobileViewBounds(views[state.level]),transforms[state.level]);
  if(!await moveCamera(target,true,320,e=>{state.mobileTransition.blend=e;})||request!==navigation)return;
  if(state.level==='model'&&state.unfold>0&&(!await unfoldTo(0)||request!==navigation))return;
  finishMobileTransition();
  mobile.returnToDiagram();
}
function finishMobileTransition(){
  if(!state.mobileTransition)return;
  const previousLevel=state.mobileTransition.from;
  ++navigation;stopMotion();state.mobileTransition=null;state.unfold=viewPaths[state.level].includes('block')?1:0;state.transitionLevel=null;
  mobile.endTransition();updateLayout();state.camera=frameCamera(views[state.level]);setPhase('idle');renderCamera();
  if(state.level==='score'||(previousLevel==='score'&&state.level==='head'))layers[state.level].querySelector(`[data-score-row="${state.scoreRow}"][data-score-col="${state.scoreCol}"]`)?.focus({preventScroll:true});
}
function selectDefault(level){
  if(hasCustomDiagrams()&&level!=='score'){select(custom().defaultSpec(level));return;}
  if(isMoE()&&level==='feedforward'){select(moe.defaultSpec());return;}
  if(isMoE()&&level==='expert'){select({...moe.expertSpec(state.expert),enter:undefined});return;}
  if(hybrid.isDelta()&&['attention','head'].includes(level)){select({key:'delta-default',title:level==='head'?`DeltaNet head ${state.head}`:'Gated DeltaNet',kind:'RECURRENT SEQUENCE MIXER',source:level==='head'?'delta_scan':'delta',parameterScope:level==='attention'?'attention':undefined,input:`${N()} × ${fmt(d())}`,output:`${N()} × ${level==='head'?128:fmt(d())}`,outputAxes:tokenAxes(level==='head'?'value-head coordinate':'residual coordinate'),description:'Each value head updates a 128 × 128 state matrix as it reads the sequence. It decays the state, writes a correction based on its key and value, then reads with its query. Three value heads share each Q/K pair. All state starts from zero on each pass.'});return;}
  if(level==='score'){select(scoreSpec());return;}
  if(level==='output'){select(finalOutputSpec());return;}
  if(level==='feedforward'&&!singleFeedforwardReference){select({key:'feedforward-detail',title:copy("title.the-feedforward-branch"),kind:copy("kind.normalize-transform-and-add"),source:'residual_ff',parameterScope:'feedforward',
    description:copy("description.this-branch-reads-the-residual-stream-after-attention-s-addition"),input:`${N()} × ${fmt(d())}`,output:`${N()} × ${fmt(d())}`});return;}
  const defaults={model:{key:'model',title:copy("title.generating-the-next-token"),kind:copy("kind.model-overview"),source:'model',description:copy("description.the-whole-context-travels-through-the-model-at-the-end-the-last-p"),note:copy("note.click-a-transformer-block-to-unfold-its-operations")},block:{title:copy("title.transformer-block-block", {block:state.block}),kind:copy("kind.one-of")+C().num_transformers+' BLOCKS',source:'block',description:copy("description.attention-lets-tokens-exchange-information-feedforward-transforms"),input:`${N()} × ${fmt(d())}`,output:`${N()} × ${fmt(d())}`},attention:{title:copy("title.combining-head-outputs"),kind:copy("kind.grouped-query-attention"),source:'attention',description:copy("description.every-head-reads-the-same-normalized-input-its-own-output-project"),note:copy("note.heads-query-heads-kvheads-k-v-pairs-headdim-dimensions-per-head", {heads:C().num_heads, kvHeads:C().num_kv_heads, headDim:h()})},head:{title:copy("title.inside-head-head", {head:state.head}),kind:copy("kind.attention-calculation"),source:'attention',outputAxes:tokenAxes('head coordinate'),description:copy("description.taking-dot-products-of-query-and-key-vectors-produces-scores-afte"),input:`${N()} × ${fmt(d())}`,output:`${N()} × ${h()}`,note:copy("note.uses-k-v-pair-kvhead-all-heads-are-computed-together-in-the-pytho", {kvHead:kvHead()})},feedforward:{title:copy("title.the-feedforward-network"),kind:copy("kind.the-feedforward-network"),source:'feedforward',description:copy("description.two-learned-projections-expand-the-current-vector-at-each-sequenc"),input:`${N()} × ${fmt(d())}`,output:`${N()} × ${fmt(d())}`}};
  select({...defaults[level],parameterScope:level==='feedforward'?'feedforward-core':level});
}
function up(){const parent=viewPaths[state.level].at(-2);if(parent)go(parent);}
function updateChrome(){
  $('.work-area').classList.toggle('kimi-routing-view',isKimi()&&state.level==='feedforward'&&!kimi.isDense());
  $('.work-area').classList.toggle('kimi-depth-view',hasCustomDiagrams()&&state.level==='depth');
  $('.work-area').classList.toggle('feedforward-view',['feedforward','expert'].includes(state.level)&&!singleFeedforwardReference);
  $('.work-area').classList.toggle('expanded-block-view',state.level==='block'&&state.blockHeight>BLOCK.height+20);
  const levels=viewPaths[state.level];
  const names={model:'Model Overview',embedding:'Input embeddings',block:`Block ${state.block}`,attention:isMamba()?'State-space mixer':hybrid.isDelta()?'DeltaNet':'Attention',head:`Head ${state.head}`,feedforward:'Feedforward',score:`Score (${state.scoreRow+1}, ${state.scoreCol+1})`,output:'Output projection',expert:state.expert===-1?((isGLM()||isDeepSeek())?'Shared expert':'Shared experts'):`Expert ${state.expert}`,depth:isGemma()?'Per-layer embedding':isDeepSeek()?(state.depthBranch==='memory'?'Engram memory':'Single-pass mHC'):isGLM()?'mHC':'Attention Residuals',indexer:isDeepSeek()?'CSA2 index and sharing':'Pool indexer'};
  $('#breadcrumbs').innerHTML=levels.map((l,i)=>`${i?'<span>/</span>':''}<button data-level="${l}">${names[l]}</button>`).join('');
  $('#up').disabled=state.level==='model';
  const viewLabels={
    model:'01 / MODEL OVERVIEW',output:'02 / OUTPUT PROJECTION',block:'02 / A TRANSFORMER BLOCK',
    attention:'03 / GROUPED-QUERY ATTENTION',head:'04 / INSIDE AN ATTENTION HEAD',
    feedforward:'03 / THE FEEDFORWARD NETWORK',score:'05 / A SINGLE ATTENTION SCORE'
  };
  if(isMamba()){viewLabels.block='02 / A MAMBA BLOCK';viewLabels.attention='03 / STATE-SPACE MIXER';viewLabels.head='04 / INSIDE A STATE-SPACE HEAD';}
  if(isGPT2()){viewLabels.embedding='02 / TOKEN AND POSITION EMBEDDINGS';viewLabels.attention='03 / MULTI-HEAD ATTENTION';viewLabels.feedforward='03 / FEEDFORWARD';}
  if(isGemma()){viewLabels.attention='03 / '+gemma.attentionName().toUpperCase();viewLabels.feedforward='03 / GATED FEEDFORWARD';viewLabels.depth='03 / PER-LAYER EMBEDDING';}
  if(isDeepSeek()){viewLabels.attention='03 / '+deepseek.attentionName().toUpperCase();viewLabels.feedforward='03 / MIXTURE OF EXPERTS';viewLabels.depth=state.depthBranch==='memory'?'03 / ENGRAM MEMORY':'03 / SINGLE-PASS mHC';viewLabels.indexer='04 / CSA2 INDEX AND SHARING';}
  if(isGLM()){viewLabels.attention='03 / '+glm.attentionName().toUpperCase();viewLabels.feedforward=glm.isDense()?'03 / DENSE FEEDFORWARD':'03 / MIXTURE OF EXPERTS';viewLabels.expert='04 / INSIDE AN EXPERT';viewLabels.depth='03 / MIXING RESIDUAL STREAMS';viewLabels.indexer='04 / POOL INDEXER';}
  if(isKimi()){viewLabels.attention='03 / '+kimi.attentionName().toUpperCase();viewLabels.feedforward=kimi.isDense()?'03 / DENSE FEEDFORWARD':'03 / LATENT MIXTURE OF EXPERTS';viewLabels.expert='04 / INSIDE AN EXPERT';viewLabels.depth='03 / ATTENTION ACROSS DEPTH';}
  if(isMoE()){viewLabels.feedforward='03 / MIXTURE OF EXPERTS';viewLabels.expert='04 / INSIDE A FEEDFORWARD EXPERT';}
  if(hybrid.isDelta()){viewLabels.attention='03 / GATED DELTANET';viewLabels.head='04 / INSIDE A DELTANET HEAD';}
  $('#view-number').textContent=viewLabels[state.level];textEditor?.refresh();
  $('#minimap').innerHTML=Array.from({length:C().num_transformers},(_,b)=>{
    const kind=blockTypeLabel(b),colored=isHybrid()||isKimi()||isGLM()||isGemma()||isDeepSeek();
    const warm=isDeepSeek()?b<C().encoder_layers:C().layer_types?.[b]!=='full_attention';
    const tint=colored?(warm?'delta-block':'full-attention-block'):'';
    return `<button data-block="${b}" class="${b===state.block&&levels.includes('block')?'active ':''}${tint}" aria-label="Go to block ${b}" title="${esc(`Block ${b}${kind?': '+kind:''}`)}"></button>`;
  }).join('');
  mobile?.updateChrome(names,levels);
}
function detailAxes(spec,direction){
  if(spec[direction+'Axes'])return spec[direction+'Axes'];
  if(!spec[direction]?.includes('×'))return null;
  const residual=tokenAxes(),head=tokenAxes('head coordinate'),expanded=tokenAxes(isMoE()?'expert intermediate coordinate':'expanded coordinate');
  const scores={rows:'query position',columns:'key position'};
  const queryOrKey=tokenAxes(spec.key?.startsWith('q-')?'query coordinate':'key coordinate');
  // These describe the tensors used by each operation, independently of its
  // learned parameter layout and without guessing from equal-sized dimensions.
  const axes={
    embedding:[null,tokenAxes('embedding coordinate')],
    output:[residual,tokenAxes('vocabulary entry')],
    last:[tokenAxes('vocabulary entry'),null],
    attention:[residual,spec.enter==='head'?head:residual],
    q:[residual,tokenAxes('query coordinate')],
    k:[residual,tokenAxes('key coordinate')],
    v:[residual,tokenAxes('value coordinate')],
    qk_norm:[queryOrKey,queryOrKey],rope:[queryOrKey,queryOrKey],
    scores:['Q: rows are query positions; columns are head coordinates. Kᵀ: rows are head coordinates; columns are key positions.',scores],
    mask:[scores,scores],softmax:[scores,scores],
    mix:['Attention weights: rows are query positions; columns are key positions. V: rows are token positions; columns are value coordinates.',head],
    sharing:[residual,head],combine:[spec.key==='combine'?residual:head,residual],
    gate:[residual,expanded],value:[residual,expanded],silu:[expanded,expanded],down:[expanded,residual],
  };
  return axes[spec.source]?.[direction==='input'?0:1]??residual;
}
function detailShape(label,shape,axes){
  if(!shape)return '';
  const explanation=!axes?'':typeof axes==='string'||axes instanceof CopyText?axes:axes.vector?copy("explanation.vector-axes",{columns:axes.columns}):copy("explanation.matrix-axes",{rows:axes.rows,columns:axes.columns});
  return `<div class="detail-shape"><small>${esc(label)}</small>${esc(shape)}${explanation?`<p class="detail-axes" ${copyAttributes(explanation)}>${richText(explanation)}</p>`:''}</div>`;
}
function vocabularyNote(spec){
  if(!['tokenizer','embedding','output','output_stage','last','sampling'].includes(spec.source)||!state.vocabularySize)return '';
  const entries=state.vocabularySize,rows=C().vocabulary_size,padding=rows-entries;
  return `<details class="sidebar-note"><summary>Vocabulary and table sizes</summary><p>The tokenizer has ${fmt(entries)} entries. The model’s embedding and output tables each have ${fmt(rows)} rows.${padding>0?` The extra ${fmt(padding)} rows pad the tables; they do not represent additional tokenizer entries.`:''}</p></details>`;
}
function parameterDetails(spec){
  const scope=spec.parameterScope||(['block','attention','feedforward','output'].includes(spec.enter)?spec.enter:null);
  const stats=parameterSummary(state.model.parameters,C(),{scope,block:spec.block??state.block,weight:spec.weight,expert:spec.expert??state.expert,outputHead:!!spec.outputHead});
  if(!stats)return '';
  return `<section class="parameter-summary" aria-label="Parameter count and weight storage"><small>${esc(stats.label)}</small><strong>${fmt(stats.count)}</strong>`
    +`<dl class="parameter-storage"><dt>16-bit weights</dt><dd>${formatBytes(parameterBytes(stats.count,16))}</dd><dt>32-bit weights</dt><dd>${formatBytes(parameterBytes(stats.count,32))}</dd></dl>`
    +`<p class="detail-axes" ${copyAttributes(copy("explanation.weight-storage"))}>${richText(copy("explanation.weight-storage"))}</p>`
    +(stats.groups.length?`<details class="parameter-breakdown"${scope==='kv-pair'?' open':''}><summary>Where the parameters are</summary><dl>${stats.groups.map(g=>`<div><dt>${esc(g.label)}</dt><dd>${fmt(g.count)}</dd></div>`).join('')}</dl>${stats.note?`<p>${esc(stats.note)}</p>`:''}</details>`:stats.note?`<p class="detail-axes">${esc(stats.note)}</p>`:'')
    +'</section>';
}
function select(spec){
  mobile?.clearReadout();mobile?.selectionChanged(spec);
  state.selected=spec;
  $('.inspector').scrollTop=0;
  setCopyText($('#selection-title'),spec.title);
  setCopyText($('#selection-description'),spec.description||'');
  const learnedShape=spec.outputHead?`${fmt(d())} × ${h()}`:spec.weight?matrixShape(spec.weight,state.model.parameters[spec.weight].length===3&&(/\.W_[qkv]$/.test(spec.weight))):'';
  const kvPair=spec.parameterScope==='kv-pair';
  const learnedDetails=kvPair?['k','v'].map(letter=>{
    const weight=spec.weight.replace(/\.W_k$/,`.W_${letter}`);
    return detailShape(letter==='k'?'SHARED KEY MATRIX':'SHARED VALUE MATRIX',matrixShape(weight,true),parameterAxes(weight,false,C()));
  }).join(''):spec.weight?detailShape(spec.outputHead?'LEARNED MATRIX SLICE':'LEARNED PARAMETER',learnedShape,parameterAxes(spec.weight,!!spec.outputHead,C())):'';
  $('#selection-details').innerHTML=detailShape('INPUT',spec.input,detailAxes(spec,'input'))+detailShape('OUTPUT',spec.output,detailAxes(spec,'output'))+learnedDetails+parameterDetails(spec)+(spec.note&&!isEmptyCopy(spec.note)?`<p class="detail-note" ${copyAttributes(spec.note)}>${richText(spec.note)}</p>`:'')+vocabularyNote(spec)
    +(spec.key?.startsWith('score-')&&state.level==='score'?`<p class="detail-note">In Python: <code>scores[${state.head}, ${state.scoreRow}, ${state.scoreCol}]</code>. ${isGPT2()?'Each head has its own query and key projections.':'Key heads have been repeated to match the query heads.'} Coordinates 1–${h()} correspond to Python indices 0–${h()-1}.</p><p class="detail-note">Choose another score in the matrix above. With the keyboard, use the arrow keys and Enter.</p>`:'');
  setCopyText($('#notation-description'),copy("explanation.notation",{positions:positionConvention()}));
  $('#selection-actions').innerHTML=(spec.weight&&state.model.weight_inspection!==false?`<button class="primary" id="inspect-weights">⌕ ${kvPair?'Inspect key weights':'Inspect real weights'}</button>`:'')+`<button id="selection-code"><span class="code-glyph">&lt;/&gt;</span> See the Python</button>`+(state.level==='head'?`<button id="previous-head" ${state.head===0?'disabled':''}>← Previous head</button><button id="next-head" ${state.head===headCount()-1?'disabled':''}>Next head →</button>`:'');
  $('#selection-code').onclick=()=>openCode(spec.source||'model');
  if(spec.weight&&state.model.weight_inspection!==false)$('#inspect-weights').onclick=()=>openWeight(spec);
  if($('#previous-head'))$('#previous-head').onclick=()=>go('head',{head:state.head-1});
  if($('#next-head'))$('#next-head').onclick=()=>go('head',{head:state.head+1});
  $('#moe-controls')?.remove();
  $('#kimi-controls')?.remove();
  if((isKimi()||isGLM()||isDeepSeek())&&['feedforward','expert','indexer'].includes(state.level)){
    $('#selection-description').insertAdjacentHTML('beforebegin',`<div id="kimi-controls"><label class="moe-control-label">Sequence position<select id="kimi-position">${Array.from({length:N()},(_,i)=>`<option value="${i}" ${i===state.moePosition?'selected':''}>Position ${i+1}</option>`).join('')}</select></label></div>`);
    $('#kimi-position').onchange=()=>go(state.level,{position:Number($('#kimi-position').value),instant:true});
    if(state.level==='expert'){
      $('#kimi-controls').insertAdjacentHTML('beforeend',`<label class="moe-control-label">Expert<select id="kimi-expert"><option value="-1" ${state.expert===-1?'selected':''}>${(isGLM()||isDeepSeek())?'Shared expert':'Shared experts'}</option>${Array.from({length:C().num_experts},(_,i)=>`<option value="${i}" ${i===state.expert?'selected':''}>Expert ${i}</option>`).join('')}</select></label>`);
      $('#kimi-expert').onchange=()=>go('expert',{expert:Number($('#kimi-expert').value),instant:true});
    }
  }
  if(isMoE()&&['feedforward','expert'].includes(state.level)){
    $('#selection-description').insertAdjacentHTML('beforebegin','<div id="moe-controls"></div>');
    $('#moe-controls').insertAdjacentHTML('beforeend',`<label class="moe-control-label">Sequence position<select id="moe-position" ${N()?'':'disabled'}>${Array.from({length:N()},(_,i)=>`<option value="${i}" ${i===state.moePosition?'selected':''}>Position ${i+1}</option>`).join('')}</select></label>`);
    $('#moe-position').onchange=()=>go(state.level,{position:Number($('#moe-position').value),instant:true});
    if(state.level==='expert'){
      $('#moe-controls').insertAdjacentHTML('beforeend',`<label class="moe-control-label">Expert<select id="moe-expert">${Array.from({length:C().num_experts},(_,i)=>`<option value="${i}" ${i===state.expert?'selected':''}>Expert ${i}</option>`).join('')}</select></label>`);
      $('#moe-expert').onchange=()=>go('expert',{expert:Number($('#moe-expert').value),instant:true});
    }
  }
  for(const [id,node]of nodes){const el=world.querySelector(`[data-node="${id}"]`);el?.classList.toggle('selected',!!spec.key&&node.key===spec.key);}
  state.sourceKey=spec.source||'model';if(state.codeOpen)loadSource(state.sourceKey);textEditor?.refresh();
}

async function loadSource(key){
  const request=++state.sourceRequest;
  try{const source=await provider.source(key);if(request!==state.sourceRequest)return;state.source=source;renderSource();}
  catch(error){if(request===state.sourceRequest)$('#source-code').textContent=error.message;}
}
function highlightPython(line){
  const pattern=/(#.*$|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b(?:def|class|return|for|in|if|else|elif|from|import|with|as|True|False|None|self)\b|\b\d+(?:\.\d+)?\b)/g;
  let out='',last=0;for(const match of line.matchAll(pattern)){out+=esc(line.slice(last,match.index));const t=match[0],kind=t.startsWith('#')?'comment':/^["']/.test(t)?'string':/^\d/.test(t)?'number':'keyword';out+=`<span class="py-${kind}">${esc(t)}</span>`;last=match.index+t.length;}return out+esc(line.slice(last));
}
function renderSource(){
  const source=state.source;if(!source)return;
  $('#code-file').textContent=source.file;$('#code-file').title=`${source.repository_path} · ${source.commit}`;$('#code-symbol').textContent=source.symbol;
  $('#code-note').textContent=['head','score'].includes(state.level)?`Head ${state.head} uses this same code; PyTorch computes the heads together.`:'';
  if(hasCustomDiagrams()&&state.level==='expert'&&state.expert===-1)$('#code-note').textContent=`${(isGLM()||isDeepSeek())?'The shared expert uses':'Shared experts use'} transformer_blocks.${state.block}.feed_forward.shared_experts.`;
  else if(state.level==='expert')$('#code-note').textContent=`Expert ${state.expert} uses transformer_blocks.${state.block}.feed_forward.experts.${state.expert}. All experts use this class, with independently learned weights.`;
  if(state.level==='score'&&state.sourceKey==='scores')$('#code-note').textContent=`The selected scalar is scores[${state.head}, ${state.scoreRow}, ${state.scoreCol}]. ${isGPT2()?'Q and K include their projection biases. Each attention head has its own keys.':'Here queries and keys are already normalized and rotated; shared key heads have been repeated to match query heads.'}`;
  if(state.selected?.outputHead){const head=state.selected.weightHead;$('#code-note').textContent=`Head ${head} uses out_proj.weight[:, ${head*h()}:${(head+1)*h()}]. The highlighted Python applies all slices together; the diagram shows the equivalent sum of per-head projections.`;}
  else if(state.selected?.key==='combine')$('#code-note').textContent='The diagram projects each head separately and sums the results. The highlighted code evaluates the same expression by concatenating first and applying one linear projection.';
  const all=$('#code-context').checked, start=all?1:Math.max(source.context_start,source.start-3),end=all?source.lines.length:Math.min(source.context_end,source.end+3);
  $('#source-code').innerHTML=source.lines.slice(start-1,end).map((line,i)=>{const n=start+i;return `<span class="code-line${n>=source.start&&n<=source.end?' highlight':''}"><span class="line-number">${n}</span>${highlightPython(line)}</span>`;}).join('');
  const highlighted=$('#source-code .highlight');if(highlighted)$('#source-code').scrollTop=highlighted.offsetTop-$('#source-code').offsetTop-35;
}
async function openCode(key=state.sourceKey){
  hideNodeTooltip();
  state.codeOpen=true;$('#code-drawer').classList.add('open');$('#code-content').hidden=false;$('#code-toggle').setAttribute('aria-expanded','true');$('#code-chevron').textContent='⌃';
  await loadSource(key);
  if(state.codeOpen)$('#code-drawer').scrollIntoView({block:'nearest',behavior:reducedMotion?'instant':'smooth'});
}
function closeCode(){
  if($('#code-content').contains(document.activeElement))$('#code-toggle').focus({preventScroll:true});
  state.codeOpen=false;$('#code-drawer').classList.remove('open');$('#code-content').hidden=true;$('#code-toggle').setAttribute('aria-expanded','false');$('#code-chevron').textContent='⌄';
}

async function tokenizePrompt(){
  const example=provider.capabilities.customContext?null:state.examples.find(e=>e.id===$('#example-select').value);
  const request=++state.tokenRequest,prompt=example?.text??$('#prompt').value;$('#token-count').textContent='Reading…';
  try{const [result,next]=example?[example,example.continuation]:await Promise.all([provider.tokenize(prompt),prompt===EXAMPLE_CONTEXT?provider.tokenize(' end'):null]);if(request!==state.tokenRequest)return;
    state.exampleNextId=next?.ids.length===1?next.ids[0]:null;
    state.exampleContext=prompt===EXAMPLE_CONTEXT;$('#example-next').hidden=!state.exampleContext;
    state.tokens=result.ids;state.pieces=result.pieces;state.vocabularySize=result.vocabulary_size;state.token=Math.max(0,result.ids.length-1);
    refreshCopyElements();
    $('#tokens').innerHTML=result.ids.map((id,i)=>`<button class="token${i===state.token?' selected':''}" data-token="${i}" title="Position ${i+1} · Python index ${i} · token ID ${id}">${esc(result.pieces[i].replaceAll('Ġ','␣').replaceAll('Ċ','↵').replaceAll('▁',isGemma()?'␣':'▁'))}<small>${id}</small></button>`).join('');
    $('#token-count').textContent=`N = ${result.ids.length}`;
    if(!result.ids.length){$('#tokens').textContent='Enter some text to see its token IDs.';}
    if(state.model){const selectedKey=state.selected?.key;await go(state.level,{instant:true});const updated=[...nodes.values()].find(n=>selectedKey&&n.key===selectedKey);if(updated)select(updated);}
  }catch(error){if(request===state.tokenRequest)$('#token-count').textContent=error.message;}
}

function hideNodeTooltip(){
  if(isOriginalQwen&&!mobile?.hasReadout){
    clearFlowHighlight(world);
    const readout=$('#connection-readout');
    if(readout)readout.dataset.active='false';
  }
  $('#node-tooltip').hidden=true;
  for(const target of canvas.querySelectorAll('[aria-describedby="node-tooltip"],[aria-describedby="connection-readout"]'))target.removeAttribute('aria-describedby');
}
function showNodeTooltip(event){
  if(mobile?.active&&!['click','keydown'].includes(event.type))return;
  const target=event.target?.closest('[data-tooltip]');
  // Moving along the same path keeps the readout and highlights steady.
  if(target?.matches('.flow-active')&&state.phase==='idle')return;
  hideNodeTooltip();
  if(!target||target.closest('[aria-hidden="true"]')||state.phase!=='idle'||textEditor?.editing||event.target?.closest('[data-copy-editable]'))return;
  if(mobile?.active){
    clearFlowHighlight(world);mobile.showReadout(target);
    if(target.matches('.tensor-flow'))highlightFlow(target,world,nodes);
    return;
  }
  if(target.matches('.tensor-flow')){
    const readout=$('#connection-readout');
    readout.querySelector('[data-flow-name]').textContent=target.dataset.flowName;
    readout.querySelector('[data-flow-shape]').textContent=target.dataset.flowShape;
    readout.querySelector('[data-flow-meaning]').textContent=target.dataset.flowMeaning;
    readout.dataset.active='true';
    highlightFlow(target,world,nodes);
    target.setAttribute('aria-describedby','connection-readout');
    return;
  }
  const tooltip=$('#node-tooltip'),rect=target.getBoundingClientRect();
  tooltip.textContent=target.dataset.tooltip;tooltip.hidden=false;
  tooltip.style.left=Math.max(8,Math.min(innerWidth-tooltip.offsetWidth-8,rect.x+rect.width/2-tooltip.offsetWidth/2))+'px';
  tooltip.style.top=Math.max(8,rect.top-tooltip.offsetHeight-7)+'px';
  target.setAttribute('aria-describedby','node-tooltip');
}
canvas.addEventListener('pointerover',showNodeTooltip);
canvas.addEventListener('focusin',showNodeTooltip);
canvas.addEventListener('pointerout',event=>showNodeTooltip({target:event.relatedTarget,clientX:event.clientX,clientY:event.clientY}));
canvas.addEventListener('pointermove',event=>{if(event.target.closest('.tensor-flow'))showNodeTooltip(event);});
canvas.addEventListener('focusout',()=>{if(!mobile?.active)hideNodeTooltip();});
window.addEventListener('scroll',()=>{
  if(mobile?.active)return;
  const focused=document.activeElement;
  if(focused?.matches(':focus-visible'))showNodeTooltip({target:focused});else hideNodeTooltip();
},{passive:true});

// Event delegation keeps diagram actions and both icons keyboard accessible.
let interruptedTransitionUntil=0,interruptedTransitionPointer=null;
document.addEventListener('pointerdown',event=>{
  if(!state.mobileTransition||!event.target.closest('.diagram-surface,#mobile-map'))return;
  interruptedTransitionPointer=event.pointerId;
  interruptedTransitionUntil=performance.now()+400;
  finishMobileTransition();
},true);
for(const type of ['pointerup','pointercancel'])document.addEventListener(type,event=>{
  if(event.pointerId!==interruptedTransitionPointer)return;
  interruptedTransitionPointer=null;interruptedTransitionUntil=performance.now()+400;
},true);
canvas.addEventListener('click',event=>{
  if(interruptedTransitionPointer!==null||performance.now()<interruptedTransitionUntil){event.preventDefault();event.stopImmediatePropagation();}
},true);
window.addEventListener('resize',finishMobileTransition);
installZoomOutGestures(canvas,{enabled:()=>!mobile?.active,canGoUp:()=>!!state.model&&state.level!=='model',isBusy:()=>state.phase!=='idle'||!!textEditor?.editing,goUp:up});
const selectingDiagramText=installSelectionClickGuard(canvas);
canvas.addEventListener('click',(event)=>{
  if(selectingDiagramText(event)||state.phase==='folding-heads')return;
  if(event.target.closest('.tensor-flow')||(mobile?.active&&event.target.closest('.dimension-label,.flow-boundary'))){showNodeTooltip(event);return;}
  mobile?.clearReadout();
  hideNodeTooltip();
  const weight=event.target.closest('[data-weight]');if(weight){const n=nodes.get(weight.dataset.weight);select(n);openWeight(n);return;}
  const code=event.target.closest('[data-code]');if(code){const n=nodes.get(code.dataset.code);select(n);openCode(n.source);return;}
  const score=event.target.closest('[data-score-row]');if(score){go('score',{row:Number(score.dataset.scoreRow),col:Number(score.dataset.scoreCol),instant:state.level==='score'});return;}
  const el=event.target.closest('[data-node]');if(!el)return;const node=nodes.get(el.dataset.node);
  if(node.action)changeAttentionView(node.action,node.group);else if(node.enter)go(node.enter,node);else select(node);
});
canvas.addEventListener('keydown',(event)=>{
  if(event.target.closest('.tensor-flow')&&['Enter',' '].includes(event.key)){event.preventDefault();showNodeTooltip(event);return;}
  const score=event.target.closest('[data-score-row]');
  if(score&&['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End'].includes(event.key)){
    event.preventDefault();const last=Math.min(N(),8)-1;
    let row=Number(score.dataset.scoreRow),col=Number(score.dataset.scoreCol);
    if(event.key==='ArrowUp')row=Math.max(0,row-1);if(event.key==='ArrowDown')row=Math.min(last,row+1);
    if(event.key==='ArrowLeft')col=Math.max(0,col-1);if(event.key==='ArrowRight')col=Math.min(last,col+1);
    if(event.key==='Home')col=0;if(event.key==='End')col=last;
    const matrix=score.closest('.attention-matrix'),next=matrix.querySelector(`[data-score-row="${row}"][data-score-col="${col}"]`);
    for(const cell of matrix.querySelectorAll('[data-score-row]')){cell.dataset.tabstop=cell===next?'0':'-1';cell.setAttribute('tabindex',cell.dataset.tabstop);}
    next?.focus({preventScroll:true});return;
  }
  if(event.key==='Enter'||event.key===' '){const el=event.target.closest('[data-code],[data-weight],[data-score-row]')||event.target.closest('[data-node]');if(el){event.preventDefault();el.dispatchEvent(new MouseEvent('click',{bubbles:true}));}}
});
$('#breadcrumbs').onclick=(event)=>{const b=event.target.closest('[data-level]');if(b)go(b.dataset.level);};
$('#minimap').onclick=(event)=>{const b=event.target.closest('[data-block]');if(b)go('block',{block:Number(b.dataset.block)});};
$('#up').onclick=up;

function installInlineNotes(){
  const popup=$('#inline-note'),body=$('#inline-note-text'),editor=$('#text-editor');
  let trigger=null,pinned=false,openTimer=0,closeTimer=0,returningFocus=false,touchTrigger=null;
  const noteAt=target=>target?.closest?.('[data-inline-note]');
  const contains=(element,target)=>!!target&&element?.contains(target);
  function clearTimers(){clearTimeout(openTimer);clearTimeout(closeTimer);}
  function position(){
    if(!trigger||popup.hidden)return;
    const anchor=trigger.getBoundingClientRect();
    popup.style.maxWidth=Math.max(0,document.documentElement.clientWidth-24)+'px';
    const bounds=popup.getBoundingClientRect();
    const left=Math.max(12,Math.min(innerWidth-bounds.width-12,anchor.left+anchor.width/2-bounds.width/2));
    let top=anchor.bottom+9;
    if(top+bounds.height>innerHeight-12)top=anchor.top-bounds.height-9;
    popup.style.left=left+'px';popup.style.top=Math.max(12,Math.min(innerHeight-bounds.height-12,top))+'px';
  }
  function describe(){
    const interactive=!!body.querySelector('a[href],button');
    popup.setAttribute('role',interactive?'dialog':'tooltip');
    if(interactive){
      popup.setAttribute('aria-label',`Note: ${trigger.textContent.trim()}`);
      trigger.setAttribute('aria-haspopup','dialog');
    }else{popup.removeAttribute('aria-label');trigger?.removeAttribute('aria-haspopup');}
    trigger?.setAttribute('aria-expanded','true');
    trigger?.setAttribute('aria-describedby','inline-note-text');
  }
  function hide(){
    clearTimers();popup.hidden=true;pinned=false;
    trigger?.setAttribute('aria-expanded','false');trigger?.removeAttribute('aria-describedby');trigger=null;
  }
  function show(next,pin=false){
    clearTimers();
    if(textEditor?.editing||!next?.isConnected)return;
    if(trigger!==next){
      hide();trigger=next;setCopyText(body,copy(next.dataset.inlineNote));textEditor?.refresh();
    }
    pinned=pinned||pin;popup.hidden=false;describe();position();
  }
  function scheduleClose(){
    clearTimers();closeTimer=setTimeout(()=>{
      if(pinned||textEditor?.editing||trigger?.matches(':hover')||popup.matches(':hover')||document.activeElement===trigger||contains(popup,document.activeElement))return;
      hide();
    },180);
  }
  function toggle(next,focusNote=false){
    if(trigger===next&&pinned)hide();
    else{show(next,true);if(focusNote)body.querySelector('a[href],button')?.focus({preventScroll:true});}
  }
  document.addEventListener('pointerover',event=>{
    if(event.pointerType==='touch'||textEditor?.editing)return;
    const next=noteAt(event.target);
    if(next){
      if(contains(next,event.relatedTarget))return;
      clearTimers();if(trigger!==next)openTimer=setTimeout(()=>show(next),180);
    }else if(contains(popup,event.target))clearTimers();
  });
  document.addEventListener('pointerout',event=>{
    const next=noteAt(event.target);
    if(next&&!contains(next,event.relatedTarget)||contains(popup,event.target)&&!contains(popup,event.relatedTarget))scheduleClose();
  });
  // Opening on touch focus can cover the word before its synthetic click lands.
  // Let that completed click open the note instead. Keyboard focus still opens it.
  document.addEventListener('pointerdown',event=>{touchTrigger=event.pointerType==='touch'?noteAt(event.target):null;},true);
  document.addEventListener('pointercancel',()=>{touchTrigger=null;},true);
  document.addEventListener('keydown',()=>{touchTrigger=null;},true);
  document.addEventListener('focusin',event=>{
    const next=noteAt(event.target);
    if(next&&!returningFocus&&next!==touchTrigger)show(next);
    else if(contains(popup,event.target))clearTimers();
  });
  document.addEventListener('focusout',event=>{
    if(noteAt(event.target)||contains(popup,event.target))scheduleClose();
  });
  // Install before the text editor: its containing introduction remains editable,
  // while the note word itself opens the separately editable passage.
  document.addEventListener('click',event=>{
    touchTrigger=null;
    const next=noteAt(event.target);
    if(next&&!textEditor?.editing){event.preventDefault();event.stopImmediatePropagation();toggle(next,event.detail===0);return;}
    if(contains(popup,event.target)||contains(editor,event.target)||event.target.closest('#edit-text,#save-text'))return;
    hide();
  },true);
  document.addEventListener('keydown',event=>{
    if(contains(editor,event.target))return;
    const next=noteAt(event.target);
    if(next&&!textEditor?.editing&&(event.key==='Enter'||event.key===' ')){
      event.preventDefault();event.stopImmediatePropagation();toggle(next,true);return;
    }
    if(event.key==='Escape'&&!popup.hidden){
      event.preventDefault();event.stopImmediatePropagation();
      const previous=trigger,focusInside=contains(popup,document.activeElement);hide();
      if(focusInside){returningFocus=true;previous?.focus({preventScroll:true});returningFocus=false;}
    }
  },true);
  window.addEventListener('resize',position);
  window.addEventListener('scroll',position,{passive:true});
  new ResizeObserver(position).observe(body);
  // Editing redraws the introduction, so keep an open note attached to the
  // replacement word after Apply, Cancel, or another passage's edit.
  new MutationObserver(()=>{
    if(trigger&&!trigger.isConnected){
      const key=trigger.dataset.inlineNote;
      trigger=[...document.querySelectorAll('[data-inline-note]')].find(el=>el.dataset.inlineNote===key);
      if(!trigger){hide();return;}
      describe();
    }
    position();
  }).observe($('.intro-copy'),{childList:true,subtree:true,characterData:true});
}

// Inline [code] markers and Markdown links also work in editable prose. In
// edit mode the text editor captures them as part of their containing passage.
document.addEventListener('click',event=>{
  if(textEditor?.enabled)return;
  const code=event.target.closest('[data-copy-code]');
  if(code){
    event.preventDefault();event.stopImmediatePropagation();
    const node=nodes.get(code.closest('[data-node]')?.dataset.node);
    if(node)select(node);
    openCode(node?.source||'model');
  }else if(event.target.closest('[data-copy-link]'))event.stopPropagation();
},true);
$('#code-toggle').onclick=()=>state.codeOpen?closeCode():openCode();$('#code-close').onclick=closeCode;$('#code-context').onchange=renderSource;$('#code-wrap').onchange=()=>$('#source-code').classList.toggle('wrap', $('#code-wrap').checked);
$('#prompt-form').onsubmit=(event)=>{event.preventDefault();tokenizePrompt();};
$('#tokens').onclick=(event)=>{const b=event.target.closest('[data-token]');if(!b)return;state.token=Number(b.dataset.token);for(const token of $('#tokens').children)token.classList.toggle('selected',token===b);select({title:copy("title.position-value1", {value1:state.token+1}),kind:copy("kind.one-token-in-the-sequence"),source:'embedding',note:copy("note.position-value1-is-sequence-index-value2-in-python-token-ids-and", {value1:state.token+1, value2:state.token}),description:copy("description.token-id-value1-selects-row-value2-of-the-embedding-table-its-emb", {value1:state.tokens[state.token], value2:state.tokens[state.token], embeddingDim:fmt(d())}),output:`1 × ${fmt(d())}`,weight:'token_embedding_layer.weight',weightRow:state.tokens[state.token]});if(mobile?.active)$('.inspector').scrollIntoView({block:'start',behavior:reducedMotion?'instant':'smooth'});};
document.addEventListener('keydown',(event)=>{if($('#weight-dialog').open||/INPUT|TEXTAREA|SELECT/.test(event.target.tagName))return;if(event.key==='Escape'){event.preventDefault();if(state.codeOpen)closeCode();else up();}});
new ResizeObserver(()=>{if(views[state.level]&&state.phase==='idle')fit(false);}).observe(canvas);

// The microscope fetches only a sampled overview and the requested 16 x 16 patch.
async function openWeight(spec){
  state.weight={name:spec.weight,head:spec.weightHead??(spec.weight.endsWith('W_q')?state.head:0),output_head:spec.outputHead?1:0};state.overview=null;state.patch=null;
  state.weightAxes=parameterAxes(spec.weight,!!spec.outputHead,C());
  for(const id of ['#weight-overview','#weight-patch'])$(id).setAttribute('aria-description',axisDescription(state.weightAxes));
  $('#weight-title').textContent=spec.title;$('#weight-description').textContent=spec.weight;
  $('#window-marker-note').textContent='';
  $('#weight-body').hidden=true;$('#weight-status').hidden=false;$('#weight-status').textContent=provider.capabilities.sampledWeights?'Reading saved samples…':'Reading the checkpoint…';
  $('#weight-row').value=spec.weightRow??0;$('#weight-col').value=0;
  if(!$('#weight-dialog').open)$('#weight-dialog').showModal();
  const request=++state.weightRequest;
  try{const overview=await provider.weights({...state.weight,overview:1});if(request!==state.weightRequest)return;
    state.overview=overview;
    if(provider.capabilities.sampledWeights){
      const options=overview.patch_options;
      $('#weight-patch-select').innerHTML=options.map((p,i)=>`<option value="${i}">Rows ${fmt(p.rows[0])}–${fmt(p.rows.at(-1))} · columns ${fmt(p.cols[0])}–${fmt(p.cols.at(-1))}</option>`).join('');
      const preferred=options.findIndex(p=>p.row===Number($('#weight-row').value)&&p.col===0);
      $('#weight-patch-select').value=String(Math.max(0,preferred));
      chooseSavedPatch();
    }
    $('#weight-row').max=overview.shape[0]-1;$('#weight-col').max=overview.shape[1]-1;
    $('#weight-meta').innerHTML=`<span>${overview.shape.map(fmt).join(' × ')} values</span>${spec.outputHead||state.model.parameters[spec.weight].length===3?`<span>${spec.outputHead||spec.weight.endsWith('W_q')?'Query head':'K/V head'} ${state.weight.head}</span>`:''}<span>${esc(overview.dtype)} checkpoint</span><span>Real learned weights</span>`;
    $('#weight-layout').textContent=overview.transposed?'Axes: input embedding coordinate (row) × head coordinate (column). This is the same per-head transpose used by weights.py.':(overview.shape[0]===1?'This learned scale is a vector, displayed as one row.':'Axes follow the stored PyTorch parameter. Linear weights are [output, input]; the forward operation uses their transpose.');
    if(spec.weight==='token_embedding_layer.weight')$('#weight-layout').textContent='Axes: token ID (row) × embedding coordinate (column). Each row is one token’s starting vector.';
    if(overview.output_head)$('#weight-layout').textContent=`Axes: residual coordinate (row) × head coordinate (column). This is out_proj.weight[:, ${overview.column_offset}:${overview.column_offset+overview.shape[1]}], displayed in its stored layout. The forward operation multiplies by its transpose. Column coordinates here are local to this head’s slice.`;
    state.weightScale=Math.max(Math.abs(overview.min),Math.abs(overview.max),1e-10);
    $('#scale-min').textContent=(-state.weightScale).toPrecision(3);$('#scale-max').textContent=state.weightScale.toPrecision(3);
    const overviewCanvas=$('#weight-overview'),proportions=matrixProportions(...overview.shape);
    overviewCanvas.width=Math.round(proportions.width);overviewCanvas.height=Math.round(proportions.height);
    overviewCanvas.style.width=proportions.width+'px';overviewCanvas.style.aspectRatio=proportions.width/proportions.height;
    $('#overview-plot').style.setProperty('--plot-width',proportions.width+'px');
    updatePlotAxes('overview',overview,state.weightAxes,true);
    $('#weight-proportions').textContent=proportions.vector?'One row of learned scale values.':`${fmt(overview.shape[0])} rows × ${fmt(overview.shape[1])} columns. `+(proportions.compressed?`${proportions.ratio>1?'Row':'Column'} axis shortened; equal-scale proportions would be ${Math.max(proportions.ratio,1/proportions.ratio).toFixed(1)}:1.`:'Both axes use the same scale.');
    drawOverview();$('#weight-body').hidden=false;$('#weight-status').hidden=true;await loadPatch();
  }catch(error){if(request===state.weightRequest){$('#weight-status').hidden=false;$('#weight-status').textContent=error.message;}}
}
function updatePlotAxes(prefix,data,axes,whole=false){
  const rowRange=whole?[0,data.shape[0]-1]:[data.rows[0],data.rows.at(-1)];
  const colRange=whole?[0,data.shape[1]-1]:[data.cols[0],data.cols.at(-1)];
  $('#'+prefix+'-rows').textContent=axes.rows;
  $('#'+prefix+'-columns').textContent=axes.columns+' →';
  const ticks=range=>`<span>${fmt(range[0])}</span>${range[1]===range[0]?'':`<span>${fmt(range[1])}</span>`}`;
  $('#'+prefix+'-row-ticks').innerHTML=ticks(rowRange);
  $('#'+prefix+'-column-ticks').innerHTML=ticks(colRange);
  $('#'+prefix+'-plot').classList.toggle('single-row',rowRange[0]===rowRange[1]);
}
function drawOverview(){
  if(!state.overview)return;
  const el=$('#weight-overview');
  drawHeatmap(el,state.overview,false);
  if(!state.patch)return;
  const box=matrixWindow(state.overview.shape,state.patch,el.width,el.height);
  const ctx=el.getContext('2d');ctx.save();
  ctx.strokeStyle='#ffffff';ctx.lineWidth=4;
  ctx.strokeRect(box.x+1,box.y+1,box.width-2,box.height-2);
  ctx.strokeStyle='#22292a';ctx.lineWidth=1.5;
  ctx.strokeRect(box.x+1,box.y+1,box.width-2,box.height-2);ctx.restore();
  $('#window-marker-note').textContent='The outlined region is the inspected window.'+(box.enlarged?' The marker is enlarged to remain visible.':'');
  el.setAttribute('aria-label',`Sampled overview of real weights. Inspected window: rows ${state.patch.rows[0]}–${state.patch.rows.at(-1)}, columns ${state.patch.cols[0]}–${state.patch.cols.at(-1)}. ${provider.capabilities.sampledWeights?'Choose a saved patch using the selector.':'Click a region or use the row and column fields.'}`);
}
function color(value){const a=Math.min(1,Math.abs(value)/state.weightScale);const dest=value<0?[184,117,86]:[43,120,129];return `rgb(${dest.map(v=>Math.round(255+(v-255)*a)).join(',')})`;}
function drawHeatmap(el,data,numbers){
  const ctx=el.getContext('2d'),nr=data.rows.length,nc=data.cols.length,cw=el.width/nc,ch=el.height/nr;
  ctx.clearRect(0,0,el.width,el.height);
  for(let r=0;r<nr;r++)for(let c=0;c<nc;c++){const v=data.values[r][c];ctx.fillStyle=color(v);ctx.fillRect(c*cw,r*ch,Math.ceil(cw),Math.ceil(ch));if(numbers){ctx.strokeStyle='#ffffff5c';ctx.lineWidth=.6;ctx.strokeRect(c*cw,r*ch,cw,ch);}}
  if(numbers){const[r,c]=state.cell;ctx.strokeStyle='#263f43';ctx.lineWidth=2;ctx.strokeRect(c*cw+1,r*ch+1,cw-2,ch-2);}
}
async function loadPatch(){
  if(!state.weight||!state.overview)return;
  const row=Number($('#weight-row').value),col=Number($('#weight-col').value);
  if(!Number.isInteger(row)||!Number.isInteger(col)||row<0||col<0||row>=state.overview.shape[0]||col>=state.overview.shape[1]){$('#cell-value').textContent='Choose a row and column inside this matrix.';return;}
  const request=++state.patchRequest,weightRequest=state.weightRequest;
  $('#cell-value').textContent='Reading values…';
  try{const patch=await provider.weights({...state.weight,row,col});if(request!==state.patchRequest||weightRequest!==state.weightRequest)return;state.patch=patch;state.cell=[0,0];$('#patch-title').textContent=`A ${patch.rows.length} × ${patch.cols.length} window`;$('#patch-range').textContent=`rows ${patch.rows[0]}–${patch.rows.at(-1)} · columns ${patch.cols[0]}–${patch.cols.at(-1)}`;$('#weight-patch').height=patch.rows.length===1?40:384;updatePlotAxes('patch',patch,state.weightAxes);drawHeatmap($('#weight-patch'),patch,true);drawOverview();showCell();}
  catch(error){if(request===state.patchRequest)$('#cell-value').textContent=error.message;}
}
function showCell(){if(!state.patch)return;const[r,c]=state.cell;$('#cell-value').textContent=`[${state.patch.rows[r]}, ${state.patch.cols[c]}] = ${state.patch.values[r][c].toPrecision(8)}`;}
function cellAt(event,el,data){const rect=el.getBoundingClientRect();return[Math.max(0,Math.min(data.rows.length-1,Math.floor((event.clientY-rect.top)/rect.height*data.rows.length))),Math.max(0,Math.min(data.cols.length-1,Math.floor((event.clientX-rect.left)/rect.width*data.cols.length)))];}
$('#weight-overview').onclick=(event)=>{if(!state.overview||provider.capabilities.sampledWeights)return;const[r,c]=cellAt(event,$('#weight-overview'),state.overview);$('#weight-row').value=state.overview.rows[r];$('#weight-col').value=state.overview.cols[c];loadPatch();};
$('#weight-patch').onpointermove=(event)=>{if(!state.patch)return;state.cell=cellAt(event,$('#weight-patch'),state.patch);drawHeatmap($('#weight-patch'),state.patch,true);showCell();};
$('#weight-patch').onkeydown=(event)=>{if(!state.patch||!event.key.startsWith('Arrow'))return;event.preventDefault();const[r,c]=state.cell;state.cell=[Math.max(0,Math.min(state.patch.rows.length-1,r+(event.key==='ArrowDown'?1:event.key==='ArrowUp'?-1:0))),Math.max(0,Math.min(state.patch.cols.length-1,c+(event.key==='ArrowRight'?1:event.key==='ArrowLeft'?-1:0)))];drawHeatmap($('#weight-patch'),state.patch,true);showCell();};
function chooseSavedPatch(){const patch=state.overview.patch_options[Number($('#weight-patch-select').value)];$('#weight-row').value=patch.row;$('#weight-col').value=patch.col;}
$('#weight-patch-select').onchange=()=>{chooseSavedPatch();loadPatch();};
$('#weight-go').onclick=loadPatch;for(const el of[$('#weight-row'),$('#weight-col')])el.onkeydown=(event)=>{if(event.key==='Enter')loadPatch();};
$('#weight-close').onclick=()=>$('#weight-dialog').close();$('#weight-dialog').addEventListener('close',()=>{++state.weightRequest;++state.patchRequest;});

async function start(){
  try{const [model,content]=await Promise.all([provider.model(),provider.text()]);state.model=model;loadTextCatalog(content.defaults,content.edits);installInlineNotes();
    if(isOriginalQwen){
      document.body.classList.add('qwen-flow-experiment');
      const readout=document.createElement('div');
      readout.id='connection-readout';readout.className='connection-readout';readout.dataset.active='false';
      readout.setAttribute('role','status');readout.setAttribute('aria-live','polite');readout.setAttribute('aria-atomic','true');
      readout.innerHTML='<div class="connection-readout-content"><div><strong data-flow-name></strong><span data-flow-shape></span></div><p data-flow-meaning></p></div>';
      $('.diagram-surface').append(readout);
    }
    setCommonCopyValues(()=>({positions:N(),embeddingDim:fmt(d()),expandedDim:fmt(C().expanded_dim),vocabularySize:fmt(C().vocabulary_size),heads:C().num_heads,kvHeads:C().num_kv_heads,headDim:h(),blocks:C().num_transformers}));
    if(provider.capabilities.editing){const {createTextEditor}=await import('./text-editor.mjs');textEditor=createTextEditor({revision:content.revision,endpoint:provider.textEndpoint,isBusy:()=>state.phase!=='idle'||!!mobile?.active,activeLayer:()=>state.level});}else refreshCopyElements();$('#model-facts').innerHTML=`<div><strong>${C().num_transformers}</strong><span>transformer blocks</span></div><div><strong>${C().num_heads}</strong><span>attention heads per block</span></div><div><strong>${fmt(d())}</strong><span>token vector embedding dimensions</span></div>`;await Promise.all([document.fonts.load('22px Kalam'),document.fonts.load('22px "STIX Two Text"'),document.fonts.load('22px "Source Sans 3"')]);if(isHybrid()){$('#model-facts').innerHTML=`<div><strong>48 + 16</strong><span>DeltaNet + gated attention blocks</span></div><div><strong>48 / 24</strong><span>heads per DeltaNet / attention block</span></div><div><strong>${fmt(d())}</strong><span>token vector embedding dimensions</span></div>`;document.title='Inside Qwen3.8 · An annotated model';canvas.setAttribute('aria-label','Qwen3.8 text model. Click an element to look inside. Escape returns one level.');document.body.classList.add('hybrid-model');}
    if(isMoE()){$('#model-facts').innerHTML=`<div><strong>48</strong><span>transformer blocks</span></div><div><strong>32</strong><span>attention heads per block</span></div><div><strong>8 / 128</strong><span>experts selected per position</span></div><div><strong>${fmt(d())}</strong><span>token vector embedding dimensions</span></div>`;document.title='Inside Qwen3 MoE · An annotated model';canvas.setAttribute('aria-label','Qwen3 mixture of experts. Click an element to look inside. Escape returns one level.');document.body.classList.add('moe-model');}
    if(isKimi()){$('#model-facts').innerHTML=`<div><strong>69 + 24</strong><span>KDA + latent attention blocks</span></div><div><strong>96</strong><span>heads per block</span></div><div><strong>16 / 896 + 2</strong><span>routed + shared experts</span></div><div><strong>${fmt(d())}</strong><span>token vector embedding dimensions</span></div>`;document.title='Inside Kimi K3 · An annotated model';canvas.setAttribute('aria-label','Kimi K3 text architecture. Click an element to look inside. Escape returns one level.');document.body.classList.add('kimi-model');}
    if(isGLM()){$('#model-facts').innerHTML=`<div><strong>34 + 11</strong><span>KDA + sparse attention blocks</span></div><div><strong>4</strong><span>residual streams</span></div><div><strong>8 / 288 + 1</strong><span>routed + shared experts</span></div><div><strong>${fmt(d())}</strong><span>token vector embedding dimensions</span></div>`;document.title='Inside GLM-5.3-Flash · An annotated model';canvas.setAttribute('aria-label','GLM-5.3-Flash text architecture. Click an element to look inside. Escape returns one level.');document.body.classList.add('glm-model');}
    if(isGemma()){$('#model-facts').innerHTML=`<div><strong>28 + 7</strong><span>sliding + global attention blocks</span></div><div><strong>8 / 1</strong><span>query / key-value heads</span></div><div><strong>256</strong><span>per-layer embedding dimensions</span></div><div><strong>${fmt(d())}</strong><span>token vector embedding dimensions</span></div>`;document.title='Inside Gemma 4 E2B';document.body.classList.add('gemma-model');canvas.setAttribute('aria-label','Gemma 4 E2B text architecture');}
    if(isDeepSeek()){$('#model-facts').innerHTML=`<div><strong>20 + 20</strong><span>causal encoder + decoder blocks</span></div><div><strong>4</strong><span>residual streams</span></div><div><strong>6 / 384 + 1</strong><span>routed + shared experts</span></div><div><strong>${fmt(d())}</strong><span>token vector embedding dimensions</span></div>`;document.title='Inside DeepSeek-V4.1-Flash';document.body.classList.add('deepseek-model');canvas.setAttribute('aria-label','DeepSeek-V4.1-Flash text architecture');}
    if(isMamba()){$('#token-count').title='N is the number of input tokens (sequence length).';$('#model-facts').innerHTML=`<div><strong>${C().num_blocks}</strong><span>Mamba blocks</span></div><div><strong>${C().num_heads}</strong><span>state-space heads per block</span></div><div><strong>${h()} × ${C().state_size}</strong><span>state dimensions per head</span></div><div><strong>${fmt(d())}</strong><span>token vector embedding dimensions</span></div>`;document.title='Inside Mamba-2 · An annotated model';canvas.setAttribute('aria-label','Mamba-2 130M architecture. Click an element to look inside. Escape returns one level.');document.body.classList.add('mamba-model');}
    if(isGPT2()){document.title='Inside GPT-2 small · An annotated model';canvas.setAttribute('aria-label','GPT-2 small architecture. Click an element to look inside. Escape returns one level.');document.body.classList.add('gpt2-model');}
    const choices=await provider.models();$('#model-select').innerHTML=choices.map(choice=>`<option value="${esc(choice.id)}">${esc(choice.label)}</option>`).join('');$('#model-select').value=modelId;
    $('.model-picker').hidden=choices.length<2;
    $('#model-summary-name').textContent=choices.find(choice=>choice.id===modelId)?.label||state.model.name;
    $('#model-select').onchange=()=>{if(textEditor?.dirty){$('#model-select').value=modelId;$('#model-switch-status').textContent='Save your text changes before switching models.';return;}const url=new URL(location.href);url.searchParams.set('model',$('#model-select').value);location.assign(url);};
    if(!provider.capabilities.customContext){
      state.examples=await provider.examples();
      $('#example-select').innerHTML=state.examples.map(e=>`<option value="${esc(e.id)}">${esc(e.text)}</option>`).join('');
      $('#example-select').onchange=tokenizePrompt;
    }
    if(provider.capabilities.sampledWeights){
      document.body.classList.add('sampled-weights');
      $('.patch-controls').hidden=true;$('.sample-patch-picker').hidden=false;
      $('#weight-overview-hint').textContent='Choose a saved patch to inspect individual values. Colors use the sampled range; values beyond it saturate.';
    }
    mobile=createMobileExplorer({canvas,world,getState:()=>state,overviewVisibleThrough:()=>{const b=blockGeometry();return b.start+4*b.step+b.closedWidth;},go,up,openCode,openWeight,refit:()=>{if(views[state.level]&&state.phase==='idle')fit(false);},onReadoutClose:()=>{clearFlowHighlight(world);hideNodeTooltip();},onMode:()=>{finishMobileTransition();textEditor?.refresh();}});
    await tokenizePrompt();registerModelTool();
    $('main').inert=false;$('main').setAttribute('aria-busy','false');document.body.dataset.appState='ready';}
  catch(error){$('#canvas-error').hidden=false;$('#canvas-error').textContent=`The model could not be opened. ${error.message} ${provider.capabilities.editing?'Start the local server with uv run visualize.py and reload this page.':'Reload this page or try another model.'}`;$('#model-facts').textContent='Model unavailable';document.body.dataset.appState='error';$('main').setAttribute('aria-busy','false');$('#startup-status').textContent=$('#canvas-error').textContent;$('#startup-status').setAttribute('role','alert');}
}

function registerModelTool(){
  const context=document.modelContext;if(!context?.registerTool)return;
  const lifetime=new AbortController();window.addEventListener('pagehide',()=>lifetime.abort(),{once:true});
  try{Promise.resolve(context.registerTool({
    name:'navigate_qwen_model',title:'Explore a Qwen3 component',
    description:"Navigate the visible local Qwen3 explorer to the whole model, a transformer block, its attention or feedforward network, a query head, or the final output projection.",
    inputSchema:{type:'object',properties:{level:{type:'string',enum:['model','block','attention','feedforward','head','output']},block:{type:'integer',minimum:0,maximum:C().num_transformers-1},head:{type:'integer',minimum:0,maximum:C().num_heads-1}},required:['level'],additionalProperties:false},
    annotations:{readOnlyHint:false,untrustedContentHint:false},
    async execute(input){
      if(!input||typeof input!=='object'||Array.isArray(input)||Object.keys(input).some(k=>!['level','block','head'].includes(k))||!['model','block','attention','feedforward','head','output'].includes(input.level))throw new Error('Choose a valid model level.');
      for(const[key,max]of[['block',C().num_transformers],['head',C().num_heads]])if(input[key]!==undefined&&(!Number.isInteger(input[key])||input[key]<0||input[key]>=max))throw new Error(`Invalid ${key} index.`);
      await go(input.level,{...input,instant:true});await new Promise(requestAnimationFrame);
      return{level:state.level,block:state.block,head:state.head,title:state.selected.title};
    }
  },{signal:lifetime.signal})).catch(error=>console.warn('Model navigation tool unavailable:',error.message));}
  catch(error){console.warn('Model navigation tool unavailable:',error.message);}
}
start();
