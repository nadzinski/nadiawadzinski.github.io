import {illustrativeRouting} from './moe-routing.mjs';

export function deepseekLayerKind(config,block){
  if(!config.compress_ratios[block])return 'Sliding-window attention';
  if(config.kv_source_layers.includes(block))return 'CSA2 · Full';
  if(config.index_source_layers.includes(block))return 'CSA2 · Reindex';
  return 'CSA2 · Reuse';
}
export function illustrativeDeepSeekRouting(position,block){
  const logits=illustrativeRouting(384,6,position,block).logits;
  const bias=illustrativeRouting(384,6,position+7,block+3).logits.map(x=>x*.08);
  const chosen=logits.map((x,expert)=>({expert,score:Math.sqrt(Math.log1p(Math.exp(x)))})).map(x=>({...x,choice:x.score+bias[x.expert]})).sort((a,b)=>b.choice-a.choice).slice(0,6);
  const sum=chosen.reduce((s,x)=>s+x.score,0);return chosen.map(x=>({...x,weight:x.score/sum*1.5}));
}
export const deepseekText={
 'model.description':'DeepSeek-V4.1-Flash has 40 text blocks: a 20-layer causal encoder followed by a 20-layer decoder. Both halves use causal attention. The decoder reads global key/value information derived at the encoder boundary, alongside its own local sliding-window keys and values. Every block has a mixture of experts and four residual streams. Engram memory adds learned information retrieved from short token sequences in blocks 1 and 14. This implementation recomputes the whole sequence on every call.',
 'block.description':'Four residual streams pass through the block. Single-pass mHC collapses them to one input for attention, distributes its output, and mixes the bypassed streams. Feedforward does the same. Each sublayer computes collapse coefficients for the following sublayer; it uses the coefficients inherited from the preceding one. Blocks 1 and 14 first add an Engram memory contribution.',
 'attention.description':'Each block produces its own queries and local K=V vectors. The local window includes the current position and up to 127 preceding positions. CSA2 blocks also read selected global vectors. Full layers produce global vectors and an index, Reindex layers recompute only the index, and Reuse layers inherit both. All 64 query heads share the same keys, values and selected positions.',
 'norm.description':'RMSNorm divides each vector by its root mean square and applies learned coordinate scales. Epsilon is 10⁻²⁰. Attention and feedforward have separate input normalization weights.',
 'query.description':'Project each 5,120-coordinate input to a 1,280-coordinate query latent and normalize it. Expand to 64 queries with 512 coordinates each. Apply RoPE to the last 64 coordinates. The query latent also supplies the indexer in layers that build an index.',
 'local.description':'Project each normalized input to a 512-coordinate vector and apply RMSNorm and RoPE. The same vector serves as both key and value for all heads. Only the current position and its causal 128-position window are visible. Each block has its own local projection.',
 'global.description':'Blocks 2, 8 and 14 form one global vector per complete pair of sequence positions, using a learned, coordinate-wise softmax pooling gate. Block 20 forms one vector per position from its attention input at the encoder boundary. A source’s global vectors are shared with subsequent blocks until the next source. The indexer receives the unrotated vectors; main attention receives their rotated versions.',
 'sharing.description':'Full layers are 2, 8, 14 and 20. They compute global K/V, index keys, and selected indices. Reindex layers 24, 28, 32 and 36 use the same global K/V and index keys, but make new queries and selections. All remaining CSA2 layers reuse the previous selection. Layers 0 and 1 use only their local window. This sharing takes place within a forward pass in our implementation.',
 'head.description':'A head compares its 512-coordinate query with the selected local and global K=V vectors. Divide scores by √512. Softmax includes one learned per-head sink logit whose associated value is zero. Mix the visible vectors with these weights, then apply inverse RoPE to the final 64 output coordinates. The 64 results are joined into eight groups for output projection.',
 'mask.description':'The local window is causal. A compressed pair is visible only when both positions are in the query’s past or include the query itself. The indexer chooses up to 512 eligible global vectors. Local and global entries are separate attention slots, even when they describe overlapping positions. Each head also has a learned softmax sink with zero value.',
 'output-groups.description':'Join eight head outputs per group, giving 4,096 coordinates. Each of eight groups has its own 4,096 → 1,024 projection. Join the eight results into 8,192 coordinates and project to the 5,120-coordinate residual width. This is a grouped low-rank output projection.',
 'index.description':'The indexer forms 32 queries of 128 coordinates from the normalized query latent. Shared index keys come from the unrotated global vectors. Apply RoPE, compute dot products, take ReLU, and combine the head scores with signed, input-dependent weights scaled by 1/√(128 × 32). Causal top-512 selection supplies one index for every main attention head.',
 'candidate.description':'At block 20, score blocks of eight global positions by their largest index score. Keep up to 2,048 candidate blocks, always including the newest visible block. Later Reindex layers search within this inherited candidate set. Each index still selects individual global positions, and enforces causality.',
 'moe.description':'At each sequence position, score all 384 experts using square-root softplus. Add a learned correction bias to choose six experts. Retrieve their original scores, normalize the selected scores to sum to one, and multiply by 1.5. Sum the weighted expert outputs and add one always-active shared expert. Each expert is a 5,120 → 2,304 → 5,120 SwiGLU network.',
 'expert.description':'The gate path is capped above at 10 and passed through SiLU. The value path is clamped between −10 and 10. Multiply the two paths coordinate by coordinate. For a routed expert, multiply this product by its routing weight before the down projection. The shared expert has no routing multiplier. Each sequence position is processed independently.',
 'hc.description':'At each position, flatten the four stream vectors and RMS-normalize the resulting 20,480 coordinates. A learned projection produces 24 coefficients: four for collapse, four for distributing a contribution, and sixteen for mixing the bypassed streams. Sigmoid controls collapse and distribution. Softmax and 20 rounds of row/column normalization constrain the 4 × 4 mixing matrix. The newly computed collapse coefficients are passed to the next sublayer.',
 'engram.description':'Engram looks up learned memory using 2-, 3- and 4-token sequences ending at the current position. Token IDs are first mapped to 99,092 normalized IDs. Each n-gram length has eight independent hash ranges. Their 24 retrieved 256-coordinate vectors are joined and projected into four keys plus one value. Each residual stream compares itself with its key to gate the shared value contribution. Memory tables are learned parameters; retrieved vectors and gates are activations.',
 'engram.hash.description':'Normalize each decoded token with Unicode normalization, accent removal, lowercase and whitespace normalization. Multiply the normalized IDs by layer-specific odd integers and XOR them across each n-gram. Reduce modulo a different prime for each hash head, then add that head’s table offset. Missing positions at the start of the sequence use the normalized padding ID.',
 'engram.gate.description':'For each of the four streams, normalize the stream and its retrieved key by their RMS values, apply learned coordinate scales, and take a dot product divided by √5,120. Apply a signed square root and sigmoid. The resulting scalar multiplies the retrieved value before it is added to that stream.',
 'output.description':'Collapse the four final streams using the last feedforward sublayer’s computed coefficients. Apply final RMSNorm, then a separate 129,280 × 5,120 output matrix to produce logits. The final sequence position supplies the next-token distribution.',
};

export function createDeepSeekDiagrams(v){
 const {state,C,N,d,fmt,copy,text,line,path,outline,grid,brokenGrid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,richText,copyAttributes}=v;
 const t=key=>copy('deepseek.'+key),td=(width=d(),positions=N())=>`${positions} × ${fmt(width)}`;
 const axes={rows:'sequence position',columns:'residual coordinate'};
 const spec=(key,title,source,description,extra={})=>({key:'deepseek-'+key,title,source,description,...extra});
 const box=(x,y,w,h,label,key,source,description,extra={})=>operation(x,y,w,h,label,spec(key,label,source,description,extra));
 const mat=(x,y,w,h,label,width,key,source,description,rows=N())=>matrix(x,y,w,h,label,td(width,rows),spec(key,label,source,description,{activation:true,rows:Math.min(rows,5)||1,cols:7,axes:{rows:'sequence position',columns:width===C().vocabulary_size?'vocabulary entry':'vector coordinate'},output:td(width,rows)}));
 const frame=(title,w=1680,h=760)=>outline(-30,-20,w,h)+text(0,25,title,29,'diagram-title','start');
 const bound=(name,w=1770,h=850)=>{views[name]={x:-100,y:-50,w,h};};
 const headSpec=head=>spec('head-'+head,`Head ${head}`,'scores',t('head.description'),{head,enter:'head',input:td(512),output:td(512)});
 const expertSpec=expert=>spec('expert-'+expert,expert===-1?'Shared expert':`Expert ${expert}`,'expert',t('expert.description'),{expert,enter:'expert',parameterScope:expert===-1?'shared-experts':'expert',input:td(d(),1),output:td(d(),1)});
 function defaultSpec(level){
  const memory=state.depthBranch==='memory';
  const info={model:['DeepSeek-V4.1-Flash','model','model','model'],block:[`Transformer block ${state.block}`,'block','block','block'],attention:[deepseekLayerKind(C(),state.block),'attention','attention','attention'],head:[`Head ${state.head}`,'scores','head',null],feedforward:['Mixture of experts','feedforward','moe','feedforward'],expert:[state.expert===-1?'Shared expert':`Expert ${state.expert}`,'expert','expert',state.expert===-1?'shared-experts':'expert'],depth:[memory?'Engram memory':'Single-pass mHC',memory?'engram':'hc',memory?'engram':'hc',memory?'engram':'hyperconnection-'+(state.depthBranch||'attention')],indexer:['CSA2 index and sharing','indexer','index',null],output:['Output projection','output_stage','output','output'],embedding:['Token embeddings','embedding','model',null]}[level];
  return spec(level,info[0],info[1],t(info[2]+'.description'),{kind:'DEEPSEEK-V4.1',parameterScope:info[3],expert:state.expert,input:level==='model'?`${N()} token IDs`:level==='expert'?td(d(),1):level==='indexer'?'Query latents and shared global keys':['block','output','depth'].includes(level)?`${N()} × 4 × ${fmt(d())}`:td(),output:['model','output'].includes(level)?td(C().vocabulary_size):level==='expert'?td(d(),1):level==='head'?td(512):level==='indexer'?`${N()} × up to 512 selected indices`:['block','depth'].includes(level)?`${N()} × 4 × ${fmt(d())}`:td(),inputAxes:['model','indexer'].includes(level)?undefined:['block','output','depth'].includes(level)?{rows:'sequence position',columns:'4 streams × residual coordinate'}:axes,outputAxes:level==='indexer'?{rows:'query position',columns:'selected global position'}:['model','output'].includes(level)?{rows:'sequence position',columns:'vocabulary entry'}:['block','depth'].includes(level)?{rows:'sequence position',columns:'4 streams × residual coordinate'}:axes});
 }
 function block(){
  let out=`<path id="block-frame" class="housing" d="${outlinePath(-15,BLOCK.frameTop,BLOCK.width,state.blockHeight)}"/>`+text(28,35,`Transformer block ${state.block}`,27,'diagram-title','start');
  out+=text(1180,35,state.block<20?'causal encoder':'decoder',20,'diagram-note','end');
  for(const y of [319,333,347,361])out+=stream(-38,572,y)+stream(717,1090,y)+stream(1227,1260,y);
  out+=text(25,290,'Four residual streams',19,'diagram-note','start');
  out+=path('M130,319 V165 H178 M130,319 V361','diagram-wire')+box(190,125,100,80,'mHC','hc-attn','hc',t('hc.description'),{enter:'depth',depthBranch:'attention'});
  out+=line(302,165,319,165);
  let sketch=outline(331,105,264,122)+text(463,132,deepseekLayerKind(C(),state.block),19,'diagram-title');
  for(const y of [148,159,170,190])sketch+=line(367,y,386,y)+outline(398,y-3,126,6,'#e8f2f1')+line(536,y,560,y);
  sketch+=text(461,181,'…',14,'diagram-note')+text(463,216,'RMSNorm · 64 heads',16,'diagram-shape');
  out+=record({...defaultSpec('attention'),key:'ds-attention',enter:'attention',box:{x:331,y:105,w:264,h:122}},sketch);
  out+=path('M607,165 H645 V283')+box(584,295,120,100,'Mix streams\n+ add output','attn-add','hc_add',t('hc.description'),{enter:'depth',depthBranch:'attention',fontSize:18});
  out+=path('M750,319 V165 H774 M750,319 V361','diagram-wire')+box(786,125,100,80,'mHC','hc-ff','hc',t('hc.description'),{enter:'depth',depthBranch:'feedforward'});
  out+=line(898,165,902,165)+box(914,105,190,122,'Mixture of\nexperts','moe','feedforward',t('moe.description'),{enter:'feedforward',fontSize:22})+text(1009,216,'RMSNorm · 6 + 1',16,'diagram-shape');
  out+=path('M1116,165 H1160 V283')+box(1105,295,110,100,'Mix streams\n+ add output','ff-add','hc_add',t('hc.description'),{enter:'depth',depthBranch:'feedforward',fontSize:18});
  if(C().engram_layers.includes(state.block))out+=box(25,405,260,48,'Engram memory','engram','engram',t('engram.description'),{enter:'depth',depthBranch:'memory',fontSize:21})+path('M55,393 V373','diagram-contribution');
  out+=`<foreignObject id="block-prose" x="28" y="495" width="1150" height="125"><div xmlns="http://www.w3.org/1999/xhtml" id="block-prose-text" class="block-prose" ${copyAttributes(t('block.description'))}>${richText(t('block.description'))}</div></foreignObject>`;
  return out;
 }
 function attention(){
  const compressed=C().compress_ratios[state.block]>0;
  let out=frame(deepseekLayerKind(C(),state.block),1740,830)+text(0,77,'From mHC · one mixed vector per sequence position',22,'diagram-note','start');
  out+=line(-95,370,3,370)+box(15,335,86,70,'RMS\nNorm','attn-norm','norm',t('norm.description'),{weight:param('attention_norm.weight'),input:td(),output:td()});
  out+=line(113,370,161,370)+box(173,322,190,96,'Query + local\nK/V projections','project','attention',t('query.description'),{fontSize:22,input:td(),output:'64 × '+td(512)+' queries; '+td(512)+' local K=V'});
  out+=line(375,370,425,370)+path('M425,170 V570','diagram-wire');
  for(const [i,y]of [170,255,340,455,570].entries()){
    const head=[0,1,2,3,63][i];out+=line(425,y,481,y)+operation(493,y-22,290,44,i===3?'… heads 3–62 …':`Head ${head}`,headSpec(head));
    out+=line(795,y,842,y)+grid(854,y-16,100,32,true,Math.min(N(),5)||1,6)+line(966,y,1005,y);
  }
  out+=path('M1005,170 V570','diagram-wire')+line(1005,370,1043,370)+box(1055,330,190,80,'Eight grouped\nprojections','groups','combine',t('output-groups.description'),{weight:param('attention.output_groups'),input:td(32768),output:td(8192),fontSize:22});
  out+=line(1257,370,1303,370)+box(1315,330,165,80,'Output\nprojection','out','combine',t('output-groups.description'),{weight:param('attention.output_proj.weight'),input:td(8192),output:td()});
  out+=line(1492,370,1775,370,'diagram-contribution')+text(1635,427,'to mHC',22,'diagram-note');
  out+=text(638,102,'64 heads · shared local K=V',24,'diagram-title');
  if(compressed){out+=C().kv_source_layers.includes(state.block)?path('M140,370 V712 H168'):line(-95,712,168,712);out+=box(180,674,455,76,'Global K/V and selected positions','shared','indexer',t('sharing.description'),{enter:'indexer',fontSize:24})+path('M647,712 H850 V627 H425 V582','diagram-arrow');out+=text(1220,722,'Shared by every head',23,'diagram-note');}
  else out+=text(850,728,'Each query reads its causal window of up to 128 positions.',24,'diagram-note');
  bound('kimiAttention',1880,930);return out;
 }
 function head(){
  let out=frame(`Attention head ${state.head}`,1820,860);
  out+=line(-95,245,3,245)+box(15,200,180,90,'Query latent\n+ RMSNorm','query-latent','query',t('query.description'),{weight:param('attention.query_down.weight'),input:td(),output:td(1280),fontSize:23});
  out+=line(207,245,243,245)+box(255,200,150,90,'Head query\n+ RoPE','query','query',t('query.description'),{input:td(1280),output:td(512),fontSize:23});
  out+=line(417,245,488,245)+mat(500,210,135,70,'Q',512,'q','query',t('query.description'));
  out+=path('M-70,245 V570 H3')+box(15,525,180,90,'Local K=V\nRMS + RoPE','kv','local_kv',t('local.description'),{weight:param('attention.local_kv.weight'),input:td(),output:td(512),fontSize:23});
  out+=line(207,570,343,570)+box(355,530,280,80,'Local window\n+ selected global vectors','visible','mask',t('mask.description'),{fontSize:22,enter:C().compress_ratios[state.block]?'indexer':undefined});
  out+=path('M647,570 H690 V357 H718')+path('M647,245 H690 V327 H718');
  out+=box(730,300,190,90,'Dot products\n÷ √512','scores','scores',t('head.description'),{fontSize:23});
  out+=line(932,345,993,345)+box(1005,300,190,90,'Softmax\n+ learned sink','softmax','softmax',t('mask.description'),{weight:param('attention.sink'),fontSize:23});
  out+=line(1207,345,1268,345)+box(1280,300,185,90,'Weighted sum\nof K=V vectors','mix','mix',t('head.description'),{fontSize:23});
  out+=path('M690,570 H1372 V402','diagram-contribution')+line(1477,345,1523,345)+box(1535,300,160,90,'Inverse\nRoPE','unrotate','combine',t('head.description'),{output:td(512)});
  out+=line(1707,345,1855,345,'diagram-contribution')+text(1668,479,'to grouped output projection',21,'diagram-note');
  out+=text(900,737,'The global source and selection depend on the block. Local K/V is computed in every block.',23,'diagram-note');
  out+=text(900,790,'K and V are the same 512-coordinate vectors. The sink contributes to the denominator only.',23,'diagram-note');
  bound('kimiHead',1980,960);return out;
 }
 function indexer(){
  let out=frame('CSA2 · global vectors, index and sharing',1770,930);
  out+=box(10,130,350,100,'Full: blocks 2, 8, 14, 20','full','global_kv',t('global.description'),{fontSize:24});
  out+=line(372,180,458,180)+box(470,130,330,100,'Reindex: 24, 28, 32, 36','reindex','indexer',t('sharing.description'),{fontSize:23});
  out+=line(812,180,898,180)+box(910,130,330,100,'Reuse: other CSA2 blocks','reuse','attention',t('sharing.description'),{fontSize:23});
  out+=text(185,281,'Produce global K/V\nand index keys',22,'diagram-note')+text(635,281,'New queries and selection\nreuse global K/V and keys',22,'diagram-note')+text(1075,281,'Reuse global K/V, keys\nand selected indices',22,'diagram-note');
  out+=text(1510,189,'Layers 0–1:\nlocal window only',24,'diagram-note');
  out+=line(-95,450,3,450)+box(15,410,245,80,'Global vectors\nfrom source layer','global','global_kv',t('global.description'),{fontSize:24})+line(272,450,318,450);
  out+=box(330,410,220,80,'Index keys\nprojection + RMS','keys','indexer',t('index.description'),{fontSize:22});
  out+=line(562,450,623,450)+box(635,405,255,90,'RoPE · dot products\nReLU · weighted sum','index-score','indexer',t('index.description'),{fontSize:22});
  out+=box(330,625,220,80,'32 index queries','queries','indexer',t('index.description'),{output:'32 × '+td(128),fontSize:23})+path('M562,665 H762 V507');
  out+=line(-95,665,318,665)+text(100,635,'From query latent',21,'diagram-note');
  out+=line(902,450,958,450)+box(970,405,285,90,'Causal mask\n+ candidate restriction','candidates','candidates',t('candidate.description'),{fontSize:23});
  out+=line(1267,450,1323,450)+box(1335,405,230,90,'Select up to\n512 positions','select','indexer',t('index.description'),{fontSize:24});
  out+=line(1577,450,1805,450,'diagram-contribution')+text(1650,558,'to every main\nattention head',23,'diagram-note');
  out+=text(850,833,'Encoder global vectors pool complete pairs. Decoder global vectors retain individual positions.',24,'diagram-note');
  bound('glmIndexer',1930,1040);return out;
 }
 function feedforward(){
  const selected=illustrativeDeepSeekRouting(state.moePosition,state.block);
  let out=frame('Mixture of experts',1710,880)+text(0,78,`Illustrative routing at position ${state.moePosition+1} · 6 of 384 experts`,22,'diagram-note','start');
  out+=line(-95,425,3,425)+box(15,390,90,70,'RMS\nNorm','ff-norm','norm',t('norm.description'),{weight:param('feed_forward_norm.weight')});
  out+=line(117,425,240,425)+path('M240,290 V690','diagram-wire');
  out+=path('M160,425 V175 H283')+box(295,135,220,80,'Router\n√softplus','router','router',t('moe.description'),{weight:param('feed_forward.router.weight'),input:td(d(),1),output:'384 scores',fontSize:24});
  out+=line(527,175,588,175)+box(600,135,240,80,'Bias · top six\nnormalize × 1.5','topk','router',t('moe.description'),{fontSize:24})+path('M852,175 H880 V235 H722 V259','moe-control');
  out+=`<rect x="270" y="245" width="965" height="440" fill="none" stroke="#9baea9" stroke-dasharray="3 6"/>`;
  selected.forEach((item,i)=>{const y=290+i*64;out+=line(240,y,308,y)+operation(320,y-19,230,38,`Expert ${item.expert}`,expertSpec(item.expert))+line(562,y,628,y)+box(640,y-19,165,38,`× ${item.weight.toFixed(3)}`,'route-'+i,'router',t('moe.description'),{fontSize:22})+line(817,y,898,y)+grid(910,y-13,150,26,true,1,8)+line(1072,y,1160,y);});
  out+=path('M1160,290 V610','diagram-wire')+line(1160,450,1288,450)+box(1300,410,180,80,'Sum six\ncontributions','sum','moe_combine',t('moe.description'),{fontSize:24});
  out+=operation(320,735,330,65,'Shared expert · always active',expertSpec(-1))+path('M240,690 V767 H308')+path('M662,767 H1580 V587','diagram-contribution');
  out+=path('M1492,450 H1580 V533','diagram-contribution')+plus(1580,560,{title:'Add shared and routed outputs',source:'moe_combine',description:t('moe.description')})+line(1607,560,1745,560,'diagram-contribution');
  out+=text(1430,830,'Contribution returns to mHC',23,'diagram-note');bound('kimiFeedforward',1870,980);return out;
 }
 function expert(){
  const prefix=state.expert===-1?'feed_forward.shared_experts.':`feed_forward.experts.${state.expert}.`;
  let out=frame(state.expert===-1?'Shared expert':`Expert ${state.expert}`,1680,700);
  out+=line(-95,315,83,315)+path('M83,315 V175 H133 M83,315 V445 H133','diagram-wire');
  out+=box(145,135,220,80,'Gate projection\ncap above at 10','gate','gate',t('expert.description'),{weight:param(prefix+'gate_proj.weight'),input:td(d(),1),output:td(2304,1),fontSize:23});
  out+=line(377,175,443,175)+box(455,135,135,80,'SiLU','silu','silu',t('expert.description'));
  out+=box(145,405,220,80,'Value projection\nclamp to ±10','value','value',t('expert.description'),{weight:param(prefix+'value_proj.weight'),input:td(d(),1),output:td(2304,1),fontSize:23});
  out+=path('M602,175 H700 V278 M377,445 H700 V352');
  out+=box(675,290,50,50,'×','multiply','silu',t('expert.description'))+line(737,315,803,315);
  out+=state.expert===-1?line(815,315,1045,315):box(815,275,230,80,'× routing weight','route','expert',t('expert.description'),{fontSize:24});
  out+=line(1057,315,1123,315)+box(1135,275,220,80,'Down projection','down','down',t('expert.description'),{weight:param(prefix+'output_proj.weight'),input:td(2304,1),output:td(d(),1),fontSize:24})+line(1367,315,1493,315);
  out+=mat(1505,300,110,30,'Contribution',d(),'contribution','expert',t('expert.description'),1)+line(1627,315,1720,315,'diagram-contribution');
  out+=text(825,600,'Each expert processes one sequence position independently.',24,'diagram-note');bound('kimiExpert',1840,800);return out;
 }
 function depth(){
  if(state.depthBranch==='memory')return memory();
  const branch=state.depthBranch==='feedforward'?'feedforward':'attention',prefix=branch==='attention'?'attention_hc':'feed_forward_hc';
  let out=frame('Single-pass mHC',1820,900)+text(0,78,`${branch} · at one sequence position`,23,'diagram-note','start');
  for(let i=0;i<4;i++)out+=stream(-95,200,180+i*28);
  out+=path('M200,180 V264 M200,222 H283','diagram-wire')+box(295,177,285,90,'Flatten · RMS\n20,480 → 24 mapping','hc','hc',t('hc.description'),{weight:param(prefix+'.mapping'),fontSize:24});
  out+=path('M592,222 H660 V132 H723')+box(735,92,285,80,'New collapse coefficients','next-pre','hc',t('hc.description'),{fontSize:23})+line(1032,132,1760,132,'moe-control')+text(1410,103,'passed to the following sublayer',22,'diagram-note');
  out+=box(295,390,285,85,'Inherited collapse weights\n× current streams','inherited','hc_pre',t('block.description'),{fontSize:23})+path('M200,264 V432 H283');
  out+=path('M-95,530 H437 V487','moe-control')+text(68,504,'from previous sublayer',21,'diagram-note');
  out+=line(592,432,723,432)+box(735,387,285,90,`RMSNorm + ${branch}`,'sublayer',branch,t('block.description'),{enter:branch,fontSize:24});
  out+=path('M660,222 V272 H1093')+box(1105,232,280,80,'Distribute contribution\n2 × sigmoid weights','post','hc_add',t('hc.description'),{fontSize:22});
  out+=path('M1032,432 H1240 V324','diagram-contribution')+path('M1397,272 H1590 V455','diagram-contribution');
  out+=path('M660,272 V700 H723')+box(735,655,285,90,'4 × 4 stream mixing\n20 Sinkhorn rounds','combine','hc',t('hc.description'),{fontSize:24});
  out+=path('M200,432 V780 H877 V757','diagram-wire')+text(400,755,'current streams',22,'diagram-note');
  out+=path('M1032,700 H1590 V509','diagram-contribution')+plus(1590,482,{source:'hc_add',title:'Add mixed streams and distributed contribution',description:t('hc.description')});
  for(let i=0;i<4;i++)out+=path(`M1617,482 H1660 V${440+i*28} H1860`,'diagram-contribution');
  bound('kimiDepth',1980,1000);return out;
 }
 function memory(){
  const memoryBlock=C().engram_layers.includes(state.block)?state.block:C().engram_layers[0];
  let out=frame(`Engram memory · block ${memoryBlock}`,1780,850);
  out+=line(-95,235,3,235)+box(15,190,225,90,'Normalize token IDs\n2-, 3-, 4-grams','ngrams','ngram',t('engram.hash.description'),{fontSize:23});
  out+=line(252,235,318,235)+box(330,190,205,90,'Eight hashes\nper n-gram length','hash','ngram',t('engram.hash.description'),{fontSize:23});
  out+=line(547,235,613,235)+box(625,190,240,90,'24 memory lookups','lookup','engram',t('engram.description'),{weight:`transformer_blocks.${memoryBlock}.engram.table.weight`,output:'24 × 256',fontSize:24});
  out+=line(877,235,943,235)+box(955,185,285,100,'Join retrieved vectors\nproject to keys + value','project','engram',t('engram.description'),{weight:`transformer_blocks.${memoryBlock}.engram.projection.weight`,input:'1 × 6,144',output:'5 × 5,120',fontSize:23});
  out+=path('M1252,235 H1435 V438')+box(1260,450,350,185,'Gate memory value\n+ add to each stream','gates','engram',t('engram.gate.description'),{fontSize:25});
  for(const y of [490,520,550,580])out+=stream(-95,1248,y)+stream(1622,1840,y);
  out+=text(500,440,'Four incoming residual streams',24,'diagram-note')+text(880,745,'The lookup depends on token IDs. The gates depend on the current stream vectors.',24,'diagram-note');
  bound('kimiDepth',1940,950);return out;
 }
 function output(){
  let out=frame('Output projection',1600,540);
  for(let i=0;i<4;i++)out+=stream(-95,105,245+i*20);
  out+=box(117,222,270,100,'Final inherited\nstream collapse','collapse','hc_pre',t('output.description'),{fontSize:25});
  out+=line(399,272,478,272)+box(490,232,160,80,'RMSNorm','norm','norm',t('norm.description'),{weight:'output_norm.weight'});
  out+=line(662,272,768,272)+box(780,212,310,120,'Output matrix\n129,280 × 5,120','matrix','output',t('output.description'),{weight:'output_layer.weight',input:td(),output:td(C().vocabulary_size),fontSize:25});
  out+=line(1102,272,1213,272)+mat(1225,235,220,75,'Logits',C().vocabulary_size,'logits','output',t('output.description'))+line(1457,272,1640,272,'diagram-contribution');
  out+=text(820,450,'The final position’s logits supply the next-token distribution.',24,'diagram-note');bound('kimiOutput',1760,660);return out;
 }
 return {defaultSpec,block,attention,head,indexer,feedforward,expert,depth,output,isDense:()=>false,isDelta:()=>false,attentionName:()=>deepseekLayerKind(C(),state.block)};
}
