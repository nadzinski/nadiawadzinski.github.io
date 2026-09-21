import {parameterAxes} from './matrix-info.mjs';
export const gemmaHeadDim=(c,b)=>c.layer_types[b]==='full_attention'?c.global_head_dim:c.head_dim;
export const gemmaFFDim=(c,b)=>c.expanded_dim*(b>=c.first_shared_block?2:1);
export function gemmaKVSource(c,b){return b<c.first_shared_block?b:c.layer_types.slice(0,c.first_shared_block).lastIndexOf(c.layer_types[b]);}
export const gemmaText={
 'model.description':'Gemma 4 E2B has 35 text transformer blocks. Four sliding-window blocks alternate with one global-attention block. Each block has gated feedforward and a further contribution from per-layer token embeddings. Blocks 15–34 reuse keys and values from earlier source blocks. The token embedding and output projection share a learned table. Final logits are softly capped at ±30.',
 'block.description':'The residual stream receives three contributions in order: attention, gated feedforward, and a gated per-layer embedding. Attention and feedforward each have RMSNorm before their calculation and on the contribution before addition. The per-layer path is gated by the current residual vector. A checkpoint scalar multiplies the completed block output.',
 'norm.description':'RMSNorm divides each vector by the square root of its mean squared coordinate value plus 10⁻⁶. Most normalizations also multiply by learned coordinate scales. Value-vector normalization has no learned scale. This checkpoint uses the stored RMSNorm scale directly.',
 'embedding.description':'Look up a 1,536-coordinate vector per token and multiply it by √1,536. These vectors start the residual stream. A second learned table supplies 35 separate 256-coordinate embeddings for each token. A projection of the initial scaled embedding supplies another 256 coordinates per layer, which are RMS-normalized. Add the lookup and projected components and multiply by 1/√2. Each block receives its own slice of this per-layer input.',
 'ple.description':'Each block receives a 256-coordinate per-layer input, prepared from the token ID and its initial embedding before the transformer blocks run. After attention and feedforward, project the current residual vector from 1,536 to 256 coordinates and apply GELU. Multiply this gate by the per-layer input coordinate by coordinate. Project back to 1,536 coordinates, RMS-normalize the contribution and add it to the residual stream.',
 'ple.table.description':'The packed per-layer embedding table has 262,144 rows and 8,960 columns: 35 slices of 256 coordinates. Looking up a token retrieves every layer’s slice; multiply the retrieved vectors by √256. This table accounts for about 2.35 billion stored values, but only the input tokens’ rows are read on a pass.',
 'attention.description':'All eight query heads share one key head and one value head. Sliding attention uses 256-coordinate heads and a causal window of 512 positions. Global attention uses 512-coordinate heads and can read every preceding position. Queries and keys have learned RMSNorm scales; values are RMS-normalized without a learned scale. RoPE rotates all local query/key coordinates and a quarter of the global coordinates. Scores use scale 1.',
 'sharing.description':'Blocks 0–14 compute their own keys and values. From block 15 onwards, sliding-attention blocks reuse block 13’s K/V and global-attention blocks reuse block 14’s K/V. Every block still makes its own queries and output projection. The sharing is across layers during one forward pass; this implementation retains no state between generation passes.',
 'head.description':'Project each normalized residual vector to this head’s query coordinates and apply RMSNorm and RoPE. The shared keys receive their own projection, RMSNorm and RoPE; shared values receive a separate projection and RMSNorm without a learned scale. Compute query–key dot products, mask unavailable positions, apply softmax and mix the value vectors. Gemma E2B uses no inverse-square-root head-size factor on these scores.',
 'rope.description':'RoPE rotates paired query/key coordinates according to sequence position. Local heads rotate all 256 coordinates using base 10,000. Global heads use proportional RoPE with base 1,000,000: 64 pairs rotate, covering 128 of the 512 coordinates. The remaining coordinate pairs have zero rotation frequency. Pairing uses the two halves of the head vector.',
 'mask.description':'Global attention allows the query’s own position and all earlier positions. Sliding attention also excludes positions at a distance of 512 or more. Unavailable scores are replaced by the dtype’s most negative finite number before float32 softmax. The example’s short sequence fits inside the window.',
 'feedforward.description':'Each position is transformed independently. The gate projection is passed through GELU, then multiplied coordinate by coordinate by a separate value projection. A down projection returns to 1,536 coordinates. Blocks 0–14 expand to 6,144 coordinates; blocks 15–34 expand to 12,288. RMSNorm surrounds the feedforward calculation, with the output normalization applied before residual addition.',
 'output.description':'Apply final RMSNorm and multiply by the transpose of the shared token embedding table to obtain 262,144 scores. Apply 30 × tanh(score / 30) coordinate by coordinate. The final sequence position’s capped logits are used to sample the next token. The shared table is counted once in parameter totals.',
};
export function createGemma4Diagrams(v){
 const {state,C,N,d,fmt,copy,text,line,path,outline,grid,brokenGrid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,attentionMatrix,richText,copyAttributes}=v;
 const t=key=>copy('gemma.'+key),hd=()=>gemmaHeadDim(C(),state.block),ff=()=>gemmaFFDim(C(),state.block),source=()=>gemmaKVSource(C(),state.block),td=(w=d())=>`${N()} × ${fmt(w)}`;
 const axes={rows:'sequence position',columns:'residual coordinate'};
 const spec=(key,title,src,desc,extra={})=>({key:'gemma-'+key,title:title||'Token embedding vectors',source:src,description:desc,...extra});
 const box=(x,y,w,h,label,key,src,desc,extra={})=>operation(x,y,w,h,label,spec(key,label,src,desc,extra));
 const vectors=(x,y,w,h,label,width,key,src,desc)=>matrix(x,y,w,h,label,td(width),spec(key,label,src,desc,{activation:true,rows:Math.min(N(),5)||1,cols:8,axes:{rows:'sequence position',columns:width===C().vocabulary_size?'vocabulary entry':'vector coordinate'},output:td(width)}));
 const frame=(title,w=1700,h=790)=>outline(-30,-20,w,h)+text(0,25,title,29,'diagram-title','start');
 const bound=(name,w=1880,h=900)=>{views[name]={x:-115,y:-50,w,h};};
 function defaultSpec(level){
  const info={model:['Gemma 4 E2B','model','model','model'],embedding:['Input and per-layer embeddings','embedding','embedding','embeddings'],block:[`Transformer block ${state.block}`,'block','block','block'],attention:[C().layer_types[state.block]==='full_attention'?'Global attention':'Sliding-window attention','attention','attention','attention'],head:[`Attention head ${state.head}`,'attention','head',null],feedforward:['Gated feedforward','feedforward','feedforward','feedforward'],depth:['Per-layer embedding contribution','ple','ple','per-layer'],output:['Output projection','output_stage','output','output']}[level];
  return spec(level,info[0],info[1],t(info[2]+'.description'),{kind:'GEMMA 4 E2B',parameterScope:info[3],input:['model','embedding'].includes(level)?`${N()} token IDs`:td(),output:level==='embedding'?`${N()} × ${fmt(d())} residual vectors; ${N()} × 35 × 256 per-layer inputs`:['model','output'].includes(level)?td(C().vocabulary_size):level==='head'?td(hd()):td(),inputAxes:['model','embedding'].includes(level)?undefined:axes,outputAxes:level==='embedding'?{rows:'sequence position',columns:'residual coordinate; layer × per-layer coordinate'}:['model','output'].includes(level)?{rows:'sequence position',columns:'vocabulary entry'}:axes});
 }
 function embeddingOverview(){
  let out=record({...defaultSpec('embedding'),key:'gemma-input',enter:'embedding',box:{x:183,y:103,w:104,h:210}},brokenGrid(188,103,65,170)+text(220,77,'Token table',22,'diagram-title')+outline(193,279,83,34,'#f5f0e5')+text(234,301,'+ per-layer',16,'diagram-title')+text(230,350,'262,144 rows',17,'diagram-shape'));
  out+=line(288,235,297,235)+vectors(309,215,97,40,'',d(),'initial','embedding',t('embedding.description'))+text(357,139,'Token embedding\nvectors × √1,536',21,'diagram-title');return out;
 }
 function embedding(){
  let out=frame('Input and per-layer embeddings',1840,870);
  out+=line(-95,260,3,260)+box(15,210,230,100,'Token table\n262,144 × 1,536','token-table','embedding',t('embedding.description'),{weight:'token_embedding_layer.weight',fontSize:24});
  out+=line(257,260,333,260)+box(345,225,180,70,'Scale × √1,536','scale','embedding',t('embedding.description'),{fontSize:23})+line(537,260,1480,260,'diagram-contribution')+text(1460,227,'Initial residual stream',24,'diagram-title')+line(1480,260,1880,260,'diagram-contribution');
  out+=path('M-70,260 V535 H3')+box(15,475,290,120,'Per-layer token table\n262,144 × 8,960','ple-table','ple_inputs',t('ple.table.description'),{weight:'per_layer_embedding.weight',fontSize:24});
  out+=line(317,535,383,535)+box(395,495,200,80,'Scale × √256\n35 slices × 256','ple-scale','ple_inputs',t('embedding.description'),{fontSize:23});
  out+=path('M665,260 V370 H738')+box(750,330,290,80,'1,536 → 8,960 projection\nscale by 1/√1,536','ple-proj','ple_inputs',t('embedding.description'),{weight:'per_layer_projection.weight',fontSize:22});
  out+=line(1052,370,1118,370)+box(1130,330,200,80,'RMSNorm\neach 256-vector','ple-norm','ple_inputs',t('embedding.description'),{weight:'per_layer_norm.weight',fontSize:23});
  out+=path('M1342,370 H1410 V508 M607,535 H1383','diagram-contribution')+plus(1410,535,{source:'ple_inputs',title:'Add lookup and projected components',description:t('embedding.description')});
  out+=line(1437,535,1493,535)+box(1505,495,210,80,'Scale × 1/√2','combine','ple_inputs',t('embedding.description'),{fontSize:24})+line(1727,535,1880,535,'diagram-contribution');
  out+=text(1630,650,'One 256-coordinate input\nfor each token and block',24,'diagram-note')+text(850,770,'Per-layer inputs are prepared from the initial embeddings before the transformer blocks run.',24,'diagram-note');
  bound('gptEmbedding',2000,970);return out;
 }
 function block(){
  let out=`<path id="block-frame" class="housing" d="${outlinePath(-15,BLOCK.frameTop,BLOCK.width,state.blockHeight)}"/>`+text(28,35,`Transformer block ${state.block}`,27,'diagram-title','start');
  out+=stream(-38,1140,340)+stream(1237,1260,340)+text(25,308,'Residual stream',21,'diagram-note','start');
  for(const [x,end]of [[65,370],[480,790],[865,1130]])out+=path(`M${x},340 V165 H${x+22}`);
  let attention=outline(99,105,250,120)+text(224,130,C().layer_types[state.block]==='full_attention'?'Global attention':'Sliding attention',21,'diagram-title');
  for(const y of [148,159,170,190])attention+=outline(152,y-3,140,6,'#e8f2f1');
  attention+=text(221,181,'…',14,'diagram-note')+text(224,216,'RMS · 8 heads · RMS',16,'diagram-shape');
  out+=record({...defaultSpec('attention'),enter:'attention',box:{x:99,y:105,w:250,h:120}},attention)+path('M361,165 H370 V313')+plus(370,340,{source:'residual',title:'Add attention contribution',description:t('block.description')});
  out+=box(514,105,250,120,'Gated feedforward','ff','feedforward',t('feedforward.description'),{enter:'feedforward',fontSize:24})+text(639,215,`RMS · ${fmt(ff())} wide · RMS`,16,'diagram-shape')+path('M776,165 H790 V313')+plus(790,340,{source:'residual_ff',title:'Add feedforward contribution',description:t('block.description')});
  out+=box(899,105,205,120,'Per-layer\nembedding gate','ple','ple',t('ple.description'),{enter:'depth',fontSize:23})+path('M1116,165 H1130 V313')+plus(1130,340,{source:'ple',title:'Add per-layer contribution',description:t('ple.description')});
  out+=box(1160,307,55,66,'Scale','layer-scale','block',t('block.description'),{weight:param('layer_scale'),fontSize:21});
  out+=`<foreignObject id="block-prose" x="28" y="451" width="1150" height="120"><div xmlns="http://www.w3.org/1999/xhtml" id="block-prose-text" class="block-prose" ${copyAttributes(t('block.description'))}>${richText(t('block.description'))}</div></foreignObject>`;
  return out;
 }
 function attention(){
  let out=frame(C().layer_types[state.block]==='full_attention'?'Global attention':'Sliding-window attention',1820,890);
  out+=stream(-95,1860,795)+text(0,762,'Residual stream',23,'diagram-note','start')+path('M-55,795 V400 H3');
  out+=box(15,365,86,70,'RMS\nNorm','input-norm','norm',t('norm.description'),{weight:param('attention_norm.weight')})+line(113,400,175,400)+path('M175,190 V590','diagram-wire');
  out+=`<rect x="220" y="100" width="485" height="560" fill="none" stroke="#9baea9" stroke-dasharray="3 6"/>`+text(462,139,'8 heads · shared K and V',24,'diagram-title');
  for(const [i,y]of [190,275,360,475,590].entries()){
   const h=[0,1,2,3,7][i];out+=line(175,y,248,y)+operation(260,y-21,215,42,i===3?'… heads 3–6 …':`Head ${h}`,spec('head-'+h,`Head ${h}`,'attention',t('head.description'),{head:h,enter:'head',output:td(hd())}))+line(487,y,538,y)+grid(550,y-16,115,32,true,Math.min(N(),5)||1,6)+line(677,y,760,y);
  }
  out+=path('M760,190 V590','diagram-wire')+line(760,400,813,400)+box(825,360,180,80,'Join head\noutputs','join','combine',t('attention.description'),{output:td(8*hd())});
  out+=line(1017,400,1083,400)+box(1095,360,210,80,'Output projection','out','combine',t('attention.description'),{weight:param('attention.output_proj.weight'),input:td(8*hd()),output:td(),fontSize:23});
  out+=line(1317,400,1383,400)+box(1395,365,110,70,'RMS\nNorm','out-norm','norm',t('block.description'),{weight:param('attention_output_norm.weight')});
  out+=path('M1517,400 H1710 V768','diagram-contribution')+plus(1710,795,{source:'residual',title:'Add attention contribution',description:t('block.description')});
  out+=record(spec('sharing','Cross-layer K/V sharing','attention',t('sharing.description')),text(1130,683,source()===state.block?'This block computes its own K and V.':`K and V come from block ${source()}.`,25,'diagram-note'));
  bound('kimiAttention',1990,995);return out;
 }
 function head(){
  const global=C().layer_types[state.block]==='full_attention',src=source(),prefix=`transformer_blocks.${src}.attention.`;
  let out=frame(`Attention head ${state.head} · ${global?'global':'512-position window'}`,2110,850);
  out+=line(-95,322,3,322)+box(15,277,190,90,'Q projection','qproj','q',t('head.description'),{weight:param('attention.query_proj.weight'),input:td(),output:td(hd()),fontSize:24});
  out+=line(217,322,268,322)+box(280,277,185,90,'RMSNorm\n+ RoPE','qnorm','rope',t('rope.description'),{fontSize:24});
  out+=line(477,322,548,322)+vectors(560,287,120,70,'Q',hd(),'q','q',t('head.description'));
  out+=box(15,490,220,100,src===state.block?'K projection\nRMSNorm + RoPE':`K from block ${src}`,'k','k',t('sharing.description'),{weight:src===state.block?prefix+'key_proj.weight':undefined,output:td(hd()),fontSize:23});
  out+=line(-95,540,3,540)+line(247,540,548,540)+vectors(560,505,120,70,'K',hd(),'k-matrix','k',t('head.description'));
  out+=line(692,322,763,322)+path('M692,540 H835 V372');
  out+=box(775,285,120,75,'Q Kᵀ','score-op','scores',t('head.description'),{fontSize:28,labelLines:['Q Kᵀ'],labelKinds:['diagram-math']});
  out+=line(907,322,968,322)+attentionMatrix(980,260,125,'scores',spec('scores','Attention scores','scores',t('head.description'),{inspectable:false}));
  out+=line(1117,322,1168,322)+box(1180,285,180,75,'Mask + softmax','softmax','softmax',t('mask.description'),{fontSize:22});
  out+=line(1372,322,1423,322)+attentionMatrix(1435,260,125,'weights',spec('weights','Attention weights','softmax',t('mask.description'),{inspectable:false}));
  out+=line(1572,322,1633,322)+box(1645,285,175,75,'Weights × V','mix','mix',t('head.description'),{fontSize:24})+line(1832,322,1888,322);
  out+=vectors(1900,302,120,40,'Head output',hd(),'head-output','mix',t('head.description'))+line(2032,322,2160,322,'diagram-contribution');
  out+=box(775,615,320,85,src===state.block?'V projection · RMSNorm':`V from block ${src}`,'v','v',t('head.description'),{weight:src===state.block?prefix+'value_proj.weight':undefined,output:td(hd()),fontSize:23})+line(-95,657,763,657)+path('M1107,657 H1732 V372');
  out+=text(1000,789,`Q/K/V width: ${hd()} · score scale: 1 · ${global?'128 rotated coordinates':'256 rotated coordinates'}`,24,'diagram-note');bound('kimiHead',2270,960);return out;
 }
 function feedforward(){
  let out=frame('Gated feedforward',1790,830);
  out+=stream(-95,1830,740)+text(0,707,'Residual stream',23,'diagram-note','start')+path('M-55,740 V350 H3');
  out+=box(15,315,90,70,'RMS\nNorm','norm','norm',t('norm.description'),{weight:param('feed_forward_norm.weight')})+line(117,350,175,350)+path('M175,350 V205 H228 M175,350 V480 H228','diagram-wire');
  out+=box(240,160,215,90,'Gate projection','gate','gate',t('feedforward.description'),{weight:param('feed_forward.gate_proj.weight'),input:td(),output:td(ff()),fontSize:24});
  out+=line(467,205,528,205)+box(540,165,175,80,'GELU','gelu','gelu',t('feedforward.description'));
  out+=box(240,435,215,90,'Value projection','value','value',t('feedforward.description'),{weight:param('feed_forward.value_proj.weight'),input:td(),output:td(ff()),fontSize:23});
  out+=path('M727,205 H820 V313 M467,480 H820 V387')+box(795,325,50,50,'×','product','feedforward',t('feedforward.description'));
  out+=line(857,350,928,350)+box(940,305,235,90,'Down projection','down','down',t('feedforward.description'),{weight:param('feed_forward.output_proj.weight'),input:td(ff()),output:td(),fontSize:24});
  out+=line(1187,350,1258,350)+box(1270,315,110,70,'RMS\nNorm','post-norm','norm',t('block.description'),{weight:param('feed_forward_output_norm.weight')});
  out+=line(1392,350,1453,350)+vectors(1465,330,125,40,'Contribution',d(),'out','feedforward',t('feedforward.description'))+path('M1602,350 H1690 V713','diagram-contribution')+plus(1690,740,{source:'residual_ff',title:'Add feedforward contribution',description:t('block.description')});
  out+=text(810,630,`Block ${state.block}: ${fmt(d())} → ${fmt(ff())} → ${fmt(d())} coordinates`,24,'diagram-note');bound('kimiFeedforward',1960,940);return out;
 }
 function depth(){
  let out=frame('Per-layer embedding contribution',1800,870);
  out+=stream(-95,1840,750)+text(0,717,'Residual after attention and feedforward',23,'diagram-note','start')+path('M-55,750 V360 H3');
  out+=box(15,315,220,90,'Gate projection\n1,536 → 256','gate','ple',t('ple.description'),{weight:param('per_layer_gate.weight'),fontSize:24})+line(247,360,318,360)+box(330,320,160,80,'GELU','gelu','ple',t('ple.description'));
  out+=box(330,105,355,90,'Per-layer input\nfrom the initial embeddings','input','ple_inputs',t('embedding.description'),{enter:'embedding',output:td(256),fontSize:23})+path('M697,150 H760 V323');
  out+=line(502,360,723,360)+box(735,335,50,50,'×','product','ple',t('ple.description'))+line(797,360,858,360);
  out+=box(870,315,260,90,'Projection\n256 → 1,536','projection','ple',t('ple.description'),{weight:param('per_layer_output.weight'),fontSize:24})+line(1142,360,1213,360);
  out+=box(1225,325,120,70,'RMS\nNorm','norm','norm',t('norm.description'),{weight:param('per_layer_output_norm.weight')})+path('M1357,360 H1640 V723','diagram-contribution')+plus(1640,750,{source:'ple',title:'Add per-layer embedding contribution',description:t('ple.description')});
  out+=text(885,585,'The gate depends on the current residual vector. The per-layer input was prepared before block 0.',24,'diagram-note');bound('kimiDepth',1970,980);return out;
 }
 function output(){
  let out=frame('Output projection',1760,590);
  out+=stream(-95,123,285)+box(135,245,150,80,'RMSNorm','norm','norm',t('norm.description'),{weight:'output_norm.weight'});
  out+=line(297,285,388,285)+record(spec('table','Shared token table','output',t('output.description'),{weight:'output_layer.weight',input:td(),output:td(C().vocabulary_size),codePosition:{x:560,y:137,size:16}}),brokenGrid(400,160,160,250)+text(480,137,'Shared token table',23,'diagram-title')+dimensions(480,450,'262,144 × 1,536',parameterAxes('output_layer.weight'),18));
  out+=line(572,285,728,285)+box(740,235,350,100,'30 × tanh(logits / 30)','softcap','softcap',t('output.description'),{fontSize:27})+line(1102,285,1223,285);
  out+=vectors(1235,245,245,80,'Capped logits',C().vocabulary_size,'logits','output',t('output.description'))+line(1492,285,1800,285,'diagram-contribution');
  out+=text(930,515,'The token embedding and output projection use the same learned table.',24,'diagram-note');bound('kimiOutput',1940,710);return out;
 }
 return {defaultSpec,embeddingOverview,embedding,block,attention,head,feedforward,depth,output,isDense:()=>true,isDelta:()=>false,attentionName:()=>C().layer_types[state.block]==='full_attention'?'Global attention':'Sliding-window attention'};
}
