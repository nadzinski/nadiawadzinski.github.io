import {illustrativeRouting} from './moe-routing.mjs';
import {parameterAxes} from './matrix-info.mjs';

export function illustrativeGLMRouting(position,block){
  const logits=illustrativeRouting(288,8,position,block).logits;
  const bias=illustrativeRouting(288,8,position+7,block+3).logits.map(x=>x*.08);
  const selected=logits.map((x,expert)=>({expert,score:1/(1+Math.exp(-x))})).map(item=>({...item,choice:item.score+bias[item.expert]})).sort((a,b)=>b.choice-a.choice).slice(0,8);
  const sum=selected.reduce((total,item)=>total+item.score,0);
  return selected.map(item=>({...item,weight:item.score/sum*2.5}));
}

export const glmText={
  "block.description": "Four residual streams pass through the block. Before each sublayer, mHC computes a weighted mixture as its input. It also mixes the bypassed streams and distributes the sublayer’s output among them. These operations happen separately at each sequence position; attention can then mix information across positions.",
  "block.footer": "Each line represents one residual stream, containing {positions} position vectors.\nmHC mixes the four streams at the same sequence position. Attention mixes across sequence positions.",
  "depth.title": "Manifold-constrained hyper-connections",
  "depth.description": "mHC reads all four streams at each position and computes three sets of coefficients: four collapse weights, four distribution weights, and a 4 × 4 stream-mixing matrix. The branch receives a weighted sum of the input streams. Its output is distributed back and added to the mixed bypass. The attention and feedforward sublayers have separate mHC parameters.",
  "depth.note": "The coefficients depend on the input at this position.\nThe four streams are separate workspaces; each contains every sequence position.",
  "attention.description": "All 64 heads receive the same mixture after mHC and RMSNorm. KDA uses a recurrent matrix state; sparse latent attention reads positions selected by its indexer. Each head’s result passes through its slice of the output projection. The summed contribution returns to mHC for distribution among the streams.",
  "attention.input": "From mHC",
  "attention.return": "Contribution to mHC",
  "attention.heads": "64 heads",
  "attention.kda-note": "One query/key/value set per head · 128 coordinates · coordinate-wise state decay",
  "attention.others": "… heads 3–62 …",
  "attention.sum": "Sum all 64",
  "attention.norm": "RMSNorm normalizes each mixed 4,096-coordinate vector and applies a learned scale. Epsilon is 10⁻⁵.",
  "kda.description": "Kimi Delta Attention uses a 128 × 128 state per head. It decays each key-coordinate row separately, reads the value associated with the current key, writes a correction, and reads an output using the current query. State is reset on every forward pass.",
  "kda.projections": "Q / K / V\nprojections",
  "kda.project.description": "Each of the 64 KDA heads has independent Q, K and V projections from 4,096 coordinates to 128. Four-position causal convolution and SiLU are applied separately to every projected coordinate.",
  "kda.conv": "Causal conv\n+ SiLU",
  "kda.conv.description": "Apply a separate learned four-position filter to each projected coordinate, then SiLU. Each filter reads this position and up to three preceding positions. Q, K and V have separate filters.",
  "kda.norm": "L2 norm\nQ and K",
  "kda.norm.description": "Q and K are L2-normalized in float32: divide by the square root of the sum of squared coordinates plus 10⁻⁶. The recurrence keeps these values in float32. Scale the query by 1/√128 when reading the state.",
  "kda.scan": "Scan sequence\npositions",
  "kda.decay": "Coordinate-wise decay",
  "kda.decay.description": "The input passes through 4,096 → 128 → 8,192 projections to produce 128 decay controls per head. For each coordinate: log α = −5 × sigmoid(exp(A_log) × (control + dt_bias)); α = exp(log α). Each key-coordinate row of the state has its own decay.",
  "kda.write": "Write strength β",
  "kda.write.description": "A separate 4,096 → 64 projection and sigmoid produce one write-strength scalar per head and position. It scales the correction written into the state.",
  "kda.gate": "Output norm\n× sigmoid gate",
  "kda.gate.description": "A low-rank 4,096 → 128 → 8,192 projection supplies 128 gate values per head. RMSNorm normalizes each state readout using a scale shared across heads, then a sigmoid gate multiplies it coordinate by coordinate. There is no activation between the two gate projections.",
  "kda.state": "Previous state",
  "kda.decay-step": "Decay rows\nS ← diag(αₜ) S",
  "kda.correction": "Read and correct\nδ = βₜ (vₜ − kₜ S)\nS ← S + kₜᵀ δ",
  "kda.read": "Read output\nyₜ = (qₜ / √128) S",
  "kda.state-note": "The updated state is carried to the next sequence position.\nThis implementation discards it at the end of the forward pass.",
  "mla.description": "Sparse latent attention uses a 1,536-coordinate query bottleneck and a shared 512-coordinate K/V bottleneck. Each of 64 heads expands these into 256-coordinate queries, keys and values. A separate pool indexer chooses the visible positions read by all main attention heads. No RoPE or output gate is applied.",
  "mla.query": "Query latent\n+ RMSNorm",
  "mla.query.description": "Project each 4,096-coordinate vector to a 1,536-coordinate query latent, apply RMSNorm, then expand to 64 queries of 256 coordinates each. The normalized query latent also supplies the indexer’s separate queries.",
  "mla.kv": "K/V latent\n+ RMSNorm",
  "mla.kv.description": "Project each input to a 512-coordinate K/V latent and apply RMSNorm. Expand it into 256-coordinate keys and values for every head. These remain individual position vectors, even though the separate indexer uses compressed pools to choose positions.",
  "mla.expand": "Per-head\nprojections",
  "mla.scores": "Q Kᵀ",
  "mla.mask": "Causal mask",
  "mla.softmax": "Scale\n+ softmax",
  "mla.scores.description": "Each main attention head computes query–key dot products. Only positions selected by the indexer remain eligible; its selection already enforces causality. Divide by √256 and apply float32 softmax, then mix the individual value vectors.",
  "mla.mix": "Weights × V",
  "mla.note": "Sparse attention · 256 query/key/value coordinates · no RoPE",
  "moe.description": "The router scores all 288 experts using sigmoid. A correction bias affects the top-eight selection. The original scores of the selected experts are renormalized and multiplied by 2.5. Each expert reads the full 4,096-coordinate vector. Sum their weighted outputs and add one always-active shared expert.",
  "moe.title": "Mixture of experts",
  "moe.illustrative": "Illustrative routing at position {position} · 8 selected from 288",
  "moe.router": "Router\n+ sigmoid",
  "moe.select": "Selection bias\n+ top eight",
  "moe.normalize": "Renormalize\n× 2.5",
  "moe.router.description": "Compute 288 logits in float32 and apply sigmoid. Add the correction bias for selection, choose eight experts, and retrieve their original sigmoid scores. Divide these by their sum, then multiply by 2.5. The routing weights sum to 2.5.",
  "moe.return.description": "Sum the weighted 4,096-coordinate outputs of the eight selected experts, then add the shared expert’s output. This contribution returns to mHC, which distributes it among the four residual streams.",
  "moe.shared": "Shared expert\nalways active",
  "moe.shared.description": "One shared 4,096 → 2,048 → 4,096 expert reads every normalized input vector. Add its output to the weighted sum of the eight routed experts. The shared path has no routing weight or 2.5 multiplier.",
  "moe.bank": "288 routed experts\n8 selected per position",
  "moe.selected": "Selected experts",
  "moe.weight": "Routing weight",
  "moe.contribution": "Expert contribution",
  "moe.sum": "Weighted sum",
  "expert.title": "Inside expert {expert}",
  "expert.description": "Each expert has its own 4,096 → 2,048 → 4,096 projections. Cap the gate projection above at 10 before SiLU, and clamp the value path to [−10, 10]. Multiply the two paths coordinate by coordinate, then project back to 4,096.",
  "expert.input": "Expert input",
  "expert.output": "Expert output",
  "expert.gate": "Gate projection",
  "expert.value": "Value projection",
  "expert.situ": "SiLU",
  "expert.bound": "Clamp to [−10, 10]",
  "expert.out": "Output projection",
  "expert.situ.description": "The gate path is capped above at 10, then passed through SiLU: g × sigmoid(g). The value path is clamped to [−10, 10]. Multiply the two paths coordinate by coordinate. Negative gate values are not clamped below.",
  "expert.note": "Every expert has its own learned projections. The same weights process every position assigned to that expert.",
  "dense.title": "Dense feedforward · block {block}",
  "dense.description": "Blocks 0–2 use a dense 4,096 → 12,288 → 4,096 feedforward network. Every position uses its three learned projections. Gate and value paths use the same clamps and SiLU as the later experts.",
  "output.description": "Take the unweighted mean of the four final residual streams at each sequence position. RMSNorm normalizes that vector, then an independent output matrix maps its 4,096 coordinates to 154,880 logits. Generation uses the tokenizer’s 154,856 entries.",
  "output.mix": "Mean of\nfour streams",
  "output.norm": "RMSNorm",
  "output.project": "Output matrix",
  "output.logits": "Logits",
  "model.description": "Token IDs select embedding vectors, which are copied into four residual streams. Forty-five transformer blocks apply KDA or sparse latent attention, followed by dense feedforward or a mixture of experts. mHC combines and updates the streams at each sublayer. Their final mean passes through RMSNorm and an output projection to produce logits.",
  "expert.selected": "This expert is selected for position {position} in the illustrative routing example.",
  "expert.unselected": "This expert is not selected for position {position} in the illustrative routing example. This view shows its structure; it contributes nothing at that position.",
  "dense.input": "Input vector",
  "dense.output": "Output vector",
  "dense.note": "Each position uses the same dense feedforward network.\nThe later routed and shared experts use the same activation functions.",
  "index.title": "Sparse-attention pool indexer",
  "index.description": "The indexer has separate projections from the main attention. Its 32 query heads score shared 128-coordinate keys compressed in groups of four. It chooses up to 512 complete pools, expands each selected pool into four original positions, and includes the unfinished tail of up to three positions. Main attention then reads the individual K/V vectors at those positions.",
  "index.pool": "For every coordinate, a learned projection supplies a score for each of the four positions. Add a learned slot bias, apply softmax over positions within the pool, and take the weighted sum of the normalized index keys. Every coordinate has its own pooling weights.",
  "index.score": "Each index query takes a dot product with a pooled key, scaled by 1/√128 and passed through ReLU. A separate 4,096 → 32 projection supplies signed head weights, scaled by 1/√32. Sum the weighted head scores to obtain one score per pool. These head weights do not use softmax.",
  "index.select": "A pool can be selected only when its final position is visible. Choose up to 512 complete pools, giving up to 2,048 positions. Append the current incomplete pool’s raw positions (at most three). All main attention heads use the same selected positions for this query.",
  "hc.mapping": "Flatten the four 4,096-coordinate vectors, RMS-normalize them without a learned scale, then apply a 16,384 → 24 projection. Learned base values and three scales transform these logits into collapse, distribution and mixing coefficients.",
  "hc.pre": "Four sigmoid coefficients (plus 10⁻⁶) weight the four input streams. Sum those weighted vectors to form the branch input. These collapse weights are not normalized to sum to one.",
  "hc.post": "Four coefficients are computed as 2 × sigmoid(logit). Multiply the sublayer’s output vector by each coefficient to get its contribution to that output stream.",
  "hc.mix": "A learned, input-dependent 4 × 4 matrix mixes the bypassed streams. Softmax and alternating row/column normalization (20 iterations) make it approximately doubly stochastic. Entry (i, j) weights input stream i in output stream j."
};

export function createGLMDiagrams(v){
  const {state,C,N,d,fmt,copy,text,line,path,outline,grid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,attentionMatrix,richText,copyAttributes}=v;
  const t=(key,vars={})=>copy('glm.'+key,{block:state.block,...vars});
  const td=(width=d(),rows=N())=>`${rows} × ${fmt(width)}`;
  const isDelta=()=>C().layer_types[state.block]==='linear_attention';
  const isDense=()=>state.block<3;
  const hd=()=>isDelta()?C().linear_head_dim:C().head_dim;
  const attentionName=()=>isDelta()?'Kimi Delta Attention':'Sparse latent attention';
  const axes={rows:'sequence position',columns:'coordinate'};
  const spec=(key,title,source,description,extra={})=>({key:'glm-'+key,title,source,description,...extra});
  const box=(x,y,w,h,label,key,source,description,extra={})=>operation(x,y,w,h,label,spec(key,label,source,description,extra));
  const projection=(x,y,w,h,label,key,source,description,suffix,extra={})=>{
    const weight=param(suffix),learnedAxes=parameterAxes(weight,false,C());
    return box(x,y,w,h,label,key,source,description,{weight,inputAxes:{rows:'sequence position',columns:learnedAxes.columns},outputAxes:{rows:'sequence position',columns:learnedAxes.vector?learnedAxes.columns:learnedAxes.rows},...extra});
  };
  const headSpec=head=>spec('head-'+head,`Head ${head}`,isDelta()?'delta':'attention',t(isDelta()?'kda.description':'mla.description'),{head,enter:'head',input:td(),output:td(hd())});
  const expertSpec=expert=>spec('expert-'+expert,expert===-1?'Shared expert':`Expert ${expert}`,'expert',t(expert===-1?'moe.shared.description':'expert.description'),{expert,enter:'expert',parameterScope:expert===-1?'shared-experts':'expert',input:td(expert===-1?d():d(),1),output:td(expert===-1?d():d(),1)});
  const rowMatrix=(x,y,w,label,width,key,description,rows=N())=>matrix(x,y,w,Math.max(1,Math.min(rows,5))*15,label,td(width,rows),spec(key,label,key.startsWith('expert')?'expert':key==='logits'?'output':key.startsWith('source')?'hc':isDelta()?'delta':'attention',description,{activation:true,rows:Math.max(1,Math.min(rows,5)),axes,output:td(width,rows)}));

  function defaultSpec(level){
    const info={
      model:['GLM-5.3-Flash · text backbone','model','model.description','model'],
      block:[`Transformer block ${state.block}`,'block','block.description','block'],
      attention:[attentionName(),isDelta()?'delta':'attention','attention.description','attention'],
      head:[`${isDelta()?'KDA':'MLA'} head ${state.head}`,isDelta()?'delta':'attention',isDelta()?'kda.description':'mla.description',null],
      feedforward:[t(isDense()?'dense.title':'moe.title'),isDense()?'expert':'feedforward',isDense()?'dense.description':'moe.description','feedforward'],
      depth:[t('depth.title'),'hc','depth.description',state.depthBranch==='attention'?'hyperconnection-attention':'hyperconnection-feedforward'],
      indexer:[t('index.title'),'indexer','index.description','indexer'],
      output:['Output projection','output_stage','output.description','output'],
      expert:[state.expert===-1?'Shared expert':`Expert ${state.expert}`,'expert',state.expert===-1?'moe.shared.description':'expert.description',state.expert===-1?'shared-experts':'expert'],
    }[level];
    const vectorAxes={rows:'sequence position',columns:['block','depth','output'].includes(level)?'4 streams × residual coordinate':'residual coordinate'};
    return spec(level,info[0],info[1],t(info[2]),{kind:level==='model'?'MODEL OVERVIEW':'GLM-5.3-FLASH',parameterScope:info[3],expert:state.expert,note:level==='expert'&&state.expert>=0?t(illustrativeGLMRouting(state.moePosition,state.block).some(item=>item.expert===state.expert)?'expert.selected':'expert.unselected',{position:state.moePosition+1}):undefined,input:['block','output','depth'].includes(level)?`${N()} × 4 × ${fmt(d())}`:level==='model'?`${N()} token IDs`:level==='expert'?td(state.expert===-1?d():d(),1):td(),output:level==='indexer'?`${N()} × ${N()} boolean mask`:['block','depth'].includes(level)?`${N()} × 4 × ${fmt(d())}`:['model','output'].includes(level)?td(C().vocabulary_size):level==='head'?td(hd()):level==='expert'?td(state.expert===-1?d():d(),1):td(),inputAxes:level==='model'?undefined:vectorAxes,outputAxes:level==='indexer'?{rows:'query position',columns:'key position'}:['model','output'].includes(level)?{rows:'sequence position',columns:'vocabulary entry'}:level==='head'?{rows:'sequence position',columns:'head coordinate'}:vectorAxes});
  }

  function block(){
    let out=`<path id="block-frame" class="housing" d="${outlinePath(-15,BLOCK.frameTop,BLOCK.width,state.blockHeight)}"/>`+text(28,35,`Transformer block ${state.block}`,27,'diagram-title','start')+dimensions(1180,35,`${N()} × 4 × ${fmt(d())}`,{rows:'sequence position',columns:'stream × residual coordinate'},19,'end');
    for(const y of [319,333,347,361])out+=stream(-38,575,y)+stream(719,1093,y)+stream(1227,1260,y);
    out+=text(30,291,'Four residual streams',19,'diagram-note','start');
    out+=path('M135,319 V165 H178 M135,319 V361','diagram-wire');
    out+=box(190,125,100,80,'mHC','attn-reader','hc',t('depth.description'),{enter:'depth',depthBranch:'attention'});
    out+=line(302,165,319,165);
    out+=record(spec('attention',attentionName(),'residual',t('attention.description'),{enter:'attention',input:td(),output:td(),box:{x:331,y:105,w:264,h:122}}),outline(331,105,264,122)+text(463,130,isDelta()?'Kimi Delta Attention':'Sparse latent attention',21,'diagram-title')+text(463,215,'RMSNorm · 64 heads',16,'diagram-shape')+path('M385,147 V188 M540,147 V188','diagram-wire')+[147,158,169,188].map(y=>line(385,y,402,y)+outline(414,y-3,93,6,isDelta()?'#f8f3e9':'#e8f2f1')+line(519,y,540,y,'diagram-contribution')).join('')+text(460,180,'…',14,'diagram-note'));
    out+=path('M607,165 H647 V283','diagram-contribution');
    out+=box(587,295,120,100,'Mix streams\n+ add output','attn-mix','hc_add',t('depth.description'),{enter:'depth',depthBranch:'attention',fontSize:18});
    out+=path('M240,217 V261 H615 V283','moe-control');
    out+=path('M751,319 V165 H774 M751,319 V361','diagram-wire');
    out+=box(786,125,100,80,'mHC','ff-reader','hc',t('depth.description'),{enter:'depth',depthBranch:'feedforward'});
    out+=line(898,165,902,165);
    out+=box(914,105,190,122,isDense()?'Dense\nfeedforward':'Mixture of\nexperts','feedforward','residual_ff',t(isDense()?'dense.description':'moe.description'),{enter:'feedforward',input:td(),output:td(),fontSize:22});
    out+=text(1009,216,isDense()?'RMSNorm · SiLU':'RMSNorm · 8 + 1',17,'diagram-shape');
    out+=path('M1116,165 H1160 V283','diagram-contribution');
    out+=box(1105,295,110,100,'Mix streams\n+ add output','ff-mix','hc_add',t('depth.description'),{enter:'depth',depthBranch:'feedforward',fontSize:18});
    out+=path('M836,217 V261 H1128 V283','moe-control');
    out+=text(1175,430,state.block===44?'to final mean':`on to block ${state.block+1}`,20,'diagram-note','end');
    out+=`<foreignObject id="block-prose" x="28" y="466" width="1150" height="100"><div xmlns="http://www.w3.org/1999/xhtml" id="block-prose-text" class="block-prose" ${copyAttributes(t('block.footer'))}>${richText(t('block.footer'))}</div></foreignObject>`;
    return out;
  }

  function attention(){
    const inputY=375,ys=[165,220,275,395,555],heads=[0,1,2,3,63];
    let out=outline(-300,-10,1440,795)+text(-268,35,attentionName(),30,'diagram-title','start')+text(1095,35,`Block ${state.block}`,20,'diagram-shape','end');
    out+=text(-268,79,t(isDelta()?'attention.kda-note':'mla.note'),19,'diagram-note','start');
    out+=line(-345,inputY,-245,inputY)+text(-264,inputY-65,t('attention.input'),20,'diagram-note');
    out+=projection(-233,inputY-35,86,70,'RMS\nNorm','input-norm','norm',t('attention.norm'),'attention_norm.weight',{input:td(),output:td()});
    out+=line(-135,inputY,-70,inputY)+path('M-70,165 V555','diagram-wire');
    out+=text(210,124,t('attention.heads'),23,'diagram-title')+text(570,124,'Output projection',23,'diagram-title');
    ys.forEach((y,index)=>{
      out+=line(-70,y,8,y);
      out+=operation(20,y-20,370,40,index===3?t('attention.others'):`Head ${heads[index]}`,headSpec(heads[index]));
      out+=line(402,y,460,y)+box(472,y-20,195,40,index===3?'60 head contributions':`Wₒ⁽${heads[index]}⁾`,'slice-'+index,isDelta()?'delta_output':'combine',t('attention.description'),{input:index===3?`60 head outputs, each ${td(hd())}`:td(hd()),output:td(),fontSize:index===3?17:undefined});
      out+=line(679,y,730,y,'diagram-contribution')+grid(742,y-15,140,30,true,Math.max(1,Math.min(N(),5)),8)+line(894,y,1025,y,'diagram-contribution');
    });
    out+=path('M1025,165 V620','diagram-contribution')+box(930,632,190,48,t('attention.sum'),'attention-sum',isDelta()?'delta_output':'combine',t('attention.description'),{output:td()});
    out+=line(1025,692,1025,720,'diagram-contribution')+line(1025,720,1190,720,'diagram-contribution')+text(1020,756,t('attention.return'),20,'diagram-note');
    if(!isDelta())out+=box(95,655,340,70,'Pool indexer','indexer','indexer',t('index.description'),{enter:'indexer',parameterScope:'indexer'})+path('M8,134 H403 V587 H8 Z','glm-selection-group')+path('M-70,375 V690 H83 M265,643 V599','moe-control');
    views.kimiAttention={x:-365,y:-35,w:1600,h:845};
    return out;
  }

  function kdaHead(){
    let out=outline(-95,-20,2100,900)+text(-60,25,`Kimi Delta Attention · head ${state.head}`,28,'diagram-title','start');
    out+=line(-145,195,-2,195)+rowMatrix(10,157,96,'Normalized\nembedding vectors',d(),'kda-input',t('attention.description'));
    const steps=[
      [170,170,t('kda.projections'),'project','delta_qkv','kda.project.description'],
      [390,190,t('kda.conv'),'conv','conv','kda.conv.description'],
      [630,170,t('kda.norm'),'norm','delta_norm','kda.norm.description'],
      [865,300,t('kda.scan'),'scan','delta_scan','kda.description'],
      [1230,255,t('kda.gate'),'gate','delta_output','kda.gate.description'],
    ];
    let previous=118;
    for(const[x,w,label,key,source,desc]of steps){out+=line(previous,195,x-12,195)+box(x,145,w,100,label,'kda-'+key,source,t(desc),{input:key==='project'?td():key==='conv'||key==='norm'||key==='scan'?`Q, K, V: each ${td(128)}`:td(128),output:['project','conv','norm'].includes(key)?`Q, K, V: each ${td(128)}`:td(128)});previous=x+w+12;}
    out+=line(previous,195,1570,195)+rowMatrix(1582,157,155,'Gated head output',128,'kda-output',t('kda.gate.description'))+line(1749,195,2045,195,'diagram-contribution')+text(1860,169,'to output projection',20,'diagram-note');
    out+=path('M127,195 V351 H237','diagram-wire')+box(249,311,490,80,t('kda.decay'),'kda-decay','delta_decay',t('kda.decay.description'),{input:td(),output:td(128)})+path('M751,351 H925 V257');
    out+=box(795,400,360,70,t('kda.write'),'kda-write','delta_gates',t('kda.write.description'),{input:td(),output:`${N()} scalars per head`})+path('M127,351 V435 H783 M1167,435 H1200 V277 H1030 V257');
    out+=path('M127,435 V489 H1335 V257','moe-control')+text(640,516,'Output gate reads the same normalized input',20,'diagram-note');
    out+=text(-55,532,'At sequence position t',25,'diagram-title','start');
    out+=matrix(0,585,140,125,t('kda.state'),'128 × 128',spec('state',t('kda.state'),'delta_scan',t('kda.description'),{activation:true,axes:{rows:'key coordinate',columns:'value coordinate'},output:'128 × 128'}));
    out+=line(152,647,208,647)+box(220,605,295,84,t('kda.decay-step'),'decay-state','delta_scan',t('kda.decay.description'));
    out+=line(527,647,582,647)+box(594,577,440,140,t('kda.correction'),'correct','delta_write',t('kda.description'));
    out+=line(1046,647,1100,647)+matrix(1112,585,140,125,'Updated state','128 × 128',spec('updated-state','Updated state','delta_write',t('kda.state-note'),{activation:true,axes:{rows:'key coordinate',columns:'value coordinate'}}));
    out+=line(1264,647,1320,647)+box(1332,605,390,84,t('kda.read'),'read','delta_read',t('kda.norm.description'))+line(1734,647,1950,647,'diagram-contribution');
    out+=path('M1264,647 H1284 V780 H-32 V647 H-12','diagram-residual')+text(875,822,t('kda.state-note'),22,'diagram-note');
    views.kimiHead={x:-160,y:-45,w:2240,h:990};
    return out;
  }

  function mlaHead(){
    let out=outline(-95,-20,2110,930)+text(-60,25,`Sparse latent attention · head ${state.head}`,28,'diagram-title','start')+text(-60,68,t('mla.note'),22,'diagram-note','start');
    out+=line(-140,220,-2,220)+rowMatrix(10,182,100,'Normalized\nembedding vectors',d(),'mla-input',t('mla.description'));
    out+=path('M122,220 H147 V150 H193 M147,220 V360 H193','diagram-wire');
    out+=projection(205,110,260,80,t('mla.query'),'mla-query','mla_q',t('mla.query.description'),'attention.q_a_proj.weight',{input:td(),output:td(1536)});
    out+=projection(205,320,260,80,t('mla.kv'),'mla-kv','mla_kv',t('mla.kv.description'),'attention.kv_a_proj.weight',{input:td(),output:td(512)});
    out+=line(477,150,520,150)+line(477,360,520,360);
    out+=box(532,110,215,80,'Query projection','mla-qb','mla_q',t('mla.query.description'),{input:td(1536),output:td(256)});
    out+=box(532,320,215,80,'Key / value\nprojections','mla-kvb','mla_kv',t('mla.kv.description'),{input:td(512),output:`K and V: each ${td(256)}`});
    out+=path('M759,150 H810 V238 H854 M759,360 H810 V282 H854');
    out+=box(866,220,160,82,t('mla.scores'),'mla-scores','scores',t('mla.scores.description'),{input:`Q: ${td(256)} · Kᵀ: 256 × ${N()}`,output:`${N()} × ${N()}`,labelLines:['Q Kᵀ'],labelKinds:['diagram-math']});
    out+=line(1038,260,1080,260)+box(1092,220,200,82,'Selected\ncausal positions','mla-mask','mask',t('index.select'),{input:`${N()} × ${N()}`,output:`${N()} × ${N()}`,fontSize:22});
    out+=line(1304,260,1346,260)+box(1358,220,180,82,t('mla.softmax'),'mla-softmax','softmax',t('mla.scores.description'),{input:`${N()} × ${N()}`,output:`${N()} × ${N()}`});
    out+=line(1550,260,1592,260)+box(1604,220,220,82,t('mla.mix'),'mla-mix','mix',t('mla.description'),{input:`weights: ${N()} × ${N()} · V: ${td(256)}`,output:td(256)});
    out+=path('M759,360 H1714 V314','diagram-wire')+line(1836,260,2060,260,'diagram-contribution')+text(1900,330,'to output projection',19,'diagram-note');
    out+=box(1042,427,300,65,'Pool indexer','indexer','indexer',t('index.description'),{enter:'indexer',parameterScope:'indexer'});
    out+=path('M147,360 V460 H1030 M1192,415 V314','moe-control');
    for(const [x,stage,title,source]of [[310,'scores','Attention scores','scores'],[815,'masked','Selected causal scores','mask'],[1320,'weights','Attention weights','softmax']]){
      out+=attentionMatrix(x,650,145,stage,spec('mla-matrix-'+stage,title,source,t('mla.scores.description'),{inspectable:false}));
    }
    out+=line(485,725,772,725)+text(626,696,'selection mask',21,'diagram-note')+line(990,725,1277,725)+text(1134,696,'scale + softmax',21,'diagram-note');
    out+=text(970,858,N()<=2051?'With this short context, all past positions fit the selection budget. Dot sizes are illustrative.':'The grids show the first eight positions. Dot sizes and selections are illustrative.',20,'diagram-note');
    views.kimiHead={x:-155,y:-45,w:2300,h:1030};
    return out;
  }

  function indexer(){
    if(isDelta()){views.glmIndexer={x:-145,y:-45,w:2080,h:990};return '';}
    let out=outline(-70,-20,1900,880)+text(-35,25,t('index.title'),29,'diagram-title','start');
    out+=text(-35,70,'32 index query heads · one shared index key · 128 coordinates',22,'diagram-note','start');
    out+=line(-125,240,-2,240)+box(10,195,225,90,'Index key projection\n+ LayerNorm','index-key','index_pool',t('index.pool'),{weight:param('attention.indexer.key_proj.weight'),input:td(),output:td(128),fontSize:22});
    out+=line(247,240,310,240)+box(322,195,310,90,'Pool four positions\ncoordinate by coordinate','index-pool','index_pool',t('index.pool'),{input:td(128),output:`${Math.floor(N()/4)} complete pools × 128`,fontSize:23});
    out+=line(644,240,710,240)+box(722,195,320,90,'Dot products\n÷ √128 → ReLU','index-scores','index_scores',t('index.score'));
    out+=box(370,375,280,80,'32 index queries','index-queries','index_scores',t('index.score'),{weight:param('attention.indexer.query_proj.weight'),input:td(1536),output:`32 × ${td(128)}`})+path('M662,415 H882 V297');
    out+=line(-125,415,358,415)+text(115,390,'From normalized query latent',20,'diagram-note');
    out+=line(1054,240,1115,240)+box(1127,195,270,90,'Weighted head sum','index-sum','index_scores',t('index.score'));
    out+=path('M-100,240 V550 H1262 V297','moe-control')+text(560,535,'Signed head weights from the normalized sublayer input · scaled by 1/√32',20,'diagram-note');
    out+=line(1409,240,1460,240)+box(1472,195,280,90,'Choose complete pools\n+ unfinished tail','index-select','index_select',t('index.select'),{output:'Up to 2,051 original positions',fontSize:22});
    out+=path('M1612,297 V635 H1770','diagram-contribution')+text(1650,686,'Selection mask for all 64 heads',20,'diagram-note');
    const position=Math.max(0,Math.min(state.moePosition,N()-1)),complete=Math.floor((position+1)/4),tail=(position+1)%4;
    out+=text(15,627,`At position ${position+1}: ${complete} complete ${complete===1?'pool':'pools'} + ${tail} tail ${tail===1?'position':'positions'}`,25,'diagram-title','start');
    out+=text(15,672,'Choose up to 512 complete pools (2,048 positions), then include the current tail.',22,'diagram-note','start');
    out+=text(15,730,N()<=2051?'At this context length, every causal position is included.':'The indexer selects from the eligible past positions for each query.',23,'diagram-note','start');
    out+=text(15,785,'Pooling is used to select positions. Main attention reads their individual key and value vectors.',22,'diagram-note','start');
    views.glmIndexer={x:-145,y:-45,w:2080,h:990};
    return out;
  }

  function feedforward(){
    if(!N()){views.kimiFeedforward={x:-100,y:-30,w:1450,h:700};return outline(-50,0,1300,600)+text(600,290,'Enter a context to see routing at a sequence position.',27,'diagram-title');}
    if(isDense())return expert(true);
    const selected=illustrativeGLMRouting(state.moePosition,state.block),ys=selected.map((_,i)=>310+i*40);
    let out=outline(-295,-15,1980,945)+text(-265,30,t('moe.title'),29,'diagram-title','start')+text(-265,73,t('moe.illustrative',{position:state.moePosition+1}),21,'diagram-note','start');
    out+=line(-350,435,-256,435)+projection(-244,400,88,70,'RMS\nNorm','moe-norm','norm',t('attention.norm'),'feed_forward_norm.weight',{input:td(),output:td()});
    out+=line(-144,435,-104,435)+rowMatrix(-92,427,92,'Mixed vector',d(),'moe-input',t('moe.description'),1)+text(-50,497,`Position ${state.moePosition+1}`,18,'diagram-note');
    out+=path('M12,435 H48 V147 H87');
    out+=projection(99,112,180,70,t('moe.router'),'moe-router','router',t('moe.router.description'),'feed_forward.router.weight',{input:td(d(),1),output:'1 × 288'});
    out+=line(291,147,333,147)+box(345,112,240,70,t('moe.select'),'moe-topk','topk',t('moe.router.description'),{input:'288 scores',output:'8 expert IDs'});
    out+=line(597,147,639,147)+box(651,112,250,70,t('moe.normalize'),'moe-norm-weights','route_norm',t('moe.router.description'),{output:'8 scalar weights'});
    out+=path('M913,147 H1160 V235','moe-control');
    out+=line(12,435,330,435)+path('M330,310 V590','diagram-wire');
    out+=path('M369,250 H1158 V626 H369 Z','gqa-kv-box')+text(525,282,t('moe.selected'),22,'diagram-title')+text(805,282,t('moe.weight'),20,'diagram-note')+text(1037,282,t('moe.contribution'),20,'diagram-note');
    selected.forEach((item,i)=>{const y=ys[i];out+=line(330,y,401,y)+operation(413,y-15,210,30,`Expert ${item.expert}`,expertSpec(item.expert))+line(635,y,715,y,'diagram-contribution');
      out+=box(727,y-15,154,30,`× ${item.weight.toFixed(3)}`,'weight-'+item.expert,'route_norm',t('moe.router.description'))+line(893,y,947,y,'diagram-contribution')+grid(959,y-11,138,22,true,1,8)+line(1109,y,1275,y,'diagram-contribution');});
    out+=path('M1275,310 V668','diagram-contribution')+box(1170,680,210,54,t('moe.sum'),'moe-sum','moe_combine',t('moe.return.description'),{output:td(d(),1)});
    out+=path('M1392,707 H1550 V778','diagram-contribution');
    out+=path('M24,435 V625 H-20 V668','diagram-wire')+operation(-170,680,300,85,t('moe.shared'),expertSpec(-1));
    out+=path('M-20,777 V805 H1523','diagram-contribution')+plus(1550,805,{source:'moe_combine',title:'Add shared and routed outputs',description:t('moe.return.description')});
    out+=line(1577,805,1730,805,'diagram-contribution')+text(1060,891,t('attention.return'),21,'diagram-note');
    views.kimiFeedforward={x:-365,y:-40,w:2160,h:1045};
    return out;
  }

  function expert(dense=false){
    if(!N()){views.kimiExpert={x:-100,y:-30,w:1450,h:700};return outline(-50,0,1300,600)+text(600,290,'Enter a context to inspect a sequence position.',27,'diagram-title');}
    const shared=state.expert===-1&&!dense;
    const width=dense||shared?d():d(),intermediate=dense?C().expanded_dim:C().expert_intermediate_dim;
    const description=t(dense?'dense.description':shared?'moe.shared.description':'expert.description');
    const prefix=dense?'feed_forward.':shared?'feed_forward.shared_experts.':`feed_forward.experts.${state.expert}.`;
    let out=outline(-30,-20,1510,620)+text(0,25,dense?t('dense.title'):shared?'Inside the shared expert':t('expert.title',{expert:state.expert}),28,'diagram-title','start');
    out+=line(-95,280,8,280)+rowMatrix(20,272,105,t(dense?'dense.input':'expert.input'),width,'expert-input',description,1);
    out+=path('M137,280 H172 V152 H230 M172,280 V405 H230','diagram-wire');
    out+=projection(242,112,205,80,'Gate projection\ncap at 10','expert-gate','gate',t('expert.situ.description'),prefix+'gate_proj.weight',{input:td(width,1),output:td(intermediate,1)});
    out+=projection(242,365,205,80,t('expert.value'),'expert-value','value',t('expert.situ.description'),prefix+'value_proj.weight',{input:td(width,1),output:td(intermediate,1)});
    out+=dimensions(344,228,`${fmt(width)} → ${fmt(intermediate)}`,{columns:'coordinates'},18)+dimensions(344,482,`${fmt(width)} → ${fmt(intermediate)}`,{columns:'coordinates'},18);
    out+=line(459,152,513,152)+box(525,112,195,80,t('expert.situ'),'situ','situ',t('expert.situ.description'));
    out+=line(459,405,513,405)+box(525,365,230,80,t('expert.bound'),'value-bound','situ',t('expert.situ.description'),{fontSize:21});
    out+=path('M732,152 H818 Q848,152 848,182 V247 M767,405 H818 Q848,405 848,375 V310');
    out+=box(819,250,58,58,'⊙','expert-product','situ',t('expert.situ.description'))+line(889,280,947,280);
    out+=projection(959,240,230,80,t('expert.out'),'expert-output-proj','down',description,prefix+'output_proj.weight',{input:td(intermediate,1),output:td(width,1)});
    out+=dimensions(1074,364,`${fmt(intermediate)} → ${fmt(width)}`,{columns:'coordinates'},18)+line(1201,280,1250,280);
    out+=rowMatrix(1262,272,125,t(dense?'dense.output':'expert.output'),width,'expert-output',description,1)+line(1399,280,1525,280,'diagram-contribution');
    out+=text(736,548,t(dense?'dense.note':'expert.note'),21,'diagram-note');
    views.kimiExpert={x:-115,y:-45,w:1690,h:735};
    if(dense){
      out+=projection(-255,245,110,70,'RMS\nNorm','dense-norm','norm',t('attention.norm'),'feed_forward_norm.weight',{input:td(),output:td()})+line(-330,280,-267,280)+line(-133,280,-95,280);
      views.kimiFeedforward={x:-350,y:-45,w:1925,h:735};
    }
    return out;
  }

  function depth(){
    const branch=state.depthBranch||'feedforward',prefix=branch==='attention'?'attention_hc':'feed_forward_hc';
    let out=outline(-40,-20,2060,1030)+text(0,26,t('depth.title'),29,'diagram-title','start')+text(1960,26,`${branch} · block ${state.block}`,20,'diagram-shape','end');
    out+=text(0,80,'Stream mixing at one sequence position',23,'diagram-note','start');
    for(let i=0;i<4;i++){const y=150+i*95;out+=line(-90,y+8,3,y+8)+rowMatrix(15,y,115,`Stream ${i}`,d(),'source-'+i,t('depth.description'),1)+line(142,y+8,190,y+8);}
    out+=path('M190,158 V443 M190,260 H235','diagram-wire');
    out+=box(247,215,270,90,'Flatten · RMS normalize\n16,384 → 24 mapping','hc-mapping','hc_mapping',t('hc.mapping'),{weight:param(prefix+'.mapping'),input:'4 × 4,096',output:'24 coefficients',fontSize:21});
    out+=path('M529,260 H570 V145 H625 M570,260 V425 H625 M570,260 V675 H625','moe-control');
    out+=box(637,105,260,80,'Collapse coefficients\nsigmoid + ε','hc-pre','hc_pre',t('hc.pre'),{output:'4 scalars',fontSize:23});
    out+=line(909,145,955,145)+box(967,105,275,80,'Weighted stream sum','hc-collapse','hc_pre',t('hc.pre'),{input:'4 × 4,096',output:'1 × 4,096',fontSize:22});
    out+=path('M190,443 V515 H1104 V197','diagram-wire')+text(490,500,'Original stream vectors',21,'diagram-note');
    out+=line(1254,145,1310,145)+box(1322,105,265,80,branch==='attention'?'RMSNorm + attention':'RMSNorm + feedforward','hc-sublayer',branch==='attention'?'residual':'residual_ff',t('block.description'),{enter:branch,input:branch==='attention'?td():'1 × 4,096',output:branch==='attention'?td():'1 × 4,096',fontSize:22});
    out+=box(637,385,260,80,'Distribution coefficients\n2 × sigmoid','hc-post','hc_post',t('hc.post'),{output:'4 scalars',fontSize:22});
    out+=line(909,425,1310,425,'moe-control')+box(1322,385,265,80,'Distribute contribution','hc-distribute','hc_add',t('hc.post'),{input:'1 × 4,096',output:'4 × 4,096',fontSize:22})+path('M1454,197 V373','diagram-contribution');
    out+=box(637,635,260,80,'Softmax + 20 rounds\nof row/column norm','hc-balance','hc_mix',t('hc.mix'),{input:'4 × 4 logits',output:'4 × 4 mixing coefficients',fontSize:22});
    out+=line(909,675,960,675)+matrix(972,615,120,120,'Stream mixing','4 × 4',spec('hc-matrix','Stream mixing coefficients','hc_mix',t('hc.mix'),{activation:true,rows:4,cols:4,axes:{rows:'input stream',columns:'output stream'}}));
    out+=line(1104,675,1150,675)+box(1162,635,290,80,'Mix bypassed streams','hc-bypass','hc_add',t('hc.mix'),{input:'4 × 4,096',output:'4 × 4,096',fontSize:22});
    out+=path('M190,515 V825 H1307 V727','diagram-wire')+text(690,808,'Original stream vectors',21,'diagram-note');
    out+=path('M1464,675 H1690 V602 M1599,425 H1690 V548','diagram-contribution')+plus(1690,575,{source:'hc_add',title:'Add distributed output and mixed bypass',description:t('depth.description'),input:'Two sets of 4 × 4,096 stream vectors',output:'4 × 4,096',outputAxes:{rows:'residual stream',columns:'residual coordinate'}});
    for(let i=0;i<4;i++){const y=455+i*95;out+=path(`M1717,575 H1770 V${y+8} H1798`,'diagram-contribution')+rowMatrix(1810,y,115,`Stream ${i}`,d(),'source-output-'+i,t('depth.description'),1)+line(1937,y+8,2060,y+8,'diagram-contribution');}
    if(branch==='attention')out+=text(970,865,'Attention reads the full sequence. Here we follow one position’s returned contribution.',23,'diagram-note');
    out+=text(970,930,t('depth.note'),24,'diagram-note');
    views.kimiDepth={x:-110,y:-45,w:2250,h:1150};
    return out;
  }

  function output(){
    let out=outline(-15,-10,1320,545)+text(15,35,'Output projection',28,'diagram-title','start');
    out+=[238,256,274,292].map(y=>stream(-90,5,y)).join('')+box(17,223,230,84,t('output.mix'),'output-mix','output_stage',t('output.description'),{input:`${N()} × 4 × ${fmt(d())}`,output:td()});
    out+=line(259,265,315,265)+box(327,223,140,84,t('output.norm'),'output-norm','norm',t('attention.norm'),{weight:'output_norm.weight',input:td(),output:td()});
    out+=line(479,265,545,265)+matrix(557,156,135,218,t('output.project'),`${fmt(C().vocabulary_size)} × ${fmt(d())}`,spec('output-weight',t('output.project'),'output',t('output.description'),{weight:'output_layer.weight',labelKind:'diagram-title',input:td(),output:td(C().vocabulary_size)}));
    out+=line(704,265,882,265)+rowMatrix(894,228,200,t('output.logits'),C().vocabulary_size,'logits',t('output.description'))+line(1106,265,1370,265,'diagram-contribution');
    out+=text(1120,352,'on to sampling',21,'diagram-note');
    out+=text(650,465,'Take the mean of the four streams independently at each sequence position.',23,'diagram-note');
    views.kimiOutput={x:-110,y:-35,w:1515,h:635};
    return out;
  }
  return {isDelta,isDense,attentionName,defaultSpec,expertSpec,block,attention,head:()=>isDelta()?kdaHead():mlaHead(),indexer,feedforward,expert:()=>expert(false),depth,output};
}
