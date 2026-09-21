import {illustrativeRouting} from './moe-routing.mjs';
import {parameterAxes} from './matrix-info.mjs';

// Illustrative scores only. Selection uses a bias; mixing uses the original scores.
export function illustrativeKimiRouting(position,block){
  const logits=illustrativeRouting(896,16,position,block).logits;
  const scores=logits.map(x=>1/(1+Math.exp(-x)));
  const bias=illustrativeRouting(896,16,position+7,block+3).logits.map(x=>x*.08);
  const selected=scores.map((score,expert)=>({expert,score,choice:score+bias[expert]})).sort((a,b)=>b.choice-a.choice).slice(0,16);
  const sum=selected.reduce((total,item)=>total+item.score,0);
  return selected.map(item=>({...item,weight:item.score/sum}));
}

export const kimiText={
  'model.description':'Token IDs select embedding vectors. The 93 transformer blocks apply attention and feedforward, with Attention Residuals mixing sources across depth. A final depth mixture, RMSNorm and output projection produce logits. Sampling at the final sequence position chooses the next token.',
  'expert.selected':'This expert is selected for position {position} in the illustrative routing example.',
  'expert.unselected':'This expert is not selected for position {position} in the illustrative routing example. This view shows its structure; it contributes nothing at that position.',
  'block.description':'Each attention and feedforward branch first mixes the available depth sources using Attention Residuals (AttnRes), then normalizes that mixture. Its output is added to the current group’s local sum. A completed group is retained as a separate source every twelve transformer blocks.',
  'block.footer':'Within a group, add each new contribution to the local sum.\nAttnRes reads the embedding, completed groups, and current local sum separately.',
  'depth.title':'Attention across depth',
  'depth.description':'At each sequence position, AttnRes assigns a weight to each available depth source. A learned query scores normalized source vectors. Softmax runs over the sources, and the resulting weights combine the original vectors. Every reader has its own query and normalization scale. Sequence positions remain separate.',
  'depth.sources':'Depth sources at the same sequence position',
  'depth.keys':'Normalize\nfor scoring',
  'depth.query':'Learned depth query',
  'depth.softmax':'Softmax\nover sources',
  'depth.values':'Original source vectors supply the values',
  'depth.note':'One weight per depth source, per sequence position.\nThe embedding stays available; each completed group contains twelve transformer blocks.',
  'attention.description':'Each head reads the same vectors after this branch’s AttnRes mixture and RMSNorm. KDA processes the sequence with a recurrent matrix state. Full-attention blocks use latent projections and causal softmax attention. All 96 head outputs pass through their slices of the output projection and are summed.',
  'attention.input':'From AttnRes',
  'attention.return':'Contribution to the current group sum',
  'attention.heads':'96 heads',
  'attention.kda-note':'One query/key/value set per head · 128 coordinates · coordinate-wise state decay',
  'attention.others':'… heads 3–94 …',
  'attention.sum':'Sum all 96',
  'attention.norm':'RMSNorm operates on each mixed 7,168-coordinate vector. It uses a learned scale directly, with epsilon 10⁻⁵.',
  'kda.description':'Kimi Delta Attention uses a 128 × 128 state per head. It decays each key-coordinate row separately, reads the value associated with the current key, writes a correction, and reads an output using the current query. State is reset on every forward pass.',
  'kda.projections':'Q / K / V\nprojections',
  'kda.project.description':'This head has its own query, key and value projections, each mapping 7,168 input coordinates to 128. All 96 heads are computed together. Unlike Qwen3.8’s grouped DeltaNet, K3 has one Q/K pair per value head.',
  'kda.conv':'Causal conv\n+ SiLU',
  'kda.conv.description':'Apply a separate learned four-position filter to each projected coordinate, then SiLU. Each filter reads this position and up to three preceding positions. Q, K and V have separate filters.',
  'kda.norm':'L2 norm\nQ and K',
  'kda.norm.description':'Normalize Q and K by the square root of their sum of squared coordinates plus 10⁻⁶. Values retain their convolved coordinates. Scale the query by 1/√128 when reading the state.',
  'kda.scan':'Scan sequence\npositions',
  'kda.decay':'Coordinate-wise decay',
  'kda.decay.description':'The input passes through 7,168 → 128 → 12,288 projections to produce 128 decay controls per head. For each coordinate: log α = −5 × sigmoid(exp(A_log) × (control + dt_bias)); α = exp(log α). Each key-coordinate row of the state has its own decay.',
  'kda.write':'Write strength β',
  'kda.write.description':'A separate 7,168 → 96 projection and sigmoid produce one write-strength scalar per head and position. It scales the correction written into the state.',
  'kda.gate':'Output norm\n× sigmoid gate',
  'kda.gate.description':'Normalize the 128-coordinate state readout and apply the learned scale shared across heads. Multiply by the sigmoid of this head’s 7,168 → 128 gate projection. The gate reads the original normalized input.',
  'kda.state':'Previous state',
  'kda.decay-step':'Decay rows\nS ← diag(αₜ) S',
  'kda.correction':'Read and correct\nδ = βₜ (vₜ − kₜ S)\nS ← S + kₜᵀ δ',
  'kda.read':'Read output\nyₜ = (qₜ / √128) S',
  'kda.state-note':'The updated state is carried to the next sequence position.\nThis implementation discards it at the end of the forward pass.',
  'mla.description':'K3’s full-attention heads use a 1,536-coordinate query bottleneck and a shared 512-coordinate K/V bottleneck. Each key has 128 head-specific coordinates plus 64 shared coordinates. Queries and keys are 192 coordinates wide; values are 128. K3 does not apply RoPE.',
  'mla.query':'Query latent\n+ RMSNorm',
  'mla.query.description':'Project the full 7,168-coordinate input down to 1,536 coordinates, normalize, then expand it into the queries for all 96 heads. Each head gets a 192-coordinate query. The bottleneck RMSNorm uses epsilon 10⁻⁶.',
  'mla.kv':'K/V latent\n+ RMSNorm',
  'mla.kv.description':'The first K/V projection produces 512 latent coordinates and 64 separate shared key coordinates. Normalize the 512-coordinate latent and expand it into a 128-coordinate key and value for each head. Append the 64 shared key coordinates to every head’s key. No rotary transform is applied.',
  'mla.shared':'64 shared key coordinates bypass the bottleneck norm',
  'mla.expand':'Per-head\nprojections',
  'mla.scores':'Q Kᵀ',
  'mla.mask':'Causal mask',
  'mla.softmax':'Scale\n+ softmax',
  'mla.scores.description':'Dot products of this head’s 192-coordinate query and key vectors form one score per pair of sequence positions. Future positions are masked. The code scales the scores by 1/√192, then applies softmax in float32.',
  'mla.mix':'Weights × V',
  'mla.gate':'× sigmoid\noutput gate',
  'mla.gate.description':'A separate 7,168 → 12,288 projection supplies 128 gate coordinates for each of the 96 heads. Its sigmoid scales each head’s output coordinate by coordinate, before the output projection.',
  'mla.note':'Full causal attention · 192 query/key coordinates · 128 value coordinates · no RoPE',
  'moe.description':'The router scores all 896 experts from the full 7,168-coordinate input. Sigmoid scores plus a selection bias choose sixteen experts; their original sigmoid scores are renormalized for mixing. Routed experts use a 3,584-coordinate latent space. Their weighted sum is normalized and projected back to 7,168, then added to two always-active shared experts.',
  'moe.title':'Stable LatentMoE',
  'moe.illustrative':'Illustrative routing at position {position} · 16 selected from 896',
  'moe.router':'Router\n+ sigmoid',
  'moe.select':'Selection bias\n+ top sixteen',
  'moe.normalize':'Renormalize\noriginal scores',
  'moe.router.description':'Compute 896 router logits in float32, then apply sigmoid. Add the learned correction bias only when choosing the top sixteen. Retrieve those experts’ original sigmoid scores and divide by their sum to get the routing weights.',
  'moe.latent':'Project to\nlatent space',
  'moe.latent.description':'One learned 7,168 → 3,584 projection supplies a smaller input vector to all selected routed experts. The router and shared experts read the full-width input before this projection.',
  'moe.return':'RMSNorm\n+ project back',
  'moe.return.description':'Sum the sixteen weighted expert outputs in the latent space. Apply RMSNorm there, then one shared 3,584 → 7,168 projection. Add the shared-expert branch’s output.',
  'moe.shared':'Two shared experts\nalways active',
  'moe.shared.description':'The shared path reads every full-width input vector. Its two experts are stored as one 7,168 → 6,144 → 7,168 feedforward network, mathematically equivalent to summing two independent networks of intermediate width 3,072. No routing weight is applied to this branch.',
  'moe.bank':'896 routed experts\n16 selected per position',
  'moe.selected':'Selected experts',
  'moe.weight':'Routing weight',
  'moe.contribution':'Latent contribution',
  'moe.sum':'Weighted sum',
  'expert.title':'Inside expert {expert}',
  'expert.description':'Each routed expert has its own 3,584 → 3,072 → 3,584 projections. The gate uses SiTU and the value path uses a bounded tanh. The output is scaled by this position’s routing weight before the selected experts are summed.',
  'expert.input':'Expert input',
  'expert.output':'Expert output',
  'expert.gate':'Gate projection',
  'expert.value':'Value projection',
  'expert.situ':'SiTU',
  'expert.bound':'25 tanh(v / 25)',
  'expert.out':'Output projection',
  'expert.situ.description':'SiTU applies 4 × tanh(gate / 4) × sigmoid(gate). The value path separately applies 25 × tanh(value / 25). Multiply the two results coordinate by coordinate, then project back to the input width.',
  'expert.note':'Every expert has its own learned projections. The same weights process every position assigned to that expert.',
  'dense.title':'Dense feedforward · block 0',
  'dense.description':'Only block 0 uses a dense feedforward network: 7,168 → 33,792 → 7,168. Its gate and value paths use the same bounded activation functions as the experts. Every position uses this network.',
  'dense.input':'Input vector',
  'dense.output':'Output vector',
  'dense.note':'Each position uses the same dense feedforward network.\nThe routed and shared experts in later blocks use the same activation functions.',
  'output.description':'After block 92, a final AttnRes reader mixes the embedding, completed groups and current group sum. RMSNorm normalizes the result, then a separate learned output matrix maps each 7,168-coordinate vector to 163,840 logits.',
  'output.mix':'Final AttnRes',
  'output.norm':'RMSNorm',
  'output.project':'Output matrix',
  'output.logits':'Logits',
};

export function createKimiDiagrams(v){
  const {state,C,N,d,fmt,copy,text,line,path,outline,grid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,attentionMatrix,richText,copyAttributes}=v;
  const t=(key,vars={})=>copy('kimi.'+key,vars);
  const td=(width=d(),rows=N())=>`${rows} × ${fmt(width)}`;
  const isDelta=()=>C().layer_types[state.block]==='linear_attention';
  const isDense=()=>state.block===0;
  const attentionName=()=>isDelta()?'Kimi Delta Attention':'Latent attention';
  const axes={rows:'sequence position',columns:'coordinate'};
  const spec=(key,title,source,description,extra={})=>({key:'kimi-'+key,title,source,description,...extra});
  const box=(x,y,w,h,label,key,source,description,extra={})=>operation(x,y,w,h,label,spec(key,label,source,description,extra));
  const projection=(x,y,w,h,label,key,source,description,suffix,extra={})=>{
    const weight=param(suffix),learnedAxes=parameterAxes(weight,false,C());
    return box(x,y,w,h,label,key,source,description,{weight,inputAxes:{rows:'sequence position',columns:learnedAxes.columns},outputAxes:{rows:'sequence position',columns:learnedAxes.vector?learnedAxes.columns:learnedAxes.rows},...extra});
  };
  const headSpec=head=>spec('head-'+head,`Head ${head}`,isDelta()?'delta':'attention',t(isDelta()?'kda.description':'mla.description'),{head,enter:'head',input:td(),output:td(128)});
  const expertSpec=expert=>spec('expert-'+expert,expert===-1?'Shared experts':`Expert ${expert}`,'expert',t(expert===-1?'moe.shared.description':'expert.description'),{expert,enter:'expert',parameterScope:expert===-1?'shared-experts':'expert',input:td(expert===-1?d():C().expert_dim,1),output:td(expert===-1?d():C().expert_dim,1)});
  const rowMatrix=(x,y,w,label,width,key,description,rows=N())=>matrix(x,y,w,Math.max(1,Math.min(rows,5))*15,label,td(width,rows),spec(key,label,key.startsWith('expert')?'expert':key==='logits'?'output':key.startsWith('source')?'depth':isDelta()?'delta':'attention',description,{activation:true,rows:Math.max(1,Math.min(rows,5)),axes,output:td(width,rows)}));

  function defaultSpec(level){
    const info={
      model:['Kimi K3 · text backbone','model','model.description','model'],
      block:[`Transformer block ${state.block}`,'block','block.description','block'],
      attention:[attentionName(),isDelta()?'delta':'attention','attention.description','attention'],
      head:[`${isDelta()?'KDA':'MLA'} head ${state.head}`,isDelta()?'delta':'attention',isDelta()?'kda.description':'mla.description',null],
      feedforward:[t(isDense()?'dense.title':'moe.title'),isDense()?'expert':'feedforward',isDense()?'dense.description':'moe.description','feedforward'],
      depth:[t('depth.title'),'depth','depth.description','depth'],
      output:['Output projection','output_stage','output.description','output'],
      expert:[state.expert===-1?'Shared experts':`Expert ${state.expert}`,'expert',state.expert===-1?'moe.shared.description':'expert.description',state.expert===-1?'shared-experts':'expert'],
    }[level];
    const vectorAxes={rows:'sequence position',columns:level==='expert'&&state.expert>=0?'expert latent coordinate':'residual coordinate'};
    return spec(level,info[0],info[1],t(info[2]),{kind:level==='model'?'MODEL OVERVIEW':'KIMI K3',parameterScope:info[3],expert:state.expert,note:level==='expert'&&state.expert>=0?t(illustrativeKimiRouting(state.moePosition,state.block).some(item=>item.expert===state.expert)?'expert.selected':'expert.unselected',{position:state.moePosition+1}):undefined,input:level==='model'?`${N()} token IDs`:level==='expert'?td(state.expert===-1?d():C().expert_dim,1):td(),output:['model','output'].includes(level)?td(C().vocabulary_size):level==='head'?td(128):level==='expert'?td(state.expert===-1?d():C().expert_dim,1):td(),inputAxes:level==='model'?undefined:vectorAxes,outputAxes:['model','output'].includes(level)?{rows:'sequence position',columns:'vocabulary entry'}:level==='head'?{rows:'sequence position',columns:'head coordinate'}:vectorAxes});
  }

  function block(){
    const boundary=state.block%12===0;
    let out=`<path id="block-frame" class="housing" d="${outlinePath(-15,BLOCK.frameTop,BLOCK.width,state.blockHeight)}"/>`+text(28,35,`Transformer block ${state.block}`,27,'diagram-title','start')+dimensions(1180,35,td(),axes,19,'end');
    out+=stream(-38,boundary?68:1235,340);
    if(boundary){out+=stream(110,1235,340)+text(103,314,'new group: 0',18,'diagram-note','start')+path('M84,315 V365','moe-control');}
    else out+=text(25,310,'current group sum',21,'diagram-note','start');
    out+=path(`M${boundary?40:135},340 V165 H178`);
    out+=box(190,125,100,80,state.block===0?'Embedding':'AttnRes','attn-reader','depth',t('depth.description'),{enter:'depth',depthBranch:'attention',fontSize:state.block===0?18:undefined});
    out+=line(302,165,319,165);
    out+=record(spec('attention',attentionName(),'residual',t('attention.description'),{enter:'attention',input:td(),output:td(),box:{x:331,y:105,w:264,h:122}}),outline(331,105,264,122)+text(463,130,attentionName(),22,'diagram-title')+text(463,215,'RMSNorm · 96 heads',16,'diagram-shape')+path('M385,147 V188 M540,147 V188','diagram-wire')+[147,158,169,188].map(y=>line(385,y,402,y)+outline(414,y-3,93,6,isDelta()?'#f8f3e9':'#e8f2f1')+line(519,y,540,y,'diagram-contribution')).join('')+text(460,180,'…',14,'diagram-note'));
    out+=path('M607,165 H647 V313','diagram-contribution')+plus(647,340,{source:'residual',title:'Add attention to local sum',description:t('block.description')});
    out+=path('M731,340 V165 H774');
    out+=box(786,125,100,80,'AttnRes','ff-reader','depth',t('depth.description'),{enter:'depth',depthBranch:'feedforward'});
    out+=line(898,165,917,165);
    out+=box(929,105,215,122,isDense()?'Dense feedforward':'LatentMoE','feedforward','residual_ff',t(isDense()?'dense.description':'moe.description'),{enter:'feedforward',input:td(),output:td(),fontSize:22});
    out+=text(1036,206,isDense()?'RMSNorm · SiTU':'RMSNorm · 16 + 2',18,'diagram-shape');
    out+=path('M1156,165 H1190 V313','diagram-contribution')+plus(1190,340,{source:'residual_ff',title:'Add feedforward to local sum',description:t('block.description')});
    out+=text(1175,389,state.block===92?'to final AttnRes':`on to block ${state.block+1}`,20,'diagram-note','end');
    out+=record(spec('summaries','Earlier depth sources','depth',t('depth.description'),{enter:'depth',depthBranch:'feedforward'}),outline(225,427,790,45,'#f8f3e9')+text(620,457,'Embedding + completed group summaries · kept for each AttnRes reader',21,'diagram-note'));
    out+=path(`${state.block===0?'':'M240,427 V217 '}M836,427 V217`,'moe-control');
    out+=`<foreignObject id="block-prose" x="28" y="486" width="1150" height="100"><div xmlns="http://www.w3.org/1999/xhtml" id="block-prose-text" class="block-prose" ${copyAttributes(t('block.footer'))}>${richText(t('block.footer'))}</div></foreignObject>`;
    return out;
  }

  function attention(){
    const inputY=375,ys=[165,220,275,395,555],heads=[0,1,2,3,95];
    let out=outline(-300,-10,1440,795)+text(-268,35,attentionName(),30,'diagram-title','start')+text(1095,35,`Block ${state.block}`,20,'diagram-shape','end');
    out+=text(-268,79,t(isDelta()?'attention.kda-note':'mla.note'),19,'diagram-note','start');
    out+=line(-345,inputY,-245,inputY)+text(-264,inputY-65,t('attention.input'),20,'diagram-note');
    out+=projection(-233,inputY-35,86,70,'RMS\nNorm','input-norm','norm',t('attention.norm'),'attention_norm.weight',{input:td(),output:td()});
    out+=line(-135,inputY,-70,inputY)+path('M-70,165 V555','diagram-wire');
    out+=text(210,124,t('attention.heads'),23,'diagram-title')+text(570,124,'Output projection',23,'diagram-title');
    ys.forEach((y,index)=>{
      out+=line(-70,y,8,y);
      out+=operation(20,y-20,370,40,index===3?t('attention.others'):`Head ${heads[index]}`,headSpec(heads[index]));
      out+=line(402,y,460,y)+box(472,y-20,195,40,index===3?'92 head contributions':`Wₒ⁽${heads[index]}⁾`,'slice-'+index,isDelta()?'delta_output':'combine',t('attention.description'),{input:index===3?`92 head outputs, each ${td(128)}`:td(128),output:td(),fontSize:index===3?17:undefined});
      out+=line(679,y,730,y,'diagram-contribution')+grid(742,y-15,140,30,true,Math.max(1,Math.min(N(),5)),8)+line(894,y,1025,y,'diagram-contribution');
    });
    out+=path('M1025,165 V620','diagram-contribution')+box(930,632,190,48,t('attention.sum'),'attention-sum',isDelta()?'delta_output':'combine',t('attention.description'),{output:td()});
    out+=line(1025,692,1025,720,'diagram-contribution')+line(1025,720,1190,720,'diagram-contribution')+text(1020,756,t('attention.return'),20,'diagram-note');
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
    let out=outline(-95,-20,2185,955)+text(-60,25,`Latent attention · head ${state.head}`,28,'diagram-title','start')+text(-60,68,t('mla.note'),22,'diagram-note','start');
    out+=line(-140,220,-2,220)+rowMatrix(10,182,100,'Normalized\nembedding vectors',d(),'mla-input',t('mla.description'));
    out+=path('M122,220 H147 V150 H193 M147,220 V360 H193','diagram-wire');
    out+=projection(205,110,260,80,t('mla.query'),'mla-query','mla_q',t('mla.query.description'),'attention.q_a_proj.weight',{input:td(),output:td(1536)});
    out+=projection(205,320,260,80,t('mla.kv'),'mla-kv','mla_kv',t('mla.kv.description'),'attention.kv_a_proj.weight',{input:td(),output:'512 latent + 64 shared coordinates'});
    out+=line(477,150,520,150)+line(477,360,520,360);
    out+=box(532,110,215,80,'Query projection','mla-qb','mla_q',t('mla.query.description'),{input:td(1536),output:td(192)});
    out+=box(532,320,215,80,'Key / value\nprojections','mla-kvb','mla_kv',t('mla.kv.description'),{input:td(512),output:`K: ${td(192)} · V: ${td(128)}`});
    out+=text(450,452,t('mla.shared'),20,'diagram-note');
    out+=path('M759,150 H810 V238 H854 M759,360 H810 V282 H854');
    out+=box(866,220,160,82,t('mla.scores'),'mla-scores','scores',t('mla.scores.description'),{input:`Q: ${td(192)} · Kᵀ: 192 × ${N()}`,output:`${N()} × ${N()}`,labelLines:['Q Kᵀ'],labelKinds:['diagram-math']});
    out+=line(1038,260,1080,260)+box(1092,220,155,82,t('mla.mask'),'mla-mask','mask',t('mla.scores.description'),{input:`${N()} × ${N()}`,output:`${N()} × ${N()}`});
    out+=line(1259,260,1301,260)+box(1313,220,180,82,t('mla.softmax'),'mla-softmax','softmax',t('mla.scores.description'),{input:`${N()} × ${N()}`,output:`${N()} × ${N()}`});
    out+=line(1505,260,1550,260)+box(1562,220,220,82,t('mla.mix'),'mla-mix','mix',t('mla.description'),{input:`weights: ${N()} × ${N()} · V: ${td(128)}`,output:td(128)});
    out+=path('M759,360 H1672 V314','diagram-wire');
    out+=path('M1794,260 H1920 V487')+box(1790,500,260,80,t('mla.gate'),'mla-gate','attention_gate',t('mla.gate.description'),{input:td(128),output:td(128)});
    out+=path('M147,360 V540 H1778','moe-control')+text(950,525,'Separate gate projection reads the same input',20,'diagram-note');
    out+=line(1920,592,1920,690,'diagram-contribution')+line(1920,690,2130,690,'diagram-contribution')+text(1850,743,'to output projection',21,'diagram-note');
    for(const [x,stage,title,source]of [[310,'scores','Attention scores','scores'],[815,'masked','Masked scores','mask'],[1320,'weights','Attention weights','softmax']]){
      out+=attentionMatrix(x,690,145,stage,spec('mla-matrix-'+stage,title,source,t('mla.scores.description'),{inspectable:false}));
    }
    out+=line(485,765,772,765)+text(626,736,'causal mask',21,'diagram-note')+line(990,765,1277,765)+text(1134,736,'scale + softmax',21,'diagram-note');
    out+=text(955,896,'Dot sizes are illustrative; the causal mask is exact.',20,'diagram-note');
    views.kimiHead={x:-155,y:-45,w:2330,h:1040};
    return out;
  }

  function feedforward(){
    if(!N()){views.kimiFeedforward={x:-100,y:-30,w:1450,h:700};return outline(-50,0,1300,600)+text(600,290,'Enter a context to see routing at a sequence position.',27,'diagram-title');}
    if(isDense())return expert(true);
    const selected=illustrativeKimiRouting(state.moePosition,state.block),ys=selected.map((_,i)=>310+i*40);
    let out=outline(-295,-15,2015,1240)+text(-265,30,t('moe.title'),29,'diagram-title','start')+text(-265,73,t('moe.illustrative',{position:state.moePosition+1}),21,'diagram-note','start');
    out+=line(-350,435,-256,435)+projection(-244,400,88,70,'RMS\nNorm','moe-norm','norm',t('attention.norm'),'feed_forward_norm.weight',{input:td(),output:td()});
    out+=line(-144,435,-104,435)+rowMatrix(-92,427,92,'Mixed vector',d(),'moe-input',t('moe.description'),1)+text(-50,497,`Position ${state.moePosition+1}`,18,'diagram-note');
    out+=path('M12,435 H48 V147 H87');
    out+=projection(99,112,180,70,t('moe.router'),'moe-router','router',t('moe.router.description'),'feed_forward.router.weight',{input:td(d(),1),output:'1 × 896'});
    out+=line(291,147,333,147)+box(345,112,240,70,t('moe.select'),'moe-topk','topk',t('moe.router.description'),{input:'896 scores',output:'16 expert IDs'});
    out+=line(597,147,639,147)+box(651,112,250,70,t('moe.normalize'),'moe-norm-weights','route_norm',t('moe.router.description'),{output:'16 scalar weights'});
    out+=path('M913,147 H1160 V235','moe-control');
    out+=projection(94,400,185,70,t('moe.latent'),'latent-down','latent_down',t('moe.latent.description'),'feed_forward.down_proj.weight',{input:td(d(),1),output:td(3584,1)});
    out+=line(12,435,82,435)+line(291,435,330,435)+path('M330,310 V910','diagram-wire');
    out+=path('M369,250 H1158 V944 H369 Z','gqa-kv-box')+text(525,282,t('moe.selected'),22,'diagram-title')+text(805,282,t('moe.weight'),20,'diagram-note')+text(1037,282,t('moe.contribution'),20,'diagram-note');
    selected.forEach((item,i)=>{const y=ys[i];out+=line(330,y,401,y)+operation(413,y-15,210,30,`Expert ${item.expert}`,expertSpec(item.expert))+line(635,y,715,y,'diagram-contribution');
      out+=box(727,y-15,154,30,`× ${item.weight.toFixed(3)}`,'weight-'+item.expert,'route_norm',t('moe.router.description'))+line(893,y,947,y,'diagram-contribution')+grid(959,y-11,138,22,true,1,8)+line(1109,y,1275,y,'diagram-contribution');});
    out+=path('M1275,310 V967','diagram-contribution')+box(1170,980,210,42,t('moe.sum'),'moe-sum','moe_combine',t('moe.return.description'),{output:td(3584,1)});
    out+=line(1275,1034,1275,1040,'diagram-contribution')+projection(1150,1052,250,70,t('moe.return'),'latent-up','moe_combine',t('moe.return.description'),'feed_forward.up_proj.weight',{input:td(3584,1),output:td(d(),1)});
    // The return path runs below the routed stack, toward the shared-path sum.
    out+=line(1412,1087,1543,1087,'diagram-contribution');
    out+=path('M24,435 V625 H-20 V668','diagram-wire');
    out+=operation(-170,680,300,85,t('moe.shared'),expertSpec(-1));
    out+=path('M-20,777 V1155 H1570 V1114','diagram-contribution')+plus(1570,1087,{source:'moe_combine',title:'Add shared and routed outputs',description:t('moe.return.description')});
    out+=line(1597,1087,1775,1087,'diagram-contribution')+text(1040,1185,t('attention.return'),21,'diagram-note');
    views.kimiFeedforward={x:-365,y:-40,w:2200,h:1285};
    return out;
  }

  function expert(dense=false){
    if(!N()){views.kimiExpert={x:-100,y:-30,w:1450,h:700};return outline(-50,0,1300,600)+text(600,290,'Enter a context to inspect a sequence position.',27,'diagram-title');}
    const shared=state.expert===-1&&!dense;
    const width=dense||shared?d():C().expert_dim,intermediate=dense?C().expanded_dim:C().expert_intermediate_dim*(shared?2:1);
    const description=t(dense?'dense.description':shared?'moe.shared.description':'expert.description');
    const prefix=dense?'feed_forward.':shared?'feed_forward.shared_experts.':`feed_forward.experts.${state.expert}.`;
    let out=outline(-30,-20,1510,620)+text(0,25,dense?t('dense.title'):shared?'Inside the shared experts':t('expert.title',{expert:state.expert}),28,'diagram-title','start');
    out+=line(-95,280,8,280)+rowMatrix(20,272,105,t(dense?'dense.input':'expert.input'),width,'expert-input',description,1);
    out+=path('M137,280 H172 V152 H230 M172,280 V405 H230','diagram-wire');
    out+=projection(242,112,205,80,t('expert.gate'),'expert-gate','gate',t('expert.situ.description'),prefix+'gate_proj.weight',{input:td(width,1),output:td(intermediate,1)});
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
    const branch=state.depthBranch||'feedforward',completed=Math.floor(state.block/12),boundary=state.block%12===0;
    let sources=['Embedding',...Array.from({length:completed},(_,i)=>`Blocks ${i*12}–${i*12+11}`)];
    if(branch==='attention'&&boundary&&state.block>0)sources=sources.slice(0,-1);
    const currentLabel=branch==='attention'&&boundary&&state.block>0?`Blocks ${state.block-12}–${state.block-1}`:'Current group sum';
    if(branch==='attention'&&state.block===0)sources=[];
    sources.push(branch==='attention'&&state.block===0?'Embedding':currentLabel);
    const height=Math.max(760,260+sources.length*90);
    let out=outline(-30,-20,1635,height)+text(0,26,t('depth.title'),29,'diagram-title','start')+text(1550,26,`${branch} reader · block ${state.block}`,20,'diagram-shape','end');
    out+=text(5,78,t('depth.sources'),23,'diagram-note','start');
    sources.forEach((label,i)=>{const y=135+i*90;out+=rowMatrix(15,y,115,label,d(),'source-'+i,t('depth.description'),1)+line(142,y+8,257,y+8,'diagram-wire');});
    const last=143+(sources.length-1)*90;
    out+=path(`M257,143 V${last}`,'diagram-wire')+path(`M257,${(143+last)/2} H285 V235 H310`);
    out+=box(322,192,200,85,t('depth.keys'),'depth-norm','depth_scores',t('depth.description'));
    out+=line(534,235,594,235)+box(606,192,230,85,'Dot with query','depth-dot','depth_scores',t('depth.description'));
    out+=box(606,94,230,46,t('depth.query'),'depth-query','depth_scores',t('depth.description'),{weight:param((branch==='attention'?'attention_mix':'feed_forward_mix')+'.query.weight')})+line(721,152,721,180);
    out+=line(848,235,908,235)+box(920,192,230,85,t('depth.softmax'),'depth-softmax','depth_scores',t('depth.description'),{output:`${N()} × ${sources.length}`,outputAxes:{rows:'sequence position',columns:'depth source'}});
    out+=path('M1162,235 H1300 V408')+box(1200,420,200,74,'Weighted sum','depth-sum','depth_mix',t('depth.description'),{input:`${sources.length} source vectors per position`,output:td()});
    out+=path(`M257,${last} V${height-114} H1300 V506`,'diagram-contribution')+text(750,height-125,t('depth.values'),21,'diagram-note');
    out+=line(1412,457,1640,457,'diagram-contribution')+text(1488,506,'to RMSNorm',20,'diagram-note');
    out+=text(805,height-55,t('depth.note'),21,'diagram-note');
    if(branch==='attention'&&state.block===0){
      out=outline(-30,-20,1635,690)+text(0,26,t('depth.title'),29,'diagram-title','start')+rowMatrix(100,280,180,'Embedding',d(),'source-embedding',t('depth.description'))+line(292,317,1250,317)+text(750,275,'The first attention branch reads the embedding directly.',25,'diagram-note')+text(1380,325,'to RMSNorm',25,'diagram-title');
    }
    views.kimiDepth={x:-65,y:-45,w:1760,h:height+95};
    return out;
  }

  function output(){
    let out=outline(-15,-10,1320,545)+text(15,35,'Output projection',28,'diagram-title','start');
    out+=line(-90,265,5,265)+box(17,223,230,84,t('output.mix'),'output-mix','depth',t('output.description'),{input:'Embedding + completed groups + final group sum',output:td()});
    out+=line(259,265,315,265)+box(327,223,140,84,t('output.norm'),'output-norm','norm',t('attention.norm'),{weight:'output_norm.weight',input:td(),output:td()});
    out+=line(479,265,545,265)+matrix(557,156,135,218,t('output.project'),`${fmt(C().vocabulary_size)} × ${fmt(d())}`,spec('output-weight',t('output.project'),'output',t('output.description'),{weight:'output_layer.weight',labelKind:'diagram-title',input:td(),output:td(C().vocabulary_size)}));
    out+=line(704,265,882,265)+rowMatrix(894,228,200,t('output.logits'),C().vocabulary_size,'logits',t('output.description'))+line(1106,265,1370,265,'diagram-contribution');
    out+=text(1120,352,'on to sampling',21,'diagram-note');
    out+=text(650,465,'Final AttnRes mixes nine depth sources independently at each position.',23,'diagram-note');
    views.kimiOutput={x:-110,y:-35,w:1515,h:635};
    return out;
  }
  return {isDelta,isDense,attentionName,defaultSpec,expertSpec,block,attention,head:()=>isDelta()?kdaHead():mlaHead(),feedforward,expert:()=>expert(false),depth,output};
}
