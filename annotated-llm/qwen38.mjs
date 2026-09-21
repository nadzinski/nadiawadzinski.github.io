// Qwen3.8-specific drawings use the same SVG vocabulary and selection machinery.
export function createQwen38Diagrams(v) {
  const {state,C,N,d,fmt,text,line,path,outline,grid,stream,operation,matrix,record,plus,dimensions,views,param}=v;
  const isDelta=()=>C().layer_types?.[state.block]==='linear_attention';
  const td=()=>`${N()} × ${fmt(d())}`;
  const spec=(key,title,source,description,extra={})=>({key,title,source,description,kind:'GATED DELTANET',...extra});
  const box=(x,y,w,h,label,key,source,description,extra={})=>operation(x,y,w,h,label,spec(key,label,source,description,extra));
  const wire=(x1,y1,x2,y2)=>line(x1,y1,x2,y2,'diagram-contribution');

  function attention(){
    const inputY=375,streamY=740,sumX=980;
    let out=outline(-310,-10,1410,840)+text(-275,34,'Gated DeltaNet',30,'diagram-title','start')
      +text(-275,75,'48 value heads share 16 query/key heads',22,'diagram-note','start')
      +text(1060,34,`Block ${state.block}`,18,'diagram-shape','end');
    out+=stream(-325,1120,streamY)+path(`M-270,${streamY} V${inputY} H-232`);
    out+=box(-220,inputY-34,86,68,'RMS\nNorm','delta-input-norm','norm','Normalize the residual vector at every position. Each coordinate is multiplied by 1 plus a learned offset.',{input:td(),output:td(),weight:param('attention_norm.weight')});
    out+=dimensions(-177,inputY+66,td(),{rows:'sequence position',columns:'residual coordinate'});
    out+=line(-122,inputY,-64,inputY)+path('M-60,183 V587','diagram-wire');
    out+=text(180,126,'Recurrent heads',22,'diagram-note')+text(540,126,'Output projection',22,'diagram-note')+text(785,113,'Contribution',22,'diagram-note')+dimensions(785,143,td(),{rows:'sequence position',columns:'residual coordinate'},17);
    const groups=[{first:0,top:165},{first:45,top:465}];
    for(const group of groups){
      out+=path(`M-2,${group.top-15} H355 V${group.top+190} H-2 Z`,'gqa-kv-box');
      out+=text(176,group.top+175,'shared Q/K · separate V and gates',18,'diagram-note');
      for(let offset=0;offset<3;offset++){
        const head=group.first+offset,y=group.top+offset*52+18;
        out+=line(-60,y,12,y);
        out+=box(24,y-18,308,36,`Head ${head} · state + output gate`,`delta-head-${head}`,'delta','Project the normalized input, apply the short causal convolution, and update this head’s state across the sequence. A SiLU gate scales the normalized output.',{enter:'head',head,input:td(),output:`${N()} × 128`});
        out+=line(344,y,459,y);
        out+=box(471,y-18,138,36,`Wₒ⁽${head}⁾`,`delta-projection-${head}`,'delta_output','This slice projects one 128-coordinate gated head output into the residual space. Summing all 48 projected contributions is equivalent to concatenating the head outputs and applying the full output matrix.',{input:`${N()} × 128`,output:td()});
        out+=wire(622,y,720,y)+grid(732,y-14,108,28,true,Math.max(1,Math.min(N(),8)),8)+wire(852,y,sumX,y);
      }
    }
    out+=box(24,365,585,52,'… heads 3–44 …','delta-other-heads','delta','These 42 heads perform the same calculation with their own values, states, and gates. Three adjacent value heads share each Q/K pair. Open this group, then use Previous head and Next head to inspect its members.',{enter:'head',head:3});
    out+=line(-60,391,12,391)+wire(622,391,sumX,391);
    out+=path(`M${sumX},183 V657`,'diagram-contribution');
    out+=box(sumX-70,665,140,42,'Sum','delta-sum','delta_output','Add the 48 projected head contributions to form the DeltaNet output.',{output:td()});
    out+=wire(sumX,707,sumX,719)+plus(sumX,streamY,{source:'residual',title:'Add the DeltaNet contribution',description:'Add the complete DeltaNet contribution to the incoming residual stream.'});
    out+=text(-265,785,'Residual stream',22,'diagram-note','start')+text(1065,785,'on to feedforward',22,'diagram-note','end');
    views.attentionLocal={x:-335,y:-25,w:1485,h:885};
    return out;
  }

  function head(){
    const group=Math.floor(state.head/3),shape=`${N()} × 128`,input=td();
    let out=outline(-95,-30,2110,850)+text(0,14,`DeltaNet head ${state.head}`,29,'diagram-title','start')
      +text(1980,14,`Block ${state.block} · Q/K group ${group}`,20,'diagram-shape','end');
    out+=text(0,54,'The state scan starts at zero for every forward pass.',22,'diagram-note','start');
    out+=line(-150,195,-2,195);
    out+=matrix(10,152,96,86,'Normalized\nembedding vectors',input,spec('delta-head-input','Input to this head','delta','This head receives the block’s normalized residual vectors.',{activation:true,rows:Math.max(1,Math.min(N(),8)),input,output:input}));
    out+=line(118,195,174,195);
    out+=box(186,150,150,90,'Q / K / V\nprojections','delta-projections','delta_qkv','Shared Q and K projections each map 5,120 coordinates to 128. This value head has its own 5,120 → 128 V projection. The implementation packs all Q/K/V projections into one linear layer.',{input,output:`Q, K, V: ${shape}`,note:'Each head projection is 5,120 × 128. The packed checkpoint parameter in_proj_qkv.weight is 10,240 × 5,120 (output × input).'});
    out+=line(348,195,398,195);
    out+=box(410,150,170,90,'Causal conv\n+ SiLU','delta-conv','conv','Apply a learned length-four filter separately to every projected coordinate, using this position and up to three previous positions. SiLU follows the convolution. Future positions are never read.',{input:`Q, K, V: ${shape}`,output:`Q, K, V: ${shape}`,note:'The packed convolution weights have shape 10,240 × 1 × 4: one four-position filter per projected coordinate.'});
    out+=line(592,195,640,195);
    out+=box(652,150,150,90,'L2 norm\nQ and K','delta-normalize','delta_norm','Normalize Q and K by their L2 length, with epsilon for numerical stability. Scale queries by 1/√128. Values retain their convolved coordinates.',{input:`Q, K: ${shape}`,output:`Q, K: ${shape}`});
    out+=text(727,270,'V passes through',18,'diagram-note');
    out+=line(814,195,873,195);
    out+=box(885,133,375,124,'Scan over sequence positions','delta-scan','delta_scan','For each position: decay the state, read the value currently associated with its key, write a correction, then read the updated state with its query. The code keeps this state in float32.',{input:`Q, K, V: ${shape}`,output:shape});
    out+=line(1272,195,1333,195);
    out+=box(1345,150,200,90,'RMSNorm\n× SiLU(gate)','delta-output-gate','delta_output_norm','Normalize each output vector and apply a learned scale shared across value heads. Multiply coordinate by coordinate by SiLU of this head’s output-gate projection. This normalization uses its stored weight directly.',{input:shape,output:shape});
    out+=line(1557,195,1611,195);
    out+=matrix(1623,155,155,80,'Gated head output',shape,spec('delta-gated-head','Gated head output','delta_output','One 128-coordinate output per position. The next view applies this head’s output-projection slice and sums all head contributions.',{activation:true,rows:Math.max(1,Math.min(N(),8)),axes:{rows:'sequence position',columns:'value-head coordinate'},output:shape}));
    out+=wire(1790,195,2060,195)+text(1930,178,'to output projection',20,'diagram-note');
    // All control projections read the same normalized input, before convolution.
    out+=path('M126,195 V327 H360','diagram-wire');
    out+=box(372,296,430,70,'Write strength β and decay α','delta-controls','delta_gates','Each value head has two scalar projections. β = sigmoid(b). The decay is α = exp(−exp(A_log) × softplus(a + dt_bias)). Both vary with the input position; A_log and dt_bias are learned per-head parameters.',{input,output:`α, β: ${N()} scalars each`,note:'Each packed a/b projection is 48 × 5,120. A_log and dt_bias each contain 48 learned scalars, one per value head.'});
    out+=path('M814,331 H1072 V270','diagram-arrow');
    out+=path('M126,327 V410 H1130','diagram-wire');
    out+=box(1142,382,270,56,'Output-gate projection','delta-z','delta_output','Project the original normalized input to a separate 128-coordinate gate for this value head. This branch bypasses the causal convolution and state recurrence.',{input,output:shape,note:'The packed output-gate projection is 6,144 × 5,120. Each of its 48 head slices maps 5,120 coordinates to 128.'});
    out+=path('M1424,410 H1445 V254','diagram-arrow');
    out+=text(0,473,'Within the scan, at position t',26,'diagram-title','start');
    out+=matrix(15,540,140,116,'Previous state', '128 × 128',spec('delta-state','State matrix','delta_scan','Each head’s state associates key coordinates (rows) with value coordinates (columns). It summarizes the prefix. The first position starts from an all-zero state.',{activation:true,axes:{rows:'key coordinate',columns:'value coordinate'},output:'128 × 128'}));
    out+=line(168,598,221,598);
    out+=box(233,554,260,88,'Decay\nS ← αₜ S','delta-decay','delta_decay','Multiply every state entry by the current position’s scalar decay α.',{input:'128 × 128',output:'128 × 128'});
    out+=line(505,598,561,598);
    out+=box(573,527,505,142,'Read and correct\nprediction = kₜ S\nδ = βₜ (vₜ − prediction)\nS ← S + kₜᵀ δ','delta-correct','delta_write','Read the value currently associated with this key. Write the difference between the current value and that prediction, scaled by β. The outer product kₜᵀ δ updates the state matrix.',{fontSize:23,input:'128 × 128 state; kₜ and vₜ vectors',output:'128 × 128 updated state'});
    out+=line(1090,598,1147,598);
    out+=matrix(1159,540,140,116,'Updated state','128 × 128',spec('delta-new-state','Updated state','delta_write','The updated state is used to read this position’s output, then passed to the next sequence position.',{activation:true,axes:{rows:'key coordinate',columns:'value coordinate'},output:'128 × 128'}));
    out+=line(1311,598,1365,598);
    out+=box(1377,554,280,88,'Read output\nyₜ = qₜ S','delta-read','delta_read','Read the updated state with the current query. The query has already been L2-normalized and scaled by 1/√128.',{input:'qₜ: 128; state: 128 × 128',output:'128-coordinate vector'});
    out+=wire(1669,598,1835,598)+text(1840,582,'to output norm',20,'diagram-note');
    out+=path('M1311,598 H1330 V750 H-12 V598 H3','diagram-residual')+text(660,783,'Carry the state to position t + 1; discard it after the pass.',22,'diagram-note');
    views.headLocal={x:-165,y:-50,w:2280,h:900};
    return out;
  }
  return {isDelta,attention,head};
}
