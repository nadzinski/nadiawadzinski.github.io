import {parameterAxes} from './matrix-info.mjs';

export const mambaText={
  'model.description':'Mamba-2 passes token embedding vectors through {blocks} state-space blocks. In each block, a short causal convolution and a recurrent state let information from earlier sequence positions affect later ones. Each block adds its result to the residual stream. Final RMSNorm and the shared token/output table produce logits for choosing the next token.',
  'block.description':'The residual stream passes through this block. RMSNorm supplies a normalized copy to a state-space mixer, whose output is added back into the stream. This model has one mixer branch per block. Its projections, convolution, recurrence and output gate together transform the sequence.',
  'mixer.description':'The input projection produces values, write vectors B, read vectors C, time steps Δ and an output gate z. A causal convolution followed by SiLU acts on the values, B and C. Each of the {heads} heads updates and reads a {headDim} × {stateSize} state matrix. Join the head outputs, multiply by SiLU(z), normalize all {expandedDim} coordinates together, and project back to the residual width.',
  'head.description':'At each sequence position, this head decays its previous state, adds an outer-product write, then reads from the updated state. Its state has {headDim} value coordinates by {stateSize} memory coordinates. All heads share the input-dependent B and C vectors; values and time steps differ by head. A learned scalar D adds a direct contribution from the current value vector.',
  'norm.description':'At each position, divide the vector by the square root of its mean squared coordinate value plus 10⁻⁵. Multiply by learned coordinate scales. Each block has its own input RMSNorm. The final output has a separate RMSNorm.',
  'projection.description':'One learned projection maps each normalized {embeddingDim}-coordinate vector to five pieces: z has {expandedDim} coordinates, the values have {expandedDim}, B and C each have {stateSize}, and the raw time steps have {heads}. The code keeps these pieces in one packed projection, then splits them. B and C are input-dependent vectors; the projection matrix is learned.',
  'conv.description':'Each of the 1,792 channels has its own learned four-tap filter and bias. A channel reads its value at the current position and the preceding three positions; missing earlier positions are zero. This mixes nearby sequence positions independently within each channel. SiLU is applied after the convolution. Only the values, B and C go through this operation.',
  'sharing.description':'The {heads} heads share one {stateSize}-coordinate write vector B and one {stateSize}-coordinate read vector C at each sequence position. Each head receives its own {headDim}-coordinate value vector and scalar time step. Each also has a separate state matrix and learned A and D scalars.',
  'steps.description':'Softplus makes each projected time step Δ positive, after adding a learned per-head bias. The time step depends on the input position and head. It controls both how much the old state decays and how strongly the current outer product is written.',
  'decay.description':'Each head has a learned scalar A = −exp(A_log), which is negative. At position t, multiply every entry of that head’s old state by exp(Δₜ A). This factor lies between zero and one and depends on the input through Δₜ.',
  'write.description':'Form the outer product of the current {headDim}-coordinate value vector x and the shared {stateSize}-coordinate write vector B. Multiply by this head’s time step Δ and add the resulting {headDim} × {stateSize} matrix to the decayed state.',
  'read.description':'Multiply the updated state matrix by the current read vector C. Each of the {headDim} state rows contributes one dot product with C, giving a {headDim}-coordinate output vector. B and C can be different at every sequence position.',
  'skip.description':'Multiply this head’s current value vector by its learned scalar D, then add that vector to the state readout. This is a direct path within the mixer. The block’s outer residual addition happens later, after gating, normalization and output projection.',
  'state.description':'This {headDim} × {stateSize} matrix is an activation that changes as a head reads the sequence. The dimensions describe value and memory coordinates. The state begins at zero for every full model call in this implementation; it is carried from one sequence position to the next within that call.',
  'join.description':'Place the {heads} head outputs side by side at each sequence position. This gives {expandedDim} coordinates. The following gated normalization spans this whole vector, so its normalization scale depends on the combined heads.',
  'gate.description':'The input projection also produces a {expandedDim}-coordinate gate z at each position. Apply SiLU to z, then multiply the joined head outputs coordinate by coordinate. Each input position has its own gate values. The learned projection that produces them is shared across positions.',
  'gated-norm.description':'After multiplying by SiLU(z), normalize the whole {expandedDim}-coordinate vector by its root mean square and apply learned coordinate scales. This checkpoint uses one normalization group spanning all {heads} heads. The gate is applied before this normalization.',
  'output-proj.description':'A learned matrix maps each gated, normalized {expandedDim}-coordinate vector back to {embeddingDim} coordinates. This gives the mixer’s contribution to the residual stream. There is no bias in this projection.',
  'output.description':'Final RMSNorm normalizes each residual vector. Multiply it by the transpose of the token embedding table to obtain {vocabularySize} logits. The input and output use the same learned table. Generation uses the final position’s logits and excludes table-padding entries.',
  'block.label':'Mamba block {block}',
  'mixer.label':'State-space mixer',
  'head.label':'State-space head {head}',
  'block.note':'The residual stream continues through all {blocks} Mamba blocks.',
  'mixer.note':'B and C are shared across heads. Each head carries its own state through the sequence.',
  'head.note':'Repeat for each position in the sequence. The initial state S₀ is zero.',
};

export function mambaHeadRows(heads){
  return [{head:0,y:255},{head:1,y:330},{head:2,y:405},{head:3,y:520,collapsed:true},{head:heads-1,y:635}];
}

export function createMamba2Diagrams(v){
  const {state,C,N,d,fmt,copy,text,line,path,outline,grid,brokenGrid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,richText,copyAttributes}=v;
  const t=(key,values={})=>copy('mamba.'+key,{stateSize:C().state_size,block:state.block,head:state.head,...values});
  const td=(width=d())=>`${N()} × ${fmt(width)}`,shape=()=>`${C().head_dim} × ${C().state_size}`;
  const axes={rows:'sequence position',columns:'residual coordinate'};
  const spec=(key,title,source,description,extra={})=>({key:'mamba-'+key,title,source,description,...extra});
  const box=(x,y,w,h,label,key,source,description,extra={})=>{
    const columns={projection:['residual coordinate','packed projection coordinate'],conv:['convolution channel','convolution channel'],steps:['head (raw time step)','head (positive time step)'],join:['head value coordinate','combined head coordinate'],gate:['expanded coordinate','expanded coordinate'],gated_norm:['expanded coordinate','expanded coordinate'],mixer_output:['expanded coordinate','residual coordinate']}[source];
    return operation(x,y,w,h,label,spec(key,label,source,description,{...(columns?{inputAxes:{rows:'sequence position',columns:columns[0]},outputAxes:{rows:'sequence position',columns:columns[1]}}:{}),...extra}));
  };
  const vectorRows=()=>Math.max(1,Math.min(N(),5));
  const vectors=(x,y,w,label,width,key,source,description,height=50)=>matrix(x,y,w,height,label,td(width),spec(key,label,source,description,{activation:true,rows:vectorRows(),axes:{rows:'sequence position',columns:width===d()?'residual coordinate':width===C().vocabulary_size?'vocabulary entry':width===d()?'residual coordinate':'expanded coordinate'},output:td(width),outputAxes:{rows:'sequence position',columns:width===C().head_dim?'head value coordinate':width===C().vocabulary_size?'vocabulary entry':width===d()?'residual coordinate':'expanded coordinate'}}));
  const headSpec=head=>spec('head-'+head,`Head ${head}`,'scan',t('head.description'),{head,enter:'head',input:`${N()} positions`,output:td(C().head_dim),outputAxes:{rows:'sequence position',columns:'head value coordinate'}});
  const stateMatrix=(x,y,key,label)=>matrix(x,y,145,105,label,shape(),spec(key,label,'scan',t('state.description'),{activation:true,rows:5,cols:8,axes:{rows:'head value coordinate',columns:'memory coordinate'},output:shape(),outputAxes:{rows:'head value coordinate',columns:'memory coordinate'}}));
  function defaultSpec(level){
    const info={model:['Mamba-2 130M','model','model.description','model'],block:[t('block.label'),'block','block.description','block'],attention:[t('mixer.label'),'mixer','mixer.description','mixer'],head:[t('head.label'),'scan','head.description',null],output:['Output projection','output_stage','output.description','output']}[level];
    return spec(level,info[0],info[1],t(info[2]),{kind:'MAMBA-2',parameterScope:info[3],input:level==='model'?`${N()} token IDs`:level==='head'?'Value vector x, shared B and C, time step Δ, previous state':td(),output:['model','output'].includes(level)?td(C().vocabulary_size):level==='head'?td(C().head_dim):td(),inputAxes:['model','head'].includes(level)?undefined:axes,outputAxes:['model','output'].includes(level)?{rows:'sequence position',columns:'vocabulary entry'}:level==='head'?{rows:'sequence position',columns:'head value coordinate'}:axes});
  }
  function block(){
    let out=`<path id="block-frame" class="housing" d="${outlinePath(-15,BLOCK.frameTop,BLOCK.width,state.blockHeight)}"/>`+text(28,35,t('block.label'),27,'diagram-title','start')+dimensions(1180,35,td(),axes,19,'end');
    out+=stream(-38,1260,340)+text(35,310,'Residual stream',23,'diagram-note','start')+path('M135,340 V165 H318');
    let schematic=outline(330,105,590,122)+text(625,132,t('mixer.label'),24,'diagram-title');
    for(const [x,w,label]of [[350,38,'RMS'],[410,48,'Proj'],[480,48,'Conv'],[705,63,'Gate'],[788,38,'RMS'],[846,52,'Proj']]){
      schematic+=outline(x,155,w,27)+text(x+w/2,174,label,15,'diagram-title');
    }
    schematic+=path('M392,169 H406 M462,169 H476 M532,169 H551 M679,169 H701 M772,169 H784 M830,169 H842','diagram-wire');
    for(const y of [150,160,170,194])schematic+=outline(555,y,119,6,'#e8f2f1');
    schematic+=text(613,187,'…',15,'diagram-note')+text(615,217,`${C().num_heads} state-space heads`,17,'diagram-shape');
    out+=record({...defaultSpec('attention'),key:'mamba-mixer',enter:'attention',box:{x:330,y:105,w:590,h:122}},schematic);
    out+=path('M932,165 H1090 V313','diagram-contribution')+plus(1090,340,{source:'residual',title:'Add mixer contribution',description:t('block.description')});
    out+=text(1190,403,state.block===C().num_blocks-1?'to final RMSNorm':`on to block ${state.block+1}`,22,'diagram-note','end');
    out+=`<foreignObject id="block-prose" x="28" y="451" width="1150" height="100"><div xmlns="http://www.w3.org/1999/xhtml" id="block-prose-text" class="block-prose" ${copyAttributes(t('block.description'))}>${richText(t('block.description'))}</div></foreignObject>`;
    return out;
  }
  function attention(){
    let out=outline(-305,-15,2265,970)+text(-270,30,t('mixer.label'),30,'diagram-title','start');
    out+=stream(-360,2010,850)+text(-270,814,'Residual stream',23,'diagram-note','start')+path('M-275,850 V460 H-252');
    out+=box(-240,425,85,70,'RMS\nNorm','input-norm','norm',t('norm.description'),{weight:param('mixer_norm.weight'),input:td(),output:td()})+line(-143,460,-102,460);
    out+=box(-90,410,160,100,'Input\nprojection','in-proj','projection',t('projection.description'),{weight:param('mixer.in_proj.weight'),input:td(),output:td(2*C().expanded_dim+2*C().state_size+C().num_heads)});
    out+=line(82,460,133,460)+box(145,425,150,70,'Conv + SiLU','conv','conv',t('conv.description'),{weight:param('mixer.conv1d.weight'),input:td(C().expanded_dim+2*C().state_size),output:td(C().expanded_dim+2*C().state_size),fontSize:22});
    out+=text(220,543,'values · B · C',22,'diagram-note')+line(307,460,445,460)+path('M445,255 V635','diagram-wire');
    out+=record(spec('sharing','Shared B and C','projection',t('sharing.description'),{box:{x:490,y:164,w:430,h:537}}),`<rect x="490" y="164" width="430" height="537" rx="5" fill="none" stroke="#9baea9" stroke-dasharray="3 6"/>`+text(705,199,`${C().num_heads} heads · shared B and C`,22,'diagram-title'));
    for(const row of mambaHeadRows(C().num_heads)){
      const {y,head,collapsed}=row;
      out+=line(445,y,513,y)+operation(525,y-21,210,42,collapsed?`… heads 3–${C().num_heads-2} …`:`Head ${head}`,headSpec(head));
      out+=line(747,y,785,y)+grid(797,y-16,85,32,true,vectorRows(),6)+line(894,y,955,y,'diagram-contribution');
    }
    out+=path('M955,255 V635','diagram-wire')+line(955,460,988,460)+box(1000,420,145,80,'Join head\noutputs','join','join',t('join.description'),{output:td(C().expanded_dim),outputAxes:{rows:'sequence position',columns:'combined head coordinate'}});
    out+=dimensions(1072,546,td(C().expanded_dim),{rows:'sequence position',columns:'combined head coordinate'},18);
    // The two branches leaving the packed projection carry its z and raw Δ slices.
    out+=path('M-10,398 V115 H1025')+text(690,97,'z · output gate',23,'diagram-note')+box(1037,80,125,70,'SiLU','gate','gate',t('gate.description'),{input:td(C().expanded_dim),output:td(C().expanded_dim)});
    out+=path('M1174,115 H1230 V423','diagram-contribution')+line(1157,460,1193,460);
    out+=box(1205,435,50,50,'×','gate-multiply','gate',t('gate.description'),{output:td(C().expanded_dim)});
    out+=line(1267,460,1308,460)+box(1320,425,90,70,'RMS\nNorm','gated-norm','gated_norm',t('gated-norm.description'),{weight:param('mixer.norm.weight'),input:td(C().expanded_dim),output:td(C().expanded_dim)});
    out+=text(1365,559,`all ${fmt(C().expanded_dim)}\ncoordinates`,21,'diagram-note')+line(1422,460,1478,460);
    out+=box(1490,420,155,80,'Output\nprojection','out-proj','mixer_output',t('output-proj.description'),{weight:param('mixer.out_proj.weight'),input:td(C().expanded_dim),output:td()})+line(1657,460,1710,460);
    out+=vectors(1722,435,120,'Contribution',d(),'contribution','mixer_output',t('output-proj.description'));
    out+=path('M1854,460 H1890 V823','diagram-contribution')+plus(1890,850,{source:'residual',title:'Add mixer contribution',description:t('block.description')});
    out+=path('M-10,522 V750 H158')+box(170,715,160,70,'Add bias\nSoftplus','steps','steps',t('steps.description'),{labelText:'Add bias\nSoftplus',weight:param('mixer.dt_bias'),input:td(C().num_heads),output:td(C().num_heads),fontSize:22});
    out+=path('M342,750 H445 V647')+text(388,720,'Δ',25,'diagram-math');
    out+=text(1020,922,t('mixer.note'),23,'diagram-note');
    views.kimiAttention={x:-385,y:-45,w:2430,h:1080};return out;
  }
  function head(){
    let out=outline(-85,-20,2190,950)+text(-50,26,t('head.label'),30,'diagram-title','start');
    out+=text(-50,77,'At sequence position t',24,'diagram-note','start');
    out+=line(-140,385,143,385,'diagram-contribution')+stateMatrix(155,332.5,'previous-state','Previous state Sₜ₋₁');
    out+=line(312,385,458,385,'diagram-contribution')+box(470,360,50,50,'×','decay-multiply','decay',t('decay.description'));
    out+=box(400,153,190,75,'exp(Δₜ Aₕ)','decay','decay',t('decay.description'),{fontSize:25});
    out+=text(495,121,'Decay factor',24,'diagram-title')+text(495,263,'Aₕ = −exp(A_log[h])',22,'diagram-math')+line(495,275,495,348);
    out+=line(532,385,838,385,'diagram-contribution')+plus(865,385,{source:'write',title:'Add the current write',description:t('write.description'),input:shape(),output:shape(),inputAxes:{rows:'head value coordinate',columns:'memory coordinate'},outputAxes:{rows:'head value coordinate',columns:'memory coordinate'}});
    out+=line(892,385,1008,385,'diagram-contribution')+stateMatrix(1020,332.5,'updated-state','Updated state Sₜ');
    out+=line(1177,385,1310,385,'diagram-contribution')+box(1322,350,150,70,'Read with Cₜ','read','read',t('read.description'),{input:shape(),output:`${C().head_dim} coordinates`,inputAxes:{rows:'head value coordinate',columns:'memory coordinate'}});
    out+=matrix(1335,168,125,28,'Read vector Cₜ',`${C().state_size}`,spec('read-vector','Read vector Cₜ','projection',t('sharing.description'),{activation:true,rows:1,cols:8,axes:{vector:true,columns:'memory coordinate'},output:`${C().state_size} coordinates`}))+line(1397,238,1397,338);
    out+=line(1484,385,1668,385,'diagram-contribution')+plus(1695,385,{source:'skip',title:'Add the direct value contribution',description:t('skip.description'),output:`${C().head_dim} coordinates`});
    out+=line(1722,385,1788,385,'diagram-contribution')+matrix(1800,368,135,34,'Head output yₜ',`${C().head_dim}`,spec('head-output','Head output yₜ','scan',t('head.description'),{activation:true,rows:1,cols:8,axes:{vector:true,columns:'head value coordinate'},output:`${C().head_dim} coordinates`}))+line(1947,385,2155,385,'diagram-contribution');
    out+=text(1970,489,'to joining, gate\nand normalization',22,'diagram-note');
    out+=line(-140,647,53,647)+matrix(65,630,145,34,'Value vector xₜ',`${C().head_dim}`,spec('value-vector','Current value vector xₜ','conv',t('projection.description'),{activation:true,rows:1,cols:8,axes:{vector:true,columns:'head value coordinate'}}));
    out+=line(222,647,398,647)+box(410,610,190,75,'Δₜ × outer\nproduct','write','write',t('write.description'),{output:shape(),outputAxes:{rows:'head value coordinate',columns:'memory coordinate'}});
    out+=matrix(440,485,130,28,'Write vector Bₜ',`${C().state_size}`,spec('write-vector','Write vector Bₜ','projection',t('sharing.description'),{activation:true,rows:1,cols:8,axes:{vector:true,columns:'memory coordinate'}}))+line(505,553,505,598);
    out+=line(612,647,780,647,'diagram-contribution')+matrix(792,595,145,105,'Current write',shape(),spec('write-matrix','Current write','write',t('write.description'),{activation:true,rows:5,cols:8,axes:{rows:'head value coordinate',columns:'memory coordinate'}}));
    out+=line(865,549,865,412,'diagram-contribution')+text(1100,640,'Δₜ xₜ ⊗ Bₜ',25,'diagram-math');
    // The direct D*x path branches from the same value vector and joins after reading.
    out+=path('M260,647 V800 H1480')+box(1492,765,150,70,'Dₕ × xₜ','skip','skip',t('skip.description'),{weight:param('mixer.D'),output:`${C().head_dim} coordinates`,note:'The parameter vector contains one D scalar for each of the 24 heads.'});
    out+=path('M1654,800 H1695 V412','diagram-contribution');
    out+=text(1110,120,'Δₜ: positive, input-dependent time step',23,'diagram-note');
    out+=text(1080,901,t('head.note'),24,'diagram-note');
    views.kimiHead={x:-165,y:-45,w:2390,h:1085};return out;
  }
  function output(){
    let out=outline(-20,-15,1430,560)+text(10,28,'Output projection',29,'diagram-title','start');
    out+=stream(-95,185,280)+box(197,237,150,86,'RMSNorm','final-norm','norm',t('norm.description'),{weight:'output_norm.weight',input:td(),output:td()});
    const x=492,y=175,w=145,height=210,weight='output_layer.weight';
    out+=line(359,280,480,280)+record(spec('output-table','Shared token table','output',t('output.description'),{weight,input:td(),output:td(C().vocabulary_size),inputAxes:axes,outputAxes:{rows:'sequence position',columns:'vocabulary entry'},codePosition:{x:x+w,y:y-42,size:16}}),brokenGrid(x,y,w,height)+text(x+w/2,y-15,'Shared token table',23,'diagram-title')+dimensions(x+w/2,y+height+28,`${fmt(C().vocabulary_size)} × ${fmt(d())}`,parameterAxes(weight),17,'middle','The token axis is shortened.'));
    out+=line(649,280,855,280)+vectors(867,242.5,210,'Logits',C().vocabulary_size,'logits','output',t('output.description'),75)+line(1089,280,1475,280,'diagram-contribution')+text(1250,252,'on to sampling',23,'diagram-note');
    out+=text(710,484,'The token embedding and output projection use the same learned table.',24,'diagram-note');views.kimiOutput={x:-115,y:-40,w:1650,h:650};return out;
  }
  function feedforward(){views.kimiFeedforward={x:0,y:0,w:1,h:1};return '';}
  function depth(){views.kimiDepth={x:0,y:0,w:1,h:1};return '';}
  return {isDense:()=>true,isDelta:()=>false,attentionName:()=> 'State-space mixer',defaultSpec,block,attention,head,output,feedforward,depth};
}
