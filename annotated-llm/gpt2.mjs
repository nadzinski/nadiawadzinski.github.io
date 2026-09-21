import {parameterAxes} from './matrix-info.mjs';

export const gptText={
  'model.description':'GPT-2 adds a learned position vector to each token embedding, then passes the sequence through twelve transformer blocks. Attention mixes information across sequence positions; feedforward transforms each position independently. Final LayerNorm and the shared token/output table produce the logits used to choose the next token.',
  'block.description':'A residual stream passes through the block. Attention reads a LayerNorm-normalized copy and adds its contribution back. Feedforward then reads another normalized copy and adds its contribution. Each attention head has independent Q, K and V projections; every position uses the same feedforward network.',
  'block.footer':'Each line in the feedforward schematic represents one position vector.\nThe residual stream continues through all twelve blocks.',
  'attention.description':'Twelve attention heads read the same normalized sequence. Each has its own query, key and value projections, including learned biases. A head’s output is mapped back to 768 coordinates by its slice of the output matrix. Sum all twelve contributions, add one output bias, and add the result to the residual stream.',
  'head.description':'This head projects each normalized input vector to 64-coordinate query, key and value vectors, adding its own learned biases. Query–key dot products give attention scores. Mask future positions, divide by √64, and apply softmax across each row. The attention weights mix this head’s value vectors.',
  'norm.description':'At each position, subtract the mean of the 768 coordinates and divide by the square root of their variance plus 10⁻⁵. Apply a learned scale and bias to each coordinate. Attention, feedforward and the final output each have their own LayerNorm parameters.',
  'input.description':'Token IDs select rows of the 50,257 × 768 token table. Sequence indices 0–1,023 select rows of a separate learned position table. Add the two vectors coordinate by coordinate at each position. These sums form the initial residual stream. The same token at different positions gets different position vectors.',
  'positions.description':'This table contains one learned 768-coordinate vector for each of 1,024 absolute positions. Python indexes these positions from zero. Its values are added once, before the first transformer block.',
  'projection.description':'Every head has a learned 768 × 64 matrix and a 64-coordinate bias for each of Q, K and V. The same matrix and bias act on every sequence position. The checkpoint packs these projections together; the educational code exposes each head explicitly.',
  'scores.description':'Each cell is the dot product of a query vector and a key vector. Its row identifies the querying sequence position, and its column identifies the position being read. These unscaled scores are input-dependent activations.',
  'mask.description':'Replace scores above the diagonal with negative infinity. Those cells refer to future positions. The diagonal and earlier positions remain available; softmax gives the masked cells zero weight.',
  'weights.description':'Divide each unmasked score by √64 and apply softmax across its row. The resulting nonnegative weights sum to one. Multiply them by this head’s value vectors to obtain one output vector per position.',
  'feedforward.description':'LayerNorm supplies a normalized vector at each position. An affine projection expands its 768 coordinates to 3,072, GELU transforms each coordinate, and a second affine projection returns to 768. Both projections have learned biases. The result is added to the residual stream.',
  'feedforward.note':'Each line represents one embedding vector. The same feedforward network transforms each sequence position independently.\nThere is no mixing across sequence positions; that happens in attention.',
  'gelu.description':'GPT-2 uses the tanh approximation to GELU: 0.5u × [1 + tanh(√(2/π) × (u + 0.044715u³))]. It acts on every expanded coordinate independently. The curve is a schematic of this function.',
  'output.description':'Final LayerNorm normalizes each position’s vector. Multiplying it by the transpose of the token embedding table gives 50,257 logits. The embedding and output stages share the same learned table, with no additional output bias. Generation samples only from the final position’s logits.',
  'output.bias':'The output projection has one 768-coordinate bias. It is added once after summing all twelve projected head outputs. The subsequent residual addition uses this complete attention contribution.',
};

export function createGPT2Diagrams(v){
  const {state,C,N,d,fmt,copy,text,line,path,outline,grid,brokenGrid,stream,operation,matrix,record,plus,dimensions,views,param,outlinePath,BLOCK,attentionMatrix,richText,copyAttributes}=v;
  const t=key=>copy('gpt2.'+key),td=(width=d())=>`${N()} × ${fmt(width)}`;
  const axes={rows:'sequence position',columns:'residual coordinate'};
  const spec=(key,title,source,description,extra={})=>({key:'gpt2-'+key,title,source,description,...extra});
  const box=(x,y,w,h,label,key,source,description,extra={})=>operation(x,y,w,h,label,spec(key,label,source,description,extra));
  const vectors=(x,y,w,label,width,key,source,description)=>matrix(x,y,w,75,label,td(width),spec(key,label,source,description,{activation:true,rows:Math.max(1,Math.min(N(),5)),axes:{rows:'sequence position',columns:width===C().vocabulary_size?'vocabulary entry':width===64?'head coordinate':width===3072?'expanded coordinate':'residual coordinate'},output:td(width)}));
  const tokenTable=(x,y,w,height,label,key,source,weight)=>record(spec(key,label,source,t(source==='embedding'?'input.description':'output.description'),{weight,input:source==='embedding'?`${N()} token IDs`:td(),output:source==='embedding'?td():td(C().vocabulary_size),codePosition:{x:x+w,y:y-42,size:16}}),brokenGrid(x,y,w,height)+text(x+w/2,y-15,label,23,'diagram-title')+dimensions(x+w/2,y+height+28,`${fmt(C().vocabulary_size)} × ${fmt(d())}`,parameterAxes(weight),17,'middle','The token axis is shortened.'));
  const headSpec=head=>spec('head-'+head,`Head ${head}`,'attention',t('head.description'),{head,enter:'head',input:td(),output:td(64)});
  function defaultSpec(level){
    const info={model:['GPT-2 small','model','model.description','model'],embedding:['Token and position embeddings','embedding','input.description','embedding'],block:[`Transformer block ${state.block}`,'block','block.description','block'],attention:['Multi-head attention','attention','attention.description','attention'],head:[`Attention head ${state.head}`,'attention','head.description',null],feedforward:['Feedforward','feedforward','feedforward.description','feedforward'],output:['Output projection','output_stage','output.description','output']}[level];
    return spec(level,info[0],info[1],t(info[2]),{kind:'GPT-2 SMALL',parameterScope:info[3],input:['model','embedding'].includes(level)?`${N()} token IDs`:td(),output:['model','output'].includes(level)?td(C().vocabulary_size):level==='head'?td(64):td(),inputAxes:['model','embedding'].includes(level)?undefined:axes,outputAxes:['model','output'].includes(level)?{rows:'sequence position',columns:'vocabulary entry'}:level==='head'?{rows:'sequence position',columns:'head coordinate'}:axes});
  }
  function embeddingOverview(){
    return record({...defaultSpec('embedding'),enter:'embedding',box:{x:185,y:167,w:220,h:135}},outline(185,167,220,135)+text(295,193,'Token + position\nembeddings',23,'diagram-title')+grid(213,249,51,29,false,5,5)+text(283,272,'+',25,'diagram-math')+grid(305,249,51,29,false,5,5))+dimensions(295,337,td(),axes,18);
  }
  function embedding(){
    let out=outline(-40,-15,1515,710)+text(-5,30,'Token and position embeddings',29,'diagram-title','start');
    out+=text(-5,94,'Look up two learned vectors, then add them at each sequence position.',23,'diagram-note','start');
    out+=line(-95,235,110,235)+text(18,211,'Token IDs',22,'diagram-note');
    out+=tokenTable(122,165,120,145,'Token table','token-table','embedding','token_embedding_layer.weight');
    out+=line(254,235,360,235)+vectors(372,197,180,'Token vectors',d(),'token-vectors','embedding',t('input.description'));
    out+=line(-95,490,110,490)+text(15,459,`Indices 0–${Math.max(0,N()-1)}`,22,'diagram-note');
    out+=matrix(122,415,120,145,'Position table',`1,024 × ${fmt(d())}`,spec('position-table','Position embedding table','positions',t('positions.description'),{weight:'position_embedding_layer.weight',labelKind:'diagram-title'}));
    out+=line(254,490,360,490)+vectors(372,452,180,'Position vectors',d(),'position-vectors','positions',t('positions.description'));
    out+=path('M564,235 H750 Q790,235 790,275 V334 M564,490 H750 Q790,490 790,450 V388');
    out+=plus(790,361,{source:'positions',title:'Add token and position vectors',description:t('input.description')})+line(817,361,980,361);
    out+=vectors(992,323,190,'Initial residual stream',d(),'initial-residual','positions',t('input.description'))+line(1194,361,1530,361,'diagram-contribution');
    out+=text(1335,335,'to block 0',23,'diagram-note')+text(730,650,'Position embeddings are added once, before the transformer blocks.',23,'diagram-note');
    views.gptEmbedding={x:-115,y:-45,w:1700,h:830};return out;
  }
  function block(){
    let out=`<path id="block-frame" class="housing" d="${outlinePath(-15,BLOCK.frameTop,BLOCK.width,state.blockHeight)}"/>`+text(28,35,`Transformer block ${state.block}`,27,'diagram-title','start')+dimensions(1180,35,td(),axes,19,'end');
    out+=stream(-38,1260,340)+text(35,310,'Residual stream',23,'diagram-note','start');
    out+=path('M135,340 V165 H319');
    out+=record(spec('attention','Multi-head attention','residual',t('attention.description'),{enter:'attention',input:td(),output:td(),box:{x:331,y:105,w:264,h:122}}),outline(331,105,264,122)+text(463,130,'Multi-head attention',23,'diagram-title')+text(463,216,'LayerNorm · 12 heads',17,'diagram-shape')+[147,158,169,190].map(y=>line(362,y,390,y)+outline(402,y-3,102,6,'#e8f2f1')+line(516,y,557,y,'diagram-contribution')).join('')+text(451,182,'…',14,'diagram-note'));
    out+=path('M607,165 H647 V313','diagram-contribution')+plus(647,340,{source:'residual',title:'Add attention contribution',description:t('block.description')});
    out+=path('M751,340 V165 H917');
    out+=record(spec('feedforward','Feedforward','residual_ff',t('feedforward.description'),{enter:'feedforward',input:td(),output:td(),box:{x:929,y:105,w:215,h:122}}),outline(929,105,215,122)+text(1036,134,'Feedforward',23,'diagram-title')+text(1036,211,'LayerNorm · GELU',18,'diagram-shape')+[949,1000,1051,1102].map((x,i)=>outline(x,161,29,21)+text(x+14.5,176,['LN','↑','G','↓'][i],13,'diagram-title')).join('')+path('M941,171 H1134','diagram-wire'));
    out+=path('M1156,165 H1190 V313','diagram-contribution')+plus(1190,340,{source:'residual_ff',title:'Add feedforward contribution',description:t('block.description')});
    out+=text(1175,400,state.block===11?'to final LayerNorm':`on to block ${state.block+1}`,22,'diagram-note','end');
    out+=`<foreignObject id="block-prose" x="28" y="451" width="1150" height="100"><div xmlns="http://www.w3.org/1999/xhtml" id="block-prose-text" class="block-prose" ${copyAttributes(t('block.description'))}>${richText(t('block.description'))}</div></foreignObject>`;
    return out;
  }
  function attention(){
    const ys=[165,220,275,395,555],heads=[0,1,2,3,11];
    let out=outline(-300,-10,1510,880)+text(-268,35,'Multi-head attention',30,'diagram-title','start')+text(-268,79,'12 independent query/key/value heads · 64 coordinates per head',22,'diagram-note','start');
    out+=stream(-345,1260,795)+path('M-275,795 V375 H-245')+text(-264,758,'Residual stream',22,'diagram-note','start');
    out+=box(-233,340,86,70,'Layer\nNorm','norm','norm',t('norm.description'),{weight:param('attention_norm.weight'),input:td(),output:td()});
    out+=line(-135,375,-70,375)+path('M-70,165 V555','diagram-wire');
    out+=text(210,124,'12 heads',23,'diagram-title')+text(570,124,'Output projection',23,'diagram-title');
    ys.forEach((y,i)=>{out+=line(-70,y,8,y)+operation(20,y-20,370,40,i===3?'… heads 3–10 …':`Head ${heads[i]}`,headSpec(heads[i]))+line(402,y,460,y);
      out+=box(472,y-20,195,40,i===3?'8 contributions':`Wₒ⁽${heads[i]}⁾`,'slice-'+i,'combine',t('attention.description'),{input:i===3?'8 head outputs':td(64),output:td()});
      out+=line(679,y,730,y,'diagram-contribution')+grid(742,y-15,140,30,true,Math.max(1,Math.min(N(),5)),8)+line(894,y,1025,y,'diagram-contribution');});
    out+=path('M1025,165 V600','diagram-contribution')+box(930,612,190,48,'Sum all 12','sum','combine',t('attention.description'),{output:td()})+line(1025,672,1025,686,'diagram-contribution');
    out+=box(930,698,190,45,'Add output bias','bias','combine',t('output.bias'),{weight:param('attention.out_proj.bias'),output:td(),fontSize:21})+line(1025,755,1025,768,'diagram-contribution')+plus(1025,795,{source:'residual',title:'Add attention to residual stream',description:t('attention.description')});
    out+=text(1170,839,'on to feedforward',22,'diagram-note','end');views.kimiAttention={x:-365,y:-35,w:1650,h:970};return out;
  }
  function head(){
    let out=outline(-95,-20,2180,840)+text(-60,25,`Attention head ${state.head}`,29,'diagram-title','start');
    out+=line(-145,260,-2,260)+vectors(10,222,95,'Normalized\nembedding vectors',d(),'head-input','norm',t('norm.description'));
    for(const [letter,y] of [['q',150],['k',340],['v',595]]){
      out+=path(`M117,260 H145 V${y} H193`,'diagram-wire')+matrix(205,y-45,95,90,'W_'+letter,`${fmt(d())} × 64`,spec(letter,'Head '+letter.toUpperCase()+' projection',letter,t('projection.description'),{weight:param('attention.W_'+letter),weightHead:state.head,input:td(),output:td(64)}));
      out+=line(312,y,337,y)+box(349,y-25,108,50,'+ b_'+letter,'bias-'+letter,letter,t('projection.description'),{input:td(64),output:td(64),fontSize:24})+line(469,y,511,y);
      out+=vectors(523,y-37.5,110,letter.toUpperCase(),64,letter+'-vectors',letter,t('projection.description'));
    }
    out+=path('M645,150 H720 V288 H754')+line(645,340,754,340);
    out+=attentionMatrix(785,240,160,'scores',spec('scores','Attention scores','scores',t('scores.description'),{input:`(${td(64)}) @ (64 × ${N()})`,output:`${N()} × ${N()}`}));
    out+=line(957,320,1000,320)+box(1012,290,80,60,'Mask','mask','mask',t('mask.description'))+line(1104,320,1142,320);
    out+=attentionMatrix(1172,240,160,'masked',spec('masked','Masked scores','mask',t('mask.description')));
    out+=line(1344,320,1380,320)+box(1392,280,115,80,'÷ √64\nSoftmax','softmax','softmax',t('weights.description'),{labelLines:['÷ √64','Softmax'],labelKinds:['diagram-math','diagram-title']})+line(1519,320,1557,320);
    out+=attentionMatrix(1587,240,160,'weights',spec('weights','Attention weights','softmax',t('weights.description')));
    out+=line(1667,412,1667,568,'diagram-contribution')+line(645,595,1630,595,'diagram-contribution')+text(1000,626,'Each row mixes this head’s value vectors.',23,'diagram-note');
    out+=box(1640,570,54,50,'×','value-mix','mix',t('weights.description'))+line(1706,595,1780,595,'diagram-contribution')+vectors(1792,557.5,130,'Head output',64,'head-output','mix',t('weights.description'))+line(1934,595,2135,595,'diagram-contribution');
    out+=text(1975,705,'to output projection',21,'diagram-note')+text(865,443,'Q Kᵀ',24,'diagram-math')+text(1070,757,'W and b are learned parameters. Q, K, V and attention scores depend on the input.',23,'diagram-note');
    views.kimiHead={x:-165,y:-50,w:2360,h:950};return out;
  }
  function feedforward(){
    const bundle=(x1,x2)=>Array.from({length:Math.min(N(),5)},(_,i)=>line(x1,242.5+(i+.5)*75/Math.min(N(),5),x2,242.5+(i+.5)*75/Math.min(N(),5),'diagram-contribution')).join('');
    let out=outline(-315,-20,1985,740)+text(-280,27,'Feedforward',29,'diagram-title','start');
    out+=stream(-370,1720,570)+path('M-275,570 V280 H-247')+text(-275,608,'Residual stream',22,'diagram-note','start');
    out+=box(-235,245,105,70,'Layer\nNorm','ff-norm','norm',t('norm.description'),{weight:param('feed_forward_norm.weight'),input:td(),output:td()})+line(-118,280,-5,280);
    out+=vectors(7,242.5,100,'Normalized\nembedding vectors',d(),'ff-input','norm',t('feedforward.description'))+bundle(119,180);
    out+=box(192,230,220,100,'Input projection\n+ bias','ff-expand','ff_up',t('feedforward.description'),{weight:param('feed_forward.input_proj.weight'),input:td(),output:td(3072)});
    out+=dimensions(302,370,'768 → 3,072',{rows:'input coordinate',columns:'expanded coordinate'},19)+bundle(424,500);
    out+=record(spec('gelu','GELU','gelu',t('gelu.description'),{input:td(3072),output:td(3072),box:{x:512,y:230,w:190,h:100}}),outline(512,230,190,100)+text(647,286,'GELU',22,'diagram-title'));
    // A small schematic of GPT-2's tanh GELU curve, overlaid within the box.
    const gelu=u=>.5*u*(1+Math.tanh(Math.sqrt(2/Math.PI)*(u+.044715*u**3)));
    out+=path('M529,312 H600 M559,319 V258','silu-axis')+path(Array.from({length:49},(_,i)=>{const u=-3+i/8;return `${i?'L':'M'}${529+(u+3)*11},${311-gelu(u)*16}`;}).join(' '),'silu-curve');
    out+=bundle(714,790)+box(802,230,230,100,'Output projection\n+ bias','ff-down','down',t('feedforward.description'),{weight:param('feed_forward.output_proj.weight'),input:td(3072),output:td()});
    out+=dimensions(917,370,'3,072 → 768',{rows:'expanded coordinate',columns:'residual coordinate'},19)+bundle(1044,1110);
    out+=vectors(1122,242.5,145,'Contribution',d(),'ff-contribution','down',t('feedforward.description'))+path('M1279,280 H1515 V543','diagram-contribution')+plus(1515,570,{source:'residual_ff',title:'Add feedforward contribution',description:t('feedforward.description')});
    out+=text(690,655,t('feedforward.note'),23,'diagram-note');views.kimiFeedforward={x:-390,y:-45,w:2180,h:860};return out;
  }
  function output(){
    let out=outline(-20,-15,1430,560)+text(10,28,'Output projection',29,'diagram-title','start');
    out+=stream(-95,185,280)+box(197,237,150,86,'LayerNorm','final-norm','norm',t('norm.description'),{weight:'output_norm.weight',input:td(),output:td()});
    out+=line(359,280,480,280)+tokenTable(492,175,145,210,'Shared token table','output-table','output','output_layer.weight');
    out+=line(649,280,855,280)+vectors(867,242.5,210,'Logits',C().vocabulary_size,'logits','output',t('output.description'))+line(1089,280,1475,280,'diagram-contribution')+text(1250,252,'on to sampling',23,'diagram-note');
    out+=text(710,484,'The token embedding and output projection use the same learned table.',24,'diagram-note');views.kimiOutput={x:-115,y:-40,w:1650,h:650};return out;
  }
  // The shared navigation infrastructure has slots for other models' deeper views.
  function depth(){views.kimiDepth={x:0,y:0,w:1,h:1};return '';}
  return {isDense:()=>true,isDelta:()=>false,attentionName:()=> 'Multi-head attention',defaultSpec,embeddingOverview,embedding,block,attention,head,feedforward,output,depth};
}
