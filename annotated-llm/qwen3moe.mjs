import {illustrativeRouting} from './moe-routing.mjs';

export function createMoEDiagrams(v){
  const {state,C,N,d,fmt,copy,text,line,path,outline,grid,stream,operation,matrix,record,plus,dimensions,views,param}=v;
  const route=()=>illustrativeRouting(C().num_experts,C().experts_per_token,state.moePosition,state.block);
  const shape=(rows=N())=>`${rows} × ${fmt(d())}`;
  const axes={rows:'sequence position',columns:'residual coordinate'};
  const expertSpec=expert=>({key:`expert-${expert}`,title:copy('moe.expert-label',{expert}),kind:'FEEDFORWARD EXPERT',source:'expert',parameterScope:'expert',expert,enter:'expert',input:shape(1),output:shape(1),inputAxes:axes,outputAxes:axes,
    description:copy('moe.expert-description',{expert}),note:copy(route().selected.some(item=>item.expert===expert)?'moe.expert-selected':'moe.expert-unselected',{expert,position:state.moePosition+1})});
  const defaultSpec=()=>({key:'moe-default',title:copy('moe.title'),kind:'MIXTURE OF EXPERTS',source:'feedforward',parameterScope:'feedforward',input:shape(),output:shape(),description:copy('moe.description'),note:copy('moe.schematic-note')});

  function preview(){
    // RMSNorm, routing, three representative experts, and the weighted sum.
    let out='<g class="feedforward-preview" aria-hidden="true">';
    out+=outline(887,156,28,18)+text(901,169,copy('label.rms'),12,'diagram-title')+line(915,165,941,165,'diagram-wire');
    out+=outline(936,131,58,16)+text(965,143,copy('moe.router'),11,'diagram-title')+path('M929,165 V139 H936 M994,139 H1005 V148','moe-control');
    out+=path('M944,153 V185 M1060,153 V185','diagram-wire');
    for(let i=0;i<3;i++){const y=153+i*16;out+=line(944,y,964,y,'diagram-wire')+outline(964,y-5,64,10,'#f7efdf')+line(1028,y,1060,y,'diagram-contribution');}
    out+=line(1060,165,1078,165,'diagram-contribution')+outline(1078,157,18,18)+text(1087,170,'Σ',13,'diagram-math')+line(1096,165,1118,165,'diagram-contribution');
    return out+'</g>';
  }

  function diagram(){
    views.moeLocal={x:-310,y:-50,w:1780,h:1020};
    if(!N())return outline(-275,-25,1700,965)+text(550,440,copy('moe.empty'),28,'diagram-title');
    const example=route(),chosen=new Set(example.selected.map(item=>item.expert)),rowShape=shape(1),streamY=870;
    let out=outline(-275,-25,1700,965)+text(-240,18,copy('moe.title'),28,'diagram-title','start')
      +text(1390,18,copy('label.block-block',{block:state.block}),20,'diagram-shape','end')
      +text(-240,60,copy('moe.routing-note'),21,'diagram-note','start');
    out+=stream(-300,1460,streamY)+path('M-252,870 V500 H-242');
    out+=operation(-230,466,86,68,copy('label.rmsnorm'),{key:'moe-input-norm',labelText:copy('label.rms-norm-lines'),source:'norm',weight:param('feed_forward_norm.weight'),description:copy('description.read-the-residual-stream-after-attention-s-addition-and-normalize'),input:shape(),output:shape()});
    out+=line(-132,500,-117,500);
    out+=matrix(-105,450,86,100,copy('moe.expert-input'),rowShape,{key:'moe-input',title:copy('moe.position-title',{position:state.moePosition+1}),source:'feedforward',activation:true,rows:1,axes,output:rowShape,description:copy('moe.position-description',{position:state.moePosition+1})});
    out+=text(-62,610,copy('moe.position-title',{position:state.moePosition+1}),22,'diagram-note');
    out+=line(-7,500,335,500,'diagram-wire')+path('M10,500 V162 H48');
    const stages=[
      {x:60,w:155,label:'moe.router',key:'router',description:'moe.router-description',input:rowShape,output:'1 × 128',weight:param('feed_forward.router.weight')},
      {x:277,w:130,label:'moe.softmax',key:'routing_probs',description:'moe.softmax-description',input:'1 × 128',output:'1 × 128'},
      {x:461,w:180,label:'moe.topk',key:'topk',description:'moe.topk-description',input:'1 × 128',output:'1 × 8'},
      {x:699,w:190,label:'moe.renormalize',key:'route_norm',description:'moe.renormalize-description',input:'1 × 8',output:'1 × 8'},
    ];
    stages.forEach((stage,i)=>{
      const outputAxes={rows:'sequence position',columns:i<2?'expert':'selected expert'};
      out+=operation(stage.x,130,stage.w,64,copy(stage.label),{key:`moe-${stage.key}`,source:stage.key,description:copy(stage.description),input:stage.input,output:stage.output,inputAxes:i===0?axes:{rows:'sequence position',columns:i===3?'selected expert':'expert'},outputAxes,weight:stage.weight});
      out+=dimensions(stage.x+stage.w/2,229,stage.output,outputAxes,18);
      if(i<3)out+=line(stage.x+stage.w+12,162,stages[i+1].x-12,162);
    });
    out+=path('M901,162 H1160 V268','moe-control')+text(1050,228,copy('moe.ids-and-weights'),20,'diagram-note');
    out+=path('M335,350 V714','diagram-wire')+path('M365,281 H1180 V790 H365 Z','gqa-kv-box');
    out+=text(525,318,copy('moe.selected-title'),23,'diagram-title')+text(830,318,copy('moe.weight-label'),20,'diagram-note')+text(1020,318,copy('label.contribution'),20,'diagram-note');
    example.selected.forEach((item,i)=>{
      const y=350+i*52;
      out+=line(335,y,403,y);
      out+=operation(415,y-18,220,36,copy('moe.expert-label',{expert:item.expert}),expertSpec(item.expert));
      out+=line(647,y,748,y,'diagram-contribution');
      out+=operation(760,y-18,140,36,`× ${item.weight.toFixed(3)}`,{key:`route-weight-${item.expert}`,title:copy('moe.weight-title',{expert:item.expert}),kind:'SCALAR MULTIPLICATION',source:'moe_combine',description:copy('moe.weight-description',{expert:item.expert,probability:item.probability.toFixed(4),weight:item.weight.toFixed(4)}),input:rowShape,output:rowShape});
      out+=line(912,y,952,y,'diagram-contribution')+grid(964,y-14,115,28,true,1,8)+line(1091,y,1300,y,'diagram-contribution');
    });
    out+=text(760,767,copy('moe.selected-note'),20,'diagram-note');
    out+=path('M1300,350 V793','diagram-contribution');
    out+=operation(1205,805,190,42,copy('moe.sum'),{key:'moe-sum',source:'moe_combine',description:copy('moe.combine-description'),input:'8 × 2,048',output:rowShape,inputAxes:{rows:'selected expert',columns:'residual coordinate'},outputAxes:axes});
    out+=line(1300,847,1300,850,'diagram-wire')+plus(1300,streamY,{key:'moe-add',source:'residual_ff',description:copy('moe.combine-description'),input:shape(),output:shape()});
    out+=text(-240,917,copy('moe.footer'),20,'diagram-note','start');
    out+=text(-70,666,copy('moe.expert-bank'),22,'diagram-title');
    for(let expert=0;expert<C().num_experts;expert++){
      const x=-230+(expert%16)*19,y=688+Math.floor(expert/16)*18;
      out+=record({...expertSpec(expert),tooltip:`Expert ${expert}${chosen.has(expert)?' · selected at this position':''}`},`<g class="moe-bank-cell${chosen.has(expert)?' routed':''}">${outline(x,y,17,15,chosen.has(expert)?'#c4ddd7':'#fff')}${text(x+8.5,y+11,expert,9,'diagram-shape')}</g>`);
    }
    out+=text(-75,850,copy('moe.bank-note'),15,'diagram-note');
    views.moeLocal={x:-310,y:-50,w:1780,h:1020};
    return out;
  }
  return {route,expertSpec,defaultSpec,preview,diagram};
}
