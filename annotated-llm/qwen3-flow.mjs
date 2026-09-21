// Connection descriptions belong to the Qwen3 diagram's formulation. In
// particular, projected head contributions remain separate until the sum.
const fmt = value => value.toLocaleString('en-US');
export function qwen3FlowInfo(kind, {config, positions, vocabularySize}, options={}) {
  const d=config.token_embedding_dim, h=config.head_dim, e=config.expanded_dim;
  const shape=width=>`${fmt(positions)} × ${fmt(width)}`;
  const vectors=(name,width,meaning)=>({name,shape:shape(width),meaning:meaning||`One ${fmt(width)}-coordinate vector per sequence position.`});
  const square=(name,meaning)=>({name,shape:shape(positions),meaning});
  const info={
    tokens:{name:'Token IDs',shape:`${fmt(positions)}`,meaning:'One integer ID per sequence position.'},
    embeddings:vectors('Token embedding vectors',d),
    residual:vectors('Residual vectors',d),
    afterAttention:vectors('Residual after attention',d,'One vector per position, including the attention contribution.'),
    afterFeedforward:vectors('Residual after feedforward',d,'One vector per position, including both contributions from this block.'),
    normalized:vectors('Normalized embedding vectors',d),
    attention:vectors('Attention contribution',d),
    feedforward:vectors('Feedforward contribution',d),
    head:vectors('Head output',h,'One mixed value vector per query position.'),
    projectedHead:vectors('Projected head contribution',d),
    q:vectors('Queries',h), k:vectors('Keys',h), v:vectors('Values',h),
    qNorm:vectors('Normalized queries',h), kNorm:vectors('Normalized keys',h),
    qRoPE:vectors('Queries after RoPE',h), kRoPE:vectors('Keys after RoPE',h),
    scores:square('Attention scores','Rows: query positions. Columns: key positions.'),
    masked:square('Masked attention scores','Query positions × key positions. Future positions contain −∞.'),
    weights:square('Attention weights','Query positions × key positions. Each row sums to 1; future weights are 0.'),
    gate:vectors('Gate projection output',e),
    activatedGate:vectors('SiLU gate output',e),
    value:vectors('Value projection output',e),
    product:vectors('Gated values',e,'One vector per position: gate × value, coordinate by coordinate.'),
    logits:vectors('Logits',config.vocabulary_size,'Rows: sequence positions. Columns: vocabulary entries.'),
    selectedLogits:{name:'Last-position logits',shape:fmt(vocabularySize||config.vocabulary_size),meaning:'One score per usable vocabulary entry; padding entries are excluded.'},
    next:{name:'Next token ID',shape:'scalar',meaning:'A single integer ID, which we then append to the context for the next pass.'},
  };
  if(kind==='contributions')return {name:'Head contributions',shape:`${options.count??config.num_heads} tensors · ${shape(d)} each`,meaning:'Separate contributions, each position × residual coordinate; added at Sum.'};
  if(!info[kind])throw new Error(`Unknown Qwen3 connection: ${kind}`);
  return info[kind];
}

export function createQwen3Flow({enabled,context,escape}) {
  const describe=(kind,options)=>qwen3FlowInfo(kind,context(),options);
  function flow(markup,kind,from='',to='',options={}) {
    if(!enabled)return markup;
    const info=describe(kind,options),label=`${info.name} · ${info.shape}\n${info.meaning}`;
    return `<g class="tensor-flow" data-flow="${kind}" data-flow-name="${escape(info.name)}" data-flow-shape="${escape(info.shape)}" data-flow-meaning="${escape(info.meaning)}" data-flow-from="${escape(from)}" data-flow-to="${escape(to)}" data-tooltip="${escape(label)}" role="img" tabindex="0" aria-label="${escape(label)}">${markup}</g>`;
  }
  function boundary(x,y,kind,label,anchor='start',stacked=false) {
    if(!enabled)return '';
    const info=describe(kind),value=`${label} · ${info.shape}`;
    const lines=stacked?`<tspan x="${x}">${escape(label)}</tspan><tspan x="${x}" dy="21">${escape(info.shape)}</tspan>`:escape(value);
    return `<g class="flow-boundary" data-tooltip="${escape(info.name+'\n'+info.meaning)}" role="img" tabindex="0" aria-label="${escape(value+'. '+info.meaning)}"><text x="${x}" y="${y}" text-anchor="${anchor}" class="flow-boundary-label" font-size="17">${lines}</text></g>`;
  }
  return {flow,boundary};
}

// Hit areas and highlights share the paths' actual geometry, including folded
// head groups and the five-line feedforward bundles. They never cover boxes.
export function prepareFlowTargets(root) {
  for(const group of root.querySelectorAll('.tensor-flow')) {
    if(group.querySelector('.ff-bundle'))group.classList.add('flow-bundle');
    const seen=new Set();
    for(const source of [...group.querySelectorAll('path')]) {
      const geometry=source.getAttribute('d')||'';
      if(seen.has(geometry))continue;
      seen.add(geometry);
      source.classList.add('flow-original');
      for(const className of ['flow-highlight','flow-hit']) {
        const clone=source.ownerDocument.createElementNS('http://www.w3.org/2000/svg','path');
        clone.setAttribute('d',source.getAttribute('d')||'');
        clone.setAttribute('class',className);
        // Keep a direct reference: layout animations can change the source d.
        clone.flowSource=source;
        group.append(clone);
      }
    }
    const bounds=group.ownerDocument.createElementNS('http://www.w3.org/2000/svg','rect');
    bounds.setAttribute('class','flow-focus-bounds');
    group.append(bounds);
  }
  syncFlowTargets(root);
}
export function syncFlowTargets(root) {
  for(const clone of root.querySelectorAll('.flow-highlight,.flow-hit'))clone.setAttribute('d',clone.flowSource.getAttribute('d')||'');
  for(const group of root.querySelectorAll('.tensor-flow')){
    const paths=[...group.querySelectorAll('.flow-original')];
    if(!paths.length)continue;
    const boxes=paths.map(path=>path.getBBox()),left=Math.min(...boxes.map(b=>b.x)),top=Math.min(...boxes.map(b=>b.y));
    const right=Math.max(...boxes.map(b=>b.x+b.width)),bottom=Math.max(...boxes.map(b=>b.y+b.height));
    const rect=group.querySelector('.flow-focus-bounds');
    // Give horizontal/vertical connections a nonzero keyboard focus box.
    for(const [name,value]of Object.entries({x:left-4,y:top-4,width:right-left+8,height:bottom-top+8}))rect.setAttribute(name,value);
  }
}

export function highlightFlow(target,root,nodes) {
  clearFlowHighlight(root);
  if(!target?.matches('.tensor-flow'))return;
  target.classList.add('flow-active');
  const keys=new Set([target.dataset.flowFrom,target.dataset.flowTo].filter(Boolean));
  const layer=target.closest('[id^="layer-"]');
  for(const element of layer.querySelectorAll('[data-node]')) {
    if(keys.has(nodes.get(element.dataset.node)?.key))element.classList.add('flow-endpoint');
  }
}
export function clearFlowHighlight(root) {
  for(const element of root.querySelectorAll('.flow-active,.flow-endpoint'))element.classList.remove('flow-active','flow-endpoint');
}
