// Axes in the layout actually returned by /api/weights. Per-head Q/K/V
// parameters have already been transposed by the checkpoint reader.
export function parameterAxes(name, outputHead=false, config={}) {
  if(config.state_space_model){
    if(name.endsWith('mixer.in_proj.weight'))return {rows:'packed projection coordinate (z, values, B, C, time steps)',columns:'residual coordinate'};
    if(name.endsWith('mixer.conv1d.weight'))return {rows:'convolution channel',columns:'filter tap; the middle tensor axis is a singleton input channel'};
    if(name.endsWith('mixer.conv1d.bias'))return {rows:'one display row',columns:'convolution channel',vector:true,role:'bias'};
    if(name.endsWith('mixer.out_proj.weight'))return {rows:'residual coordinate',columns:'expanded coordinate'};
    if(name.endsWith('mixer.norm.weight'))return {rows:'one display row',columns:'expanded coordinate',vector:true};
    if(/\.(D|A_log|dt_bias)$/.test(name))return {rows:'one display row',columns:'state-space head',vector:true,role:name.endsWith('dt_bias')?'bias':'scale'};
  }
  if(config.gemma4){
    if(name==='per_layer_embedding.weight')return {rows:'token ID',columns:'layer × per-layer embedding coordinate'};
    if(name==='per_layer_projection.weight')return {rows:'layer × per-layer embedding coordinate',columns:'initial embedding coordinate'};
    if(name==='per_layer_norm.weight')return {rows:'one display row',columns:'per-layer embedding coordinate',vector:true};
    if(/attention\.(query|key)_norm\.weight$/.test(name))return {rows:'one display row',columns:'head coordinate',vector:true};
    if(name.endsWith('.layer_scale'))return {rows:'one display row',columns:'block scalar',vector:true};
  }
  if(config.deepseek_v41){
    if(name.endsWith('engram.table.weight'))return {rows:'n-gram hash bucket',columns:'memory embedding coordinate'};
    if(name.endsWith('engram.projection.weight'))return {rows:'four keys and one value × residual coordinate',columns:'joined memory coordinate'};
    if(name.endsWith('attention.output_groups'))return {rows:'output projection group',columns:'projection output coordinate × grouped head coordinate'};
    if(name.endsWith('attention.sink'))return {rows:'one display row',columns:'attention head',vector:true,role:'bias'};
  }
  const expertCoordinate=config.attn_res_block_size?'expert latent coordinate':'residual coordinate';
  if(name==='token_embedding_layer.weight')return {rows:'token ID',columns:'embedding coordinate'};
  if(name==='position_embedding_layer.weight')return {rows:'absolute position index',columns:'embedding coordinate'};
  if(name==='output_layer.weight')return {rows:'vocabulary entry',columns:'residual coordinate'};
  if(name.endsWith('.W_gate'))return {rows:'input embedding coordinate',columns:'gate coordinate'};
  if(name.endsWith('.bias'))return {rows:'one display row',columns:'output coordinate',vector:true,role:'bias'};
  const projection=name.match(/\.W_([qkv])$/);
  if(projection)return {rows:'input embedding coordinate',columns:({q:'query',k:'key',v:'value'})[projection[1]]+' coordinate'};
  if(name.endsWith('.attention.out_proj.weight'))return {rows:'residual coordinate',columns:outputHead?'head coordinate':'combined head coordinate'};
  if(name.endsWith('feed_forward.router.weight'))return {rows:'expert',columns:'residual coordinate'};
  if(/\.experts\.\d+\.(gate|value)_proj\.weight$/.test(name))return {rows:'expert intermediate coordinate',columns:expertCoordinate};
  if(/\.experts\.\d+\.output_proj\.weight$/.test(name))return {rows:expertCoordinate,columns:'expert intermediate coordinate'};
  if(config.attn_res_block_size){
    if(name.endsWith('feed_forward.down_proj.weight'))return {rows:'expert latent coordinate',columns:'residual coordinate'};
    if(name.endsWith('feed_forward.up_proj.weight'))return {rows:'residual coordinate',columns:'expert latent coordinate'};
    if(/shared_experts\.(gate|value)_proj\.weight$/.test(name))return {rows:'shared-expert intermediate coordinate',columns:'residual coordinate'};
    if(name.endsWith('shared_experts.output_proj.weight'))return {rows:'residual coordinate',columns:'shared-expert intermediate coordinate'};
    if(name.endsWith('latent_norm.weight'))return {rows:'one display row',columns:'expert latent coordinate',vector:true};
  }
  if(/feed_forward\.(gate|value)_proj\.weight$/.test(name))return {rows:'expanded coordinate',columns:'residual coordinate'};
  if(name.endsWith('feed_forward.output_proj.weight'))return {rows:'residual coordinate',columns:'expanded coordinate'};
  if(name.endsWith('norm.weight'))return {rows:'one display row',columns:/\.[qk]_norm\.weight$/.test(name)?'head coordinate':'residual coordinate',vector:true};
  return {rows:'output coordinate',columns:'input coordinate'};
}

export function axisDescription(axes) {
  return axes.vector?`A learned ${axes.role==='bias'?'bias':'scale'} vector, shown as one row. Columns: ${axes.columns}.`:`Rows: ${axes.rows}. Columns: ${axes.columns}.`;
}

// Shapes come from Python's named parameters. Visual branches include their
// sibling input RMSNorm; shared K/V tensors are counted once in aggregates.
export function parameterSummary(parameters,config,{scope,block=0,expert=0,weight,outputHead=false}={}) {
  const elements=shape=>shape.reduce((total,size)=>total*size,1);
  if(scope==='kv-pair'){
    const groups=['k','v'].map(letter=>{
      const shape=parameters[`transformer_blocks.${block}.attention.W_${letter}`];
      return shape?{label:letter==='k'?'Shared key matrix':'Shared value matrix',count:elements(shape.slice(1))}:null;
    });
    if(groups.some(group=>!group))return null;
    return {count:groups.reduce((sum,group)=>sum+group.count,0),label:'Parameters in this shared K/V pair',groups,
      note:'Counts both matrices once, even though multiple query heads use them.'};
  }
  if(weight){
    const shape=parameters[weight];
    if(!shape)return null;
    const headTensor=shape.length===3&&/\.W_(q|k|v|gate)$/.test(weight);
    const slice=outputHead?[shape[0],config.head_dim]:headTensor?shape.slice(1):shape;
    const sliced=outputHead||headTensor;
    const shared=weight.match(/\.W_([kv])$/);
    return {count:elements(slice),label:sliced?'Parameters in this matrix slice':shape.length===1?'Parameters in this vector':'Parameters in this matrix',groups:[],
      note:shared?`The count covers one shared ${shared[1]==='k'?'key':'value'} projection. Heads that use it share these parameters.`:sliced?'The count covers the displayed head’s matrix slice.':null};
  }
  const prefix=`${config.state_space_model?'blocks':'transformer_blocks'}.${block}.`;
  const group=(label,prefixes)=>({label,names:Object.keys(parameters).filter(name=>prefixes.some(p=>name===p||name.startsWith(p+'.')))});
  const branch=(label,module,norm)=>group(label,[prefix+module,prefix+norm]);
  const duplicatedTiedTable=scope==='model'&&config.tie_word_embeddings!==false&&!config.shared_output_storage;
  let groups,note;
  if(scope==='model'){
    groups=[group(duplicatedTiedTable?'Token embeddings / output matrix (shared)':'Token embeddings',['token_embedding_layer']),...(config.learned_position_embeddings?[group('Position embeddings',['position_embedding_layer'])]:[]),group(config.state_space_model?'Mamba blocks':'Transformer blocks',[config.state_space_model?'blocks':'transformer_blocks']),group('Final normalization',['output_norm']),...(config.shared_output_storage||duplicatedTiedTable?[]:[group('Output matrix',['output_layer'])])];
    const listed=new Set(groups.flatMap(g=>g.names));
    if(config.shared_output_storage||duplicatedTiedTable)listed.add('output_layer.weight');
    const other=Object.keys(parameters).filter(name=>!listed.has(name));
    if(other.length)groups.push({label:'Other parameters',names:other});
  }else if(scope==='block'){
    groups=[branch('Attention, including RMSNorm','attention','attention_norm'),branch('Feedforward, including RMSNorm','feed_forward','feed_forward_norm')];
    if(config.state_space_model)groups=[branch('State-space mixer, including RMSNorm','mixer','mixer_norm')];
    if(config.hc_mult)groups.push(group('Hyper-connections',[prefix+'attention_hc',prefix+'feed_forward_hc']));
    if(config.deepseek_v41)groups.push(group('Engram memory',[prefix+'engram']));
    if(config.gemma4)groups.push(group('Contribution normalization',[prefix+'attention_output_norm',prefix+'feed_forward_output_norm']),group('Per-layer embedding contribution',[prefix+'per_layer_gate',prefix+'per_layer_output',prefix+'per_layer_output_norm']),group('Block scalar',[prefix+'layer_scale']));
    if(config.attn_res_block_size)groups.push(group('Attention Residuals',[prefix+'attention_mix',prefix+'feed_forward_mix']));
  }else if(scope==='embeddings'){
    groups=[group('Main token table',['token_embedding_layer']),group('Per-layer token table',['per_layer_embedding']),group('Per-layer projection and norm',['per_layer_projection','per_layer_norm'])];
  }else if(scope==='per-layer'){
    groups=[group('Per-layer contribution',[prefix+'per_layer_gate',prefix+'per_layer_output',prefix+'per_layer_output_norm'])];
  }else if(scope==='engram'){
    groups=[group('Engram memory',[prefix+'engram'])];
  }else if(scope==='mixer'){
    groups=[group('Input RMSNorm',[prefix+'mixer_norm']),group('Input projection',[prefix+'mixer.in_proj']),group('Causal convolution',[prefix+'mixer.conv1d']),group('Per-head decay, time-step bias and direct scale',[prefix+'mixer.A_log',prefix+'mixer.dt_bias',prefix+'mixer.D']),group('Gated normalization',[prefix+'mixer.norm']),group('Output projection',[prefix+'mixer.out_proj'])];
    note='The recurrent states are activations. They add no learned parameters. B and C share one projection group across all heads.';
  }else if(scope==='attention'){
    groups=[group('Input RMSNorm',[prefix+'attention_norm']),group('Query projections',[prefix+'attention.W_q']),group('Shared key projections',[prefix+'attention.W_k']),group('Shared value projections',[prefix+'attention.W_v']),group('Query/key normalization',[prefix+'attention.q_norm',prefix+'attention.k_norm']),group('Output projection',[prefix+'attention.out_proj'])];
    note='Includes the input RMSNorm. Shared key/value projections and normalization scales are each counted once.';
    if(config.learned_position_embeddings){
      groups=[group('Input LayerNorm',[prefix+'attention_norm']),group('Head projections, biases and output projection',[prefix+'attention'])];note='Counts all twelve heads and their biases, plus the output bias added after the head sum.';
    }else if(config.gemma4){
      groups=[group('Input and contribution RMSNorm',[prefix+'attention_norm',prefix+'attention_output_norm']),group('Attention projections and Q/K normalization',[prefix+'attention'])];note=block>=config.first_shared_block?'This block reuses earlier K/V activations. Its count includes its own query and output projections.':'Includes eight query heads, one key head, one value head, and their normalization.';
    }else if(config.hc_mult){
      groups=[group('Input RMSNorm',[prefix+'attention_norm']),group('Attention projections, controls, indexer and normalization',[prefix+'attention'])];
      note='Counts all 64 heads. The hyper-connection is counted separately in the block.';
    }else if(config.attn_res_block_size){
      groups=[group('Input RMSNorm',[prefix+'attention_norm']),group('Attention projections, controls and normalization',[prefix+'attention'])];
      note='Counts all 96 heads. The AttnRes reader is counted separately in the block.';
    }else if(config.layer_types?.[block]==='linear_attention'){
      groups=[group('Input RMSNorm',[prefix+'attention_norm']),group('Q/K/V projection',[prefix+'attention.in_proj_qkv']),group('Causal convolution',[prefix+'attention.conv1d']),group('Decay and write gates',[prefix+'attention.in_proj_a',prefix+'attention.in_proj_b',prefix+'attention.A_log',prefix+'attention.dt_bias']),group('Output gate and normalization',[prefix+'attention.in_proj_z',prefix+'attention.norm']),group('Output projection',[prefix+'attention.out_proj'])];
      note='Shared query/key projections are counted once. The recurrent state is an activation, so it adds no learned parameters.';
    }else if(config.layer_types){
      groups.push(group('Sigmoid gate projections',[prefix+'attention.W_gate']));
    }
  }else if(scope==='feedforward'||scope==='feedforward-core'){
    groups=[...(scope==='feedforward'?[group('Input RMSNorm',[prefix+'feed_forward_norm'])]:[]),group('Gate projection',[prefix+'feed_forward.gate_proj']),group('Value projection',[prefix+'feed_forward.value_proj']),group('Down projection',[prefix+'feed_forward.output_proj'])];
    if(config.learned_position_embeddings){groups=[group('Input LayerNorm',[prefix+'feed_forward_norm']),group('Input projection and bias',[prefix+'feed_forward.input_proj']),group('Output projection and bias',[prefix+'feed_forward.output_proj'])];}
    if(config.gemma4)groups.push(group('Contribution RMSNorm',[prefix+'feed_forward_output_norm']));
    if(config.num_experts&&(!(config.attn_res_block_size||config.hc_mult)||block>=config.first_dense_blocks)){
      groups=[...(scope==='feedforward'?[group('Input RMSNorm',[prefix+'feed_forward_norm'])]:[]),group('Router',[prefix+'feed_forward.router']),group('All experts',[prefix+'feed_forward.experts'])];
      note=`All ${config.num_experts} experts are stored. Each position selects ${config.experts_per_token}; the other experts can be selected at other positions.`;
      if(config.hc_mult){groups.push(group('Selection correction bias',[prefix+'feed_forward.selection_bias']),group('Shared expert',[prefix+'feed_forward.shared_experts']));note+=' One shared expert is always active.';}
      if(config.attn_res_block_size){
        groups.push(group('Selection correction bias',[prefix+'feed_forward.selection_bias']),group('Latent down/up projections and normalization',[prefix+'feed_forward.down_proj',prefix+'feed_forward.up_proj',prefix+'feed_forward.latent_norm']),group('Shared experts',[prefix+'feed_forward.shared_experts']));
        note+=' The two shared experts are always active.';
      }
    }
  }else if(scope==='expert'){
    const expertPrefix=prefix+`feed_forward.experts.${expert}.`;
    groups=[group('Gate projection',[expertPrefix+'gate_proj']),group('Value projection',[expertPrefix+'value_proj']),group('Output projection',[expertPrefix+'output_proj'])];
    note='This count covers one expert. Every expert has its own learned projection weights.';
  }else if(scope==='shared-experts'){
    groups=[group(config.hc_mult?'Shared expert':'Two shared experts',[prefix+'feed_forward.shared_experts'])];
  }else if(scope==='embedding'){
    groups=[group('Token table',['token_embedding_layer']),group('Position table',['position_embedding_layer'])];
  }else if(scope==='indexer'){
    groups=[group('Pool indexer',[prefix+'attention.indexer'])];
  }else if(scope?.startsWith('hyperconnection-')){
    groups=[group('Mapping, base and scales',[prefix+(scope.endsWith('attention')?'attention_hc':'feed_forward_hc')])];
  }else if(scope==='depth'){
    groups=[group('Attention reader',[prefix+'attention_mix']),group('Feedforward reader',[prefix+'feed_forward_mix'])];
    note='Both readers in this transformer block. Each has its own learned query and normalization scale.';
  }else if(scope==='output'){
    groups=[group('Final RMSNorm',['output_norm']),group('Output matrix',['output_layer'])];
    if(config.attn_res_block_size)groups.unshift(group('Final Attention Residuals reader',['output_mix']));
  }else return null;
  const names=new Set(groups.flatMap(g=>g.names));
  const count=[...names].reduce((sum,name)=>sum+elements(parameters[name]),0);
  if(duplicatedTiedTable){
    // Qwen3-0.6B ties these weights; this implementation retains both copies.
    const storedCount=count+elements(parameters['output_layer.weight']);
    note=`Qwen3-0.6B shares the learned weights between its embedding and output tables. Counting those weights once gives ${count.toLocaleString('en-US')} parameters (about ${(count/1e9).toFixed(1)} billion). This simple implementation stores separate copies of the two tables, so the sample code actually requires ${storedCount.toLocaleString('en-US')} stored weight values.`;
  }
  if(scope==='model'&&config.tie_word_embeddings===false)note='The text model has separate learned embedding and output tables. This count excludes the vision encoder and auxiliary multi-token-prediction modules.';
  if(scope==='model'&&config.num_experts)note=`This includes every expert and the separate embedding and output tables. About 3.3 billion parameters are active per token, with ${config.experts_per_token} of ${config.num_experts} experts selected at each layer. All expert weights still need to be stored.`;
  if(scope==='model'&&config.attn_res_block_size)note='This count covers the text backbone and all stored experts, excluding the vision encoder and projector. Block 0 is dense; the remaining blocks select sixteen routed experts and always evaluate the shared experts. Storage estimates describe hypothetical 16/32-bit tensors; the released routed-expert checkpoint uses packed MXFP4.';
  if(scope==='model'&&config.hc_mult)note='This count covers the text backbone and all stored experts, excluding the vision encoder and auxiliary prediction layer. Blocks 0–2 are dense. Later blocks select eight routed experts and evaluate one shared expert. The released checkpoint uses block FP8 weights; these storage estimates describe hypothetical 16/32-bit tensors.';
  if(scope==='model'&&config.deepseek_v41)note='The text backbone includes all 384 routed experts per block, one shared expert per block, the two Engram memory tables and their projections. Six routed experts are selected at each position. Vision and DSpark are omitted. The release uses mixed FP8/FP4 storage; the estimates show hypothetical floating-point tensors.';
  if(scope==='model'&&config.shared_output_storage)note='GPT-2 shares one learned table between token embedding and output projection. This count includes that table once, along with position embeddings, projection biases and all LayerNorm scales and biases.';
  if(scope==='model'&&config.gemma4)note='Counts the shared main token/output table once, all per-layer embeddings, and the tensors used by the text backbone. The released checkpoint also contains unused K/V projection copies in the sharing layers; those copies and the audio/vision modules are excluded.';
  if(scope==='model'&&config.state_space_model)note='Mamba-2 shares one learned table between token embedding and output projection. This count includes that table once, all state-space blocks and normalization scales. Recurrent states are activations and are not counted as parameters.';
  if(config.learned_position_embeddings)groups=groups.map(g=>({...g,label:g.label.replaceAll('RMSNorm','LayerNorm')}));
  return {count,label:scope==='model'&&!duplicatedTiedTable?'Stored weight values':'Learned parameters',
    groups:groups.map(g=>({label:g.label,count:g.names.reduce((sum,name)=>sum+elements(parameters[name]),0)})),note};
}

export function parameterBytes(count,bits){return count*bits/8;}

export function formatBytes(bytes){
  const units=['B','KiB','MiB','GiB'];
  let unit=0,value=bytes;
  while(value>=1024&&unit<units.length-1){value/=1024;unit++;}
  return `${value.toLocaleString('en-US',{maximumFractionDigits:unit?2:0})} ${units[unit]}`;
}
