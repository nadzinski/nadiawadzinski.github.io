// The diagrams use the same interface with the local API and exported files.
export function expandParameters(tree) {
  const parameters = {};
  function visit(ref, prefix) {
    if (ref < 0) { parameters[prefix] = [...tree.shapes[-ref-1]]; return; }
    for (const [name, child] of tree.nodes[ref]) visit(child, prefix ? `${prefix}.${name}` : name);
  }
  visit(tree.root, '');
  return parameters;
}

export const weightKey = ({name, head=0, output_head=false}) => `${name}|${head}|${Number(Boolean(Number(output_head)))}`;

export function resolveModelId(requested, config={}) {
  if (config.mode==='static' && config.models?.length) {
    return config.models.includes(requested) ? requested : config.models[0];
  }
  return requested || 'qwen-3-dense-no-kv';
}

export function createDataProvider({modelId, mode='local', baseURL, fetcher=globalThis.fetch}) {
  const cache = new Map();
  const base = new URL(baseURL);
  const capabilities = Object.freeze({editing:mode==='local', customContext:mode==='local', sampledWeights:mode==='static'});
  async function json(path, cached=false) {
    const url = new URL(path, base).href;
    if (cached && cache.has(url)) return cache.get(url);
    const pending = (async()=>{
      const response = await fetcher(url);
      const data = await response.json().catch(error=>{if(response.ok)throw error;return {};});
      if (!response.ok) throw new Error(data.error || `Could not read ${path} (${response.status}).`);
      if (data.error) throw new Error(data.error);
      return data;
    })();
    if (cached) cache.set(url, pending);
    try { return await pending; } catch(error) { cache.delete(url); throw error; }
  }
  const local = (kind, args={}) => json(`api/${kind}?`+new URLSearchParams({model:modelId,...args}));
  const modelFile = name => json(`data/models/${modelId}/${name}.json`, true);
  async function samples(args) {
    const index = await json('data/prepared/weights/index.json', true);
    const key = weightKey(args), file = index[key];
    if (!file) throw new Error('No saved weight sample for this matrix.');
    const bundle = await json('data/prepared/'+file, true);
    return bundle[key];
  }
  return {
    capabilities,
    textEndpoint: new URL(`api/text?model=${encodeURIComponent(modelId)}`, base).href,
    models: () => mode==='local' ? local('models') : json('data/models.json', true),
    text: () => mode==='local' ? local('text') : modelFile('text'),
    async model() {
      if (mode==='local') return local('model');
      const model = await modelFile('model');
      if (model.parameter_tree) { model.parameters=expandParameters(model.parameter_tree); delete model.parameter_tree; }
      return model;
    },
    async source(key) {
      if (mode==='local') return local('source', {key});
      const index=await modelFile('sources'), entry=index[key];
      if (!entry) throw new Error(`Unknown source: ${key}`);
      const file=await json(`data/models/${modelId}/source/${entry.file}.json`, true);
      return {...entry, lines:file.lines};
    },
    examples: () => json(`data/prepared/examples/${modelId}.json`, true),
    tokenize: text => {
      if (mode!=='local') throw new Error('Choose one of the prepared examples.');
      return local('tokenize', {text});
    },
    async weights(args) {
      if (mode==='local') return local('weights', args);
      const data=await samples(args);
      if (args.overview) return {...data.overview, patch_options:data.patches.map(p=>({row:p.rows[0],col:p.cols[0],rows:p.rows,cols:p.cols}))};
      const patch=data.patches.find(p=>p.rows[0]===args.row && p.cols[0]===args.col);
      if (!patch) throw new Error('Choose one of the saved patches.');
      return patch;
    },
  };
}
