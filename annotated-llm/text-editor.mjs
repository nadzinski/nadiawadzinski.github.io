import {textCatalog,validateText,placeholders,updateCopyElement,commonCopyValues,renderTemplate,refreshCopyElements} from './text-content.mjs';

export function createTextEditor({revision,endpoint='/api/text',isBusy=()=>false,onMode=()=>{},activeLayer=()=>null}){
  const toggle=document.querySelector('#edit-text'),save=document.querySelector('#save-text'),status=document.querySelector('#text-save-status');
  const panel=document.querySelector('#text-editor'),field=panel.querySelector('textarea'),message=panel.querySelector('.text-editor-message');
  let enabled=false,current=null,saving=false,savedRevision=revision;
  const originalAttributes=new WeakMap();
  const refreshCopies=()=>refreshCopyElements();
  function dirty(){return textCatalog.dirty||!!current&&field.value!==current.initial;}
  function updateStatus(text){
    save.hidden=!enabled&&!textCatalog.dirty;save.disabled=saving||!dirty();
    if(text!==undefined)status.textContent=text;
    else if(!saving)status.textContent=dirty()?'Unsaved changes':'';
    toggle.textContent=enabled?'Done editing':'Edit text';toggle.setAttribute('aria-pressed',String(enabled));
  }
  function refresh(){
    for(const el of document.querySelectorAll('[data-copy]')){
      const layer=el.closest('[id^="layer-"]');
      const available=enabled&&!isBusy()&&(!layer||layer.id==='layer-'+activeLayer())&&!el.closest('[aria-hidden="true"]');
      if(available){
        if(!originalAttributes.has(el))originalAttributes.set(el,{tabindex:el.getAttribute('tabindex'),role:el.getAttribute('role'),label:el.getAttribute('aria-label')});
        el.dataset.copyEditable='true';el.setAttribute('tabindex','0');el.setAttribute('role','button');
        el.setAttribute('aria-label','Edit text: '+el.textContent.trim().slice(0,100));
      }else if(originalAttributes.has(el)){
        const old=originalAttributes.get(el);
        for(const [name,value]of [['tabindex',old.tabindex],['role',old.role],['aria-label',old.label]]){if(value===null)el.removeAttribute(name);else el.setAttribute(name,value);}
        delete el.dataset.copyEditable;originalAttributes.delete(el);
      }
    }
  }
  function preview(value){
    for(const el of document.querySelectorAll('[data-copy]'))if(el.dataset.copy===current.id)updateCopyElement(el,value);
  }
  function close(){
    panel.hidden=true;current=null;updateStatus();refresh();
  }
  function commit(){
    if(!current)return true;
    try{textCatalog.set(current.id,field.value);}
    catch(error){message.textContent=error.message;message.classList.add('error');field.focus();return false;}
    refreshCopies();close();return true;
  }
  function cancel(){if(!current)return;preview(current.initial);const target=current.target;close();target.isConnected&&target.focus({preventScroll:true});}
  function position(target){
    const bounds=target.getBoundingClientRect(),style=getComputedStyle(target),matrix=target.getScreenCTM?.()||target.closest('foreignObject')?.getScreenCTM();
    const scale=matrix?Math.hypot(matrix.a,matrix.b):1;
    const size=Math.max(12,Math.min(40,parseFloat(style.fontSize)*scale));
    const width=Math.min(innerWidth-24,Math.max(200,bounds.width+16));
    const centered=style.textAnchor==='middle'||style.textAlign==='center';
    panel.style.width=width+'px';
    panel.style.left=Math.max(12,Math.min(innerWidth-width-12,centered?bounds.left+bounds.width/2-width/2:bounds.left-7))+'px';
    panel.style.top=Math.max(12,Math.min(innerHeight-180,bounds.top-7))+'px';
    field.style.fontFamily=style.fontFamily;field.style.fontSize=size+'px';field.style.fontStyle=style.fontStyle;
    field.style.fontWeight=style.fontWeight;field.style.lineHeight=(size*1.23)+'px';field.style.textAlign=centered?'center':style.textAnchor==='end'?'right':'left';
    field.wrap=target.namespaceURI==='http://www.w3.org/2000/svg'?'off':'soft';resizeField();
  }
  function resizeField(){
    field.style.height='auto';field.style.height=Math.min(Math.max(44,field.scrollHeight+2),Math.max(90,innerHeight-240))+'px';
    const rect=panel.getBoundingClientRect();if(rect.bottom>innerHeight-12)panel.style.top=Math.max(12,innerHeight-rect.height-12)+'px';
  }
  function begin(target){
    if(isBusy()||!enabled||target.dataset.copyEditable!=='true')return;
    if(current?.target===target)return;
    if(!commit())return;
    const id=target.dataset.copy,initial=textCatalog.template(id);
    current={id,initial,target};field.value=initial;panel.hidden=false;
    const tokens=[...new Set([...placeholders(initial),...Object.keys(commonCopyValues())])];
    const localValues=JSON.parse(target.dataset.copyValues||'{}');
    panel.querySelector('.text-editor-values').textContent=tokens.length?'Available values (reusable): '+tokens.map(t=>'{'+t+'} = '+renderTemplate('{'+t+'}',localValues)).join(' · '):'';
    message.textContent='⌘/Ctrl+Enter to apply · Esc to cancel';message.classList.remove('error');
    panel.querySelector('.text-editor-format').textContent='__underline__ · *italic* · **bold** · [label](https://…) · [code]'+(Object.hasOwn(target.dataset,'copyBlocks')?' · 1. item on each line for a numbered list':'')+(initial.includes('(note:')?' · [word](note:…) links to a note':'');
    document.querySelector('#node-tooltip').hidden=true;
    position(target);field.focus();updateStatus();
  }
  field.addEventListener('input',()=>{
    preview(field.value);resizeField();updateStatus();
    try{validateText(textCatalog.defaults[current.id],field.value);message.textContent='⌘/Ctrl+Enter to apply · Esc to cancel';message.classList.remove('error');}
    catch(error){message.textContent=error.message;message.classList.add('error');}
  });
  panel.addEventListener('keydown',event=>{
    if(event.key==='Escape'){event.preventDefault();event.stopPropagation();cancel();}
    if(event.key==='Enter'&&(event.ctrlKey||event.metaKey)){event.preventDefault();event.stopPropagation();commit();}
  });
  panel.querySelector('[data-editor-apply]').onclick=commit;
  panel.querySelector('[data-editor-cancel]').onclick=cancel;
  panel.querySelector('[data-editor-restore]').onclick=()=>{
    if(!current)return;field.value=textCatalog.defaults[current.id];preview(field.value);resizeField();updateStatus();
    message.textContent='Original wording restored. Apply to keep it.';message.classList.remove('error');field.focus();
  };
  toggle.onclick=()=>{
    if(isBusy())return;
    if(!commit())return;enabled=!enabled;document.body.classList.toggle('editing-text',enabled);onMode(enabled);refresh();updateStatus();
  };
  document.addEventListener('click',event=>{
    if(isBusy()||!enabled||panel.contains(event.target)||event.target===toggle||event.target===save)return;
    const target=event.target.closest('[data-copy-editable="true"]');
    if(target){event.preventDefault();event.stopImmediatePropagation();begin(target);}
    else if(current&&!commit()){event.preventDefault();event.stopImmediatePropagation();}
  },true);
  document.addEventListener('keydown',event=>{
    if(isBusy()||!enabled||panel.contains(event.target))return;
    const target=event.target.closest('[data-copy-editable="true"]');
    if(target&&(event.key==='Enter'||event.key===' ')){event.preventDefault();event.stopImmediatePropagation();begin(target);}
  },true);
  save.onclick=async()=>{
    if(!commit()||saving||!textCatalog.dirty)return;
    const edits={...textCatalog.edits};saving=true;updateStatus('Saving…');
    try{
      const response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({revision:savedRevision,edits})});
      const result=await response.json();if(!response.ok)throw new Error(result.error||'Could not save the text.');
      savedRevision=result.revision;textCatalog.savedAs(edits);saving=false;updateStatus(textCatalog.dirty?'Unsaved changes':'Saved to project');
    }catch(error){saving=false;updateStatus('Could not save. '+error.message);}
  };
  window.addEventListener('beforeunload',event=>{if(dirty()){event.preventDefault();event.returnValue='';}});
  window.addEventListener('resize',()=>{if(current)position(current.target);});
  window.addEventListener('scroll',()=>{if(current)position(current.target);},true);
  refreshCopies();updateStatus();
  return {refresh,get dirty(){return dirty();},get editing(){return !!current;},get enabled(){return enabled;},refreshCopies,reposition(){if(current)position(current.target);}};
}
