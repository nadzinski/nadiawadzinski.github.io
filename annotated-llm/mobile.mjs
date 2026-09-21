// Mobile presentation uses the same SVG layers, source, and authored prose.
export const MOBILE_QUERY = '(max-width: 1000px), (max-height: 550px) and (pointer: coarse)';
export function mobileCamera(box, transformScale, width, height, zoom=1) {
  const padding=14, available=Math.max(80,height-padding*2);
  const fitted=Math.min(.8/transformScale,available/box.h);
  const minimum=Math.min(1,Math.max(1,width-padding*2)/(box.w*fitted));
  zoom=Math.max(minimum,Math.min(1,zoom));
  const k=fitted*zoom;
  const drawingWidth=Math.max(width,Math.ceil(box.w*k+padding*2));
  return {width:drawingWidth,k,x:(drawingWidth-box.w*k)/2-box.x*k,y:(height-box.h*k)/2-box.y*k,zoom};
}

// One finger belongs to native scrolling; two fingers resize the current drawing.
export function installDiagramPinch(target,{enabled,begin,update,now=()=>performance.now()}) {
  let pinch=null,suppressClickUntil=0;
  const points=event=>{
    const [a,b]=event.touches;
    return {distance:Math.hypot(a.clientX-b.clientX,a.clientY-b.clientY),x:(a.clientX+b.clientX)/2,y:(a.clientY+b.clientY)/2};
  };
  function start(event){
    if(!enabled()||event.touches.length!==2)return;
    event.preventDefault();const p=points(event);
    pinch={distance:Math.max(1,p.distance),state:begin(p)};
  }
  function move(event){
    if(!enabled()||!pinch||event.touches.length!==2)return;
    event.preventDefault();const p=points(event);update(pinch.state,p.distance/pinch.distance,p);
  }
  function finish(event){if(pinch&&event.touches.length!==2){pinch=null;suppressClickUntil=now()+400;}}
  function click(event){if(enabled()&&(pinch||now()<suppressClickUntil)){event.preventDefault();event.stopImmediatePropagation();}}
  function gesture(event){if(enabled())event.preventDefault();}
  const handlers={touchstart:start,touchmove:move,touchend:finish,touchcancel:finish,gesturestart:gesture,gesturechange:gesture,gestureend:gesture};
  for(const [name,handler]of Object.entries(handlers))target.addEventListener(name,handler,{passive:false});
  target.addEventListener('click',click,true);
  return ()=>{for(const [name,handler]of Object.entries(handlers))target.removeEventListener(name,handler);target.removeEventListener('click',click,true);};
}

export function createMobileExplorer({canvas,world,getState,overviewVisibleThrough,go,up,openCode,openWeight,refit,onReadoutClose,onMode}) {
  const $=s=>document.querySelector(s), media=matchMedia(MOBILE_QUERY),desktopLabel=canvas.getAttribute('aria-label');
  const viewport=$('.diagram-surface'),wrap=$('.canvas-wrap'),work=$('.work-area'),inspector=$('.inspector');
  const stage=document.createElement('section');stage.id='mobile-stage';stage.setAttribute('aria-label','Model diagram');
  work.insertBefore(stage,wrap);stage.append(wrap);
  const toolbar=document.createElement('div');toolbar.className='mobile-toolbar mobile-only';
  toolbar.innerHTML=`<button data-mobile-back aria-label="Back to parent component"><span aria-hidden="true">←</span><span class="mobile-back-text"> Back</span></button><div class="mobile-location"><small></small><strong></strong></div><button data-mobile-path aria-expanded="false" aria-controls="mobile-path">Navigate ▾</button><button class="mobile-code" data-mobile-code aria-label="See Python">〈/〉</button><button data-mobile-tools aria-label="Introduction, model, legend, and inspection" aria-expanded="false" aria-controls="mobile-tools">⋯</button><div id="mobile-path" class="mobile-menu" hidden><nav aria-label="Model navigation"></nav><label><span data-mobile-block-label>Transformer block</span><select id="mobile-block" aria-label="Choose a transformer block"></select></label></div><div id="mobile-tools" class="mobile-menu" hidden><strong>The annotated LLM</strong><div data-mobile-model></div><button data-mobile-intro>Introduction</button><button data-mobile-explanation>Explanation ↓</button><button data-mobile-weight hidden>⌕ Inspect weights</button><div data-mobile-facts></div><div data-mobile-legend></div></div>`;
  // Let navigation stick throughout the diagram, explanation, code, and context.
  work.before(toolbar);
  const overview=document.createElement('section');overview.className='mobile-overview mobile-only';
  overview.innerHTML='<div class="mobile-overview-label"><span>Full diagram</span><button class="mobile-intro-shortcut" aria-label="Read the introduction">Introduction</button></div><div id="mobile-map" tabindex="0" role="group" aria-label="Full diagram. Tap or drag to move the visible area; arrow keys also move it."></div>';
  wrap.before(overview);
  const readout=document.createElement('div');readout.id='mobile-readout';readout.className='mobile-only';readout.hidden=true;
  readout.innerHTML='<div role="status" aria-live="polite"><strong></strong><code></code><p></p></div><button aria-label="Close tensor explanation">×</button>';stage.after(readout);
  const jump=document.createElement('button');jump.className='mobile-jump mobile-only';jump.textContent='Go to visualization ↓';$('.intro h1').after(jump);
  const introEnd=document.createElement('button');introEnd.className='mobile-jump mobile-intro-end mobile-only';introEnd.textContent='Go to visualization ↓';$('.intro').append(introEnd);
  const back=document.createElement('button');back.className='mobile-return mobile-only';back.textContent='↑ Back to diagram';inspector.append(back);
  const intro=$('.intro-copy'),introNote=document.createElement('p');introNote.className='mobile-only mobile-intro-note';
  introNote.innerHTML="I've made every effort to make this usable on mobile, but it's still <strong>best viewed on a larger screen</strong>.";
  function placeIntroNote(){
    const paragraphs=[...intro.children].filter(node=>node!==introNote&&node.tagName==='P');
    const anchor=paragraphs.find(node=>/\bTo start\b/i.test(node.textContent))||paragraphs.at(-1);
    if(anchor&&anchor.nextElementSibling!==introNote)anchor.after(introNote);
  }
  placeIntroNote();new MutationObserver(placeIntroNote).observe(intro,{childList:true,subtree:true,characterData:true});

  const homes=new Map();
  for(const element of [$('.model-picker'),$('.model-summary'),$('.legend'),$('#code-drawer')]){
    const home=document.createComment('desktop position');element.before(home);homes.set(element,home);
  }
  const map=$('#mobile-map'),ns='http://www.w3.org/2000/svg';
  let active=false,currentKey=null,mapDirty=true,box=null,camera=null,miniature=null,marker=null,drag=null,selected=null,transitioning=false;
  const positions=new Map(),zooms=new Map();
  const key=()=>{const s=getState();return [s.level,...(['model','embedding','output'].includes(s.level)?[]:[s.block]),...(['head','score'].includes(s.level)?[s.head]:[]),...(s.level==='expert'?[s.expert]:[]),...(s.level==='depth'?[s.depthBranch]:[])].join(':');};
  function savePosition(){if(active&&currentKey&&!transitioning)positions.set(currentKey,viewport.scrollLeft);}
  function closeMenus(){for(const id of ['path','tools']){$('#mobile-'+id).hidden=true;toolbar.querySelector('[data-mobile-'+id+']').setAttribute('aria-expanded','false');}}
  const scrollBehavior=()=>matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth';
  function returnToDiagram(){
    closeMenus();
    const inset=parseFloat(getComputedStyle(stage).scrollMarginTop)||0;
    window.scrollTo({top:window.scrollY+stage.getBoundingClientRect().top-inset,behavior:scrollBehavior()});
  }
  function showIntroduction(){closeMenus();window.scrollTo({top:0,behavior:scrollBehavior()});}
  function clearReadout(){if(readout.hidden)return;readout.hidden=true;onReadoutClose();}
  function setMode(){
    savePosition();active=media.matches;document.body.classList.toggle('mobile-layout',active);closeMenus();clearReadout();
    if(active){
      $('[data-mobile-model]').append($('.model-picker'));
      $('[data-mobile-facts]').append($('.model-summary'));
      $('[data-mobile-legend]').append($('.legend'));
      readout.after($('#code-drawer'));
      canvas.setAttribute('aria-label','Model diagram. Tap a component to look inside. Use Back or Navigate to return.');
    }else{
      for(const [element,home]of homes)home.after(element);
      if(desktopLabel===null)canvas.removeAttribute('aria-label');else canvas.setAttribute('aria-label',desktopLabel);
      canvas.style.removeProperty('width');viewport.scrollLeft=0;
    }
    currentKey=null;mapDirty=true;onMode?.(active);refit();
  }
  function updateChrome(names,levels){
    const state=getState();
    const blockLabel=$('#model-select').value==='mamba-2-130m-no-cache'?'Mamba block':'Transformer block';
    names={...names,block:`${blockLabel} ${state.block}`};
    toolbar.querySelector('.mobile-location').dataset.level=state.level;
    toolbar.querySelector('[data-mobile-block-label]').textContent=blockLabel;
    $('#mobile-block').setAttribute('aria-label',`Choose a ${blockLabel.toLowerCase()}`);
    toolbar.querySelector('.mobile-location small').textContent=$('#model-select').selectedOptions[0]?.textContent||'';
    toolbar.querySelector('.mobile-location strong').textContent=names[state.level];
    toolbar.querySelector('.mobile-location strong').title=names[state.level];
    toolbar.querySelector('[data-mobile-back]').disabled=levels.length===1;
    const path=$('#mobile-path nav');path.replaceChildren();
    for(const level of levels){const button=document.createElement('button');button.textContent=names[level];button.dataset.level=level;if(level===state.level)button.setAttribute('aria-current','page');path.append(button);}
    const picker=$('#mobile-block');picker.replaceChildren();
    for(const source of document.querySelectorAll('#minimap button')){const option=document.createElement('option');option.value=source.dataset.block;option.textContent=source.title.replace(/^Block\b/,blockLabel);option.selected=Number(option.value)===state.block;picker.append(option);}
    const blockTypes=new Set([...picker.options].map(option=>option.textContent.replace(/^.*?block \d+/,'')));
    picker.closest('label').hidden=blockTypes.size<2;
  }
  function selectionChanged(spec){selected=spec;const weight=toolbar.querySelector('[data-mobile-weight]');weight.hidden=!spec?.weight||getState().model?.weight_inspection===false;}
  function cameraFor(bounds,transform){
    const width=viewport.clientWidth,height=viewport.clientHeight;
    let zoom=zooms.get(key());
    if(zoom===undefined&&getState().level==='model'&&overviewVisibleThrough){
      const fitted=mobileCamera(bounds,transform.s,width,height);
      // Start with context and the first five blocks visible, including the invitation.
      zoom=(width-28)/((overviewVisibleThrough()-bounds.x)*fitted.k);
    }
    return mobileCamera(bounds,transform.s,width,height,zoom??1);
  }
  function fitCamera(bounds,transform){
    const next=key();
    box=bounds;camera=cameraFor(bounds,transform);
    // Only retain explicit pinch choices; the default adapts to the screen width.
    if(zooms.has(next))zooms.set(next,camera.zoom);
    canvas.style.width=camera.width+'px';
    if(next!==currentKey){currentKey=next;viewport.scrollLeft=positions.get(next)||0;}
    return {x:camera.x,y:camera.y,k:camera.k};
  }
  function refreshMap(layer){
    if(!active||transitioning||!box||!camera)return;
    if(mapDirty||!miniature){
      mapDirty=false;miniature=document.createElementNS(ns,'svg');miniature.setAttribute('aria-hidden','true');
      miniature.append(canvas.querySelector('defs').cloneNode(true));const clone=layer.cloneNode(true);clone.style.opacity=1;miniature.append(clone);
      const ids=new Map([...miniature.querySelectorAll('[id]')].map(node=>[node.id,'mobile-map-'+node.id]));
      for(const node of [miniature,...miniature.querySelectorAll('*')]){
        node.removeAttribute('tabindex');node.removeAttribute('role');
        for(const attribute of [...node.attributes]){
          if(attribute.name.startsWith('data-')||attribute.name==='aria-describedby'){node.removeAttribute(attribute.name);continue;}
          let value=attribute.value.replace(/url\(#([^)]+)\)/g,(match,id)=>ids.has(id)?`url(#${ids.get(id)})`:match);
          if(attribute.name==='id')value=ids.get(value);
          if(['href','xlink:href'].includes(attribute.name)&&value.startsWith('#')&&ids.has(value.slice(1)))value='#'+ids.get(value.slice(1));
          node.setAttribute(attribute.name,value);
        }
      }
      marker=document.createElementNS(ns,'rect');marker.setAttribute('class','mobile-map-window');marker.setAttribute('vector-effect','non-scaling-stroke');miniature.append(marker);map.replaceChildren(miniature);
    }
    miniature.setAttribute('viewBox',`${box.x} ${box.y} ${box.w} ${box.h}`);updateMarker();
  }
  function updateMarker(){
    if(!active||transitioning||!marker||!camera)return;
    const left=Math.max(box.x,(viewport.scrollLeft-camera.x)/camera.k),right=Math.min(box.x+box.w,(viewport.scrollLeft+viewport.clientWidth-camera.x)/camera.k);
    for(const [name,value]of Object.entries({x:left,y:box.y,width:Math.max(0,right-left),height:box.h}))marker.setAttribute(name,value);
  }
  function point(event){return new DOMPoint(event.clientX,event.clientY).matrixTransform(miniature.getScreenCTM().inverse());}
  function moveMap(event){if(!drag||!camera)return;const p=point(event);viewport.scrollLeft=(p.x-drag.offset)*camera.k+camera.x;updateMarker();}
  map.addEventListener('pointerdown',event=>{
    if(transitioning||!miniature||event.button!==0||!event.isPrimary)return;
    const p=point(event),left=Number(marker.getAttribute('x')),width=Number(marker.getAttribute('width'));
    drag={offset:p.x>=left&&p.x<=left+width?p.x-left:width/2};map.setPointerCapture(event.pointerId);moveMap(event);
  });
  map.addEventListener('pointermove',moveMap);for(const type of ['pointerup','pointercancel','lostpointercapture'])map.addEventListener(type,()=>{drag=null;});
  map.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();viewport.scrollLeft=event.key==='Home'?0:event.key==='End'?viewport.scrollWidth:viewport.scrollLeft+(event.key==='ArrowLeft'?-100:100);}});
  viewport.addEventListener('scroll',()=>{savePosition();updateMarker();},{passive:true});
  installDiagramPinch(viewport,{enabled:()=>active&&!!camera,
    begin(point){
      const x=point.x-viewport.getBoundingClientRect().left;
      return {key:key(),zoom:camera.zoom,worldX:(viewport.scrollLeft+x-camera.x)/camera.k};
    },
    update(start,ratio,point){
      if(start.key!==key())return;
      zooms.set(start.key,start.zoom*ratio);refit();
      const x=point.x-viewport.getBoundingClientRect().left;
      viewport.scrollLeft=start.worldX*camera.k+camera.x-x;savePosition();updateMarker();
    },
  });
  toolbar.addEventListener('click',event=>{
    const button=event.target.closest('button');if(!button)return;
    if(button.matches('[data-mobile-back]'))up();
    if(button.matches('[data-mobile-code]'))openCode();
    if(button.matches('[data-mobile-path],[data-mobile-tools]')){const id=button.hasAttribute('data-mobile-path')?'path':'tools',open=$('#mobile-'+id).hidden;closeMenus();$('#mobile-'+id).hidden=!open;button.setAttribute('aria-expanded',String(open));}
    if(button.dataset.level){closeMenus();go(button.dataset.level);}
    if(button.matches('[data-mobile-intro]'))showIntroduction();
    if(button.matches('[data-mobile-explanation]')){closeMenus();inspector.scrollIntoView({block:'start',behavior:scrollBehavior()});}
    if(button.matches('[data-mobile-weight]')){closeMenus();openWeight(selected);}
  });
  $('#mobile-block').addEventListener('change',event=>{closeMenus();go('block',{block:Number(event.target.value)});});
  document.addEventListener('pointerdown',event=>{if(active&&!toolbar.contains(event.target))closeMenus();});
  document.addEventListener('keydown',event=>{if(!active||event.key!=='Escape')return;if(!readout.hidden){event.preventDefault();event.stopImmediatePropagation();clearReadout();}else if(!$('#mobile-path').hidden||!$('#mobile-tools').hidden){event.preventDefault();event.stopImmediatePropagation();closeMenus();}},true);
  readout.querySelector('button').onclick=clearReadout;jump.onclick=introEnd.onclick=back.onclick=returnToDiagram;
  overview.querySelector('.mobile-intro-shortcut').onclick=showIntroduction;
  new ResizeObserver(()=>{if(active)refit();}).observe(viewport);
  media.addEventListener('change',setMode);setMode();
  return {get active(){return active;},get hasReadout(){return !readout.hidden;},fitCamera,refreshMap,updateChrome,selectionChanged,clearReadout,returnToDiagram,
    beginTransition(){
      const start={...getState().camera};start.x-=viewport.scrollLeft;
      transitioning=true;canvas.style.width=viewport.clientWidth+'px';viewport.scrollLeft=0;
      return start;
    },
    transitionCamera(bounds,transform){
      const frame=cameraFor(bounds,transform);
      const pan=Math.max(0,Math.min(positions.get(key())||0,frame.width-viewport.clientWidth));
      return {x:frame.x-pan,y:frame.y,k:frame.k};
    },
    endTransition(){transitioning=false;currentKey=null;mapDirty=true;},
    invalidateMap(){mapDirty=true;},prepareNavigation(){savePosition();clearReadout();closeMenus();currentKey=null;},
    showReadout(target){
      const lines=(target.dataset.tooltip||'').split('\n');readout.querySelector('strong').textContent=target.dataset.flowName||lines.shift();
      readout.querySelector('code').textContent=target.dataset.flowShape||'';readout.querySelector('p').textContent=target.dataset.flowMeaning||lines.join('\n');readout.hidden=false;
    },
  };
}
