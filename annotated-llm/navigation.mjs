// Native SVG text selection ends with a click too. Ignore that click without
// preventing the selection itself or blocking later clicks and keyboard actions.
export function installSelectionClickGuard(target,{getSelection=()=>target.ownerDocument.getSelection()}={}) {
  let pointer=null;
  const snapshot=()=>{
    const s=getSelection();
    return [s?.anchorNode,s?.anchorOffset,s?.focusNode,s?.focusOffset];
  };
  target.addEventListener('pointerdown',event=>{
    pointer=event.button===0?{x:event.clientX,y:event.clientY,selection:snapshot(),dragged:false}:null;
  });
  target.addEventListener('pointermove',event=>{
    if(pointer&&event.buttons&1&&Math.hypot(event.clientX-pointer.x,event.clientY-pointer.y)>4)pointer.dragged=true;
  });
  target.addEventListener('pointercancel',()=>{pointer=null;});
  return event=>{
    if(event.detail===0)return false; // Enter, Space, or assistive activation.
    const start=pointer;pointer=null;
    if(!start)return false;
    if(start.dragged||Math.hypot(event.clientX-start.x,event.clientY-start.y)>4)return true;
    const s=getSelection();
    return !!s&&!s.isCollapsed&&(target.contains(s.anchorNode)||target.contains(s.focusNode))
      &&snapshot().some((value,i)=>value!==start.selection[i]);
  };
}

// Pinch momentum belongs to the gesture that started it, even when the
// navigation animation has finished (or reduced motion skips the animation).
export function installZoomOutGestures(target,{canGoUp,isBusy,goUp,enabled=()=>true,now=()=>performance.now()}) {
  const quietTime=300;
  let lastWheel=-Infinity,distance=0,wheelUsed=false;
  let pinching=false,pinchUsed=false,initialScale=1;

  function wheel(event) {
    if(!enabled()){distance=0;wheelUsed=false;return;}
    // Ordinary wheel and two-finger scrolling belong to the page, even during
    // navigation or immediately after a pinch. Only zoom gestures go up a level.
    if(!event.ctrlKey&&!event.metaKey)return;
    if(event.shiftKey||!event.deltaY||Math.abs(event.deltaX)>Math.abs(event.deltaY))return;
    const time=now();
    if(time-lastWheel>quietTime){distance=0;wheelUsed=isBusy();}
    lastWheel=time;
    if(pinching||wheelUsed){event.preventDefault();return;}
    if(isBusy()){event.preventDefault();wheelUsed=true;return;}
    if(!canGoUp()){
      // The overview is the outermost diagram view; pinching cannot go further.
      event.preventDefault();
      return;
    }
    event.preventDefault();
    if(event.deltaY<0){distance=0;return;}
    const unit=event.deltaMode===1?16:event.deltaMode===2?target.clientHeight:1;
    distance+=event.deltaY*unit;
    if(distance>=12){
      wheelUsed=true;
      goUp();
    }
  }

  // Safari reports trackpad pinches as GestureEvents. Chromium and Firefox
  // report them as Ctrl+wheel, handled above. Share the latch to avoid repeats.
  function gestureStart(event) {
    if(!enabled()){pinching=false;return;}
    event.preventDefault();
    pinching=true;pinchUsed=wheelUsed&&now()-lastWheel<=quietTime||isBusy()||!canGoUp();
    initialScale=event.scale||1;
    wheelUsed=true;lastWheel=now();
  }
  function gestureChange(event) {
    if(!enabled()||!pinching)return;
    event.preventDefault();
    if(isBusy())pinchUsed=true;
    if(!pinchUsed&&canGoUp()&&event.scale/initialScale<=.9){
      pinchUsed=true;
      goUp();
    }
  }
  function gestureEnd(event) {
    if(!enabled()||!pinching)return;
    event.preventDefault();pinching=false;wheelUsed=true;lastWheel=now();
  }

  const handlers={wheel,gesturestart:gestureStart,gesturechange:gestureChange,gestureend:gestureEnd};
  for(const [type,handler]of Object.entries(handlers))target.addEventListener(type,handler,{passive:false});
  return ()=>{for(const [type,handler]of Object.entries(handlers))target.removeEventListener(type,handler);};
}
