// Geometry shared by the diagram and its regression checks. World coordinates
// stay fixed while the selected block unfolds; only its neighbors slide aside.
export const BLOCK = Object.freeze({start:465,step:28,closedWidth:20,top:150,closedHeight:130,frameTop:-8,width:1240,height:540,streamY:235,streamLocalY:340});

export function blockLayout(index, count, progress, height=BLOCK.height, geometry=BLOCK) {
  const p=Math.max(0,Math.min(1,progress));
  const openScale=BLOCK.closedHeight/height;
  const openWidth=BLOCK.width*openScale;
  const width=geometry.closedWidth+(openWidth-geometry.closedWidth)*p;
  const growth=width-geometry.closedWidth;
  const positions=Array.from({length:count},(_,i)=>geometry.start+i*geometry.step+(i>index?growth/2:-growth/2));
  const scale=width/BLOCK.width;
  // Wrapped prose can make the open block taller. Align its neighbors to its
  // final frame while keeping the residual connection at the same height.
  const openTop=BLOCK.streamY+(BLOCK.frameTop-BLOCK.streamLocalY)*openScale;
  const neighborY=(openTop-BLOCK.top)*p;
  return {width,growth,positions,neighborY,left:-growth/2,right:growth/2,
    transform:{x:positions[index]+15*scale,y:BLOCK.streamY-BLOCK.streamLocalY*scale,s:scale}};
}

export function matrixProportions(rows, cols) {
  const ratio=rows/cols;
  if(rows===1)return {width:384,height:28,ratio,compressed:true,vector:true};
  const displayRatio=Math.max(.25,Math.min(4,ratio));
  return {width:ratio>=1?384/displayRatio:384,height:ratio>=1?384:384*displayRatio,
    ratio,compressed:displayRatio!==ratio,vector:false};
}

// Locate a contiguous patch in the full matrix, not in the sampled overview's
// 64 x 64 index space. Tiny extents get a visible marker, clamped to the canvas.
export function matrixWindow(shape, patch, width, height, minimum=6) {
  const axis=(total,indices,pixels)=>{
    const start=indices[0]/total*pixels,end=(indices.at(-1)+1)/total*pixels;
    const actual=end-start,size=Math.min(pixels,Math.max(minimum,actual));
    return {start:Math.max(0,Math.min(pixels-size,(start+end-size)/2)),size,enlarged:size>actual+1e-9};
  };
  const x=axis(shape[1],patch.cols,width),y=axis(shape[0],patch.rows,height);
  return {x:x.start,y:y.start,width:x.size,height:y.size,enlarged:x.enlarged||y.enlarged};
}

// Keep shared K/V groups together, with at most eight rows in each vertical stack.
export function attentionLayout(heads, kvHeads) {
  const per=heads/kvHeads, rowsPerBank=Math.max(per,Math.floor(8/per)*per);
  const bankCount=Math.ceil(heads/rowsPerBank), bankWidth=820;
  const rows=Array.from({length:heads},(_,head)=>{
    const bank=Math.floor(head/rowsPerBank),row=head%rowsPerBank;
    return {head,bank,group:Math.floor(head/per),x:bank*bankWidth+170,
      y:176+row*40+Math.floor(row/per)*8};
  });
  const bottom=Math.max(...rows.map(row=>row.y)),width=(bankCount-1)*bankWidth+660;
  return {rows,bankCount,bankWidth,per,bottom,width,sumX:width*.58,sumY:bottom+62,
    streamY:bottom+145,box:{x:-325,y:-15,w:width+370,h:bottom+220}};
}

// A single column, with disclosure groups that never split a shared K/V pair.
export const GQA=Object.freeze({inputX:-60,headX:8,headWidth:152,projectionX:280,projectionWidth:110,contributionX:468,contributionWidth:86,sumX:680,rowStep:40,pairGap:32,firstRow:20});
const headOffset=(offset,per)=>GQA.firstRow+offset*GQA.rowStep+Math.floor(offset/per)*GQA.pairGap;

export function foldableAttentionLayout(heads, kvHeads, activeGroup=0) {
  const per=heads/kvHeads,groupSize=per*Math.max(1,Math.floor(4/per));
  const active=Math.max(0,Math.min(Math.ceil(heads/groupSize)-1,activeGroup)),groups=[],rows=[];
  let top=164;
  for(let first=0,index=0;first<heads;first+=groupSize,index++) {
    const count=Math.min(groupSize,heads-first),open=index===active;
    const headerHeight=open?0:48,detailHeight=open?groupSize*GQA.rowStep+groupSize/per*GQA.pairGap:0,height=headerHeight+detailHeight;
    const group={index,first,last:first+count-1,count,open,expansion:open?1:0,top,height,headerHeight,detailHeight,mid:top+height/2,
      port:top+(open?GQA.firstRow:24),lastPort:top+(open?headOffset(count-1,per):24),
      firstKV:Math.floor(first/per),lastKV:Math.floor((first+count-1)/per)};
    group.pairs=Array.from({length:count/per},(_,i)=>{
      const pairTop=i*(per*GQA.rowStep+GQA.pairGap);
      return {kv:group.firstKV+i,first:first+i*per,last:first+(i+1)*per-1,top:pairTop,height:per*GQA.rowStep+24,labelY:pairTop+per*GQA.rowStep+14};
    });
    groups.push(group);
    for(let offset=0;offset<count;offset++)rows.push({head:first+offset,group:Math.floor((first+offset)/per),
      disclosure:index,visible:open,x:GQA.headX,localY:headOffset(offset,per),y:open?top+headOffset(offset,per):group.mid});
    top+=height+14;
  }
  const bottom=top-14,width=760,sumY=bottom+54,streamY=bottom+139;
  const connections=groups.flatMap(group=>[group.port,group.lastPort]);
  // K/V notes live inside the head boxes, leaving the middle input fixed.
  const inputY=(groups[0].top+bottom)/2;
  return {groups,rows,per,groupSize,width,bottom,sumX:GQA.sumX,sumY,streamY,
    inputY,connections,
    box:{x:-325,y:-15,w:1120,h:streamY+80}};
}

// Close the old group completely, pause with every packet folded, then open
// the new one from its settled position. The outer frame stays fixed.
export function attentionAccordionFrame(from,to,progress,stage='closing') {
  const p=Math.max(0,Math.min(1,progress));
  let top=from.groups[0].top;
  const groups=to.groups.map((target,index)=>{
    const start=from.groups[index];
    const expansion=stage==='closing'?start.expansion*(1-p):stage==='opening'?target.expansion*p:0;
    const headerHeight=48*(1-expansion),detailHeight=Math.max(start.detailHeight,target.detailHeight)*expansion,height=headerHeight+detailHeight;
    const port=top+Math.min(24,headerHeight+20);
    const lastDetail=headerHeight+Math.min(detailHeight,headOffset(target.count-1,to.per));
    const group={...target,top,height,expansion,headerHeight,detailHeight,mid:top+height/2,port,
      lastPort:top+24+(lastDetail-24)*Math.min(1,detailHeight/20)};
    top+=height+14;
    return group;
  });
  const rows=to.rows.map(row=>{
    const group=groups[row.disclosure];
    return {...row,y:group.top+group.headerHeight+row.localY,visible:group.expansion===1};
  });
  return {...to,groups,rows,connections:groups.flatMap(group=>[group.port,group.lastPort])};
}
