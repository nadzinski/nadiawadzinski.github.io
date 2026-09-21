// A deterministic teaching example, independent of model weights or activations.
export function illustrativeRouting(experts, active, position, block) {
  if(!Number.isInteger(experts)||!Number.isInteger(active)||active<1||active>experts)throw new Error('Invalid expert counts');
  const logits=Array.from({length:experts},(_,expert)=>{
    let seed=Math.imul(expert+1,0x45d9f3b)^Math.imul(position+1,0x27d4eb2d)^Math.imul(block+1,0x165667b1);
    seed=Math.imul(seed^(seed>>>16),0x45d9f3b);
    return ((seed^(seed>>>16))>>>0)/4294967296*8-4;
  });
  const highest=Math.max(...logits),exponentials=logits.map(value=>Math.exp(value-highest));
  const total=exponentials.reduce((a,b)=>a+b,0),probabilities=exponentials.map(value=>value/total);
  const selected=probabilities.map((probability,expert)=>({expert,probability})).sort((a,b)=>b.probability-a.probability||a.expert-b.expert).slice(0,active);
  const selectedMass=selected.reduce((sum,item)=>sum+item.probability,0);
  return {logits,probabilities,selected:selected.map(item=>({...item,weight:item.probability/selectedMass}))};
}
