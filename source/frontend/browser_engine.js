/* PharmaOps browser-native execution engine: zero dependencies, file:// compatible.
   Independently implemented discrete-event simulator, search-based scheduler, and random forests.
   This is NOT the Python SimPy library, NOT Google OR-Tools CP-SAT, and NOT scikit-learn.
   Synthetic educational model only; no production or GxP use. */
(function(root, factory){
  const api=factory();
  if(typeof module==='object' && module.exports) module.exports=api;
  root.PharmaOpsBrowser=api;
})(typeof globalThis!=='undefined'?globalThis:this, function(){'use strict';
const STAGES=['dispensing','granulation','blending','compression','coating','packaging','qc'];
const LABELS={dispensing:'Dispensing',granulation:'Wet granulation',blending:'Final blending',compression:'Tablet compression',coating:'Film coating',packaging:'Blister packaging',qc:'QC release'};
const CAP={dispensing:1,granulation:2,blending:1,compression:2,coating:1,packaging:2,qc:1};
const BASE={dispensing:1.35,granulation:4.4,blending:1.6,compression:5.1,coating:4.3,packaging:3.2,qc:2.6};
const POWER={dispensing:4,granulation:27,blending:9,compression:23,coating:32,packaging:18,qc:6};
const NODE_POS={warehouse:[80,175],dispensing:[240,100],granulation:[425,100],blending:[605,100],compression:[790,100],coating:[790,310],packaging:[605,310],qc:[425,310],dispatch:[240,310],NW:[235,195],NC:[490,195],NE:[780,195],SW:[235,245],SC:[490,245],SE:[780,245]};
const LINKS=[['warehouse','NW',15],['dispensing','NW',8],['granulation','NC',10],['blending','NC',8],['compression','NE',8],['coating','SE',8],['packaging','SC',9],['qc','SC',8],['dispatch','SW',8],['NW','NC',22],['NC','NE',22],['NE','SE',24],['SE','SC',22],['SC','SW',22],['SW','NW',27],['NC','SC',29]];
const SCENARIOS={normal:['Normal operations','Standard demand and equipment conditions'],demand:['Demand surge','Tighter order deadlines'],dryer:['Granulator outage','Reduced granulation machine availability'],quality:['QC hold','Additional testing time'],supplier:['Material supply delay','Delayed material release'],energy:['Energy price spike','Higher electricity price']};
const PRODUCTS=[['Aster 10 mg','AS-010',235,92,.94],['Navira 20 mg','NV-020',275,113,1.08],['Serava XR 50 mg','SR-050',320,147,1.24],['Meridian 5 mg','MD-005',255,103,1.0]];
const FEATURE_NAMES=['size_kg','product_complexity','equipment_health','ambient_humidity','previous_line_load','operator_skill'];
const round=(v,n=3)=>Number(Number(v).toFixed(n));
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function rng(seed){let a=(seed>>>0)||1;return function(){a=(a+0x6D2B79F5)>>>0;let t=Math.imul(a^(a>>>15),1|a);t^=t+Math.imul(t^(t>>>7),61|t);return ((t^(t>>>14))>>>0)/4294967296;};}
function normal(rand){let u=Math.max(1e-9,rand()),v=rand();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);}
function shuffle(a,r){for(let i=a.length-1;i>0;i--){let j=Math.floor(r()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function sampleUnique(n,k,r){const s=new Set();while(s.size<Math.min(n,k))s.add(Math.floor(r()*n));return [...s];}
function featureRow(b){return [b.kilograms,b.complexity,b.equipment_health,b.humidity,b.prior_load,b.operator_skill];}
function sigmoid(x){return 1/(1+Math.exp(-x));}
function syntheticData(){const rand=rng(20261118),X=[],Y=[],D=[];
  for(let i=0;i<1750;i++){
    const x=[210+220*rand(),.9+.45*rand(),.54+.46*rand(),38+37*rand(),.55+1.05*rand(),.5+.5*rand()];
    const [size,complexity,health,humidity,load,skill]=x;
    const y=Math.max(.65,.59+.00058*size+.30*(complexity-.85)+.42*(1-health)+.0034*Math.max(0,humidity-49)+.15*Math.max(0,load-.85)-.14*(skill-.5)+.10*Number(health<.7&&load>1.13)+.035*normal(rand));
    const prob=sigmoid(-3.6+7.5*(1-health)+2.3*Math.max(0,load-1)+.038*Math.max(0,humidity-55)+1.2*(complexity-1));
    X.push(x);Y.push(y);D.push(Number(rand()<prob));
  }return {X,Y,D};
}
function trainForest(X,y,{seed=1,trees=24,depth=7,minLeaf=9,kind='reg'}={}){
 const rand=rng(seed),n=X.length,importance=[0,0,0,0,0,0];
 function impurity(arr){let sum=0,sq=0;for(let v of arr){sum+=v;sq+=v*v;}const mean=sum/arr.length;return kind==='reg'?(sq/arr.length-mean*mean):mean*(1-mean);}
 function branch(ids,d){let sum=0;for(const id of ids)sum+=y[id];const pred=sum/ids.length;
   if(d>=depth||ids.length<2*minLeaf||impurity(ids.map(id=>y[id]))<1e-7)return {p:pred};
   let best=null;
   for(const f of sampleUnique(6,3,rand)){
     const candidates=sampleUnique(ids.length,12,rand);
     for(const candidate of candidates){const threshold=X[ids[candidate]][f];let left=[],right=[];
       for(const id of ids)(X[id][f]<=threshold?left:right).push(id);
       if(left.length<minLeaf||right.length<minLeaf)continue;
       const parent=impurity(ids.map(id=>y[id]));
       const gain=parent-(left.length*impurity(left.map(id=>y[id]))+right.length*impurity(right.map(id=>y[id])))/ids.length;
       if(gain>(best?.gain||1e-9))best={f,threshold,gain,left,right};
     }
   }
   if(!best)return {p:pred};
   importance[best.f]+=best.gain*ids.length;
   return {f:best.f,t:best.threshold,l:branch(best.left,d+1),r:branch(best.right,d+1)};
 }
 const forest=[];for(let t=0;t<trees;t++){const boot=Array.from({length:n},()=>Math.floor(rand()*n));forest.push(branch(boot,0));}
 function predict(x){let sum=0;for(const tree of forest){let node=tree;while(node.f!==undefined)node=x[node.f]<=node.t?node.l:node.r;sum+=node.p;}return sum/trees;}
 const total=importance.reduce((a,b)=>a+b,0)||1;return {predict,importance:importance.map(x=>x/total)};
}
let fitted=null;
function auc(labels,scores){const pairs=labels.map((y,i)=>[scores[i],y]).sort((a,b)=>a[0]-b[0]);let nPos=labels.reduce((s,x)=>s+x,0),nNeg=labels.length-nPos;if(!nPos||!nNeg)return 0.5;
let ranks=0;for(let i=0;i<pairs.length;){let j=i+1;while(j<pairs.length&&pairs[j][0]===pairs[i][0])j++;for(let k=i;k<j;k++)if(pairs[k][1])ranks+=(i+j+1)/2;i=j;}return (ranks-nPos*(nPos+1)/2)/(nPos*nNeg);}
function trainModels(){if(fitted)return fitted;const {X,Y,D}=syntheticData(),inds=shuffle(Array.from({length:X.length},(_,i)=>i),rng(917));const split=Math.floor(inds.length*.78),train=inds.slice(0,split),test=inds.slice(split);
const trainX=train.map(i=>X[i]), reg=train.map(i=>Y[i]), cl=train.map(i=>D[i]);
const regModel=trainForest(trainX,reg,{seed:17,trees:27,depth:8,minLeaf:10,kind:'reg'});
const clsModel=trainForest(trainX,cl,{seed:29,trees:29,depth:7,minLeaf:13,kind:'clf'});
let ae=0,se=0,mean=test.reduce((s,i)=>s+Y[i],0)/test.length,tot=0,probs=[],ys=[];
for(const i of test){const p=regModel.predict(X[i]),e=p-Y[i];ae+=Math.abs(e);se+=e*e;tot+=(Y[i]-mean)**2;probs.push(clamp(clsModel.predict(X[i]),.001,.999));ys.push(D[i]);}
const imp={};FEATURE_NAMES.forEach((name,i)=>imp[name]=round(regModel.importance[i]));
fitted={regModel,clsModel,metrics:{training_records:train.length,test_records:test.length,duration_mae:round(ae/test.length,4),duration_r2:round(1-se/tot),downtime_auc:round(auc(ys,probs)),data_type:'SYNTHETIC',features:FEATURE_NAMES,feature_importance:imp,estimator:'Browser-native Random Forest regressor + classifier (trained on load)'}};
return fitted;
}
function buildBatches(scenario,demand){const indices=[0,1,2,3,0,2,1,3,0,1,2,3,0,1,2,3,0,2,1,3,0,1,2,3];
return indices.map((pi,i)=>{const [product,sku,price,materials,complexity]=PRODUCTS[pi],kg=Math.trunc((260+i%4*32+pi*18)*demand);let due=[58,93,44,121,76,38,104,88,59,98,72,50][i%12]+Math.floor(i/12)*83;
if(scenario==='demand')due-=18+i%4*2;
return {id:'BT26-'+(5101+i),product,sku,kilograms:kg,due_hour:Math.max(18,due),priority:1+i%4,price_per_kg:price,material_cost_per_kg:materials,complexity,equipment_health:round(.96-i%5*.071),humidity:42+(i*7)%28,prior_load:round(.7+i%5*.19),operator_skill:round(.62+i%4*.105)};
});}
function predictBatches(batches){const model=trainModels();const predictions={};for(const b of batches){let x=featureRow(b);const m=round(clamp(model.regModel.predict(x),.65,1.6)),p=round(clamp(model.clsModel.predict(x),0,1));predictions[b.id]={duration_multiplier:m,downtime_probability:p,risk_level:p>=.55?'high':p>=.30?'medium':'low'};}return predictions;}
function release(scenario,i){return scenario==='supplier'&&[1,4,7].includes(i)?17.5:0;}
function cap(stage,scenario){return scenario==='dryer'&&stage==='granulation'?1:CAP[stage];}
function stageDurations(batches,pred,scenario){let o={};batches.forEach((b,i)=>{
 const scale=(b.kilograms/300)**.48*b.complexity*pred[b.id].duration_multiplier;
 for(const st of STAGES){let mult=1;if(scenario==='dryer'&&st==='granulation'&&i%3===1)mult=1.58;if(scenario==='quality'&&st==='qc'&&i%3===0)mult=2.2;
 o[b.id+'|'+st]=Math.max(.35,BASE[st]*scale*mult+.35);
 }});return o;}
const GRAPH={};for(const node of Object.keys(NODE_POS))GRAPH[node]=[];for(const [a,b,d] of LINKS){GRAPH[a].push([b,d]);GRAPH[b].push([a,d]);}
function traffic(a,b,scenario){const key=[a,b].sort().join('|');if(key==='NC|NW')return ['demand','supplier'].includes(scenario)?5:1.2;if(key==='NE|SE')return ['demand','dryer'].includes(scenario)?4.5:1;if(key==='NC|SC')return scenario==='quality'?1.4:1;return 1;}
const routeCache=new Map();
function route(src,dst,scenario,policy){const key=[src,dst,scenario,policy].join('|');if(routeCache.has(key))return routeCache.get(key);
 const heap=[[0,src,[src]]],best={[src]:0};while(heap.length){heap.sort((a,b)=>a[0]-b[0]);const [score,node,path]=heap.shift();if(score>best[node]+1e-9)continue;
 if(node===dst){let distance=0,weighted=0;for(let i=1;i<path.length;i++){const u=path[i-1],v=path[i];let d=GRAPH[u].find(x=>x[0]===v)[1];distance+=d;weighted+=d*traffic(u,v,scenario);}let out={path,distance,travel_hours:round(.11+.01*weighted,4)};routeCache.set(key,out);return out;}
 for(const [next,dist] of GRAPH[node]){let cost=score+dist*(policy==='adaptive'?traffic(node,next,scenario):1);if(cost<(best[next]??Infinity)-1e-9){best[next]=cost;heap.push([cost,next,[...path,next]]);}}
 }throw Error('No path in factory corridors');}
class MinHeap{constructor(){this.a=[];} push(v){let a=this.a,i=a.length;a.push(v);while(i>0){let p=(i-1)>>1;if(a[p].t<v.t||a[p].t===v.t&&a[p].seq<=v.seq)break;a[i]=a[p];i=p;}a[i]=v;}pop(){let a=this.a,out=a[0],x=a.pop();if(a.length){let i=0;while(2*i+1<a.length){let c=2*i+1;if(c+1<a.length&&(a[c+1].t<a[c].t||(a[c+1].t===a[c].t&&a[c+1].seq<a[c].seq)))c++;if(a[c].t>x.t||a[c].t===x.t&&a[c].seq>=x.seq)break;a[i]=a[c];i=c;}a[i]=x;}return out;}get length(){return this.a.length;}}
function simulate(batches,pred,scenario,order,seed,energyRate,policy='distance',durationCache=null){
 const duration=durationCache||stageDurations(batches,pred,scenario),available=Object.fromEntries(STAGES.map(s=>[s,Array(cap(s,scenario)).fill(0)])),tugger=[0,0];
 const eventQueue=new MinHeap(),events=[],transports=[],completed={},position=Object.fromEntries(order.map((i,pos)=>[i,pos]));let seq=0;
 const push=(t,i,k,kind)=>eventQueue.push({t,seq:seq++,i,k,kind});
 batches.forEach((b,i)=>push(Math.max(release(scenario,i),(position[i]??i)*.16),i,0,'stage'));
 const randoms=Array.from({length:batches.length},(_,i)=>STAGES.map((s,k)=>{let rr=rng((seed*1000003+i*9176+k*2791+307)>>>0);return clamp(1+normal(rr)*.10,.76,1.37);}));
 while(eventQueue.length){const {t,i,k,kind}=eventQueue.pop(),b=batches[i],stage=STAGES[k];
  if(kind==='move'){
   const src=STAGES[k-1],decision=policy==='ppo'?globalThis.PharmaOpsAdvanced.ppoAction(src,stage,scenario,t,tugger,b.due_hour):null,
     r=route(src,stage,scenario,decision?.route||policy),veh=decision?decision.vehicle:(tugger[0]<=tugger[1]?0:1),st=Math.max(t,tugger[veh]),end=st+r.travel_hours;
   transports.push({batch_id:b.id,sku:b.sku,source:src,destination:stage,start:round(st),end:round(end),queue_hours:round(st-t),vehicle:veh,path:r.path,distance:r.distance});
   tugger[veh]=end;push(end,i,k,'stage');continue;
  }
  let slots=available[stage],machine=0;for(let j=1;j<slots.length;j++)if(slots[j]<slots[machine])machine=j;
  const start=Math.max(t,slots[machine]);const dur=Math.max(.2,duration[b.id+'|'+stage]*randoms[i][k]);const end=start+dur;
  events.push({batch_id:b.id,sku:b.sku,product:b.product,stage,start:round(start),end:round(end),queue_hours:round(start-t),machine,duration:round(dur)});
  slots[machine]=end;
  if(k+1<STAGES.length)push(end,i,k+1,'move');else completed[b.id]=round(end);
 }
 return summarize(batches,scenario,events,transports,completed,energyRate);
}
const HORIZON=168;
function percentile(arr,q){if(!arr.length)return 0;const xs=arr.slice().sort((a,b)=>a-b);return xs[Math.max(0,Math.ceil(q*xs.length)-1)];}
function summarize(batches,scenario,events,transports,completed,energyRate){events.sort((a,b)=>a.start-b.start),transports.sort((a,b)=>a.start-b.start);
 const finished=batches.filter(b=>completed[b.id]<=HORIZON),onTime=finished.filter(b=>completed[b.id]<=b.due_hour),throughput=finished.reduce((s,b)=>s+b.kilograms,0);
 let rev=0,mat=0;for(const b of finished){rev+=b.kilograms*b.price_per_kg;mat+=b.kilograms*b.material_cost_per_kg;}
 const busy=e=>Math.max(0,Math.min(HORIZON,e.end)-Math.max(0,e.start));
 const energy=events.reduce((s,e)=>s+busy(e)*POWER[e.stage]*energyRate,0),labor=events.reduce((s,e)=>s+busy(e)*118,0);
 const late=batches.reduce((s,b)=>s+Math.max(0,Math.min(HORIZON,completed[b.id]??HORIZON)-b.due_hour)*b.priority*130,0);
 const fixed=26800,handling=transports.reduce((s,m)=>s+(m.end-m.start)*30,0),contribution=rev-mat-energy-labor-late-fixed-handling;
 const utilization=Object.fromEntries(STAGES.map(s=>[s,round(100*events.filter(e=>e.stage===s).reduce((tot,e)=>tot+busy(e),0)/(HORIZON*cap(s,scenario)),1)]));
 const zones=Object.fromEntries(STAGES.map(s=>[s,{completed:events.filter(e=>e.stage===s&&e.end<=HORIZON).length,utilization:utilization[s]}]));
 const lots=batches.map(b=>({id:b.id,sku:b.sku,product:b.product,kilograms:b.kilograms,due:b.due_hour,completed:completed[b.id],on_time:completed[b.id]<=b.due_hour,released:completed[b.id]<=HORIZON,priority:b.priority}));
 const waits=transports.map(m=>m.queue_hours),hours=transports.reduce((s,m)=>s+m.end-m.start,0);
 const logistics={move_count:transports.length,avg_wait_hours:round(waits.reduce((s,v)=>s+v,0)/Math.max(1,waits.length)),p95_wait_hours:round(percentile(waits,.95)),max_wait_hours:round(Math.max(0,...waits)),avg_travel_hours:round(hours/Math.max(1,transports.length)),total_wait_hours:round(waits.reduce((s,v)=>s+v,0)),transport_hours:round(hours),distance_units:round(transports.reduce((s,m)=>s+m.distance,0))};
 const kpis={throughput_kg:throughput,batches_released:finished.length,batches_total:batches.length,service_pct:round(100*onTime.length/batches.length,1),avg_utilization_pct:round(Object.values(utilization).reduce((s,v)=>s+v,0)/STAGES.length,1),revenue:round(rev,2),material_cost:round(mat,2),energy_cost:round(energy,2),labor_cost:round(labor,2),late_penalties:round(late,2),fixed_overhead:fixed,handling_cost:round(handling,2),contribution:round(contribution,2),power_kwh:round(energy/energyRate,1),queued_hours:round(events.reduce((s,e)=>s+e.queue_hours,0),1),transport_queue_hours:logistics.total_wait_hours,max_completion_hour:round(Math.max(...Object.values(completed)),1)};
 return {engine:'Browser-native priority-event discrete-event simulator (not SimPy)',kpis,events,transports,logistics,zones,lots,horizon_hours:HORIZON};
}
function optimize(batches,preds,scenario,seed,rate){const durations=stageDurations(batches,preds,scenario),base=Array.from({length:batches.length},(_,i)=>i);
 const priority=base.slice().sort((a,b)=>(batches[a].due_hour-12*batches[a].priority)-(batches[b].due_hour-12*batches[b].priority));
 const options=[base.slice(),priority.slice(),base.slice().sort((a,b)=>batches[a].due_hour-batches[b].due_hour)];
 const cache=new Map();let evaluations=0;
 const objective=order=>{const key=order.join(',');if(cache.has(key))return cache.get(key);const s=simulate(batches,preds,scenario,order,seed,rate,'adaptive',durations);
   const obj=s.kpis.contribution+150*s.kpis.service_pct-1.6*s.kpis.queued_hours;evaluations++;cache.set(key,obj);return obj;};
 let winner=options[0],bestScore=objective(winner);for(const x of options.slice(1)){const v=objective(x);if(v>bestScore){winner=x;bestScore=v;}}
 const rr=rng(seed+5654);let incumbent=winner.slice();
 for(let iter=0;iter<165;iter++){
   let trial=incumbent.slice(),a=Math.floor(rr()*trial.length),b=Math.floor(rr()*trial.length);if(a===b)continue;
   if(iter%5===0){const x=trial.splice(a,1)[0];trial.splice(b,0,x);}else[trial[a],trial[b]]=[trial[b],trial[a]];
   const val=objective(trial);if(val>bestScore+1e-7){bestScore=val;winner=trial.slice();incumbent=trial.slice();}
   else if(iter%31===0)incumbent=priority.slice();
 }
 return {order:winner,meta:{solver:'Browser-native local-search scheduler (not OR-Tools)',status:'HEURISTIC / FEASIBLE',objective_value:round(bestScore,2),best_bound:null,wall_seconds:null,optimal:false,evaluations}};
}
function checkResult(result,batches,context){let errors=[];const byBatch=new Map(),stageSorted={};for(const e of result.events){if(!byBatch.has(e.batch_id))byBatch.set(e.batch_id,[]);byBatch.get(e.batch_id).push(e);(stageSorted[e.stage]??=[]).push(e);if(e.end<e.start-1e-5)errors.push('Event duration');}
for(const b of batches){const series=(byBatch.get(b.id)||[]).sort((a,c)=>STAGES.indexOf(a.stage)-STAGES.indexOf(c.stage));if(series.length!==STAGES.length||series.some((e,i)=>e.stage!==STAGES[i]))errors.push('Stage route '+b.id);
 for(let i=0;i<series.length-1;i++)if(series[i].end>series[i+1].start+.011)errors.push('Overlapping batch process '+b.id);
 const moves=result.transports.filter(m=>m.batch_id===b.id).sort((a,c)=>STAGES.indexOf(a.source)-STAGES.indexOf(c.source));if(moves.length!==STAGES.length-1)errors.push('Missing transport '+b.id);
 for(let i=0;i<Math.min(moves.length,series.length-1);i++)if(moves[i].start+.011<series[i].end||moves[i].end>series[i+1].start+.011)errors.push('Invalid transfer precedence '+b.id);
 }
for(const st of STAGES){const entries=(stageSorted[st]||[]);for(let k=0;k<cap(st,context);k++){const m=entries.filter(x=>x.machine===k).sort((a,b)=>a.start-b.start);for(let j=1;j<m.length;j++)if(m[j].start+.011<m[j-1].end)errors.push('Equipment double-booked '+st);}}
for(let v=0;v<2;v++){const moves=result.transports.filter(m=>m.vehicle===v).sort((a,b)=>a.start-b.start);for(let j=1;j<moves.length;j++)if(moves[j].start+.011<moves[j-1].end)errors.push('Tugger double-booked '+v);}
const k=result.kpis;if(Math.abs(k.contribution-(k.revenue-k.material_cost-k.energy_cost-k.labor_cost-k.late_penalties-k.fixed_overhead-k.handling_cost))>.15)errors.push('Financial reconciliation');
if(k.batches_released>24||k.throughput_kg<0)errors.push('Invalid throughput');return errors;
}
function factorySpec(scenario){return {facility:'North Campus · Oral Solid Dosage Unit (fictional)',stages:STAGES.map(s=>({id:s,name:LABELS[s],machines:CAP[s],available_assets:cap(s,scenario),nominal_hours:BASE[s]})),scenarios:Object.fromEntries(Object.entries(SCENARIOS).map(([k,v])=>[k,{name:v[0],detail:v[1]}])),products:PRODUCTS.map(p=>({name:p[0],code:p[1]})),logistics_network:{nodes:NODE_POS,links:LINKS,vehicles:2},horizon_hours:HORIZON,units:'kg, hours, illustrative USD'};}
function runCase(settings={}){const started=Date.now();let scenario=settings.scenario||'normal',demand_factor=Number(settings.demand_factor??1),energy_rate=Number(settings.energy_rate??.22),seed=Number(settings.seed??41),monte_carlo=Number(settings.monte_carlo??8),routing_policy=settings.routing_policy||'adaptive';
if(!SCENARIOS[scenario])throw Error('Unknown scenario');if(!['adaptive','ppo'].includes(routing_policy))throw Error('Unknown routing policy');if(routing_policy==='ppo'&&!globalThis.PharmaOpsAdvanced?.ppoAction)throw Error('PPO engine not loaded');if(demand_factor<.8||demand_factor>1.35)throw Error('Demand outside limits');if(energy_rate<.08||energy_rate>.8)throw Error('Energy tariff outside limits');if(monte_carlo<0||monte_carlo>20)throw Error('Monte Carlo trials outside limits');
const batches=buildBatches(scenario,demand_factor),pred=predictBatches(batches),rate=energy_rate*(scenario==='energy'?1.65:1);
const baseOrder=batches.map((_,i)=>i),winner=optimize(batches,pred,scenario,seed,rate);
const baseline=simulate(batches,pred,scenario,baseOrder,seed,rate,'distance'),optimized=simulate(batches,pred,scenario,winner.order,seed,rate,routing_policy);
const trials=[];for(let i=0;i<monte_carlo;i++){const s=seed+i+13,ba=simulate(batches,pred,scenario,baseOrder,s,rate,'distance').kpis,op=simulate(batches,pred,scenario,winner.order,s,rate,'adaptive').kpis;
trials.push({trial:i+1,baseline_contribution:ba.contribution,optimized_contribution:op.contribution,baseline_service:ba.service_pct,optimized_service:op.service_pct,delta_contribution:round(op.contribution-ba.contribution,2)});}
const capex=145000,support=24000,capture=.42,delta=optimized.kpis.contribution-baseline.kpis.contribution,annual=delta*260/7*capture-support,discount=.12,npv=-capex+[1,2,3].reduce((s,y)=>s+annual/(1+discount)**y,0);
const economics={capex,annual_support:support,annual_net_benefit:round(annual,2),capture_rate:capture,discount_rate:discount,npv_3yr:round(npv,2),payback_years:annual>0?round(capex/annual,2):null,baseline_contribution:baseline.kpis.contribution,optimized_contribution:optimized.kpis.contribution,delta_contribution:round(delta,2),note:'Synthetic scenario valuation only; not measured savings or a forecast.'};
const failures=[...checkResult(baseline,batches,scenario).map(s=>'baseline: '+s),...checkResult(optimized,batches,scenario).map(s=>'candidate: '+s)];
const metrics=trainModels().metrics;
return {meta:{project:'PharmaOps Twin Pro',version:'4.0 Browser-Native Hybrid',scenario,scenario_label:SCENARIOS[scenario][0],engine_mode:'BROWSER_NATIVE',simpy_installed:false,ortools_installed:false,pyodide_installed:false,scikit_learn_installed:false,routing_policy,ppo_active:routing_policy==='ppo',simulation_engine:baseline.engine,optimizer:{...winner.meta,wall_seconds:round((Date.now()-started)/1000,2)},synthetic_data:true,live_telemetry:false,warning:null,disclaimer:'Offline live computation in JavaScript. Does not run Python SimPy, OR-Tools, or scikit-learn.'},factory:factorySpec(scenario),settings:{scenario,demand_factor,energy_rate,seed,routing_policy},baseline,optimized,predictions:pred,ml_metrics:metrics,trials,economics,checks:{passed:failures.length===0,failures,test_count:2*(3+4*batches.length)+2*STAGES.length}};
}
return {runCase,trainModels,buildBatches,route,checkResult,version:'4.0',engines:{simulation:'Browser-native DES',optimization:'Browser-native search heuristic',ml:'Browser-native random forests'}};
});
