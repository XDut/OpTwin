/* PharmaOps v4: independent offline JavaScript implementations of asset-level
   predictive maintenance, supervised model benchmarks, exact-coalition feature
   attributions relative to a fixed background vector, and tabular-feature
   clipped-policy-gradient (PPO) material dispatch. Synthetic data only.
   NOT Python/scikit-learn, SHAP Python package, XGBoost, LightGBM, or OR-Tools. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;root.PharmaOpsAdvanced=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){'use strict';
const PI=Math.PI, FEATURES=['Vibration (mm/s)','Bearing temp (C)','Motor current (A)','Process pressure (bar)','Equipment load','Hours since service','Vibration 12h slope','Temperature 12h slope'];
const ASSET_TYPES=[['GR-201','High-shear granulator'],['GR-202','Fluid bed granulator'],['CP-401','Tablet compression press'],['CP-402','Rotary compression press'],['CO-501','Film coating pan'],['PK-601','Blister packaging line'],['BL-301','Bin blender'],['QC-701','QC testing station']];
const rnd=(v,d=3)=>Number(Number(v).toFixed(d)),clamp=(v,l,h)=>Math.max(l,Math.min(h,v));
function rng(seed){let a=seed>>>0;return function(){a=(a+0x6D2B79F5)>>>0;let t=Math.imul(a^(a>>>15),1|a);t^=t+Math.imul(t^(t>>>7),61|t);return((t^(t>>>14))>>>0)/4294967296;};}
function gauss(r){return Math.sqrt(-2*Math.log(Math.max(r(),1e-9)))*Math.cos(2*PI*r());}
function mean(x){return x.reduce((a,b)=>a+b,0)/(x.length||1);}
function percent(xs,p){if(!xs.length)return 0;const a=xs.slice().sort((x,y)=>x-y),z=(a.length-1)*p,lo=Math.floor(z);return a[lo]+(a[Math.min(a.length-1,lo+1)]-a[lo])*(z-lo);}
function sigmoid(x){return x>25?1:x< -25?0:1/(1+Math.exp(-x));}
function auc(y,s){const pairs=y.map((v,i)=>[s[i],v]).sort((a,b)=>a[0]-b[0]);let pos=y.reduce((a,b)=>a+b,0),neg=y.length-pos;if(!pos||!neg)return null;let rank=0;for(let i=0;i<pairs.length;){let j=i+1;while(j<pairs.length&&pairs[j][0]===pairs[i][0])j++;for(let k=i;k<j;k++)if(pairs[k][1])rank+=(i+j+1)/2;i=j;}return(rank-pos*(pos+1)/2)/(pos*neg);}
function regressionMetrics(y,p){let se=0,ae=0,v=mean(y),ss=0;for(let i=0;i<y.length;i++){ae+=Math.abs(y[i]-p[i]);se+=(y[i]-p[i])**2;ss+=(y[i]-v)**2;}return{mae:rnd(ae/y.length),rmse:rnd(Math.sqrt(se/y.length)),r2:rnd(1-se/(ss||1))};}
function classificationMetrics(y,p){let tp=0,fp=0,tn=0,fn=0,brier=0;for(let i=0;i<y.length;i++){const z=p[i]>=.5?1:0;brier+=(y[i]-p[i])**2;if(y[i]&&z)tp++;if(!y[i]&&z)fp++;if(!y[i]&&!z)tn++;if(y[i]&&!z)fn++;}const ece=Array.from({length:10},()=>[]);for(let i=0;i<y.length;i++)ece[Math.min(9,Math.floor(p[i]*10))].push(i);let cal=0;for(const b of ece)if(b.length)cal+=b.length/y.length*Math.abs(mean(b.map(i=>p[i]))-mean(b.map(i=>y[i])));return{roc_auc:auc(y,p)==null?null:rnd(auc(y,p)),brier:rnd(brier/y.length),ece:rnd(cal),f1:rnd(2*tp/Math.max(1,2*tp+fp+fn)),precision:rnd(tp/Math.max(1,tp+fp)),recall:rnd(tp/Math.max(1,tp+fn)),confusion:{tp,fp,tn,fn},calibration:ece.map((b,i)=>({range:`${i*10}-${i*10+10}%`,count:b.length,predicted:b.length?rnd(mean(b.map(k=>p[k]))):null,observed:b.length?rnd(mean(b.map(k=>y[k]))):null}))};}
// Assets deliberately have separate train and hold-out identities to prevent same-asset train/test leakage.
function telemetry(seed=928){const r=rng(seed),rows=[];const nAssets=48;
 for(let a=0;a<nAssets;a++){
  const kind=a%ASSET_TYPES.length,life=110+Math.floor(r()*150),tempBias=(r()-.5)*7,vibBias=(r()-.5)*.7,hardness=.83+r()*.4,loadBias=.45+r()*.38,history=[];
  for(let h=4;h<=168;h+=4){const load=clamp(loadBias+.08*Math.sin(h/17+a*.3)+gauss(r)*.035,.3,1),age=h/life;
   const vibration=clamp(1.15+6.7*age*age+load*.8+vibBias+gauss(r)*.32,.7,15),temp=41+27*load+19*age*age+tempBias+gauss(r)*1.8;
   const current=7+10.3*load+4.5*age*age+gauss(r)*.65,pressure=3.4+.4*load-.75*age*age+gauss(r)*.055;
   const failureHours=clamp(life-h,0,240),failure24=Number(failureHours<=24),asset=ASSET_TYPES[kind][0]+'-'+String(Math.floor(a/8)+1);
   const previous=history[Math.max(0,history.length-3)];
   const vibSlope=previous?(vibration-previous.vibration)/(h-previous.h):0;
   const tempSlope=previous?(temp-previous.temp)/(h-previous.h):0;
   rows.push({asset,kind,h,life,x:[vibration,temp,current,pressure,load,h,vibSlope,tempSlope],rul:failureHours,y:failure24});
   history.push({h,vibration,temp});
  }
 }return rows;}
function trainTree(X,y,opt={}){const nFeat=X[0].length,seed=opt.seed||11,r=rng(seed),leaf=opt.leaf||14,depth=opt.depth||6,cls=opt.cls||false;
 function varImp(indices){let sum=0,sq=0;for(const i of indices){sum+=y[i];sq+=y[i]*y[i];}return sq-(sum*sum)/indices.length;}
 function split(ids,d){const val=mean(ids.map(i=>y[i]));if(d>=depth||ids.length<2*leaf)return {v:val};
  const fset=[];while(fset.length<Math.min(nFeat,Math.ceil(Math.sqrt(nFeat))+1)){let f=Math.floor(r()*nFeat);if(!fset.includes(f))fset.push(f);}
  let best=null,score0=varImp(ids);
  for(const f of fset){const sorted=ids.slice().sort((a,b)=>X[a][f]-X[b][f]);let lsum=0,lsq=0,tsum=0,tsq=0;for(const ix of sorted){tsum+=y[ix];tsq+=y[ix]**2;}
   const stride=Math.max(1,Math.floor(ids.length/24));for(let k=1;k<sorted.length-1;k++){const yy=y[sorted[k-1]];lsum+=yy;lsq+=yy**2;
    if(k<leaf||sorted.length-k<leaf||k%stride!==0||X[sorted[k-1]][f]===X[sorted[k]][f])continue;
    const left=lsq-lsum*lsum/k,right=tsq-lsq-(tsum-lsum)**2/(sorted.length-k),gain=score0-left-right;
    if(gain>Math.max(1e-8,best?.gain||0))best={f,t:(X[sorted[k-1]][f]+X[sorted[k]][f])/2,gain};}
  }
  if(!best)return{v:val};let left=[],right=[];for(const i of ids)(X[i][best.f]<=best.t?left:right).push(i);
  return{f:best.f,t:best.t,l:split(left,d+1),r:split(right,d+1)};
 }
 const indices=opt.indices||Array.from({length:y.length},(_,i)=>i),tree=split(indices,0);
 return x=>{let n=tree;while(n.f!==undefined)n=x[n.f]<=n.t?n.l:n.r;return cls?clamp(n.v,.001,.999):n.v;};
}
function forest(X,y,opt={}){const r=rng(opt.seed||31),trees=[];for(let t=0;t<(opt.trees||18);t++){const ix=Array.from({length:y.length},()=>Math.floor(r()*y.length));trees.push(trainTree(X,y,{indices:ix,depth:opt.depth||7,leaf:opt.leaf||13,seed:31+t*17,cls:opt.cls}));}
 return{predict(x){return trees.reduce((sum,f)=>sum+f(x),0)/trees.length;},name:'Random Forest'};}
function stump(X,res){let best=null;const n=X.length,fs=X[0].length,global=mean(res);let baseline=res.reduce((v,x)=>v+(x-global)**2,0);
 for(let f=0;f<fs;f++){const order=Array.from({length:n},(_,i)=>i).sort((a,b)=>X[a][f]-X[b][f]);const stride=Math.max(1,Math.floor(n/30));let sum=0,sq=0,all=0,allSq=0;for(const y of res){all+=y;allSq+=y*y;}
  for(let i=1;i<n-1;i++){const y=res[order[i-1]];sum+=y;sq+=y*y;if(i<30||n-i<30||i%stride||X[order[i-1]][f]===X[order[i]][f])continue;
   const l=sum/i,r=(all-sum)/(n-i),gain=baseline-(sq-sum*sum/i+(allSq-sq)-(all-sum)**2/(n-i));
   if(gain>Math.max(1e-7,best?.gain||0))best={f,t:(X[order[i-1]][f]+X[order[i]][f])/2,l,r,gain};}
 }return best;}
function gbdt(X,y,opt={}){const cls=opt.cls||false,lr=opt.lr||.12,n=opt.steps||50,avg=mean(y),bias=cls?Math.log(clamp(avg,.001,.999)/(1-clamp(avg,.001,.999))):avg;const scores=y.map(()=>bias),stumps=[];
 for(let i=0;i<n;i++){const residual=y.map((v,j)=>v-(cls?sigmoid(scores[j]):scores[j])),s=stump(X,residual);if(!s)break;
  stumps.push(s);for(let j=0;j<y.length;j++)scores[j]+=lr*(X[j][s.f]<=s.t?s.l:s.r);
 }return{predict(x){let p=bias;for(const s of stumps)p+=lr*(x[s.f]<=s.t?s.l:s.r);return cls?clamp(sigmoid(p),.001,.999):p;},name:'Gradient-boosted trees (custom)'};}
function logistic(X,y){const n=X.length,d=X[0].length,weights=new Array(d+1).fill(0),mins=Array.from({length:d},(_,f)=>Math.min(...X.map(x=>x[f]))),maxs=Array.from({length:d},(_,f)=>Math.max(...X.map(x=>x[f])));
 const scale=x=>x.map((v,j)=>(v-mins[j])/(maxs[j]-mins[j]+1e-6));const z=X.map(scale);
 for(let ep=0;ep<260;ep++){let grad=weights.map(()=>0);for(let i=0;i<n;i++){const p=sigmoid(weights[0]+z[i].reduce((sum,v,j)=>sum+v*weights[j+1],0)),e=p-y[i];grad[0]+=e;for(let j=0;j<d;j++)grad[j+1]+=e*z[i][j];}
 for(let j=0;j<weights.length;j++)weights[j]-=.8*(grad[j]/n+.001*weights[j]);}
 return{predict(x){let v=scale(x);return clamp(sigmoid(weights[0]+v.reduce((s,z,j)=>s+z*weights[j+1],0)),.001,.999);},name:'Logistic regression'};}
function mlp(X,y,opt={}){const r=rng(137),d=X[0].length,H=10,mins=Array.from({length:d},(_,f)=>Math.min(...X.map(x=>x[f]))),maxs=Array.from({length:d},(_,f)=>Math.max(...X.map(x=>x[f])));
 const scale=x=>x.map((v,j)=>2*(v-mins[j])/(maxs[j]-mins[j]+1e-6)-1),Z=X.map(scale);const W=Array.from({length:H},()=>Array.from({length:d+1},()=> (r()-.5)*.75)),V=Array.from({length:H+1},()=> (r()-.5)*.33);
 for(let ep=0;ep<42;ep++){const rate=.025/(1+ep*.025);for(let k=0;k<Z.length;k++){const i=Math.floor(r()*Z.length),x=Z[i],h=W.map(w=>Math.tanh(w[0]+x.reduce((s,v,j)=>s+w[j+1]*v,0))),p=sigmoid(V[0]+h.reduce((s,v,j)=>s+V[j+1]*v,0)),dz=p-y[i],oldV=V.slice();V[0]-=rate*dz;for(let j=0;j<H;j++)V[j+1]-=rate*dz*h[j];for(let j=0;j<H;j++){const dh=dz*oldV[j+1]*(1-h[j]**2);W[j][0]-=rate*dh;for(let f=0;f<d;f++)W[j][f+1]-=rate*dh*x[f];}}}
 return{predict(x){const z=scale(x),h=W.map(w=>Math.tanh(w[0]+z.reduce((s,v,j)=>s+w[j+1]*v,0)));return clamp(sigmoid(V[0]+h.reduce((s,v,j)=>s+V[j+1]*v,0)),.001,.999);},name:'One-hidden-layer neural net'};}
const SAMPLE_IDS=['GR-201','GR-202','CP-401','CP-402','CO-501','PK-601','BL-301','QC-701'];
let trained=null;
function trainMaintenance(){if(trained)return trained;const started=Date.now(),rows=telemetry(),train=rows.filter(x=>Number(x.asset.split('-').pop())<=4),test=rows.filter(x=>Number(x.asset.split('-').pop())>4),X=train.map(z=>z.x),Y=train.map(z=>z.y),R=train.map(z=>z.rul),xTest=test.map(z=>z.x),yTest=test.map(z=>z.y),rTest=test.map(z=>z.rul);
 const classifierModels=[forest(X,Y,{seed:66,trees:18,depth:8,cls:true}),gbdt(X,Y,{cls:true,steps:42,lr:.16}),mlp(X,Y),logistic(X,Y)];
 const regressorModels=[forest(X,R,{seed:83,trees:20,depth:8}),gbdt(X,R,{steps:60,lr:.14})];
 const classification=classifierModels.map(m=>{const now=performanceNow(),p=xTest.map(x=>m.predict(x)),ms=performanceNow()-now;return{name:m.name,metrics:classificationMetrics(yTest,p),latency_ms_per_1000:rnd(ms*1000/xTest.length,3)};});
 const regression=regressorModels.map(m=>{const now=performanceNow(),p=xTest.map(x=>m.predict(x)),ms=performanceNow()-now;return{name:m.name,metrics:regressionMetrics(rTest,p),latency_ms_per_1000:rnd(ms*1000/xTest.length,3)};});
 classification.sort((a,b)=>(b.metrics.roc_auc||0)-(a.metrics.roc_auc||0));regression.sort((a,b)=>a.metrics.rmse-b.metrics.rmse);
 const model=classifierModels.find(m=>m.name===classification[0].name),rulModel=regressorModels.find(m=>m.name===regression[0].name),background=Array.from({length:FEATURES.length},(_,i)=>mean(X.map(x=>x[i])));
 const snapshots=Array.from({length:8},(_,i)=>{const asset=rows.filter(z=>z.kind===i && z.h===136 && z.asset.endsWith('-6'))[0];return{asset_id:SAMPLE_IDS[i],type:ASSET_TYPES[i][1],hour:136,signals:asset.x.map(x=>rnd(x,2)),actual_rul_hours:asset.rul,predicted_rul_hours:rnd(clamp(rulModel.predict(asset.x),0,240),1),risk_24h:rnd(model.predict(asset.x),4),trend:rows.filter(z=>z.kind===i&&z.asset===asset.asset&&z.h>=72&&z.h<=168).map(z=>({hour:z.h,vibration:rnd(z.x[0],2),temperature:rnd(z.x[1],1),current:rnd(z.x[2],1),risk:rnd(model.predict(z.x),3)}))};});
 trained={dataset:{rows:rows.length,training_rows:train.length,test_rows:test.length,train_assets:32,test_assets:16,split:'Grouped by simulated asset identity (no shared equipment between train and holdout)',source:'Deterministic synthetic vibration, bearing temperature, current, pressure, load, operating hours; no plant sensors'},model,rulModel,background,snapshots,classification,regression,training_ms:Date.now()-started,feature_names:FEATURES,holdout:{y:yTest,r:rTest,x:xTest}};return trained;
}
function performanceNow(){return typeof performance!=='undefined'&&performance.now?performance.now():Date.now();}
const FACT=[1,1,2,6,24,120,720,5040,40320];
function explanation(x,model,background){const n=x.length,N=1<<n,vals=new Array(N).fill(0);for(let mask=0;mask<N;mask++){const z=background.map((b,j)=>(mask&(1<<j))?x[j]:b);vals[mask]=model.predict(z);}const shap=Array(n).fill(0);for(let mask=0;mask<N;mask++)for(let i=0;i<n;i++)if(!(mask&(1<<i))){let size=0;for(let j=0;j<n;j++)size+=Boolean(mask&(1<<j));shap[i]+=FACT[size]*FACT[n-size-1]/FACT[n]*(vals[mask|(1<<i)]-vals[mask]);}return{base:rnd(vals[0],5),prediction:rnd(vals[N-1],5),shap:shap.map(v=>rnd(v,5)),reconstruction_error:rnd(Math.abs(vals[0]+shap.reduce((s,v)=>s+v,0)-vals[N-1]),8),method:'Exact subset Shapley contributions with fixed mean-feature reference (not Python SHAP library; not causal)'};}
function conditionForecast(scenario='normal',demand=1){const m=trainMaintenance();return m.snapshots.map(s=>{
 const x=s.signals.slice();const overload=Math.max(0,demand-1);x[4]=clamp(x[4]+overload*.45,0,1.3);x[2]+=overload*2;x[1]+=overload*6;
 if(scenario==='dryer'&&s.asset_id.startsWith('GR')){x[0]+=2.0;x[1]+=5.5;x[6]+=.15;x[7]+=.27;}
 if(scenario==='energy'){x[1]+=1.7;}
 if(scenario==='quality'&&s.asset_id.startsWith('QC'))x[4]=clamp(x[4]+.16,0,1.3);
 return{...s,signals:x.map(z=>rnd(z,2)),risk_24h:rnd(m.model.predict(x),4),predicted_rul_hours:rnd(clamp(m.rulModel.predict(x),0,240),1),scenario};
 });}
function maintenanceExplain(assetIndex=0,scenario='normal',demand=1){const m=trainMaintenance(),chosen=conditionForecast(scenario,demand)[clamp(assetIndex,0,m.snapshots.length-1)],local=explanation(chosen.signals,m.model,m.background),global=new Array(FEATURES.length).fill(0);
 for(const a of m.snapshots){const s=explanation(a.signals,m.model,m.background);s.shap.forEach((v,i)=>global[i]+=Math.abs(v)/m.snapshots.length);}
 return{asset:chosen,local,global:global.map(v=>rnd(v,5)),features:FEATURES};}
// PPO: clipped policy-gradient actor and temporal-difference critic on a stochastic
// two-tugger dispatch MDP. Four decisions = (distance/traffic route) x (vehicle 0/1).
const ACTIONS=['Shortest · Tugger 1','Shortest · Tugger 2','Traffic-aware · Tugger 1','Traffic-aware · Tugger 2'];
function tasks(seed=111,scenario='normal',count=64){const r=rng(seed),pairs=[['warehouse','dispensing'],['dispensing','granulation'],['granulation','blending'],['blending','compression'],['compression','coating'],['coating','packaging'],['packaging','qc']];return Array.from({length:count},(_,i)=>{const [source,dest]=pairs[Math.floor(r()*pairs.length)],request=i*.24+r()*.10,urgent=r()>.66,deadline=request+(urgent?.75:2.1)+r()*.35;return{source,dest,request,deadline,urgent};});}
function dispatchFeatures(task,avail,options){const shortest=options.short,adaptive=options.traffic;return[1,clamp((avail[0]-avail[1])/2,-3,3),clamp((avail[1]-avail[0])/2,-3,3),clamp((shortest.travel_hours-adaptive.travel_hours)*2,-2,2),clamp(shortest.distance/120,0,2),task.urgent?1:0,clamp((task.deadline-task.request)/3,0,2)];}
function softmax(logits){const m=Math.max(...logits),v=logits.map(x=>Math.exp(x-m)),sum=v.reduce((a,b)=>a+b,0);return v.map(x=>x/sum);}
function dot(x,y){return x.reduce((s,v,i)=>s+v*y[i],0);}
function policyProbs(w,x){return softmax(w.map(row=>dot(row,x)));}
function optionsFor(task,scenario){const E=typeof globalThis!=='undefined'?globalThis.PharmaOpsBrowser:null;if(E){return{short:E.route(task.source,task.dest,scenario,'distance'),traffic:E.route(task.source,task.dest,scenario,'adaptive')};}
 // Fallback for isolated node-model tests: same topology-independent proxy cost.
 const s=task.source.length*.12+task.dest.length*.07;return{short:{travel_hours:s,distance:65},traffic:{travel_hours:s*.92,distance:71}};}
function rollout(w,v,seed,scenario='normal',mode='sample',rseed=123,record=false){const rr=rng(rseed),avail=[0,0],history=[];let rewardSum=0,wait=0,late=0,travel=0,actionCounts=[0,0,0,0];for(const job of tasks(seed,scenario)){
 const opt=optionsFor(job,scenario),feat=dispatchFeatures(job,avail,opt),prob=policyProbs(w,feat),action=mode==='shortest'?(avail[0]<=avail[1]?0:1):mode==='adaptive'?(avail[0]<=avail[1]?2:3):mode==='greedy'?prob.indexOf(Math.max(...prob)):(()=>{let z=rr(),idx=0;for(let k=0;k<prob.length;k++){z-=prob[k];if(z<=0){idx=k;break;}}return idx;})();
 const vehicle=action%2,rt=action>=2?opt.traffic:opt.short;const start=Math.max(job.request,avail[vehicle]),finish=start+rt.travel_hours,delay=Math.max(0,finish-job.deadline),queue=start-job.request;
 const reward=-(queue*2.5+rt.travel_hours*1.8+delay*4)*.2;avail[vehicle]=finish;rewardSum+=reward;wait+=queue;late+=delay;travel+=rt.travel_hours;actionCounts[action]++;
 if(record)history.push({features:feat,action,oldProb:prob[action],reward,value:dot(v,feat),job,vehicle,start,finish,queue,late:delay});}
 return{reward:rewardSum,mean_wait:wait/64,late_hours:late,travel_hours:travel,action_counts:actionCounts,history};}
let trainedPPO=null;
function trainPPO(opt={}){if(trainedPPO&&!opt.retrain)return trainedPPO;const r=rng(opt.seed||2026),dim=7,w=Array.from({length:4},()=>Array.from({length:dim},()=> (r()-.5)*.06)),v=Array(dim).fill(0);const gamma=.96,lambda=.92,clip=.2,epochs=Number(opt.episodes||170),history=[];
 // Behavior-cloning warm start against feasible, congestion-aware dispatch.
 // PPO subsequently optimizes its own on-policy rollouts, not merely replaying labels.
 for(let warm=0;warm<55;warm++){
   const avail=[0,0];for(const job of tasks(700+warm*17,['normal','demand','dryer','quality'][warm%4])){
     const rt=optionsFor(job,['normal','demand','dryer','quality'][warm%4]),x=dispatchFeatures(job,avail,rt),p=policyProbs(w,x),target=avail[0]<=avail[1]?2:3;
     for(let a=0;a<4;a++)for(let j=0;j<dim;j++)w[a][j]+=.017*((a===target?1:0)-p[a])*x[j];
     const k=target%2;avail[k]=Math.max(avail[k],job.request)+rt.traffic.travel_hours;
   }
 }
 for(let ep=0;ep<epochs;ep++){const rolloutData=rollout(w,v,ep*73+64,['normal','demand','dryer','quality'][ep%4],'sample',ep+900,true),steps=rolloutData.history;
  let gae=0,adv=Array(steps.length).fill(0),ret=Array(steps.length).fill(0);for(let t=steps.length-1;t>=0;t--){const next=t===steps.length-1?0:steps[t+1].value,delta=steps[t].reward+gamma*next-steps[t].value;gae=delta+gamma*lambda*gae;adv[t]=gae;ret[t]=gae+steps[t].value;}
  const am=mean(adv),std=Math.sqrt(mean(adv.map(x=>(x-am)**2))+1e-7);
  for(let pass=0;pass<3;pass++)for(let t=0;t<steps.length;t++){const s=steps[t],a=s.action,p=policyProbs(w,s.features),ratio=p[a]/Math.max(1e-8,s.oldProb),A=clamp((adv[t]-am)/std,-3,3);
   if(!((A>0&&ratio>1+clip)||(A<0&&ratio<1-clip))){const coef=.0011*ratio*A;for(let k=0;k<4;k++)for(let j=0;j<dim;j++)w[k][j]+=coef*((k===a?1:0)-p[k])*s.features[j];}
   const err=clamp(ret[t]-dot(v,s.features),-12,12);for(let j=0;j<dim;j++)v[j]+=.0006*err*s.features[j];
  }
  if(ep%20===0)history.push({episode:ep+1,training_reward:rnd(rolloutData.reward)});
 }
 const scores={};for(const [label,mode]of[['PPO policy','greedy'],['Shortest-path FIFO','shortest'],['Congestion-aware heuristic','adaptive']]){
  const results=Array.from({length:24},(_,j)=>rollout(w,v,9019+j*71,['normal','demand','dryer','quality'][j%4],mode,0,false));
  scores[label]={average_episode_reward:rnd(mean(results.map(a=>a.reward))),average_queue_hours:rnd(mean(results.map(a=>a.mean_wait))),average_late_hours:rnd(mean(results.map(a=>a.late_hours))),average_travel_hours:rnd(mean(results.map(a=>a.travel_hours)))};
 }
 trainedPPO={weights:w,value_weights:v,train_episodes:epochs,heldout_episodes:24,train_curve:history,benchmark:scores,algorithm:'Clipped PPO surrogate, linear categorical actor, value critic, GAE; browser-native implementation',actions:ACTIONS,method:'Separate synthetic transport episodes. Independent of the local-search production scheduler; not OR-Tools.'};return trainedPPO;
}
function ppoAction(src,dst,scenario,t,avail,deadline=t+1.5){const trained=trainPPO(),job={source:src,dest:dst,request:t,deadline,urgent:deadline-t<1},opt=optionsFor(job,scenario),p=policyProbs(trained.weights,dispatchFeatures(job,avail,opt)),choice=p.indexOf(Math.max(...p));return{vehicle:choice%2,route:choice>=2?'adaptive':'distance',action:choice,confidence:rnd(p[choice],3)};}
function report(){const m=trainMaintenance(),rl=trainPPO();return{project:'PharmaOps Twin Pro v4',synthetic:true,executed_in_browser:true,dependencies:{simpy:false,ortools:false,pyodide:false,scikit_learn:false,shap_package:false,xgboost:false,lightgbm:false},maintenance:{train_samples:m.dataset.training_rows,test_samples:m.dataset.test_rows,classification:m.classification,regression:m.regression},ppo:{training_episodes:rl.train_episodes,evaluation_episodes:rl.heldout_episodes,benchmark:rl.benchmark},disclaimer:'Educational research prototype. Requires plant-data integration and validation before use in manufacturing.'};}
return{trainMaintenance,conditionForecast,maintenanceExplain,trainPPO,ppoAction,report,telemetry,explanation,classificationMetrics,regressionMetrics,trainTree,forest,gbdt,logistic,mlp,version:'4.0',featureNames:FEATURES,assetNames:SAMPLE_IDS};
});
