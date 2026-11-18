const assert=require('node:assert/strict');
const engine=require('../frontend/browser_engine.js');
let passes=0, failures=[];
function check(name, fn){try{fn();passes++;console.log('PASS',name);}catch(err){failures.push([name,String(err)]);console.error('FAIL',name,err.message);}}
const standard=engine.runCase({scenario:'normal'});
check('Fully offline engine returns 24 completed batch traces',()=>assert.equal(standard.optimized.lots.length,24));
check('168 traceable processing events per strategy',()=>assert.equal(standard.optimized.events.length,168));
check('144 material transfer events per strategy',()=>assert.equal(standard.optimized.transports.length,144));
check('All resource, transport and finance constraints pass',()=>assert.equal(standard.checks.passed,true));
check('ML trained on synthetic generated records and tested on independent holdout',()=>assert(standard.ml_metrics.duration_r2>0.5&&standard.ml_metrics.downtime_auc>0.6));
check('Optimization tries at least 75 candidate schedules',()=>assert(standard.meta.optimizer.evaluations>=75));
check('8 randomized paired-seed trials run',()=>assert.equal(standard.trials.length,8));
check('Synthetic data and exact model origin visibly recorded',()=>assert(standard.meta.simpy_installed===false&&standard.meta.ortools_installed===false&&standard.meta.engine_mode==='BROWSER_NATIVE'));
check('Rerun deterministic with same seed',()=>{let r=engine.runCase({scenario:'normal'});assert.deepEqual(r.optimized.events,standard.optimized.events);assert.deepEqual(r.trials,standard.trials);});
check('Demand slider causes new batch sizes',()=>{let r=engine.runCase({scenario:'normal',demand_factor:1.2});assert(r.optimized.kpis.throughput_kg>standard.optimized.kpis.throughput_kg);});
check('Electricity tariff changes financial outcome',()=>{let r=engine.runCase({scenario:'normal',energy_rate:.62});assert(r.optimized.kpis.energy_cost>standard.optimized.kpis.energy_cost*2.5);});
check('Granulator outage reduces stage capacity',()=>{let r=engine.runCase({scenario:'dryer'});assert.equal(r.factory.stages.find(s=>s.id==='granulation').available_assets,1);assert.equal(r.checks.passed,true);});
check('Supplier delay changes release timing',()=>{let r=engine.runCase({scenario:'supplier'});assert(r.baseline.events.find(e=>e.batch_id==='BT26-5102'&&e.stage==='dispensing').start>=17.5);});
check('Quality hold increases QC time',()=>{let r=engine.runCase({scenario:'quality'});assert(r.optimized.events.some(e=>e.stage==='qc'&&e.duration>6));});
check('Energy spike scenario applies tariff multiplier',()=>{let r=engine.runCase({scenario:'energy'});assert(r.baseline.kpis.energy_cost>standard.baseline.kpis.energy_cost*1.5);});
for(const scenario of Object.keys({normal:1,demand:1,dryer:1,quality:1,supplier:1,energy:1})){
 for(const demand_factor of [.8,1.35]){
 const r=engine.runCase({scenario,demand_factor,energy_rate:.54,seed:97});
 check(`Scenario ${scenario} / demand ${demand_factor} validates`,()=>assert.deepEqual(r.checks.failures,[]));
 }}
check('Rejects unsupported settings',()=>assert.throws(()=>engine.runCase({scenario:'nonsense'}),/Unknown scenario/));
console.log('RESULT:',passes,'passed,',failures.length,'failed');
if(failures.length)process.exitCode=1;
