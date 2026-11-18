"""Synthetic scenario orchestration with real library usage where available."""
from __future__ import annotations
import importlib.util
import math
from .plant import build_batches,factory_spec,SCENARIOS,STAGES,active_capacity
from .predictive import predict_batches,train_models
from .optimizer import make_baseline,optimize_cp_sat,optimize_greedy_preview
from .simulator import simulate_simpy,simulate_preview,simulated_consistency

HAS_SIMPY=importlib.util.find_spec('simpy') is not None
HAS_ORTOOLS=importlib.util.find_spec('ortools') is not None

def run_case(scenario='normal',demand_factor=1.0,energy_rate=.22,seed=41,monte_carlo=8,solver_seconds=3):
    if scenario not in SCENARIOS: raise ValueError('Unknown scenario')
    if not(.8<=demand_factor<=1.35): raise ValueError('Demand factor outside .8–1.35')
    if not(.08<=energy_rate<=.8): raise ValueError('Energy rate outside .08–.8')
    if not(0<=monte_carlo<=20):raise ValueError('Monte Carlo trials outside 0–20')
    batches=build_batches(scenario,demand_factor)
    predictions,metrics=predict_batches(batches)
    baseline_plan=make_baseline(batches)
    solver_warning=None
    if HAS_ORTOOLS:
        try: optimal_plan,solver=optimize_cp_sat(batches,predictions,scenario,solver_seconds)
        except Exception as exc:
            solver_warning=f'OR-Tools planning error: {exc}'
            optimal_plan,solver=optimize_greedy_preview(batches,predictions,scenario)
    else:
        optimal_plan,solver=optimize_greedy_preview(batches,predictions,scenario)
    for schedule in optimal_plan.values():schedule["routing"]=True
    sim=simulate_simpy if HAS_SIMPY else simulate_preview
    rate=energy_rate*(1.65 if scenario=='energy' else 1)
    baseline=sim(batches,predictions,scenario,baseline_plan,seed,rate)
    optimized=sim(batches,predictions,scenario,optimal_plan,seed,rate)
    trials=[]
    for i in range(monte_carlo):
        s=seed+i+13
        a=sim(batches,predictions,scenario,baseline_plan,s,rate)['kpis']
        b=sim(batches,predictions,scenario,optimal_plan,s,rate)['kpis']
        trials.append({'trial':i+1,'baseline_contribution':a['contribution'],
                       'optimized_contribution':b['contribution'],
                       'baseline_service':a['service_pct'],'optimized_service':b['service_pct'],
                       'delta_contribution':round(b['contribution']-a['contribution'],2)})
    capex=145000;yearly_support=24000;capture=.42
    delta=optimized['kpis']['contribution']-baseline['kpis']['contribution']
    annual=delta*(260/7)*capture-yearly_support
    discount=.12
    npv=-capex+sum(annual/((1+discount)**year) for year in range(1,4))
    economics={'capex':capex,'annual_support':yearly_support,'annual_net_benefit':round(annual,2),
               'capture_rate':capture,'discount_rate':discount,'npv_3yr':round(npv,2),
               'payback_years':round(capex/annual,2) if annual>0 else None,
               'baseline_contribution':baseline['kpis']['contribution'],
               'optimized_contribution':optimized['kpis']['contribution'],
               'delta_contribution':round(delta,2),
               'note':'Illustrative scenario valuation, not measured savings or a financial forecast.'}
    checks=simulated_consistency(baseline,batches)+simulated_consistency(optimized,batches)
    for name,result in [('baseline',baseline),('optimized',optimized)]:
        for st,z in result['zones'].items():
            if z['utilization']>100.05:checks.append(f'Overcapacity: {name} {st}')
    return {'meta':{'project':'PharmaOps Twin Pro','version':'2.1',
                    'scenario':scenario,'scenario_label':SCENARIOS[scenario]['name'],
                    'engine_mode':'NATIVE' if HAS_SIMPY and HAS_ORTOOLS and not solver_warning else 'DEGRADED_PREVIEW',
                    'simpy_installed':HAS_SIMPY,'ortools_installed':HAS_ORTOOLS,
                    'simulation_engine':baseline['engine'],'optimizer':solver,'warning':solver_warning,
                    'synthetic_data':True,'live_telemetry':False,
                    'disclaimer':'Synthetic educational decision support; not connected to MES/SCADA and not GMP validated.'},
            'factory':factory_spec(scenario),'settings':{'scenario':scenario,'demand_factor':demand_factor,'energy_rate':energy_rate,'seed':seed},
            'baseline':baseline,'optimized':optimized,'predictions':predictions,'ml_metrics':metrics,
            'trials':trials,'economics':economics,
            'checks':{'passed':len(checks)==0,'failures':checks,'test_count':2*(3+4*len(batches))+2*len(STAGES)}}
