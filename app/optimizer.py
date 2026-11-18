"""Real OR-Tools CP-SAT multi-stage interval scheduling.

A solution assigns planned start/end slots to every batch at every station.
Per-stage cumulative resources model shared machines; precedence enforces GMP-like
sequential routing (without claiming validated GMP process constraints).
"""
from __future__ import annotations
from .plant import STAGES, CAPACITY, active_capacity, BASE_HOURS, scenario_multiplier, material_release

SLOT_HOURS=.25

def durations(batches,predictions,scenario):
    out={}
    for i,b in enumerate(batches):
        scale=(b.kilograms/300)**.48 * b.complexity * predictions[b.id]['duration_multiplier']
        for st in STAGES:
            out[(b.id,st)]=max(.35, BASE_HOURS[st]*scale*scenario_multiplier(st,scenario,i)+.35)
    return out

def make_baseline(batches):
    return {b.id:{'release':float(i)*.11,'stages':{}} for i,b in enumerate(batches)}

def optimize_cp_sat(batches,predictions,scenario,time_limit=4):
    from ortools.sat.python import cp_model
    model=cp_model.CpModel()
    dur=durations(batches,predictions,scenario)
    horizon=1200   # 300 hours at 15 min per slot
    starts={}; ends={}; intervals_by_stage={st:[] for st in STAGES}
    delays=[]; finishes=[]
    for i,b in enumerate(batches):
        previous=None
        for stage in STAGES:
            key=(b.id,stage)
            ticks=max(1,round(dur[key]/SLOT_HOURS))
            start=model.NewIntVar(0,horizon,f's_{i}_{stage}')
            end=model.NewIntVar(0,horizon+ticks,f'e_{i}_{stage}')
            interval=model.NewIntervalVar(start,ticks,end,f'i_{i}_{stage}')
            starts[key]=start;ends[key]=end
            intervals_by_stage[stage].append(interval)
            if previous is not None:
                model.Add(start>=previous)
            else:
                model.Add(start>=int(round(material_release(scenario,i)/SLOT_HOURS)))
            previous=end
        lateness=model.NewIntVar(0,horizon+200,f'late_{i}')
        due=round(b.due_hour/SLOT_HOURS)
        model.AddMaxEquality(lateness,[0,previous-due])
        delays.append(b.priority*lateness)
        finishes.append(previous)
    for st in STAGES:
        model.AddCumulative(intervals_by_stage[st],[1]*len(batches),active_capacity(st,scenario))
    # Weighted tardiness primary, cycle time secondary.
    model.Minimize(30*sum(delays)+sum(finishes))
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=time_limit
    solver.parameters.num_search_workers=4
    solver.parameters.random_seed=43
    status=solver.Solve(model)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        raise RuntimeError(f'CP-SAT returned no feasible schedule: {solver.StatusName(status)}')
    plan={}
    for i,b in enumerate(batches):
        times={st:round(float(solver.Value(starts[(b.id,st)])*SLOT_HOURS),2) for st in STAGES}
        plan[b.id]={'release':times['dispensing'],'stages':times}
    return plan,{'solver':'Google OR-Tools CP-SAT','status':solver.StatusName(status),
                 'objective_value':round(float(solver.ObjectiveValue()),2),
                 'best_bound':round(float(solver.BestObjectiveBound()),2),
                 'wall_seconds':round(float(solver.WallTime()),3),'optimal':status==cp_model.OPTIMAL}

def optimize_greedy_preview(batches,predictions,scenario):
    """Explicitly marked preview fallback used only if OR-Tools isn't installed."""
    dur=durations(batches,predictions,scenario)
    resources={s:[0.0]*active_capacity(s,scenario) for s in STAGES}
    plan={}
    for i,b in sorted(enumerate(batches), key=lambda ib:ib[1].due_hour):
        previous=material_release(scenario,i)
        times={}
        for st in STAGES:
            slot=min(range(len(resources[st])),key=lambda j:resources[st][j])
            start=max(previous,resources[st][slot]); times[st]=round(start,2)
            previous=start+dur[(b.id,st)]
            resources[st][slot]=previous
        plan[b.id]={'release':times['dispensing'],'stages':times}
    return plan,{'solver':'Greedy fallback / OR-Tools unavailable','status':'PREVIEW ONLY',
                 'objective_value':None,'best_bound':None,'wall_seconds':0,'optimal':False}
