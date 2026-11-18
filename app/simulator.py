"""Batch-flow simulation with a SimPy engine and a clearly marked preview fallback."""
from __future__ import annotations
import random
import math
import importlib.util
from .plant import STAGES, CAPACITY, active_capacity, POWER_KW, material_release
from .optimizer import durations
from .logistics import TUGGER_COUNT,shortest_route,logistics_summary

HORIZON=168.0

def _runtime_duration(base,seed,i,k,scenario,stage):
    rng=random.Random(seed*1000003+i*9176+k*2791+307)
    jitter=max(.76,min(1.37,1+rng.gauss(0,.10)))
    # Scenario effects already appear in the nominal duration calculation.
    return max(.20,base*jitter)

def simulate_simpy(batches,preds,scenario,plan,seed=40,energy_rate=.22):
    import simpy
    env=simpy.Environment()
    resources={s:simpy.Resource(env,capacity=active_capacity(s,scenario)) for s in STAGES}
    machine_ids={s:simpy.Store(env,capacity=active_capacity(s,scenario)) for s in STAGES}
    for st in STAGES:
        for machine_id in range(active_capacity(st,scenario)):machine_ids[st].items.append(machine_id)
    d=durations(batches,preds,scenario)
    events=[]; moves=[]; completions={}
    tugger_resource=simpy.Resource(env,capacity=TUGGER_COUNT)
    tugger_ids=simpy.Store(env,capacity=TUGGER_COUNT)
    for tug_id in range(TUGGER_COUNT):tugger_ids.items.append(tug_id)
    def journey(i,b):
        release=max(plan[b.id]['release'],material_release(scenario,i))
        yield env.timeout(max(0,release-env.now))
        for k,st in enumerate(STAGES):
            if k:
                source=STAGES[k-1]
                route=shortest_route(source,st,scenario,policy="adaptive" if plan[b.id].get("routing") else "distance")
                queued=env.now
                with tugger_resource.request() as claim:
                    yield claim
                    tug=yield tugger_ids.get()
                    start_move=env.now
                    yield env.timeout(route["travel_hours"])
                    moves.append({"batch_id":b.id,"sku":b.sku,"source":source,"destination":st,
                                  "start":round(start_move,3),"end":round(env.now,3),
                                  "queue_hours":round(start_move-queued,3),"vehicle":tug,
                                  "path":route["path"],"distance":route["distance"]})
                    yield tugger_ids.put(tug)
            desired=plan[b.id]['stages'].get(st,0)
            if env.now<desired:
                yield env.timeout(desired-env.now)
            queued=env.now
            with resources[st].request() as req:
                yield req
                machine_id=yield machine_ids[st].get()
                start=env.now
                duration=_runtime_duration(d[(b.id,st)],seed,i,k,scenario,st)
                yield env.timeout(duration)
                events.append({'batch_id':b.id,'sku':b.sku,'product':b.product,
                               'stage':st,'start':round(start,3),'end':round(env.now,3),
                               'queue_hours':round(start-queued,3),'machine':machine_id,
                               'duration':round(duration,3)})
                yield machine_ids[st].put(machine_id)
        completions[b.id]=round(env.now,3)
    for i,b in enumerate(batches): env.process(journey(i,b))
    env.run(until=360)
    return summarize(batches,scenario,events,completions,energy_rate,'SimPy discrete-event simulation',moves)

def simulate_preview(batches,preds,scenario,plan,seed=40,energy_rate=.22):
    """Offline fallback: chronological event queue with shared machines and tuggers.

    This is NOT SimPy. Events are dispatched by earliest ready time; each job
    blocks its assigned resource until completion. Native SimPy is used when available.
    """
    import heapq
    d=durations(batches,preds,scenario)
    stage_available={st:[0.0]*active_capacity(st,scenario) for st in STAGES}
    tug_available=[0.0]*TUGGER_COUNT
    events=[];moves=[];completions={};queue=[]
    sequence=0
    def push(ready,i,k,kind):
        nonlocal sequence
        heapq.heappush(queue,(ready,sequence,i,k,kind))
        sequence+=1
    for i,b in enumerate(batches):
        first=max(plan[b.id]['release'],material_release(scenario,i),
                  plan[b.id]['stages'].get(STAGES[0],0))
        push(first,i,0,'stage')
    while queue:
        ready,_,i,k,kind=heapq.heappop(queue)
        b=batches[i]; st=STAGES[k]
        if kind=='move':
            source=STAGES[k-1]
            route=shortest_route(source,st,scenario,policy='adaptive' if plan[b.id].get('routing') else 'distance')
            tug=min(range(TUGGER_COUNT),key=lambda j:tug_available[j])
            start=max(ready,tug_available[tug]);end=start+route['travel_hours']
            moves.append({'batch_id':b.id,'sku':b.sku,'source':source,'destination':st,
                          'start':round(start,3),'end':round(end,3),
                          'queue_hours':round(start-ready,3),'vehicle':tug,
                          'path':route['path'],'distance':route['distance']})
            tug_available[tug]=end
            push(max(end,plan[b.id]['stages'].get(st,0)),i,k,'stage')
            continue
        desired=max(ready,plan[b.id]['stages'].get(st,0))
        slot=min(range(len(stage_available[st])),key=lambda j:stage_available[st][j])
        start=max(desired,stage_available[st][slot])
        duration=_runtime_duration(d[(b.id,st)],seed,i,k,scenario,st)
        end=start+duration
        events.append({'batch_id':b.id,'sku':b.sku,'product':b.product,'stage':st,
                       'start':round(start,3),'end':round(end,3),
                       'queue_hours':round(start-desired,3),'machine':slot,
                       'duration':round(duration,3)})
        stage_available[st][slot]=end
        if k+1<len(STAGES):push(end,i,k+1,'move')
        else:completions[b.id]=round(end,3)
    return summarize(batches,scenario,events,completions,energy_rate,
                     'Preview chronological scheduler / SimPy unavailable',moves)

def summarize(batches,scenario,events,completions,energy_rate,engine,moves):
    ev=sorted(events,key=lambda e:(e['start'],e['stage']))
    finished=[b for b in batches if completions.get(b.id,float('inf'))<=HORIZON]
    ontime=[b for b in finished if completions[b.id]<=b.due_hour]
    produced=sum(b.kilograms for b in finished)
    revenue=sum(b.kilograms*b.price_per_kg for b in finished)
    materials=sum(b.kilograms*b.material_cost_per_kg for b in finished)
    energy=sum(max(0,min(HORIZON,e['end'])-min(HORIZON,e['start']))*POWER_KW[e['stage']]*energy_rate for e in ev if e['start']<HORIZON)
    labor=sum(max(0,min(HORIZON,e['end'])-min(HORIZON,e['start']))*118 for e in ev if e['start']<HORIZON)
    overdue=sum( max(0,min(HORIZON,completions.get(b.id,HORIZON))-b.due_hour)*b.priority*130 for b in batches )
    # The illustrative fixed overhead is shared between the two strategies.
    overhead=26800
    # Synthetic cost of running two internal material-handling tuggers (operator + power).
    handling_cost=round(sum(m["end"]-m["start"] for m in moves)*30,2)
    contribution=revenue-materials-energy-labor-overdue-overhead-handling_cost
    utilization={}
    for st in STAGES:
        busy=sum(max(0,min(HORIZON,e['end'])-max(0,e['start'])) for e in ev if e['stage']==st)
        utilization[st]=round(100*busy/(HORIZON*active_capacity(st,scenario)),1)
    zones={st:{'completed':len([e for e in ev if e['stage']==st and e['end']<=HORIZON]),
               'utilization':utilization[st]} for st in STAGES}
    lot_info=[]
    for b in batches:
        ct=completions.get(b.id)
        lot_info.append({'id':b.id,'sku':b.sku,'product':b.product,'kilograms':b.kilograms,
                         'due':b.due_hour,'completed':ct,'on_time':bool(ct and ct<=b.due_hour),
                         'released':bool(ct and ct<=HORIZON),'priority':b.priority})
    kpis={'throughput_kg':produced,'batches_released':len(finished),
          'batches_total':len(batches),'service_pct':round(100*len(ontime)/len(batches),1),
          'avg_utilization_pct':round(sum(utilization.values())/len(utilization),1),
          'revenue':round(revenue,2),'material_cost':round(materials,2),
          'energy_cost':round(energy,2),'labor_cost':round(labor,2),
          'late_penalties':round(overdue,2),'fixed_overhead':overhead,
          'handling_cost':handling_cost,
          'contribution':round(contribution,2),'power_kwh':round(energy/energy_rate,1),
          'queued_hours':round(sum(e['queue_hours'] for e in ev),1),
          'transport_queue_hours':round(sum(m['queue_hours'] for m in moves),1),
          'max_completion_hour':round(max(completions.values(),default=0),1)}
    return {'engine':engine,'kpis':kpis,'events':ev,'transports':sorted(moves,key=lambda m:m['start']),
            'logistics':logistics_summary(moves),'zones':zones,'lots':lot_info,'horizon_hours':HORIZON}

def simulated_consistency(result,batches):
    """Checks invariants on either simulator output."""
    issues=[]; k=result['kpis'];events=result['events']
    id_set={b.id for b in batches}
    moves=result.get('transports',[])
    for b in batches:
        transfers=sorted([m for m in moves if m['batch_id']==b.id],key=lambda m:m['start'])
        if len(transfers)!=len(STAGES)-1:issues.append('Missing material transfers '+b.id)
        for move in transfers:
            if move['end']<move['start'] or move['queue_hours'] < -1e-6:
                issues.append('Invalid transport event '+b.id)
        manufacturing=sorted([e for e in events if e['batch_id']==b.id],key=lambda e:e['start'])
        for i in range(min(len(transfers),len(manufacturing)-1)):
            if transfers[i]['start']+0.011<manufacturing[i]['end'] or transfers[i]['end']>manufacturing[i+1]['start']+.011:
                issues.append('Transport precedes production or overlaps next step '+b.id)
    for tug in range(TUGGER_COUNT):
        vehicle_moves=sorted([m for m in moves if m['vehicle']==tug],key=lambda m:m['start'])
        if any(vehicle_moves[i]['end']>vehicle_moves[i+1]['start']+.011 for i in range(len(vehicle_moves)-1)):
            issues.append('Transport vehicle double booked '+str(tug))
    if any(e['batch_id'] not in id_set or e['end']<e['start'] for e in events):issues.append('Invalid event identity/time')
    if k['batches_released']>len(batches) or k['throughput_kg']<0:issues.append('Impossible batch throughput')
    for b in batches:
        path=sorted([e for e in events if e['batch_id']==b.id],key=lambda e:e['start'])
        if [e['stage'] for e in path]!=STAGES: issues.append('Process stage order '+b.id)
        if any(path[i]['end']>path[i+1]['start']+0.01 for i in range(len(path)-1)):
            issues.append('Overlapping batch stages '+b.id)
    if abs(k['contribution']-(k['revenue']-k['material_cost']-k['energy_cost']-k['labor_cost']-k['late_penalties']-k['fixed_overhead']-k['handling_cost']))>.1:
        issues.append('Financial reconciliation')
    return issues
