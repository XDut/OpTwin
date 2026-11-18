"""Original, synthetic OSD internal material-flow network with auditable shortest-path routing.

Routes are illustrative corridors, not a real facility design or GMP material-flow approval.
Dijkstra finds the distance baseline or minimum modeled travel-time path; no RL is claimed.
"""
from __future__ import annotations
import heapq

# Two fictional electric tuggers serve transfers between controlled process stages.
TUGGER_COUNT = 2
# Distances are arbitrary layout units; one unit is not a real-world meter.
# Stage and corridor waypoint nodes (corridors do not imply access to process rooms).
NODE_POS = {
    'warehouse':(80,175), 'dispensing':(240,100), 'granulation':(425,100),
    'blending':(605,100), 'compression':(790,100), 'coating':(790,310),
    'packaging':(605,310), 'qc':(425,310), 'dispatch':(240,310),
    'NW':(235,195),'NC':(490,195),'NE':(780,195),
    'SW':(235,245),'SC':(490,245),'SE':(780,245),
}

LINKS = [
 ('warehouse','NW',15),('dispensing','NW',8),('granulation','NC',10),
 ('blending','NC',8),('compression','NE',8),('coating','SE',8),
 ('packaging','SC',9),('qc','SC',8),('dispatch','SW',8),
 ('NW','NC',22),('NC','NE',22),('NE','SE',24),('SE','SC',22),
 ('SC','SW',22),('SW','NW',27),('NC','SC',29),
 ('NW','SW',27), # retained as a single shared corridor segment, not a second edge
]
# Avoid double-counting an edge in the public visualization.
LINKS = [e for i,e in enumerate(LINKS) if {e[0],e[1]} not in [set(x[:2]) for x in LINKS[:i]]]
GRAPH = {n:[] for n in NODE_POS}
for a,b,d in LINKS:
    GRAPH[a].append((b,d)); GRAPH[b].append((a,d))

# Traffic factors represent modeled cart/people activity in corridor links.
def traffic(a,b,scenario):
    key=frozenset((a,b))
    if key==frozenset(('NW','NC')):
        return 5.0 if scenario in ('demand','supplier') else 1.2
    if key==frozenset(('NE','SE')):
        return 4.5 if scenario in ('demand','dryer') else 1.0
    if key==frozenset(('NC','SC')):
        return 1.4 if scenario=='quality' else 1.0
    return 1.0

def shortest_route(source,destination,scenario='normal',policy='adaptive'):
    """Return route, modeled distance and travel time. All routes remain inside the graph."""
    if source not in GRAPH or destination not in GRAPH:
        raise ValueError('Unknown logistics node')
    if policy not in ('adaptive','distance'):
        raise ValueError('Unknown routing policy')
    heap=[(0,source,(source,))]; best={source:0}
    while heap:
        score,node,path=heapq.heappop(heap)
        if score>best.get(node,float('inf'))+1e-9: continue
        if node==destination:
            distance=0; weighted=0
            for u,v in zip(path,path[1:]):
                length=next(d for nxt,d in GRAPH[u] if nxt==v)
                distance+=length; weighted+=length*traffic(u,v,scenario)
            return {'path':list(path),'distance':distance,'travel_hours':round(.11+.010*weighted,4),
                    'corridor_cost':round(weighted,2)}
        for neighbor,dist in GRAPH[node]:
            extra=dist*(traffic(node,neighbor,scenario) if policy=='adaptive' else 1)
            new=score+extra
            if new<best.get(neighbor,float('inf'))-1e-9:
                best[neighbor]=new; heapq.heappush(heap,(new,neighbor,path+(neighbor,)))
    raise RuntimeError('No route between nodes')

def logistics_summary(moves):
    if not moves:return {'move_count':0,'avg_wait_hours':0,'p95_wait_hours':0,'avg_travel_hours':0,
                         'transport_hours':0,'total_wait_hours':0,'distance_units':0,'max_wait_hours':0}
    waits=sorted(m['queue_hours'] for m in moves)
    p95=waits[max(0,__import__('math').ceil(.95*len(waits))-1)]
    return {'move_count':len(moves),'avg_wait_hours':round(sum(waits)/len(waits),3),
            'p95_wait_hours':round(p95,3),'max_wait_hours':round(max(waits),3),
            'avg_travel_hours':round(sum(m['end']-m['start'] for m in moves)/len(moves),3),
            'total_wait_hours':round(sum(waits),3),
            'transport_hours':round(sum(m['end']-m['start'] for m in moves),3),
            'distance_units':round(sum(m['distance'] for m in moves),1)}
