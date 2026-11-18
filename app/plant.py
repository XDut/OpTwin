"""Synthetic pharmaceutical plant definition; no regulated process parameters or real records."""
from __future__ import annotations
from dataclasses import dataclass, asdict

STAGES = ['dispensing','granulation','blending','compression','coating','packaging','qc']
STAGE_LABEL = {'dispensing':'Dispensing','granulation':'Wet granulation','blending':'Final blending','compression':'Tablet compression','coating':'Film coating','packaging':'Blister packaging','qc':'QC release'}
CAPACITY = {'dispensing':1,'granulation':2,'blending':1,'compression':2,'coating':1,'packaging':2,'qc':1}
BASE_HOURS = {'dispensing':1.35,'granulation':4.4,'blending':1.6,'compression':5.1,'coating':4.3,'packaging':3.2,'qc':2.6}
POWER_KW = {'dispensing':4,'granulation':27,'blending':9,'compression':23,'coating':32,'packaging':18,'qc':6}
def active_capacity(stage, scenario='normal'):
    return 1 if scenario == 'dryer' and stage == 'granulation' else CAPACITY[stage]

ZONE_IDS = {'warehouse':'RAW MATERIALS','dispensing':'DISPENSING','granulation':'GRANULATION','blending':'BLENDING','compression':'COMPRESSION','coating':'COATING','packaging':'PACKAGING','qc':'QC LAB','dispatch':'DISPATCH'}
SCENARIOS = {
  'normal': {'name':'Normal operations','detail':'Standard demand and equipment conditions'},
  'demand': {'name':'Demand surge','detail':'Tighter order deadlines and increased line pressure'},
  'dryer': {'name':'Granulator outage','detail':'Higher downtime on granulation assets'},
  'quality': {'name':'QC hold','detail':'Additional testing time for selected batches'},
  'supplier': {'name':'Material delay','detail':'Selected batches released later due to material availability'},
  'energy': {'name':'Energy price spike','detail':'Higher electricity rates across the site'},
}

@dataclass
class Batch:
    id: str
    product: str
    sku: str
    kilograms: int
    due_hour: float
    priority: int
    price_per_kg: int
    material_cost_per_kg: int
    complexity: float
    equipment_health: float
    humidity: float
    prior_load: float
    operator_skill: float

PRODUCTS = [
 ('Aster 10 mg','AS-010',235,92,0.94),
 ('Navira 20 mg','NV-020',275,113,1.08),
 ('Serava XR 50 mg','SR-050',320,147,1.24),
 ('Meridian 5 mg','MD-005',255,103,1.00),
]

def build_batches(scenario='normal', demand_factor=1.0):
    products = [0,1,2,3,0,2,1,3,0,1,2,3] * 2
    result = []
    for i, pi in enumerate(products):
        product, sku, price, materials, complexity = PRODUCTS[pi]
        kgs = int((260 + (i % 4)*32 + pi*18) * demand_factor)
        due = [58,93,44,121,76,38,104,88,59,98,72,50][i % 12] + (i//12)*83
        if scenario == 'demand': due -= 18 + (i % 4)*2
        result.append(Batch(
            id=f'BT26-{5101+i}', product=product, sku=sku, kilograms=kgs,
            due_hour=float(max(18, due)), priority=1+(i%4), price_per_kg=price,
            material_cost_per_kg=materials, complexity=complexity,
            equipment_health=round(.96-(i%5)*.071,3),
            humidity=float(42+(i*7)%28), prior_load=round(.7+(i%5)*.19,2),
            operator_skill=round(.62+(i%4)*.105,2)
        ))
    return result

def scenario_multiplier(stage, scenario, batch_index):
    if scenario == 'dryer' and stage == 'granulation' and batch_index % 3 == 1:
        return 1.58
    if scenario == 'quality' and stage == 'qc' and batch_index % 3 == 0:
        return 2.20
    return 1.0

def material_release(scenario, batch_index):
    return 17.5 if scenario == 'supplier' and batch_index in (1,4,7) else 0.0

def factory_spec(scenario='normal'):
    from .logistics import NODE_POS,LINKS,TUGGER_COUNT
    return {'facility':'North Campus · Oral Solid Dosage Unit (fictional)',
            'stages':[{'id':s,'name':STAGE_LABEL[s],'machines':CAPACITY[s], 'available_assets':active_capacity(s,scenario), 'nominal_hours':BASE_HOURS[s]} for s in STAGES],
            'scenarios':SCENARIOS, 'products':[{'name':p[0],'code':p[1]} for p in PRODUCTS],
            'logistics_network':{'nodes':NODE_POS,'links':LINKS,'vehicles':TUGGER_COUNT},
            'horizon_hours':168,'units':'kg, hours, illustrative USD'}
