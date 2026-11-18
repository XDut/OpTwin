"""FastAPI facade; start from project root: uvicorn backend.app:app --reload"""
from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel,Field
from .service import run_case,HAS_SIMPY,HAS_ORTOOLS
from .plant import factory_spec
from .predictive import train_models

ROOT=Path(__file__).resolve().parent.parent
FRONTEND=ROOT/'frontend'
app=FastAPI(title='PharmaOps Twin Pro',version='2.0',description='Synthetic pharma plant modeling + optimization')
app.mount('/assets',StaticFiles(directory=str(FRONTEND)),name='assets')

class ScenarioRequest(BaseModel):
    scenario:str='normal'
    demand_factor:float=Field(default=1,ge=.8,le=1.35)
    energy_rate:float=Field(default=.22,ge=.08,le=.8)
    seed:int=Field(default=41,ge=1,le=100000)
    monte_carlo:int=Field(default=8,ge=0,le=20)
    solver_seconds:float=Field(default=8,ge=.5,le=12)

@app.get('/')
def home():return FileResponse(FRONTEND/'index.html')

@app.get('/api/health')
def health():
    _,_,ml=train_models()
    return {'ok':True,'simpy_installed':HAS_SIMPY,'ortools_installed':HAS_ORTOOLS,
            'ml_trained':True,'ml_metrics':ml,'synthetic_data':True,'live_connected':False}

@app.get('/api/factory')
def factory():return factory_spec()

@app.post('/api/run')
def run(req:ScenarioRequest):
    try:return run_case(**req.model_dump() if hasattr(req,'model_dump') else req.dict())
    except ValueError as e: raise HTTPException(status_code=422,detail=str(e))
    except Exception as e: raise HTTPException(status_code=500,detail=f'Model execution failed: {e}')
