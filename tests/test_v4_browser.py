"""Tests a standalone HTML through document injection: sandbox blocks file:// navigation."""
from pathlib import Path
from playwright.sync_api import sync_playwright
root=Path(__file__).resolve().parent.parent
html=root.parent/'PharmaOps_Twin_Pro_v4_Hybrid_AI_Offline.html'
assert html.is_file()
with sync_playwright() as p:
  browser=p.chromium.launch(headless=True, executable_path='/usr/bin/chromium', args=['--no-sandbox'])
  page=browser.new_page(viewport={'width':1440,'height':920}, accept_downloads=True)
  errors=[]
  page.on('pageerror', lambda x:errors.append(str(x)))
  page.set_content(html.read_text(), wait_until='load')
  page.locator('#kpiRow .kpi-card').first.wait_for(timeout=40000)
  assert len(page.locator('.zone-room').all())==9
  assert page.locator('[data-tab]').count()==10
  page.screenshot(path=str(root.parent/'PharmaOps_v4_Factory.png'),full_page=True)
  page.locator('[data-tab="maintenance"]').click()
  page.locator('#maintenanceContent .pm-asset').first.wait_for(timeout=30000)
  assert page.locator('.pm-asset').count()==8
  assert 'Predicted remaining useful life' in page.locator('#maintenanceContent').inner_text()
  page.locator('.pm-asset[data-asset="4"]').click()
  assert 'CO-501' in page.locator('#maintenanceContent').inner_text()
  page.screenshot(path=str(root.parent/'PharmaOps_v4_Maintenance.png'),full_page=True)
  with page.expect_download() as d:page.locator('#downloadSensorExample').click()
  assert d.value.suggested_filename.endswith('.csv')
  page.locator('[data-tab="explain"]').click()
  page.locator('#explainAsset').wait_for(timeout=30000)
  page.locator('#explainAsset').select_option('4')
  assert 'Shapley' in page.locator('#explainContent').inner_text()
  page.screenshot(path=str(root.parent/'PharmaOps_v4_Explainability.png'),full_page=True)
  page.locator('[data-tab="rl"]').click()
  assert page.locator('.policy-comparison').count()==3
  page.screenshot(path=str(root.parent/'PharmaOps_v4_PPO.png'),full_page=True)
  page.locator('#applyPpo').click()
  page.wait_for_function('document.querySelector("#runBtn").disabled === false',timeout=30000)
  assert page.locator('#routingPolicy').input_value()=='ppo'
  assert 'TRAINED PPO' in page.locator('#rlContent').inner_text()
  with page.expect_download() as d:page.locator('#exportBtn').click()
  import json
  export=json.loads(Path(d.value.path()).read_text())
  assert export['settings']['routing_policy']=='ppo'
  assert export['checks']['passed'] is True
  assert len(export['optimized']['transports'])==144
  assert export['advanced']['ppo']['training_episodes']>=150
  page.locator('[data-tab="benchmark"]').click()
  assert 'ROC-AUC' in page.locator('#benchmarkContent').inner_text()
  assert page.locator('#benchmarkContent tbody tr').count()>=6
  with page.expect_download() as d:page.locator('#exportBenchmark').click()
  assert d.value.suggested_filename.endswith('.json')
  page.screenshot(path=str(root.parent/'PharmaOps_v4_Benchmarks.png'),full_page=True)
  page.locator('[data-tab="validation"]').click()
  assert 'SimPy: not bundled' in page.locator('#validationContent').inner_text()
  assert 'Pyodide: not bundled' in page.locator('#validationContent').inner_text()
  assert '17' not in page.locator('#validationContent #v4provenance').inner_text()
  page.locator('#scenario').select_option('dryer')
  page.locator('#runBtn').click()
  page.wait_for_function('document.querySelector("#runBtn").disabled === false',timeout=30000)
  page.locator('[data-tab="maintenance"]').click()
  assert 'GR-201' in page.locator('#maintenanceContent').inner_text()
  assert not errors,errors
  mobile=browser.new_page(viewport={'width':390,'height':844})
  merr=[]
  mobile.on('pageerror',lambda x:merr.append(str(x)))
  mobile.set_content(html.read_text(),wait_until='load')
  mobile.locator('#kpiRow .kpi-card').first.wait_for(timeout=40000)
  mobile.locator('[data-tab="maintenance"]').click()
  mobile.locator('#maintenanceContent .pm-asset').first.wait_for(timeout=20000)
  overflow=mobile.evaluate('document.documentElement.scrollWidth-document.documentElement.clientWidth')
  assert overflow==0,f'mobile horizontal overflow = {overflow}'
  mobile.screenshot(path=str(root.parent/'PharmaOps_v4_Mobile.png'),full_page=True)
  assert not merr,merr
  print('PASS: standalone HTML injected without server; 10 tabs; 8 maintenance assets; live SHAP explanations; trained PPO changes real transport; 4 ML models+2 RUL models; exports; validation provenance; responsive mobile; zero page errors')
  browser.close()
