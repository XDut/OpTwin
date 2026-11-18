from pathlib import Path
root=Path(__file__).resolve().parent
frontend=root/'frontend'
html=(frontend/'index.html').read_text()
css=(frontend/'style.css').read_text()+'\n'+(frontend/'advanced_style.css').read_text()
modules=['browser_engine.js','advanced_engine.js','advanced_ui.js','app_live.js']
html=html.replace('<link rel="stylesheet" href="/assets/style.css">','<style>\n'+css+'\n</style>')
inline='\n'.join('<script>\n'+(frontend/module).read_text().replace('</script','<\\/script')+'\n</script>' for module in modules)
html=html.replace('<script src="/assets/app.js"></script>',inline)
assert 'src="/assets/' not in html
assert 'href="/assets/' not in html
out=root.parent/'PharmaOps_Twin_Pro_v4_Hybrid_AI_Offline.html'
out.write_text(html,encoding='utf8')
print(out,out.stat().st_size)
