<script lang="ts">
  import { onDestroy } from 'svelte';
  let lr=0.12,p=3.8; let timer:number|undefined;
  $: loss=p*p; $: gradient=2*p; $: update=(+lr)*gradient; $: nextP=p-update; $: nextLoss=nextP*nextP;
  function step(){p=nextP;if(!Number.isFinite(p)||Math.abs(p)>100)stop()}
  function reset(){stop();p=3.8}
  function run(){if(timer)return;timer=window.setInterval(step,180)}
  function stop(){if(timer){clearInterval(timer);timer=undefined}}
  onDestroy(stop);
</script>
<section class="lab" aria-labelledby="gd-title">
  <header><div><span>Step-size lab</span><h3 id="gd-title">Predict one update before taking it</h3></div><div class="buttons"><button onclick={step}>Take step</button><button class="run" onclick={timer?stop:run}>{timer?'Pause':'Run'}</button><button onclick={reset}>Reset</button></div></header>
  <div class="landscape"><div class="parabola"></div><div class="ball" style={`left:${Math.max(2,Math.min(98,(p+5)*10))}%;bottom:${Math.min(84,10+loss*3)}%`}><i></i><span>p={p.toFixed(3)}<br/>L={loss.toFixed(3)}</span></div><div class="axis"></div></div>
  <div class="control"><label>Learning rate η <b>{(+lr).toFixed(2)}</b><input aria-label="Learning rate" type="range" min="0.01" max="1.1" step="0.01" bind:value={lr}/></label><div class:danger={+lr>=1}>{+lr<.06?'Converges slowly':+lr===1?'Flips forever':+lr>1?'Diverges':'Converges'}</div></div>
  <div class="arithmetic">
    <article><small>gradient 2p</small><b>{gradient.toFixed(3)}</b></article><i>× η →</i>
    <article><small>update η · gradient</small><b>{update.toFixed(3)}</b></article><i>subtract →</i>
    <article><small>next parameter</small><b>{nextP.toFixed(3)}</b></article><i>gives →</i>
    <article><small>next loss</small><b>{nextLoss.toFixed(3)}</b></article>
  </div>
  <footer><code>{nextP.toFixed(3)} = {p.toFixed(3)} − {(+lr).toFixed(2)} × {gradient.toFixed(3)}</code><span>{nextLoss<loss?'This step lowers loss.':nextLoss===loss?'This step preserves loss and flips sides.':'This step raises loss.'}</span></footer>
</section>
<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}header{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem}header span{font:800 .6rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0}.buttons{display:flex;gap:.35rem}.buttons button{border:1px solid var(--border);border-radius:7px;background:var(--bg);padding:.4rem .55rem;color:var(--text);font-size:.68rem}.buttons .run{background:var(--accent);color:white;border-color:var(--accent)}.landscape{height:250px;position:relative;overflow:hidden;background:var(--ink)}.parabola{position:absolute;width:95%;height:420px;border:3px solid #626d7e;border-top:0;border-radius:0 0 50% 50%;left:2.5%;top:-250px}.axis{position:absolute;bottom:9%;left:4%;right:4%;border-top:1px solid #465064}.ball{position:absolute;transform:translateX(-50%);transition:.16s ease}.ball i{display:block;width:18px;height:18px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 7px #df5b3533}.ball span{position:absolute;white-space:nowrap;left:25px;top:-8px;color:white;font:.65rem var(--font-mono)}.control{display:grid;grid-template-columns:1fr auto;gap:1rem;align-items:end;padding:.8rem 1rem;border-bottom:1px solid var(--border)}.control label{display:grid;grid-template-columns:1fr auto;color:var(--text-muted);font-size:.68rem}.control input{grid-column:1/-1;accent-color:var(--accent)}.control>div{padding:.45rem .6rem;background:#e9f5e8;color:#2d6730;border-radius:7px;font-size:.67rem;font-weight:750}.control>div.danger{background:#fde7e2;color:#a23f2c}.arithmetic{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;align-items:center;padding:.8rem 1rem}.arithmetic article{padding:.4rem .6rem}.arithmetic small{display:block;color:var(--text-muted);font-size:.54rem}.arithmetic b{font:700 .82rem var(--font-mono)}.arithmetic>i{font:normal .55rem var(--font-mono);color:var(--text-muted)}footer{display:flex;justify-content:space-between;gap:1rem;padding:.7rem 1rem;background:var(--accent-soft);border-top:1px solid var(--border);font-size:.68rem}footer code{font-size:.67rem}@media(max-width:650px){header{align-items:flex-start;flex-direction:column}.control{grid-template-columns:1fr}.arithmetic{grid-template-columns:1fr}.arithmetic>i{transform:rotate(90deg);text-align:center}footer{flex-direction:column}}
</style>
