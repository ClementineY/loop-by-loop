<script lang="ts">
  import { optimizerPath } from '../../lib/optimization';
  let lr = 0.05;
  let beta = 0.8;
  let step = 6;
  const total = 24;
  const start:[number,number] = [4,2];
  $: plain = optimizerPath(start,total,+lr,0,12);
  $: momentum = optimizerPath(start,total,+lr,+beta,12);
  $: current = momentum[Math.min(step,total)];
  const sx=(x:number)=>280+x*55, sy=(y:number)=>145-y*55;
  const path=(states:ReturnType<typeof optimizerPath>)=>states.map((s,i)=>`${i?'L':'M'} ${sx(s.point[0]).toFixed(1)} ${sy(s.point[1]).toFixed(1)}`).join(' ');
  $: momentumPath=path(momentum.slice(0, Math.min(step,total)+1));
  $: plainPath=path(plain.slice(0, Math.min(step,total)+1));
</script>
<section class="lab" aria-labelledby="momentum-title">
  <header><div><span>Memory lab</span><h3 id="momentum-title">Momentum remembers consistent directions</h3></div><button onclick={() => step=Math.min(total,step+1)}>Next step →</button></header>
  <div class="plot">
    <svg viewBox="0 0 560 290" role="img" aria-label="Gradient descent and momentum paths through a narrow loss valley">
      {#each [220,170,125,85,50] as rx,i}<ellipse cx="280" cy="145" rx={rx} ry={rx/12} class={`ring r${i}`}/>{/each}
      <line class="axis" x1="30" y1="145" x2="530" y2="145"/><line class="axis" x1="280" y1="25" x2="280" y2="265"/>
      <path class="plain" d={plainPath}/><path class="momentum" d={momentumPath}/>
      <circle class="plain-dot" cx={sx(plain[Math.min(step,total)].point[0])} cy={sy(plain[Math.min(step,total)].point[1])} r="5"/>
      <circle class="momentum-dot" cx={sx(current.point[0])} cy={sy(current.point[1])} r="7"/>
      <text x="35" y="25">steep across the valley</text><text x="375" y="276">shallow along the valley</text>
    </svg>
  </div>
  <div class="controls">
    <label>Learning rate <b>{(+lr).toFixed(3)}</b><input type="range" min="0.01" max="0.12" step="0.005" bind:value={lr} oninput={() => step=0}/></label>
    <label>Momentum β <b>{(+beta).toFixed(2)}</b><input type="range" min="0" max="0.95" step="0.05" bind:value={beta} oninput={() => step=0}/></label>
    <label>Inspect step <b>{step}</b><input type="range" min="0" max={total} step="1" bind:value={step}/></label>
  </div>
  <div class="readout">
    <article><span>Current gradient</span><b>[{current.gradient[0].toFixed(2)}, {current.gradient[1].toFixed(2)}]</b><p>What the surface says now</p></article>
    <article><span>Stored velocity</span><b>[{current.velocity[0].toFixed(2)}, {current.velocity[1].toFixed(2)}]</b><p>Past gradients carried forward</p></article>
    <article><span>Current loss</span><b>{current.loss.toFixed(4)}</b><p>Orange path at step {step}</p></article>
  </div>
  <footer><i></i> momentum <i></i> plain gradient descent. Across the narrow direction, signs alternate and cancel; along the valley, consistent gradients accumulate speed.</footer>
</section>
<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}header{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem}header span,.readout span{font:800 .6rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0 0}header button{border:0;border-radius:8px;background:var(--accent);color:white;padding:.55rem .7rem;font-weight:750}.plot{padding:.5rem 1rem;background:#121925}.plot svg{display:block;width:100%}.ring{fill:none;stroke:#313c4d}.axis{stroke:#3f4a5c}.plain,.momentum{fill:none;stroke-width:2}.plain{stroke:#8a96a7;stroke-dasharray:4}.momentum{stroke:var(--accent-light);stroke-width:3}.plain-dot{fill:#aab3c1}.momentum-dot{fill:var(--accent);stroke:#ffd0bf;stroke-width:2}.plot text{fill:#8995a7;font:10px var(--font-mono)}.controls{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;padding:.8rem 1rem;border-bottom:1px solid var(--border)}.controls label{display:grid;grid-template-columns:1fr auto;color:var(--text-muted);font-size:.67rem}.controls input{grid-column:1/-1;accent-color:var(--accent)}.readout{display:grid;grid-template-columns:repeat(3,1fr)}.readout article{padding:.8rem;border-right:1px solid var(--border)}.readout article:last-child{border:0}.readout b{display:block;margin:.2rem 0;font:700 .85rem var(--font-mono)}.readout p,footer{margin:0;color:var(--text-muted);font-size:.65rem}footer{padding:.7rem 1rem;border-top:1px solid var(--border)}footer i{display:inline-block;width:18px;border-top:3px solid var(--accent);margin:0 .3rem}footer i+ i{border-color:#8a96a7;border-top-style:dashed}@media(max-width:620px){header{align-items:flex-start;flex-direction:column}.controls,.readout{grid-template-columns:1fr}.readout article{border-right:0;border-bottom:1px solid var(--border)}}
</style>
