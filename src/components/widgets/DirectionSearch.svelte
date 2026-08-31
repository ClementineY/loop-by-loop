<script lang="ts">
  import { gradientDirectionDelta, randomDirectionDeltas } from '../../lib/optimization';
  let dimension = 10;
  let seed = 7;
  const trials = 120;
  const step = 0.1;
  const low = -0.12;
  const high = 0.12;
  const bins = 20;
  $: deltas = randomDirectionDeltas(+dimension, trials, step, seed);
  $: counts = Array.from({length: bins}, (_, i) => deltas.filter(v => v >= low + i*(high-low)/bins && (i === bins-1 ? v <= high : v < low + (i+1)*(high-low)/bins)).length);
  $: best = Math.min(...deltas);
  $: improved = deltas.filter(v => v < 0).length;
  $: gd = gradientDirectionDelta(step);
  $: maxCount = Math.max(...counts, 1);
  const x = (value:number) => 42 + ((value-low)/(high-low))*476;
</script>

<section class="lab" aria-labelledby="search-title">
  <header><div><span>Direction lab</span><h3 id="search-title">One informed step versus 120 guesses</h3></div><button onclick={() => seed += 1}>New random directions</button></header>
  <div class="dimension" role="group" aria-label="Parameter count">
    {#each [2,10,100,1000] as d}<button class:active={dimension===d} onclick={() => dimension=d}>{d} parameters</button>{/each}
  </div>
  <div class="plot">
    <svg viewBox="0 0 560 230" role="img" aria-label={`Histogram of loss changes from random directions in ${dimension} dimensions`}>
      <line class="axis" x1="42" y1="185" x2="518" y2="185"/><line class="zero" x1={x(0)} y1="25" x2={x(0)} y2="190"/>
      {#each counts as count,i}<rect x={43+i*(476/bins)} y={185-(count/maxCount)*135} width={476/bins-2} height={(count/maxCount)*135}/>{/each}
      <line class="gd" x1={x(gd)} y1="20" x2={x(gd)} y2="190"/><text class="gd-label" x={x(gd)+5} y="32">gradient step {gd.toFixed(3)}</text>
      <text x="42" y="210">loss decreases ←</text><text x="418" y="210">→ loss increases</text><text x={x(0)-10} y="225">0</text>
    </svg>
  </div>
  <div class="readout">
    <article><span>Random moves that helped</span><b>{improved} / {trials}</b><p>Directions landing left of zero</p></article>
    <article><span>Best random move</span><b>{best.toFixed(3)}</b><p>Best of this particular sample</p></article>
    <article class="informed"><span>Gradient move</span><b>{gd.toFixed(3)}</b><p>Uses all partial derivatives once</p></article>
  </div>
  <footer>As dimensions grow, random directions become almost perpendicular to the gradient. The orange line stays put because the gradient deliberately chooses the steepest local descent for this step length.</footer>
</section>

<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}header{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem}header span,.readout span{font:800 .6rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0 0}header button,.dimension button{border:1px solid var(--border);border-radius:7px;background:var(--bg);color:var(--text);padding:.45rem .6rem;font-size:.67rem}.dimension{display:grid;grid-template-columns:repeat(4,1fr);border-block:1px solid var(--border)}.dimension button{border:0;border-right:1px solid var(--border);border-radius:0}.dimension button.active{background:var(--accent);color:white}.plot{padding:1rem;background:#121925}.plot svg{display:block;width:100%}.plot rect{fill:#657286}.axis{stroke:#657286}.zero{stroke:#d7dce4;stroke-dasharray:4}.gd{stroke:var(--accent-light);stroke-width:3}.plot text{fill:#8d99aa;font:10px var(--font-mono)}.plot .gd-label{fill:var(--accent-light);font-weight:700}.readout{display:grid;grid-template-columns:repeat(3,1fr)}.readout article{padding:.85rem;border-right:1px solid var(--border)}.readout article:last-child{border:0}.readout .informed{background:var(--accent-soft)}.readout b{display:block;margin:.2rem 0;font:750 1rem var(--font-mono)}.readout p{margin:0;color:var(--text-muted);font-size:.66rem}footer{padding:.8rem 1rem;border-top:1px solid var(--border);color:var(--text-muted);font-size:.7rem;line-height:1.5}@media(max-width:600px){header{align-items:flex-start;flex-direction:column}.dimension{grid-template-columns:1fr 1fr}.readout{grid-template-columns:1fr}.readout article{border-right:0;border-bottom:1px solid var(--border)}}
</style>
