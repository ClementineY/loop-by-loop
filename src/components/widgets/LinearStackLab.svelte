<script lang="ts">
  let activation = false;
  let w1 = 1.4;
  let b1 = 0.2;
  let w2 = -1.1;
  let b2 = 0.4;
  const relu=(x:number)=>Math.max(0,x);
  const f=(x:number)=>w2*(activation?relu(w1*x+b1):(w1*x+b1))+b2;
  const px=(x:number)=>45+((x+3)/6)*470, py=(y:number)=>145-y*38;
  $: curve=Array.from({length:121},(_,i)=>{const x=-3+i/20;return `${i?'L':'M'} ${px(x).toFixed(1)} ${py(f(x)).toFixed(1)}`}).join(' ');
  $: collapsedW=w2*w1; $: collapsedB=w2*b1+b2;
</script>
<section class="lab" aria-labelledby="stack-title">
  <header><div><span>Composition lab</span><h3 id="stack-title">Can two layers make a bend?</h3></div><div class="toggle"><button class:active={!activation} onclick={()=>activation=false}>Linear only</button><button class:active={activation} onclick={()=>activation=true}>Insert ReLU</button></div></header>
  <div class="formula">
    <article><small>layer 1</small><code>h = {w1.toFixed(1)}x {b1>=0?'+':'−'} {Math.abs(b1).toFixed(1)}</code></article><i>→</i>
    {#if activation}<article class="bend"><small>activation</small><code>h = max(0, h)</code></article><i>→</i>{/if}
    <article><small>layer 2</small><code>y = {w2.toFixed(1)}h {b2>=0?'+':'−'} {Math.abs(b2).toFixed(1)}</code></article>
  </div>
  <div class="plot"><svg viewBox="0 0 560 290" role="img" aria-label={activation?'Piecewise linear function with a ReLU bend':'Straight line produced by two linear layers'}>
    <line class="axis" x1="35" y1="145" x2="525" y2="145"/><line class="axis" x1="280" y1="20" x2="280" y2="270"/><path d={curve}/>
    {#if activation}<circle cx={px(-b1/w1)} cy={py(f(-b1/w1))} r="6"/><text x={px(-b1/w1)+10} y={py(f(-b1/w1))-8}>the bend</text>{/if}
  </svg></div>
  <div class="controls">
    <label>w₁ <b>{w1.toFixed(1)}</b><input type="range" min="-2" max="2" step="0.1" bind:value={w1}/></label>
    <label>b₁ <b>{b1.toFixed(1)}</b><input type="range" min="-1" max="1" step="0.1" bind:value={b1}/></label>
    <label>w₂ <b>{w2.toFixed(1)}</b><input type="range" min="-2" max="2" step="0.1" bind:value={w2}/></label>
    <label>b₂ <b>{b2.toFixed(1)}</b><input type="range" min="-1" max="1" step="0.1" bind:value={b2}/></label>
  </div>
  <footer>{#if activation}<strong>ReLU creates two regions.</strong> Each side is linear, but the network as a whole can now bend.{:else}<strong>The stack collapses:</strong> <code>y = {collapsedW.toFixed(2)}x {collapsedB>=0?'+':'−'} {Math.abs(collapsedB).toFixed(2)}</code>. Moving any slider changes the line, never its straightness.{/if}</footer>
</section>
<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}header{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem}header span{font:800 .6rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0 0}.toggle{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}.toggle button{border:0;background:var(--bg);color:var(--text-muted);padding:.45rem .6rem;font-size:.67rem}.toggle button.active{background:var(--accent);color:white}.formula{display:flex;align-items:center;justify-content:center;gap:.5rem;padding:.8rem;border-block:1px solid var(--border)}.formula article{padding:.5rem .65rem;background:var(--bg);border-radius:7px}.formula .bend{background:var(--accent-soft);border:1px solid var(--accent)}.formula small{display:block;color:var(--text-muted);font:.52rem var(--font-mono)}.formula code{font-size:.67rem}.formula i{color:var(--text-muted)}.plot{background:#121925;padding:.5rem 1rem}.plot svg{display:block;width:100%}.axis{stroke:#455064}.plot path{fill:none;stroke:var(--accent-light);stroke-width:3}.plot circle{fill:var(--accent)}.plot text{fill:var(--accent-light);font:10px var(--font-mono)}.controls{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;padding:.8rem 1rem}.controls label{display:grid;grid-template-columns:1fr auto;color:var(--text-muted);font-size:.65rem}.controls input{grid-column:1/-1;accent-color:var(--accent)}footer{padding:.8rem 1rem;background:var(--accent-soft);border-top:1px solid var(--border);font-size:.72rem}footer code{font-size:.68rem}@media(max-width:650px){header{align-items:flex-start;flex-direction:column}.formula{align-items:stretch;flex-direction:column}.formula>i{transform:rotate(90deg);text-align:center}.controls{grid-template-columns:1fr 1fr}}
</style>
