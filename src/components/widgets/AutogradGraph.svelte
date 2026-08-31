<script lang="ts">
  import { V } from '../../lib/autograd';
  let x=2,w=-1,b=.5,backward=false;
  $: graph=build(+x,+w,+b,backward);
  function build(xn:number,wn:number,bn:number,run:boolean){const xv=V(xn),wv=V(wn),bv=V(bn);const mul=xv.mul(wv);const add=mul.add(bv);const y=add.pow(2);if(run)y.backward();return [{name:'x',op:'input',v:xv},{name:'w',op:'input',v:wv},{name:'×',op:'multiply',v:mul},{name:'b',op:'input',v:bv},{name:'+',op:'add',v:add},{name:'y',op:'square',v:y}]}
  function changed(){backward=false}
</script>
<section class="lab" aria-labelledby="auto-title">
  <div class="head"><div><span>Interactive lab</span><h3 id="auto-title">Follow the gradient</h3><code>y = (x·w + b)²</code></div><button onclick={()=>backward=true}>Run backward →</button></div>
  <div class="sliders"><label>x <b>{x}</b><input type="range" min="-3" max="3" step=".5" bind:value={x} oninput={changed}/></label><label>w <b>{w}</b><input type="range" min="-3" max="3" step=".5" bind:value={w} oninput={changed}/></label><label>b <b>{b}</b><input type="range" min="-2" max="2" step=".5" bind:value={b} oninput={changed}/></label></div>
  <div class="graph" aria-label="Computation graph">
    {#each graph as node,i}<article class:output={node.name==='y'}><small>{node.op}</small><strong>{node.name}</strong><span>value <b>{node.v.data.toFixed(2)}</b></span><span class:lit={backward}>grad <b>{backward?node.v.grad.toFixed(2):'—'}</b></span></article>{#if i<graph.length-1}<i aria-hidden="true">→</i>{/if}{/each}
  </div>
  <p class="caption">{backward?'Gradients accumulate from right to left. Each node only needs its local derivative.':'Change an input, predict which gradients will change, then run backward.'}</p>
</section>
<style>
  .lab{margin:2rem 0;padding:1rem;border:1px solid var(--border);border-radius:14px;background:var(--surface);overflow:hidden}.head{display:flex;justify-content:space-between;align-items:center;gap:1rem}.head span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}.head h3{margin:.2rem 0}.head code{font-size:.72rem;color:var(--text-muted)}.head button{border:0;border-radius:8px;background:var(--accent);color:white;padding:.55rem .7rem;font-weight:750;font-size:.75rem}.sliders{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0}.sliders label{display:grid;grid-template-columns:1fr auto;font-size:.7rem;color:var(--text-muted)}.sliders input{grid-column:1/-1;accent-color:var(--accent)}.graph{display:flex;align-items:center;gap:.45rem;padding:1rem;background:var(--ink);border-radius:10px;overflow:auto}.graph article{flex:0 0 83px;border:1px solid #3a4556;background:#1b2330;border-radius:8px;padding:.55rem;color:white}.graph article.output{border-color:var(--accent);background:#36241f}.graph small{display:block;color:#8994a5;font:.55rem var(--font-mono)}.graph strong{display:block;font:800 1.1rem var(--font-mono);margin:.25rem 0}.graph span{display:flex;justify-content:space-between;color:#9ca6b5;font-size:.58rem}.graph span b{color:white;font-family:var(--font-mono)}.graph span.lit b{color:var(--accent-light)}.graph>i{color:#6f7a8d;font-style:normal}.caption{color:var(--text-muted);font-size:.75rem;margin:.7rem .2rem 0}@media(max-width:520px){.sliders{grid-template-columns:1fr}.head{align-items:flex-start}.head code{display:none}}
</style>
