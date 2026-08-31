<script lang="ts">
  import { mean } from '../../lib/optimization';
  const gradients=[-6,-4,-2,-1,0,1,3,4,5,6,8,10];
  let batchSize=2;
  let offset=0;
  $: ordered=gradients.map((_,i)=>gradients[(i+offset)%gradients.length]);
  $: batch=ordered.slice(0,+batchSize);
  $: estimate=mean(batch);
  $: full=mean(gradients);
  const pos=(v:number)=>8+((v+7)/18)*84;
</script>
<section class="lab" aria-labelledby="batch-title">
  <header><div><span>Gradient-estimate lab</span><h3 id="batch-title">How many examples should vote on one update?</h3></div><button onclick={()=>offset=(offset+3)%gradients.length}>Shuffle examples</button></header>
  <div class="votes">
    {#each ordered as gradient,i}<div class:included={i<batchSize}><small>example {i+1}</small><i style={`left:${pos(gradient)}%`}></i><b>{gradient>0?'+':''}{gradient}</b></div>{/each}
  </div>
  <div class="axis"><span>← decrease</span><i style={`left:${pos(0)}%`}></i><span>increase →</span></div>
  <div class="controls"><label>Batch size <b>{batchSize}</b><input type="range" min="1" max="12" step="1" bind:value={batchSize}/></label><article><small>batch estimate</small><b>{estimate>=0?'+':''}{estimate.toFixed(2)}</b></article><article><small>full-data gradient</small><b>+{full.toFixed(2)}</b></article><article><small>estimation error</small><b>{Math.abs(estimate-full).toFixed(2)}</b></article></div>
  <footer>Outlined examples are waiting; orange examples vote in this batch. Larger batches usually reduce noise, but require more work before each parameter update.</footer>
</section>
<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}header{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem}header span{font:800 .6rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0 0}header button{border:1px solid var(--border);border-radius:7px;background:var(--bg);color:var(--text);padding:.45rem .6rem}.votes{padding:1rem;background:#121925}.votes>div{position:relative;height:25px;border-bottom:1px solid #2c3544;color:#7f8a9c}.votes small{position:absolute;left:0;font:9px var(--font-mono)}.votes i{position:absolute;top:4px;width:7px;height:7px;border:1px solid #6f7b8e;border-radius:50%}.votes b{position:absolute;right:0;font:10px var(--font-mono)}.votes .included{color:white}.votes .included i{background:var(--accent);border-color:var(--accent-light);box-shadow:0 0 0 3px #df5b3533}.axis{position:relative;display:flex;justify-content:space-between;padding:.5rem 1rem;color:var(--text-muted);font-size:.6rem}.axis i{position:absolute;height:100%;top:0;border-left:1px dashed var(--border)}.controls{display:grid;grid-template-columns:1.4fr repeat(3,1fr);gap:1rem;padding:.8rem 1rem;border-top:1px solid var(--border)}.controls label{display:grid;grid-template-columns:1fr auto;color:var(--text-muted);font-size:.67rem}.controls input{grid-column:1/-1;accent-color:var(--accent)}.controls article{border-left:1px solid var(--border);padding-left:.8rem}.controls small{display:block;color:var(--text-muted);font-size:.55rem}.controls article b{font:700 .86rem var(--font-mono)}footer{padding:.7rem 1rem;border-top:1px solid var(--border);color:var(--text-muted);font-size:.68rem}@media(max-width:620px){header{align-items:flex-start;flex-direction:column}.controls{grid-template-columns:1fr 1fr}.controls label{grid-column:1/-1}.votes i{max-left:90%}}
</style>
