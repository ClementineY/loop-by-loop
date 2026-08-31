<script lang="ts">
  import {linearParameterCount} from '../../lib/curriculum';
  let batch=4,inputs=2,hidden=8,outputs=1;
  $: first=linearParameterCount(+inputs,+hidden); $: second=linearParameterCount(+hidden,+outputs); $: total=first+second;
</script>
<section class="lab" aria-labelledby="flow-title">
  <header><div><span>Shape lab</span><h3 id="flow-title">Follow one batch through every layer</h3></div><strong>{total} trainable values</strong></header>
  <div class="pipeline">
    <article><small>input batch</small><b>[{batch}, {inputs}]</b><div>{#each Array(Math.min(+inputs,8)) as _}<i></i>{/each}</div></article><span>Linear({inputs}, {hidden})<em>{first} params</em></span>
    <article><small>hidden features</small><b>[{batch}, {hidden}]</b><div>{#each Array(Math.min(+hidden,12)) as _}<i></i>{/each}</div></article><span>ReLU<em>0 params</em></span>
    <article><small>same shape</small><b>[{batch}, {hidden}]</b><div>{#each Array(Math.min(+hidden,12)) as _}<i class="active"></i>{/each}</div></article><span>Linear({hidden}, {outputs})<em>{second} params</em></span>
    <article><small>logits</small><b>[{batch}, {outputs}]</b><div>{#each Array(Math.min(+outputs,8)) as _}<i></i>{/each}</div></article>
  </div>
  <div class="accounting"><code>({inputs} × {hidden} weights + {hidden} biases) + ({hidden} × {outputs} weights + {outputs} biases) = {total}</code><span>The batch dimension changes how many examples flow through, not how many parameters the model owns.</span></div>
  <div class="controls"><label>batch <b>{batch}</b><input type="range" min="1" max="8" step="1" bind:value={batch}/></label><label>input features <b>{inputs}</b><input type="range" min="1" max="8" step="1" bind:value={inputs}/></label><label>hidden width <b>{hidden}</b><input type="range" min="2" max="12" step="1" bind:value={hidden}/></label><label>outputs <b>{outputs}</b><input type="range" min="1" max="6" step="1" bind:value={outputs}/></label></div>
</section>
<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}header{display:flex;justify-content:space-between;align-items:center;padding:1rem}header span{font:800 .6rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0}header strong{font:700 .75rem var(--font-mono);color:var(--accent)}.pipeline{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;gap:.6rem;align-items:center;padding:1rem;background:#121925;color:white}.pipeline article{padding:.6rem;border:1px solid #394456;border-radius:8px;background:#1b2432}.pipeline small{display:block;color:#8995a7;font:.52rem var(--font-mono)}.pipeline b{font:.7rem var(--font-mono)}.pipeline article div{display:flex;gap:3px;flex-wrap:wrap;margin-top:.5rem}.pipeline article i{width:10px;height:20px;border-radius:2px;background:#667489}.pipeline article i.active{background:var(--accent)}.pipeline>span{font:.56rem var(--font-mono);color:#a4adba;text-align:center}.pipeline em{display:block;color:var(--accent-light);font-style:normal}.accounting{display:flex;justify-content:space-between;gap:1rem;padding:.7rem 1rem;background:var(--accent-soft)}.accounting code{font-size:.63rem}.accounting span{max-width:310px;color:var(--text-muted);font-size:.65rem}.controls{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;padding:.8rem 1rem}.controls label{display:grid;grid-template-columns:1fr auto;color:var(--text-muted);font-size:.64rem}.controls input{grid-column:1/-1;accent-color:var(--accent)}@media(max-width:760px){.pipeline{grid-template-columns:1fr}.pipeline>span{padding:.3rem}.accounting{flex-direction:column}.controls{grid-template-columns:1fr 1fr}}
</style>
