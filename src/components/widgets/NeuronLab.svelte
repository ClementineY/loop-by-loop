<script lang="ts">
  import { neuronForward } from '../../lib/curriculum';
  let x1=1.5,x2=-1,w1=.8,w2=-.4,b=.2,activation='relu';
  $: result=neuronForward(+x1,+x2,+w1,+w2,+b,activation as 'relu'|'tanh'); $: c1=result.contribution1; $: c2=result.contribution2; $: z=result.z; $: output=result.output;
  const fmt=(n:number)=>`${n>=0?'+':''}${n.toFixed(2)}`;
</script>
<section class="lab" aria-labelledby="neuron-title">
  <header><div><span>Neuron lab</span><h3 id="neuron-title">Multiply, add, then activate</h3></div><select bind:value={activation} aria-label="Neuron activation"><option value="relu">ReLU</option><option value="tanh">Tanh</option></select></header>
  <div class="diagram">
    <div class="inputs"><article><small>input x₁</small><b>{(+x1).toFixed(2)}</b></article><article><small>input x₂</small><b>{(+x2).toFixed(2)}</b></article><article><small>bias</small><b>{fmt(+b)}</b></article></div>
    <div class="contributions"><span>x₁ × w₁ = <b>{fmt(c1)}</b></span><span>x₂ × w₂ = <b>{fmt(c2)}</b></span><span>always added = <b>{fmt(+b)}</b></span></div>
    <div class="sum"><small>pre-activation</small><b>z = {z.toFixed(2)}</b></div><i>→</i><div class="output"><small>{activation}</small><b>{output.toFixed(3)}</b></div>
  </div>
  <div class="formula"><code>z = ({(+x1).toFixed(2)} × {(+w1).toFixed(2)}) + ({(+x2).toFixed(2)} × {(+w2).toFixed(2)}) + {(+b).toFixed(2)} = {z.toFixed(2)}</code><strong>output = {activation}(z) = {output.toFixed(3)}</strong></div>
  <div class="controls">
    <label>x₁ <b>{(+x1).toFixed(1)}</b><input type="range" min="-2" max="2" step=".1" bind:value={x1}/></label><label>w₁ <b>{(+w1).toFixed(1)}</b><input type="range" min="-2" max="2" step=".1" bind:value={w1}/></label>
    <label>x₂ <b>{(+x2).toFixed(1)}</b><input type="range" min="-2" max="2" step=".1" bind:value={x2}/></label><label>w₂ <b>{(+w2).toFixed(1)}</b><input type="range" min="-2" max="2" step=".1" bind:value={w2}/></label>
    <label>bias <b>{(+b).toFixed(1)}</b><input type="range" min="-2" max="2" step=".1" bind:value={b}/></label>
  </div>
  <footer>Weights control how strongly each input votes. The bias shifts the threshold. The activation decides how the combined evidence leaves the neuron.</footer>
</section>
<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}header{display:flex;justify-content:space-between;align-items:center;padding:1rem}header span{font:800 .6rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0}header select{border:1px solid var(--border);border-radius:7px;background:var(--bg);color:var(--text);padding:.4rem}.diagram{display:grid;grid-template-columns:110px 1fr 100px auto 90px;align-items:center;gap:.7rem;padding:1rem;background:#121925;color:white}.inputs,.contributions{display:grid;gap:.5rem}.inputs article,.sum,.output{padding:.55rem;border:1px solid #3a4557;border-radius:8px;background:#1b2432}.inputs small,.sum small,.output small{display:block;color:#8f9bad;font:.53rem var(--font-mono)}.inputs b,.sum b,.output b{font:.75rem var(--font-mono)}.contributions span{padding:.5rem;border-bottom:1px solid #384355;color:#9aa5b5;font:.61rem var(--font-mono)}.contributions b{color:var(--accent-light)}.sum{border-color:var(--accent)}.output{background:#38241f;border-color:var(--accent)}.diagram>i{font-style:normal;color:var(--accent-light)}.formula{display:flex;justify-content:space-between;gap:1rem;padding:.7rem 1rem;background:var(--accent-soft);border-bottom:1px solid var(--border)}.formula code{font-size:.65rem}.formula strong{font-size:.7rem}.controls{display:grid;grid-template-columns:repeat(5,1fr);gap:.8rem;padding:.8rem 1rem}.controls label{display:grid;grid-template-columns:1fr auto;color:var(--text-muted);font-size:.64rem}.controls input{grid-column:1/-1;accent-color:var(--accent)}footer{padding:.7rem 1rem;border-top:1px solid var(--border);color:var(--text-muted);font-size:.68rem}@media(max-width:680px){.diagram{grid-template-columns:1fr}.diagram>i{transform:rotate(90deg);text-align:center}.formula{flex-direction:column}.controls{grid-template-columns:1fr 1fr}}
</style>
