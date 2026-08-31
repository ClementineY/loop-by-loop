<script lang="ts">
  import { relu, sigmoid, tanh, leakyRelu, dRelu, dSigmoid, dTanh, dLeakyRelu } from '../../lib/activations';
  let activation = 'relu';
  let x = 0.5;
  const funcs:Record<string,(n:number)=>number>={relu,sigmoid,tanh,leakyRelu};
  const derivs:Record<string,(n:number)=>number>={relu:dRelu,sigmoid:dSigmoid,tanh:dTanh,leakyRelu:dLeakyRelu};
  const names:Record<string,string>={relu:'ReLU',sigmoid:'Sigmoid',tanh:'Tanh',leakyRelu:'Leaky ReLU'};
  const formulas:Record<string,string>={relu:'max(0, x)',sigmoid:'1 / (1 + e⁻ˣ)',tanh:'(eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)',leakyRelu:'max(0.01x, x)'};
  const px=(n:number)=>42+((n+6)/12)*476;
  const py=(n:number)=>(activation==='relu'||activation==='leakyRelu')?235-n*34:145-n*90;
  $: value=funcs[activation](+x); $: derivative=derivs[activation](+x);
  $: survival=Math.abs(derivative)**6;
  $: curve=Array.from({length:121},(_,i)=>{const n=-6+i/10;return `${i?'L':'M'} ${px(n).toFixed(1)} ${py(funcs[activation](n)).toFixed(1)}`}).join(' ');
  $: tangentLeft=Math.max(-6,+x-1); $: tangentRight=Math.min(6,+x+1);
</script>
<section class="lab" aria-labelledby="act-title">
  <header><div><span>Activation lab</span><h3 id="act-title">What passes forward—and backward?</h3></div><select bind:value={activation} aria-label="Activation function"><option value="relu">ReLU</option><option value="sigmoid">Sigmoid</option><option value="tanh">Tanh</option><option value="leakyRelu">Leaky ReLU</option></select></header>
  <div class="formula"><strong>{names[activation]}</strong><code>f(x) = {formulas[activation]}</code><span>f({(+x).toFixed(1)}) = {value.toFixed(3)}</span></div>
  <div class="plot"><svg viewBox="0 0 560 290" role="img" aria-label={`${names[activation]} curve at x ${x}`}>
    <line class="axis" x1="35" y1={py(0)} x2="525" y2={py(0)}/><line class="axis" x1="280" y1="20" x2="280" y2="270"/>
    <path class="curve" d={curve}/><line class="tangent" x1={px(tangentLeft)} y1={py(value+derivative*(tangentLeft-x))} x2={px(tangentRight)} y2={py(value+derivative*(tangentRight-x))}/><circle cx={px(+x)} cy={py(value)} r="7"/>
    <text x={Math.min(px(+x)+12,430)} y={Math.max(py(value)-12,22)}>local slope {derivative.toFixed(3)}</text>
  </svg></div>
  <div class="controls"><label>x <b>{(+x).toFixed(1)}</b><input type="range" min="-6" max="6" step="0.1" bind:value={x}/></label><article><small>forward output</small><b>{value.toFixed(3)}</b></article><article><small>local slope</small><b>{derivative.toFixed(3)}</b></article><article><small>after 6 identical slopes</small><b>{survival.toExponential(2)}×</b></article></div>
  <footer>The gray curve, orange tangent, formula, output, and slope all come from the selected function. Change the menu and verify that every representation changes together.</footer>
</section>
<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}header{display:flex;justify-content:space-between;align-items:center;padding:1rem}header span{font:800 .6rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0 0}header select{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:.45rem}.formula{display:flex;align-items:center;gap:1rem;padding:.7rem 1rem;border-block:1px solid var(--border);background:var(--accent-soft)}.formula strong{color:var(--accent)}.formula span{margin-left:auto;font:650 .7rem var(--font-mono)}.plot{padding:.5rem 1rem;background:#121925}.plot svg{display:block;width:100%}.axis{stroke:#455064}.curve{fill:none;stroke:#738095;stroke-width:3}.tangent{stroke:var(--accent-light);stroke-width:3}.plot circle{fill:var(--accent);stroke:#ffd0bf;stroke-width:2}.plot text{fill:var(--accent-light);font:10px var(--font-mono)}.controls{display:grid;grid-template-columns:1.5fr repeat(3,1fr);gap:1rem;padding:.8rem 1rem}.controls label{display:grid;grid-template-columns:1fr auto;color:var(--text-muted);font-size:.68rem}.controls input{grid-column:1/-1;accent-color:var(--accent)}.controls article{border-left:1px solid var(--border);padding-left:.8rem}.controls small{display:block;color:var(--text-muted);font-size:.55rem}.controls article b{font:700 .86rem var(--font-mono)}footer{padding:.7rem 1rem;border-top:1px solid var(--border);color:var(--text-muted);font-size:.68rem}@media(max-width:650px){header,.formula{align-items:flex-start;flex-direction:column}.formula span{margin:0}.controls{grid-template-columns:1fr 1fr}.controls label{grid-column:1/-1}}
</style>
