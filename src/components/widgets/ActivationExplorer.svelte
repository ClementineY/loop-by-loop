<script lang="ts">
  import { onMount } from 'svelte';
  import { relu, sigmoid, tanh, leakyRelu, dRelu, dSigmoid, dTanh, dLeakyRelu } from '../../lib/activations';
  let canvas: HTMLCanvasElement;
  let activation = 'relu';
  let x = 0.5;
  const funcs: Record<string,(n:number)=>number> = { relu, sigmoid, tanh, leakyRelu };
  const derivs: Record<string,(n:number)=>number> = { relu:dRelu, sigmoid:dSigmoid, tanh:dTanh, leakyRelu:dLeakyRelu };
  $: value = funcs[activation](+x);
  $: derivative = derivs[activation](+x);
  $: if (canvas) draw();
  function draw(){const ctx=canvas.getContext('2d');if(!ctx)return;const d=devicePixelRatio||1,w=canvas.clientWidth,h=260;canvas.width=w*d;canvas.height=h*d;ctx.scale(d,d);ctx.clearRect(0,0,w,h);ctx.strokeStyle='#687181';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.stroke();ctx.strokeStyle=getComputedStyle(canvas).getPropertyValue('--accent')||'#df5b35';ctx.lineWidth=3;ctx.beginPath();for(let px=0;px<w;px++){const vx=(px/w)*12-6;const vy=funcs[activation](vx);const py=h/2-vy*34;px?ctx.lineTo(px,py):ctx.moveTo(px,py)}ctx.stroke();const px=((+x+6)/12)*w,py=h/2-value*34;ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(px,py,6,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#df5b35';ctx.lineWidth=3;ctx.stroke();}
  onMount(draw);
</script>
<section class="lab" aria-labelledby="act-title">
  <div class="head"><div><span>Interactive lab</span><h3 id="act-title">Activation explorer</h3></div><select bind:value={activation} aria-label="Activation function"><option value="relu">ReLU</option><option value="sigmoid">Sigmoid</option><option value="tanh">Tanh</option><option value="leakyRelu">Leaky ReLU</option></select></div>
  <canvas bind:this={canvas} aria-label={`${activation} activation curve with x ${x}`}></canvas>
  <div class="readout"><label>x = <strong>{(+x).toFixed(1)}</strong><input type="range" min="-6" max="6" step="0.1" bind:value={x}/></label><div><small>output</small><b>{value.toFixed(3)}</b></div><div><small>local slope</small><b>{derivative.toFixed(3)}</b></div></div>
</section>
<style>
  .lab{margin:2rem 0;padding:1rem;border:1px solid var(--border);border-radius:14px;background:var(--ink);color:white;--accent:#df5b35}.head{display:flex;justify-content:space-between;align-items:center}.head span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent-light)}.head h3{margin:.2rem 0;font-size:1.1rem}.head select{background:#222a38;color:white;border:1px solid #3a4455;border-radius:8px;padding:.45rem}.lab canvas{display:block;width:100%;height:260px;margin:1rem 0;background:#151c28;border-radius:10px}.readout{display:grid;grid-template-columns:2fr 1fr 1fr;gap:1rem;align-items:end}.readout label{display:grid;grid-template-columns:auto 1fr;gap:.2rem .6rem;color:#aab1c0;font-size:.75rem}.readout label input{grid-column:1/-1;accent-color:var(--accent)}.readout>div{display:flex;flex-direction:column;border-left:1px solid #394253;padding-left:1rem}.readout small{font-size:.62rem;color:#8f99aa;text-transform:uppercase}.readout b{font:700 1.1rem var(--font-mono);color:var(--accent-light)}@media(max-width:560px){.readout{grid-template-columns:1fr 1fr}.readout label{grid-column:1/-1}}
</style>
