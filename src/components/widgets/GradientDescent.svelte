<script lang="ts">
  import { onDestroy } from 'svelte';
  import { sgdStep, type Optimizer } from '../../lib/gradientDescent';
  let lr=0.12,momentum=0,p=3.8,velocity=0,steps:number[]=[];let timer:number|undefined;
  $: loss=p*p;
  function step(){const opt:Optimizer={params:[p],velocity:[velocity]};sgdStep(opt,[2*p],+lr,+momentum);p=opt.params[0];velocity=opt.velocity[0];steps=[...steps,loss].slice(-20)}
  function reset(){stop();p=3.8;velocity=0;steps=[]}
  function run(){if(timer)return;timer=window.setInterval(()=>{step();if(!Number.isFinite(p)||Math.abs(p)>20)stop()},180)}
  function stop(){if(timer){clearInterval(timer);timer=undefined}}
  onDestroy(stop);
</script>
<section class="lab" aria-labelledby="gd-title">
  <div class="head"><div><span>Interactive lab</span><h3 id="gd-title">Loss landscape</h3></div><div class="buttons"><button onclick={step}>Step</button><button class="run" onclick={timer?stop:run}>{timer?'Pause':'Run'}</button><button onclick={reset}>Reset</button></div></div>
  <div class="landscape"><div class="parabola"></div><div class="ball" style={`left:${Math.max(2,Math.min(98,(p+5)*10))}%;bottom:${Math.min(84,10+loss*3)}%`}><i></i><span>p={p.toFixed(2)}<br/>L={loss.toFixed(2)}</span></div><div class="axis"></div></div>
  <div class="controls"><label>Learning rate <b>{(+lr).toFixed(2)}</b><input type="range" min="0.01" max="1.1" step="0.01" bind:value={lr}/></label><label>Momentum <b>{(+momentum).toFixed(2)}</b><input type="range" min="0" max="0.95" step="0.05" bind:value={momentum}/></label><div class:danger={+lr>=1}>{+lr<.06?'Very slow':+lr>=1?'Likely to diverge':+lr>.6?'May oscillate':'Stable region'}</div></div>
</section>
<style>
  .lab{margin:2rem 0;padding:1rem;border:1px solid var(--border);border-radius:14px;background:var(--surface)}.head{display:flex;justify-content:space-between;align-items:center;gap:1rem}.head span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}.head h3{margin:.2rem 0}.buttons{display:flex;gap:.35rem}.buttons button{border:1px solid var(--border);border-radius:7px;background:var(--bg);padding:.35rem .55rem;font-size:.72rem;color:var(--text)}.buttons .run{background:var(--accent);color:white;border-color:var(--accent)}.landscape{height:250px;position:relative;overflow:hidden;background:var(--ink);border-radius:10px;margin:1rem 0}.parabola{position:absolute;width:95%;height:420px;border:3px solid #626d7e;border-top:0;border-radius:0 0 50% 50%;left:2.5%;top:-250px}.axis{position:absolute;bottom:9%;left:4%;right:4%;border-top:1px solid #465064}.ball{position:absolute;transform:translateX(-50%);transition:.16s ease}.ball i{display:block;width:18px;height:18px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 7px #df5b3533}.ball span{position:absolute;white-space:nowrap;left:25px;top:-8px;color:white;font:.65rem var(--font-mono)}.controls{display:grid;grid-template-columns:1fr 1fr auto;gap:1rem;align-items:end}.controls label{display:grid;grid-template-columns:1fr auto;font-size:.7rem;color:var(--text-muted)}.controls input{grid-column:1/-1;accent-color:var(--accent)}.controls>div{padding:.45rem .6rem;background:#e9f5e8;color:#2d6730;border-radius:7px;font-size:.68rem;font-weight:750}.controls>div.danger{background:#fde7e2;color:#a23f2c}@media(max-width:600px){.controls{grid-template-columns:1fr}.head{align-items:flex-start;flex-direction:column}}
</style>
