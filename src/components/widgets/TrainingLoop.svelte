<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { MLP } from '../../lib/mlp';
  import { makeMoons } from '../../lib/data';
  let canvas:HTMLCanvasElement;
  let hidden=6,lr=.08,epoch=0,loss=0,running=false,frame=0;
  const data=makeMoons(70,.12,4);
  let net=new MLP(2,hidden,8);
  function reset(){running=false;cancelAnimationFrame(frame);net=new MLP(2,+hidden,8);epoch=0;loss=0;draw()}
  function tick(){if(!running)return;loss=net.trainEpoch(data.X,data.y,+lr);epoch++;draw();if(epoch<100)frame=requestAnimationFrame(tick);else running=false}
  function toggle(){running=!running;if(running)frame=requestAnimationFrame(tick);else cancelAnimationFrame(frame)}
  function map([x,y]:[number,number],w:number,h:number):[number,number]{return [(x+1.4)/3.8*w,(1.5-y)/2.4*h]}
  function draw(){if(!canvas)return;const ctx=canvas.getContext('2d');if(!ctx)return;const d=devicePixelRatio||1,w=canvas.clientWidth,h=300;canvas.width=w*d;canvas.height=h*d;ctx.scale(d,d);ctx.clearRect(0,0,w,h);const step=14;for(let px=0;px<w;px+=step)for(let py=0;py<h;py+=step){const x=px/w*3.8-1.4,y=1.5-py/h*2.4;const pred=net.forward([x,y]).data;ctx.fillStyle=pred>.5?'rgba(223,91,53,.16)':'rgba(107,122,145,.13)';ctx.fillRect(px,py,step+1,step+1)}data.X.forEach((p,i)=>{const [px,py]=map(p,w,h);ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);if(data.y[i]){ctx.fillStyle='#df5b35';ctx.fill()}else{ctx.fillStyle='#151c28';ctx.fill();ctx.strokeStyle='#f1f4f7';ctx.lineWidth=1.5;ctx.stroke()}})}
  function hiddenChanged(){reset()}
  onMount(draw);onDestroy(()=>{if(typeof cancelAnimationFrame!=='undefined')cancelAnimationFrame(frame)});
</script>
<section class="lab" aria-labelledby="train-title">
  <div class="head"><div><span>Interactive lab</span><h3 id="train-title">Train a tiny network</h3></div><div class="metrics"><div><small>epoch</small><b>{epoch}</b></div><div><small>loss</small><b>{epoch?loss.toFixed(3):'—'}</b></div></div></div>
  <canvas bind:this={canvas} aria-label="Two-class moons dataset and the neural network decision boundary"></canvas>
  <div class="controls"><label>Learning rate <b>{(+lr).toFixed(2)}</b><input type="range" min=".01" max=".25" step=".01" bind:value={lr}/></label><label>Hidden neurons <b>{hidden}</b><input type="range" min="2" max="12" step="1" bind:value={hidden} onchange={hiddenChanged}/></label><button class="train" onclick={toggle}>{running?'Pause':'Train 100 epochs'}</button><button onclick={reset}>Reset</button></div>
  <p><span class="filled"></span> class 1 <span class="hollow"></span> class 0 · Background color shows the network’s current prediction.</p>
</section>
<style>
  .lab{margin:2rem 0;padding:1rem;border:1px solid var(--border);border-radius:14px;background:var(--surface);box-shadow:0 16px 45px var(--shadow)}.head{display:flex;justify-content:space-between;align-items:center}.head span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}.head h3{margin:.2rem 0}.metrics{display:flex;gap:1.2rem}.metrics div{display:flex;flex-direction:column}.metrics small{text-transform:uppercase;color:var(--text-subtle);font-size:.55rem}.metrics b{font:750 .9rem var(--font-mono)}canvas{display:block;width:100%;height:300px;margin:1rem 0;background:var(--ink);border-radius:10px}.controls{display:grid;grid-template-columns:1fr 1fr auto auto;gap:.7rem;align-items:end}.controls label{display:grid;grid-template-columns:1fr auto;font-size:.68rem;color:var(--text-muted)}.controls input{grid-column:1/-1;accent-color:var(--accent)}.controls button{height:35px;border:1px solid var(--border);background:var(--bg);color:var(--text);border-radius:7px;padding:0 .65rem;font-size:.69rem;font-weight:720}.controls .train{background:var(--accent);color:white;border-color:var(--accent)}.lab>p{color:var(--text-subtle);font-size:.65rem;margin:.7rem .2rem 0}.lab>p span{display:inline-block;width:8px;height:8px;border-radius:50%;margin-left:.5rem}.filled{background:var(--accent)}.hollow{border:1px solid var(--text)}@media(max-width:650px){.controls{grid-template-columns:1fr 1fr}.metrics{gap:.6rem}}
</style>
