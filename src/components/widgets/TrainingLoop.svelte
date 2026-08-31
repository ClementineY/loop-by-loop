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
  function draw(){
    if(!canvas)return;const ctx=canvas.getContext('2d');if(!ctx)return;
    const d=devicePixelRatio||1,w=canvas.clientWidth,h=300;canvas.width=w*d;canvas.height=h*d;ctx.scale(d,d);ctx.clearRect(0,0,w,h);
    const sampleStep=7,columns=Math.ceil(w/sampleStep),rows=Math.ceil(h/sampleStep),samples:number[][]=[];
    const field=document.createElement('canvas');field.width=columns;field.height=rows;const fieldCtx=field.getContext('2d');if(!fieldCtx)return;const pixels=fieldCtx.createImageData(columns,rows);
    for(let rowIndex=0;rowIndex<rows;rowIndex++){const row:number[]=[];for(let columnIndex=0;columnIndex<columns;columnIndex++){
      const x=columnIndex/(columns-1)*3.8-1.4,y=1.5-rowIndex/(rows-1)*2.4;const score=net.forward([x,y]).data;const confidence=Math.max(0,Math.min(1,score));row.push(confidence);
      const red=[223,91,53],blue=[74,104,148],index=(rowIndex*columns+columnIndex)*4;for(let channel=0;channel<3;channel++)pixels.data[index+channel]=Math.round(blue[channel]+(red[channel]-blue[channel])*confidence);pixels.data[index+3]=105;
    }samples.push(row)}
    fieldCtx.putImageData(pixels,0,0);ctx.fillStyle='#121925';ctx.fillRect(0,0,w,h);ctx.imageSmoothingEnabled=true;ctx.imageSmoothingQuality='high';ctx.drawImage(field,0,0,w,h);
    ctx.strokeStyle='rgba(255,255,255,.72)';ctx.lineWidth=1.2;ctx.beginPath();
    for(let r=0;r<samples.length-1;r++)for(let c=0;c<samples[r].length-1;c++){const here=samples[r][c]>=.5,right=samples[r][c+1]>=.5,down=samples[r+1][c]>=.5,x0=c/(columns-1)*w,x1=(c+1)/(columns-1)*w,y0=r/(rows-1)*h,y1=(r+1)/(rows-1)*h;if(here!==right){ctx.moveTo(x1,y0);ctx.lineTo(x1,y1)}if(here!==down){ctx.moveTo(x0,y1);ctx.lineTo(x1,y1)}}ctx.stroke();
    data.X.forEach((p,i)=>{const [px,py]=map(p,w,h);ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fillStyle=data.y[i]?'#df5b35':'#243652';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=1.4;ctx.stroke()});
  }
  function hiddenChanged(){reset()}
  onMount(draw);onDestroy(()=>{if(typeof cancelAnimationFrame!=='undefined')cancelAnimationFrame(frame)});
</script>
<section class="lab" aria-labelledby="train-title">
  <div class="head"><div><span>Interactive lab</span><h3 id="train-title">Train a tiny network</h3></div><div class="metrics"><div><small>epoch</small><b>{epoch}</b></div><div><small>loss</small><b>{epoch?loss.toFixed(3):'—'}</b></div></div></div>
  <canvas bind:this={canvas} aria-label="Two-class moons dataset and the neural network decision boundary"></canvas>
  <div class="controls"><label>Learning rate <b>{(+lr).toFixed(2)}</b><input type="range" min=".01" max=".25" step=".01" bind:value={lr}/></label><label>Hidden neurons <b>{hidden}</b><input type="range" min="2" max="12" step="1" bind:value={hidden} onchange={hiddenChanged}/></label><button class="train" onclick={toggle}>{running?'Pause':'Train 100 epochs'}</button><button onclick={reset}>Reset</button></div>
  <div class="legend"><span><i class="blue"></i> predicts class 0</span><span><i class="boundary"></i> decision boundary</span><span><i class="red"></i> predicts class 1</span><strong>Dots are examples; the smooth background is model confidence.</strong></div>
</section>
<style>
  .lab{margin:2rem 0;padding:1rem;border:1px solid var(--border);border-radius:14px;background:var(--surface);box-shadow:0 16px 45px var(--shadow)}.head{display:flex;justify-content:space-between;align-items:center}.head span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}.head h3{margin:.2rem 0}.metrics{display:flex;gap:1.2rem}.metrics div{display:flex;flex-direction:column}.metrics small{text-transform:uppercase;color:var(--text-subtle);font-size:.55rem}.metrics b{font:750 .9rem var(--font-mono)}canvas{display:block;width:100%;height:300px;margin:1rem 0;background:var(--ink);border-radius:10px}.controls{display:grid;grid-template-columns:1fr 1fr auto auto;gap:.7rem;align-items:end}.controls label{display:grid;grid-template-columns:1fr auto;font-size:.68rem;color:var(--text-muted)}.controls input{grid-column:1/-1;accent-color:var(--accent)}.controls button{height:35px;border:1px solid var(--border);background:var(--bg);color:var(--text);border-radius:7px;padding:0 .65rem;font-size:.69rem;font-weight:720}.controls .train{background:var(--accent);color:white;border-color:var(--accent)}.legend{display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin:.7rem .2rem 0;color:var(--text-subtle);font-size:.62rem}.legend span{display:flex;align-items:center;gap:.3rem}.legend i{display:inline-block;width:12px;height:8px;border-radius:3px}.legend .blue{background:#4a6894}.legend .red{background:#df5b35}.legend .boundary{height:0;border-top:2px solid var(--text)}.legend strong{margin-left:auto;color:var(--text-muted);font-weight:600}@media(max-width:650px){.controls{grid-template-columns:1fr 1fr}.metrics{gap:.6rem}.legend strong{width:100%;margin:0}}
</style>
