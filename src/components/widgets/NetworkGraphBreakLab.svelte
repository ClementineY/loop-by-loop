<script lang="ts">
  import {onMount} from 'svelte';
  import {networkGraphBreakPlan,type NetworkCaptureMode} from '../../lib/curriculum';

  let canvas:HTMLCanvasElement,step=0,predicate=true,playing=false,frame=0;
  let animation=0,timer:ReturnType<typeof setInterval>|undefined;
  const stories=[
    {title:'1 · Python runs the network',note:'Activations light one route. Autograd records what ran.',mode:'eager-network' as NetworkCaptureMode,line:0},
    {title:'2 · The value reaches if',note:'Python needs a real True or False before it can continue.',mode:'python-network' as NetworkCaptureMode,line:1},
    {title:'3 · Capture stops here',note:'The network still runs, but the compiler now has two regions.',mode:'python-network' as NetworkCaptureMode,line:1},
    {title:'4 · Make the fork explicit',note:'torch.cond keeps both branches inside one compiled graph.',mode:'cond-network' as NetworkCaptureMode,line:2}
  ];
  const pythonCode=[
    'h = torch.relu(self.stem(x))',
    'if h.mean() > 0:',
    '    h = self.expert_a(h)',
    'else:',
    '    h = self.expert_b(h)',
    'return self.head(h)'
  ],condCode=[
    'h = torch.relu(self.stem(x))',
    'pred = h.mean() > 0',
    'h = torch.cond(pred, self.expert_a,',
    '               self.expert_b, (h,))',
    'return self.head(h)'
  ];
  $: story=stories[step];
  $: plan=networkGraphBreakPlan(story.mode,predicate);
  $: code=step===3?condCode:pythonCode;

  const layers={
    input:[[70,105],[70,160],[70,215]],
    stem:[[225,75],[225,130],[225,190],[225,245]],
    gate:[[390,160]],
    a:[[555,80],[555,125],[555,170]],
    b:[[555,200],[555,245]],
    head:[[710,125],[710,195]],
    loss:[[835,160]]
  } as const;

  function draw(){
    if(!canvas)return;
    const dpr=Math.min(devicePixelRatio||1,2),rect=canvas.getBoundingClientRect();
    if(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr)){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr)}
    const ctx=canvas.getContext('2d');if(!ctx)return;
    const sx=rect.width/900,sy=rect.height/320;ctx.setTransform(dpr*sx,0,0,dpr*sy,0,0);ctx.clearRect(0,0,900,320);
    const activeBranch=predicate?'a':'b',pulse=(frame%180)/180;
    const edge=(from:readonly number[],to:readonly number[],active:boolean,color='#76b9ff')=>{
      ctx.beginPath();ctx.moveTo(from[0],from[1]);ctx.lineTo(to[0],to[1]);ctx.strokeStyle=active?color:'rgba(130,150,176,.13)';ctx.lineWidth=active?1.7:1;ctx.stroke();
      if(active){const t=pulse,x=from[0]+(to[0]-from[0])*t,y=from[1]+(to[1]-from[1])*t;ctx.beginPath();ctx.arc(x,y,2.8,0,Math.PI*2);ctx.fillStyle=color;ctx.shadowColor=color;ctx.shadowBlur=12;ctx.fill();ctx.shadowBlur=0}
    };
    const connect=(from:readonly (readonly number[])[],to:readonly (readonly number[])[],active:boolean,color?:string)=>from.forEach(a=>to.forEach(b=>edge(a,b,active,color)));
    const beforeDecision=step!==1||pulse<.68;
    connect(layers.input,layers.stem,beforeDecision);
    connect(layers.stem,layers.gate,beforeDecision);
    const afterDecision=step===0||step===3||step===2;
    connect(layers.gate,layers.a,afterDecision&&activeBranch==='a',step===3?'#77e0b5':'#76b9ff');
    connect(layers.gate,layers.b,afterDecision&&activeBranch==='b',step===3?'#77e0b5':'#ffb45d');
    connect(layers.a,layers.head,afterDecision&&activeBranch==='a',step===3?'#77e0b5':'#76b9ff');
    connect(layers.b,layers.head,afterDecision&&activeBranch==='b',step===3?'#77e0b5':'#ffb45d');
    connect(layers.head,layers.loss,afterDecision,step===3?'#77e0b5':'#76b9ff');

    if(step===2){
      ctx.fillStyle='rgba(77,142,210,.07)';ctx.fillRect(35,35,390,250);ctx.strokeStyle='rgba(118,185,255,.55)';ctx.strokeRect(35,35,390,250);
      ctx.fillStyle='rgba(165,109,207,.07)';ctx.fillRect(515,35,355,250);ctx.strokeStyle='rgba(201,151,240,.55)';ctx.strokeRect(515,35,355,250);
      ctx.setLineDash([5,7]);ctx.strokeStyle='#ff7860';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(470,30);ctx.lineTo(470,290);ctx.stroke();ctx.setLineDash([]);
      label(ctx,'FX graph 1',52,58,'#76b9ff');label(ctx,'FX graph 2',532,58,'#d9a4ff');label(ctx,'Python',446,308,'#ff8b72');
    }
    if(step===3){
      ctx.fillStyle='rgba(80,205,156,.055)';ctx.fillRect(35,35,835,250);ctx.strokeStyle='rgba(119,224,181,.62)';ctx.lineWidth=1.5;ctx.strokeRect(35,35,835,250);
      ctx.setLineDash([4,5]);ctx.strokeStyle='rgba(119,224,181,.55)';ctx.strokeRect(510,55,95,215);ctx.setLineDash([]);label(ctx,'one FX graph',52,58,'#77e0b5');label(ctx,'cond',533,290,'#77e0b5');
    }
    if(step===1){ctx.beginPath();ctx.arc(390,160,28+5*Math.sin(frame/14),0,Math.PI*2);ctx.strokeStyle='#ff7860';ctx.lineWidth=2;ctx.stroke();label(ctx,'Python needs bool',329,215,'#ff8b72')}

    const node=(point:readonly number[],active:boolean,color='#76b9ff',radius=8)=>{
      ctx.beginPath();ctx.arc(point[0],point[1],radius,0,Math.PI*2);ctx.fillStyle=active?'#111925':'#111925';ctx.fill();ctx.strokeStyle=active?color:'rgba(148,164,187,.42)';ctx.lineWidth=active?2:1.2;ctx.shadowColor=active?color:'transparent';ctx.shadowBlur=active?10:0;ctx.stroke();ctx.shadowBlur=0;
    };
    layers.input.forEach(p=>node(p,true));layers.stem.forEach(p=>node(p,true));node(layers.gate[0],true,step===1||step===2?'#ff7860':'#f1d35d',11);
    layers.a.forEach(p=>node(p,activeBranch==='a',step===3?'#77e0b5':'#76b9ff'));
    layers.b.forEach(p=>node(p,activeBranch==='b',step===3?'#77e0b5':'#ffb45d'));
    layers.head.forEach(p=>node(p,afterDecision,step===3?'#77e0b5':'#76b9ff'));layers.loss.forEach(p=>node(p,afterDecision,'#f1d35d',10));
    label(ctx,'input',52,275);label(ctx,'stem',206,275);label(ctx,'if',382,275);label(ctx,'expert A',527,295);label(ctx,'expert B',527,312);label(ctx,'head',693,275);label(ctx,'loss',820,275);
  }
  function label(ctx:CanvasRenderingContext2D,text:string,x:number,y:number,color='rgba(190,203,220,.75)'){ctx.fillStyle=color;ctx.font='12px ui-monospace, SFMono-Regular, Menlo, monospace';ctx.fillText(text,x,y)}
  function select(next:number){step=next;frame=0;draw()}
  function togglePlay(){playing=!playing;if(playing){timer=setInterval(()=>select((step+1)%stories.length),2100)}else if(timer)clearInterval(timer)}
  onMount(()=>{
    const observer=new ResizeObserver(draw);observer.observe(canvas);
    const loop=()=>{frame++;draw();animation=requestAnimationFrame(loop)};loop();
    return()=>{observer.disconnect();cancelAnimationFrame(animation);if(timer)clearInterval(timer)};
  });
</script>

<section class="story" aria-labelledby="graph-story-title">
  <header><div><span>Animated graph</span><h3 id="graph-story-title">Watch one graph become two</h3></div><button onclick={togglePlay}>{playing?'Pause':'Play story'}</button></header>
  <div class="stage">
    <canvas bind:this={canvas} aria-hidden="true"></canvas>
    <span class="visual-label">{story.title}. {story.note} Selected path: {plan.expert}.</span>
    <div class="branch"><span>input chooses</span><button class:active={predicate} onclick={()=>predicate=true}>expert A</button><button class:active={!predicate} onclick={()=>predicate=false}>expert B</button></div>
  </div>
  <div class="steps" aria-label="Choose story step">{#each stories as item,index}<button class:active={step===index} onclick={()=>select(index)}><i>{index+1}</i><span>{item.title.replace(/^\d · /,'')}</span></button>{/each}</div>
  <div class="caption"><b>{story.title}</b><span>{story.note}</span></div>
  <pre aria-label="Relevant PyTorch code">{#each code as line,index}<code class:hot={index===story.line}>{line}</code>{/each}</pre>
</section>

<style>
  .story{margin:1.5rem 0;border:1px solid var(--border);border-radius:16px;overflow:hidden;background:#090d14;color:#e8edf4;box-shadow:0 16px 45px var(--shadow)}header{display:flex;align-items:center;justify-content:space-between;padding:.85rem 1rem;border-bottom:1px solid #253043}header span{font:800 .56rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:#76b9ff}header h3{margin:.15rem 0 0;color:#f1f5fa}header button,.branch button{border:1px solid #334157;background:#111925;color:#c0cad8;padding:.4rem .6rem;font-size:.54rem}header button{border-color:#76b9ff;color:#9fd0ff}.stage{position:relative}.stage canvas{display:block;width:100%;height:330px}.visual-label{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.branch{position:absolute;top:.7rem;right:.75rem;display:flex;align-items:center;gap:.25rem}.branch span{margin-right:.25rem;color:#75849a;font-size:.48rem}.branch button.active{border-color:#f1d35d;color:#f1d35d}.steps{display:grid;grid-template-columns:repeat(4,1fr);border-block:1px solid #253043}.steps button{display:flex;align-items:center;gap:.4rem;border:0;border-right:1px solid #253043;background:#0d131d;color:#738299;padding:.55rem;text-align:left;font-size:.5rem}.steps button:last-child{border:0}.steps button.active{background:#162131;color:#eef3f8}.steps i{display:grid;place-items:center;width:19px;height:19px;border:1px solid currentColor;border-radius:50%;font-style:normal}.caption{display:flex;align-items:baseline;gap:.65rem;padding:.7rem 1rem}.caption b{color:#f1d35d;font:.62rem var(--font-mono)}.caption span{color:#a8b4c4;font-size:.54rem}pre{display:grid;grid-template-columns:1fr 1fr;margin:0;padding:.65rem 1rem 1rem;border-top:1px solid #1d2737;column-gap:1rem;color:#67768a;font:.5rem/1.55 var(--font-mono)}pre code{padding:0 .3rem;border-left:2px solid transparent;white-space:pre}pre code.hot{border-color:#ff7860;background:#201919;color:#f0f3f7}@media(max-width:700px){.stage canvas{height:290px}.branch{position:static;padding:.55rem .7rem;border-top:1px solid #253043}.steps{grid-template-columns:1fr 1fr}.caption{align-items:flex-start;flex-direction:column;gap:.2rem}pre{grid-template-columns:1fr}}@media(max-width:420px){header{align-items:flex-start;gap:.5rem;flex-direction:column}header button{width:100%}.stage canvas{height:250px}.steps button span{display:none}}
</style>
