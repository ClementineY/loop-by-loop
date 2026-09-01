<script lang="ts">
  import {networkGraphBreakPlan,type NetworkCaptureMode} from '../../lib/curriculum';
  const modes:{id:NetworkCaptureMode;label:string}[]=[
    {id:'eager-network',label:'Eager'},
    {id:'python-network',label:'compile + Python if'},
    {id:'cond-network',label:'compile + torch.cond'}
  ];
  let mode:NetworkCaptureMode='python-network',predicate=true,phase:'forward'|'backward'='forward';
  $: plan=networkGraphBreakPlan(mode,predicate);
  $: code=mode==='cond-network'?`class TinyRouter(nn.Module):
    def forward(self, x):
        h = F.relu(self.stem(x))
        pred = h.mean() > 0
        h = torch.cond(
            pred, self.expert_a,
            self.expert_b, (h,)
        )
        return self.head(h)`:`class TinyRouter(nn.Module):
    def forward(self, x):
        h = F.relu(self.stem(x))
        if h.mean() > 0:
            h = self.expert_a(h)
        else:
            h = self.expert_b(h)
        return self.head(h)`;
  $: regionLabel=mode==='eager-network'?'runtime autograd graph':mode==='python-network'?'two FX regions + Python':'one FX graph with cond';
</script>

<section class="lab" aria-labelledby="network-tree-title">
  <header><div><span>Model graph lab</span><h3 id="network-tree-title">Where does a network graph break?</h3></div><div class="phase" aria-label="Choose graph direction"><button class:active={phase==='forward'} onclick={()=>phase='forward'}>Forward</button><button class:active={phase==='backward'} onclick={()=>phase='backward'}>Backward</button></div></header>
  <nav aria-label="Choose execution mode">{#each modes as item}<button class:active={mode===item.id} onclick={()=>mode=item.id}>{item.label}</button>{/each}</nav>
  <div class="workspace">
    <div class="code"><div class="filebar">tiny_router.py <span>{regionLabel}</span></div><pre><code>{code}</code></pre></div>
    <div class="diagram">
      <div class="controls"><span>activation mean</span><button class:active={predicate} onclick={()=>predicate=true}>&gt; 0</button><button class:active={!predicate} onclick={()=>predicate=false}>≤ 0</button></div>
      <div class="capture" class:broken={mode==='python-network'}>
        {#if mode==='eager-network'}<b>Python runs first · autograd records the selected route</b>
        {:else if mode==='python-network'}<b><span>compiled region 1</span><i>GRAPH BREAK · Python reads the boolean</i><span>compiled continuation</span></b>
        {:else}<b>one compiled FX graph · cond contains two branch subgraphs</b>{/if}
      </div>
      <div class:reverse={phase==='backward'} class="tree" aria-label={`${phase} graph through ${plan.expert}`}>
        <GraphNode label="input x" detail="[batch, features]" region={mode==='python-network'?'prefix':'whole'}/>
        <Connector/>
        <GraphNode label="stem" detail="Linear → ReLU" region={mode==='python-network'?'prefix':'whole'}/>
        <Connector/>
        <GraphNode label="h.mean() > 0" detail={mode==='python-network'?'tensor value needed by Python':mode==='cond-network'?'cond predicate stays in graph':'Python chooses the route'} region={mode==='python-network'?'break':'whole'} diamond/>
        <div class="fork" aria-label="expert branches">
          <div class:selected={predicate} class:inactive={!predicate}><GraphNode label="expert A" detail="Linear → GELU" region={mode==='python-network'?'continuation':mode==='cond-network'?'branch':'whole'}/><small>{mode==='cond-network'?'captured subgraph':predicate?'executed':'not executed'}</small></div>
          <div class:selected={!predicate} class:inactive={predicate}><GraphNode label="expert B" detail="Linear → Tanh" region={mode==='python-network'?'continuation':mode==='cond-network'?'branch':'whole'}/><small>{mode==='cond-network'?'captured subgraph':!predicate?'executed':'not executed'}</small></div>
        </div>
        <div class="merge">{phase==='forward'?'↘  merge  ↙':'↖  selected gradient  ↗'}</div>
        <GraphNode label="head" detail="Linear → prediction" region={mode==='python-network'?'continuation':'whole'}/>
        <Connector/>
        <GraphNode label="loss" detail={phase==='forward'?'scalar output':'seed gradient = 1'} region={mode==='python-network'?'continuation':'whole'}/>
      </div>
      <div class="trace"><small>{phase} route</small><div>{#each (phase==='forward'?['input','stem',plan.expert,'head','loss']:plan.backward) as step}<span>{step}</span>{/each}</div></div>
      <p>{#if mode==='eager-network'}Only the selected expert becomes part of this call's autograd graph. The next call may record the other route.
      {:else if mode==='python-network'}The network itself still works. The break means the compiler cannot optimize stem, expert, and head as one region; backward later crosses those region boundaries through autograd.
      {:else}Both experts are represented in the compiled graph, but only <b>{plan.expert}</b> executes now and only that selected route receives gradients for this call.{/if}</p>
    </div>
  </div>
  <footer><span><i class="swatch prefix"></i>first FX region</span><span><i class="swatch break"></i>Python boundary</span><span><i class="swatch continuation"></i>continuation FX region</span><span><i class="swatch branch"></i>structured subgraph</span></footer>
</section>

{#snippet GraphNode(label:string,detail:string,region:string,diamond=false)}<div class:diamond class={`node ${region}`}><b>{label}</b><span>{detail}</span></div>{/snippet}
{#snippet Connector()}<i class="connector">{phase==='forward'?'↓':'↑'}</i>{/snippet}

<style>
  .lab{margin:1.5rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden;box-shadow:0 16px 45px var(--shadow)}header{display:flex;align-items:center;justify-content:space-between;padding:1rem}header span{font:800 .58rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0}.phase{display:flex}.phase button,.controls button{border:1px solid var(--border);background:var(--bg);color:var(--text);padding:.4rem .55rem;font-size:.54rem}.phase button.active,.controls button.active{background:var(--accent);border-color:var(--accent);color:white}nav{display:grid;grid-template-columns:repeat(3,1fr);border-block:1px solid var(--border)}nav button{border:0;border-right:1px solid var(--border);padding:.65rem;background:var(--bg);color:var(--text-muted);font-size:.58rem}nav button:last-child{border:0}nav button.active{background:var(--accent);color:white}.workspace{display:grid;grid-template-columns:.78fr 1.3fr;min-height:660px;background:#111925;color:white}.code{border-right:1px solid #303b4c}.filebar{display:flex;justify-content:space-between;padding:.55rem .7rem;background:#1b2634;color:#a0adbd;font:.55rem var(--font-mono)}.filebar span{color:#748398}.code pre{box-sizing:border-box;margin:0;padding:1rem;overflow:auto;color:#dce4ed;font:.57rem/1.65 var(--font-mono)}.diagram{display:flex;flex-direction:column;padding:.8rem}.controls{display:flex;align-items:center;gap:.3rem}.controls>span{margin-right:auto;color:#8e9caf;font-size:.51rem}.controls button{border-color:#3d4a5d;background:#182332;color:#bdc7d5}.controls button.active{border-color:#df7656;background:#182332;color:#ff9b79}.capture{margin:.65rem 0;padding:.45rem .55rem;border:1px solid #526987;border-radius:5px;background:#182535;text-align:center}.capture b{font:.5rem var(--font-mono);color:#b9c6d6}.capture.broken b{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:.35rem}.capture.broken span{color:#a8c2e2}.capture.broken i{padding:.25rem .35rem;border:1px solid #df6540;background:#32231f;color:#ff9b79;font-style:normal}.tree{display:flex;flex:1;flex-direction:column;align-items:center;justify-content:center}.node{box-sizing:border-box;width:155px;padding:.5rem .6rem;border:1px solid #637997;border-radius:6px;background:#192433;text-align:center}.node b,.node span{display:block}.node b{font-size:.56rem}.node span{margin-top:.18rem;color:#95a4b6;font-size:.47rem}.node.prefix{border-color:#6d91bd;background:#18283b}.node.continuation{border-color:#9b7db5;background:#282138}.node.break{border-color:#df6540;background:#32231f}.node.branch{border-color:#69a391;background:#17302d}.node.diamond{width:180px}.connector{height:18px;color:#77879a;font:normal .7rem/18px var(--font-mono)}.fork{position:relative;display:grid;grid-template-columns:1fr 1fr;gap:1.8rem;width:min(100%,390px);padding-top:22px}.fork:before{content:"";position:absolute;top:8px;left:25%;right:25%;height:12px;border-top:1px solid #64758a;border-inline:1px solid #64758a}.fork>div{display:flex;flex-direction:column;align-items:center;opacity:.32}.fork>div.selected{opacity:1}.fork>div.inactive .node{border-style:dashed}.fork small{margin-top:.22rem;color:#758498;font-size:.44rem}.merge{height:25px;color:#758498;font:.48rem/25px var(--font-mono)}.reverse .fork>div:not(.selected){opacity:.16}.trace{margin-top:.45rem;padding-top:.55rem;border-top:1px solid #303b4b}.trace small{color:#758498;font:.46rem var(--font-mono)}.trace div{display:flex;gap:.25rem;flex-wrap:wrap;margin-top:.3rem}.trace span{padding:.2rem .35rem;background:#1e2a39;color:#b6c1cf;font:.46rem var(--font-mono)}.trace span+span:before{content:"→";margin-right:.35rem;color:#6f7e91}.diagram>p{min-height:2.2rem;margin:.55rem 0 0;color:#9eabba;font-size:.52rem;line-height:1.45}footer{display:flex;gap:1rem;flex-wrap:wrap;padding:.65rem 1rem;background:var(--accent-soft)}footer span{display:flex;align-items:center;gap:.3rem;color:var(--text-muted);font:.49rem var(--font-mono)}.swatch{width:11px;height:11px;border:1px solid}.swatch.prefix{border-color:#6d91bd;background:#18283b}.swatch.break{border-color:#df6540;background:#32231f}.swatch.continuation{border-color:#9b7db5;background:#282138}.swatch.branch{border-color:#69a391;background:#17302d}@media(max-width:760px){.workspace{grid-template-columns:1fr}.code{border-right:0;border-bottom:1px solid #303b4c}.diagram{min-height:650px}.capture.broken b{grid-template-columns:1fr}.fork{gap:.6rem}.node{width:130px}footer{gap:.55rem}}@media(max-width:420px){header{align-items:flex-start;gap:.5rem;flex-direction:column}.phase{width:100%}.phase button{flex:1}.filebar span{display:none}.fork{gap:.25rem}.node{width:112px;padding:.45rem .25rem}.node.diamond{width:150px}.capture.broken i{font-size:.44rem}}
</style>
