<script lang="ts">
  import {compileCornerCasePlan,type CompileCornerCase} from '../../lib/curriculum';

  const cases:{id:CompileCornerCase;label:string;question:string}[]=[
    {id:'fixed-loop',label:'Fixed loop',question:'Does a loop stay a loop?'},
    {id:'tensor-loop',label:'Tensor while',question:'Who decides when to stop?'},
    {id:'scalar-item',label:'.item()',question:'Where does the scalar live?'},
    {id:'shape-branch',label:'Shape branch',question:'Can “dynamic” remove every guard?'},
    {id:'cond-contract',label:'cond contract',question:'Can branches return anything?'}
  ];
  let kind:CompileCornerCase='fixed-loop',option:number|boolean=4;
  $: plan=compileCornerCasePlan(kind,option);
  $: current=cases.find(item=>item.id===kind)!;
  $: code=kind==='fixed-loop'?`@torch.compile\ndef f(x, steps: int):\n    for _ in range(steps):\n        x = torch.sin(x) + x\n    return x`:
    kind==='tensor-loop'?Boolean(option)?`from torch._higher_order_ops import while_loop\n\ndef body(i, x): return i - 1, x.sin()\ndef cond(i, x): return i.sum() > 0\n\n# structured, but currently inference-only\ny = while_loop(cond, body, (count, x))`:`@torch.compile\ndef f(x, tolerance):\n    while x.norm() > tolerance:\n        x = x / 2\n    return x`:
    kind==='scalar-item'?Boolean(option)?`torch._dynamo.config.capture_scalar_outputs = True\n\n@torch.compile\ndef f(x):\n    scale = x.sum().item()\n    return x * scale`:`@torch.compile\ndef f(x):\n    scale = x.sum().item()\n    return x * scale`:
    kind==='shape-branch'?`@torch.compile(dynamic=True)\ndef f(x):\n    if x.shape[0] <= 16:\n        return small_path(x)\n    return large_path(x)`:
    Boolean(option)?`def yes(x): return x.sin()   # [N, D]\ndef no(x):  return x.cos()   # [N, D]\n\ny = torch.cond(pred, yes, no, (x,))`:`def yes(x): return x.sin()       # [N, D]\ndef no(x):  return x.mean(dim=1) # [N]\n\ny = torch.cond(pred, yes, no, (x,))`;

  function choose(next:CompileCornerCase){
    kind=next;
    option=next==='fixed-loop'?4:next==='shape-branch'?16:next==='cond-contract';
  }
</script>

<section class="lab" aria-labelledby="corner-title">
  <header><span>Corner-case lab</span><h3 id="corner-title">{current.question}</h3></header>
  <nav aria-label="Choose a compile corner case">
    {#each cases as item}<button class:active={kind===item.id} onclick={()=>choose(item.id)}>{item.label}</button>{/each}
  </nav>
  <div class="body">
    <div class="code"><div class="filebar">edge_case.py</div><pre><code>{code}</code></pre></div>
    <div class="visual">
      <div class="controlrow">
        {#if kind==='fixed-loop'}
          <span>iterations</span>{#each [2,4,8] as count}<button class:active={option===count} onclick={()=>option=count}>{count}</button>{/each}
        {:else if kind==='tensor-loop'}
          <span>expression</span><button class:active={!option} onclick={()=>option=false}>Python while</button><button class:active={Boolean(option)} onclick={()=>option=true}>while_loop</button>
        {:else if kind==='scalar-item'}
          <span>scalar capture</span><button class:active={!option} onclick={()=>option=false}>off</button><button class:active={Boolean(option)} onclick={()=>option=true}>on</button>
        {:else if kind==='shape-branch'}
          <span>batch size</span><button class:active={option===16} onclick={()=>option=16}>16</button><button class:active={option===64} onclick={()=>option=64}>64</button>
        {:else}
          <span>branch outputs</span><button class:active={Boolean(option)} onclick={()=>option=true}>compatible</button><button class:active={!option} onclick={()=>option=false}>mismatched</button>
        {/if}
      </div>
      <div class="result"><small>compiler outcome</small><b class:error={plan.outcome==='capture error'}>{plan.outcome}</b>{#if plan.guard}<code>{plan.guard}</code>{/if}</div>

      {#if kind==='fixed-loop'}
        <div class="unrolled" aria-label={`${option} loop iterations unrolled into ${plan.nodeCount} graph operations`}>
          {#each plan.regions as region,index}<div class="iteration"><small>{index+1}</small><span>sin</span><i>→</i><span>add</span></div>{/each}
        </div>
        <p class="takeaway">The loop disappears. Dynamo makes a straight-line graph and guards the iteration count. More iterations mean a proportionally larger graph.</p>
      {:else if kind==='tensor-loop' && !option}
        <div class="flow"><Node label="FX prefix" detail="norm"/><Arrow/><Node danger label="GRAPH BREAK" detail="tensor controls while"/><Arrow/><Node dashed label="Python loop" detail="test, run, test again"/></div>
        <p class="takeaway">A changing tensor decides how often Python loops, so there is no fixed trip count to unroll.</p>
      {:else if kind==='tensor-loop'}
        <div class="structured"><b>while_loop operator</b><div><Node label="condition graph" detail="continue?"/><Node label="body graph" detail="next carried state"/></div></div>
        <p class="takeaway"><code>while_loop</code> preserves the loop, but its current prototype implementation is inference-only: it is not an autograd replacement for training loops.</p>
      {:else if kind==='scalar-item' && !option}
        <div class="flow"><Node label="FX region 1" detail="sum"/><Arrow/><Node danger label="Python scalar" detail=".item()"/><Arrow/><Node label="FX region 2" detail="multiply"/></div>
        <p class="takeaway"><code>.item()</code> is not harmless bookkeeping: by default it moves a value out of tensor computation and breaks capture.</p>
      {:else if kind==='scalar-item'}
        <div class="flow"><Node label="sum" detail="tensor"/><Arrow/><Node label="captured scalar" detail="symbolic value"/><Arrow/><Node label="multiply" detail="same graph"/></div>
        <p class="takeaway">Scalar-output capture can keep this expression together, subject to backend support. It does not automatically make a later tensor-dependent Python <code>if</code> structured.</p>
      {:else if kind==='shape-branch'}
        <div class="shape"><div class:chosen={Number(option)<=16}><b>N ≤ 16</b><span>small path</span></div><div class:chosen={Number(option)>16}><b>N &gt; 16</b><span>large path</span></div></div>
        <p class="takeaway">A symbolic batch dimension may still acquire a range guard because the graph took one shape-dependent branch. Crossing the boundary needs another valid graph.</p>
      {:else if kind==='cond-contract' && option}
        <div class="structured"><b>cond</b><div><Node label="true graph" detail="output [N, D]"/><Node label="false graph" detail="output [N, D]"/></div></div>
        <p class="takeaway">Both subgraphs fit one output contract, so the predicate can select either at runtime.</p>
      {:else}
        <div class="branches bad"><Node label="true graph" detail="output [N, D]"/><span>≠</span><Node danger label="false graph" detail="output [N]"/></div>
        <p class="takeaway">Structured control flow is not “capture any two functions.” Branch outputs must be compatible, and branch functions cannot rely on unsupported mutation or aliasing.</p>
      {/if}
    </div>
  </div>
</section>

{#snippet Node(label:string,detail:string,danger=false,dashed=false)}<div class:danger class:dashed class="node"><b>{label}</b><span>{detail}</span></div>{/snippet}
{#snippet Arrow()}<i class="arrow">→</i>{/snippet}

<style>
  .lab{margin:1.5rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden;box-shadow:0 16px 45px var(--shadow)}header{padding:1rem}header span{font:800 .58rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0 0}nav{display:grid;grid-template-columns:repeat(5,1fr);border-block:1px solid var(--border)}nav button{border:0;border-right:1px solid var(--border);padding:.65rem .4rem;background:var(--bg);color:var(--text-muted);font-size:.56rem}nav button:last-child{border:0}nav button.active{background:var(--accent);color:white}.body{display:grid;grid-template-columns:.82fr 1.35fr;min-height:430px;background:#111925;color:white}.code{border-right:1px solid #303b4c}.filebar{padding:.55rem .7rem;background:#1b2634;color:#a0adbd;font:.55rem var(--font-mono)}pre{box-sizing:border-box;margin:0;padding:1rem;overflow:auto;color:#dce4ed;font:.57rem/1.65 var(--font-mono)}.visual{display:flex;flex-direction:column;padding:.8rem}.controlrow{display:flex;align-items:center;gap:.3rem;flex-wrap:wrap}.controlrow>span{margin-right:auto;color:#8e9caf;font-size:.51rem}.controlrow button{border:1px solid #3d4a5d;background:#182332;color:#bdc7d5;padding:.38rem .55rem;font-size:.52rem}.controlrow button.active{border-color:#df7656;color:#ff9b79}.result{display:flex;align-items:center;gap:.55rem;margin:.7rem 0;padding:.55rem .65rem;border:1px solid #354255;border-radius:5px;background:#192432}.result small{color:#79889b;font:.47rem var(--font-mono)}.result b{font-size:.58rem}.result b.error{color:#ff9b79}.result code{margin-left:auto;color:#ff9b79;font-size:.48rem}.flow,.branches{display:flex;flex:1;align-items:center;justify-content:center;gap:.45rem}.node{min-width:100px;flex:1;padding:.7rem;border:1px solid #687e9d;border-radius:6px;background:#192433}.node b,.node span{display:block}.node b{font-size:.57rem}.node span{margin-top:.2rem;color:#97a4b5;font-size:.49rem}.node.danger{border-color:#df6540;background:#32231f}.node.dashed{border-style:dashed}.arrow{color:#77879a;font-style:normal}.unrolled{display:flex;align-items:center;gap:.35rem;flex:1;overflow:auto}.iteration{min-width:62px;padding:.45rem;border-top:2px solid #687e9d;background:#192433}.iteration small{display:block;color:#79889b;font-size:.44rem}.iteration span,.iteration i{font:normal .48rem var(--font-mono)}.iteration i{color:#77879a}.structured{flex:1;margin-top:.3rem;padding:.75rem;border:1px solid #566a88;border-radius:7px}.structured>b{font:.6rem var(--font-mono)}.structured>div{display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin-top:1rem}.shape{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;align-items:stretch;flex:1}.shape div{display:flex;flex-direction:column;justify-content:center;padding:1rem;border:1px dashed #3d4a5d;border-radius:6px;opacity:.45}.shape div.chosen{border:2px solid #df7656;background:#32231f;opacity:1}.shape b,.shape span{display:block}.shape b{font:.65rem var(--font-mono)}.shape span{margin-top:.3rem;color:#97a4b5;font-size:.5rem}.branches.bad>span{color:#ff9b79}.takeaway{min-height:2.2rem;margin:.55rem 0 0;color:#9eabba;font-size:.52rem;line-height:1.45}.takeaway code{color:#e0e6ed}@media(max-width:760px){nav{grid-template-columns:repeat(3,1fr)}.body{grid-template-columns:1fr}.code{border-right:0;border-bottom:1px solid #303b4c}.visual{min-height:380px}.result{align-items:flex-start;flex-wrap:wrap}.result code{width:100%;margin-left:0}.unrolled{min-height:150px}.flow{min-height:180px;overflow:auto;justify-content:flex-start}.node{min-width:110px}}@media(max-width:450px){nav{grid-template-columns:1fr 1fr}.structured>div,.shape{grid-template-columns:1fr}.branches{flex-direction:column}.branches.bad>span{transform:rotate(90deg)}}
</style>
