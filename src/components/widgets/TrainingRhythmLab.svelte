<script lang="ts">
  type Task = 'regression' | 'binary' | 'multiclass' | 'reconstruction';
  const tasks: {id:Task; label:string; output:string; loss:string; question:string}[] = [
    {id:'regression',label:'Regression',output:'one number',loss:'MSELoss',question:'How far is the predicted value from the target?'},
    {id:'binary',label:'Binary class',output:'one logit',loss:'BCEWithLogitsLoss',question:'How much evidence supports class 1?'},
    {id:'multiclass',label:'3 classes',output:'three logits',loss:'CrossEntropyLoss',question:'Does the target class have the strongest logit?'},
    {id:'reconstruction',label:'Reconstruct',output:'an image-shaped tensor',loss:'MSELoss',question:'How closely does the output reproduce every input value?'}
  ];
  const stages = [
    ['zero_grad()', 'clear gradients left by the previous batch'],
    ['model(x)', 'produce an output with the current parameters'],
    ['loss_fn(output, y)', 'turn task-specific error into one scalar'],
    ['loss.backward()', 'measure every parameter’s responsibility'],
    ['optimizer.step()', 'use those gradients to update parameters']
  ];
  let task:Task='regression',stage=1;
  $: selected=tasks.find(item=>item.id===task)!;
  $: updated=stage===4;
  const before=[.05,.72,.18,.9,.34,.8,.12,.65,.14,.84,.28,.75,.08,.6,.2,.88];
  const target=[0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0];
  $: pixels=before.map((value,index)=>updated ? value+.62*(target[index]-value) : value);
</script>

<section class="lab" aria-labelledby="rhythm-title">
  <header>
    <div><span>Interactive comparison</span><h3 id="rhythm-title">One rhythm, four learning problems</h3></div>
    <p><b>{selected.output}</b><small>{selected.loss}</small></p>
  </header>
  <nav aria-label="Choose a learning problem">
    {#each tasks as item}<button class:active={task===item.id} onclick={()=>{task=item.id;stage=1}}>{item.label}</button>{/each}
  </nav>
  <div class="body">
    <ol aria-label="Training step">
      {#each stages as item,index}
        <li class:active={stage===index} class:done={stage>index}>
          <button onclick={()=>stage=index}><i>{index+1}</i><code>{item[0]}</code><small>{item[1]}</small></button>
        </li>
      {/each}
    </ol>
    <div class="stage">
      <div class="prompt"><small>The task-specific question</small><strong>{selected.question}</strong></div>
      {#if task==='regression'}
        <svg viewBox="0 0 420 220" role="img" aria-label="Regression line moving closer to a target point after an optimizer step">
          <line class="axis" x1="40" y1="185" x2="390" y2="185"/><line class="axis" x1="40" y1="25" x2="40" y2="185"/>
          <path class="fit" d={updated?'M 40 176 L 390 38':'M 40 168 L 390 100'}/>
          <circle class="target" cx="300" cy="55" r="7"/><circle class="prediction" cx="300" cy={updated?73:116} r="7"/>
          <line class="error" x1="300" y1="55" x2="300" y2={updated?73:116}/><text x="310" y="52">target = 5</text><text x="310" y={updated?82:125}>prediction = {updated?'3.75':'2.50'}</text>
        </svg>
      {:else if task==='binary'}
        <svg viewBox="0 0 420 220" role="img" aria-label="Binary classification score moving toward the positive target">
          <line class="scale" x1="55" y1="120" x2="365" y2="120"/><text x="48" y="150">class 0</text><text x="330" y="150">class 1</text>
          <circle class="target" cx="365" cy="120" r="8"/><circle class="prediction" cx={updated?226:179} cy="120" r="9"/>
          <path class="arrow" d={updated?'M 179 91 L 218 91':'M 179 91 L 179 91'}/><text x="145" y="62">p(class 1) = {updated?'0.55':'0.40'}</text>
        </svg>
      {:else if task==='multiclass'}
        <div class="bars" role="img" aria-label="Three class probabilities with class one as the target">
          {#each (updated?[.44,.40,.16]:[.56,.28,.16]) as probability,index}
            <div class:target-bar={index===1}><span style={`height:${probability*180}px`}></span><b>{Math.round(probability*100)}%</b><small>class {index}{index===1?' · target':''}</small></div>
          {/each}
        </div>
      {:else}
        <div class="reconstruction" role="img" aria-label="Input pixels and reconstructed pixels becoming more similar">
          <div><small>input</small><div class="pixel-grid">{#each target as value}<i style={`opacity:${.12+.88*value}`}></i>{/each}</div></div>
          <b>→</b>
          <div><small>model output</small><div class="pixel-grid">{#each pixels as value}<i style={`opacity:${.12+.88*value}`}></i>{/each}</div></div>
        </div>
      {/if}
      <footer><code>{selected.loss}</code><b>{stage<2?'not computed yet':updated?'smaller after update':'one scalar error'}</b></footer>
    </div>
  </div>
  <aside><b>Changes with the task:</b> output shape, target meaning, loss, and metric. <b>Stays the same:</b> the five-step parameter lifecycle.</aside>
</section>

<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden;box-shadow:0 16px 45px var(--shadow)}header{display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:1rem 1.1rem}header span,.prompt small{font:800 .59rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0}header p{margin:0;text-align:right}header p b,header p small{display:block;font:.68rem var(--font-mono)}header p small{margin-top:.2rem;color:var(--text-muted)}nav{display:grid;grid-template-columns:repeat(4,1fr);border-block:1px solid var(--border)}nav button{border:0;border-right:1px solid var(--border);padding:.65rem;background:var(--bg);color:var(--text-muted);font-weight:720}nav button:last-child{border:0}nav button.active{background:var(--accent);color:white}.body{display:grid;grid-template-columns:minmax(210px,.8fr) 1.4fr}.body ol{list-style:none;margin:0;padding:.7rem;border-right:1px solid var(--border)}li button{display:grid;grid-template-columns:24px 1fr;gap:.05rem .55rem;width:100%;padding:.56rem;border:0;border-radius:8px;background:transparent;color:var(--text);text-align:left}li button:hover,li.active button{background:var(--accent-soft)}li i{grid-row:1/3;display:grid;place-items:center;width:22px;height:22px;border:1px solid var(--border);border-radius:50%;font:normal .58rem var(--font-mono)}li.active i{background:var(--accent);border-color:var(--accent);color:white}li.done i{border-color:var(--accent);color:var(--accent)}li code{font-size:.67rem}li small{color:var(--text-subtle);font-size:.57rem;line-height:1.3}.stage{min-width:0;background:#121925;color:white}.prompt{padding:.85rem 1rem 0}.prompt small,.prompt strong{display:block}.prompt strong{margin-top:.25rem;font-size:.76rem}.stage svg{display:block;width:100%;height:220px}.axis,.scale{stroke:#566278;stroke-width:1}.fit{fill:none;stroke:#8ea6ca;stroke-width:3}.target{fill:#df5b35}.prediction{fill:#8ea6ca;stroke:white;stroke-width:2}.error{stroke:#df5b35;stroke-width:2;stroke-dasharray:5}.arrow{fill:none;stroke:#8ea6ca;stroke-width:2}.stage text{fill:#aab4c3;font:11px var(--font-mono)}.bars{height:220px;display:flex;justify-content:center;align-items:flex-end;gap:2rem;padding:20px}.bars div{height:190px;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;gap:.3rem}.bars span{display:block;width:54px;background:#71819a;border-radius:5px 5px 0 0;transition:height .25s}.bars .target-bar span{background:#df5b35}.bars b{font:.68rem var(--font-mono)}.bars small{color:#99a5b6;font-size:.57rem}.reconstruction{height:220px;display:flex;align-items:center;justify-content:center;gap:1.5rem}.reconstruction>div>small{display:block;margin-bottom:.4rem;color:#99a5b6;font:.6rem var(--font-mono)}.pixel-grid{display:grid;grid-template-columns:repeat(4,30px);gap:4px}.pixel-grid i{display:block;width:30px;height:30px;background:#f2f4f7;border-radius:3px;transition:opacity .25s}.stage footer{display:flex;justify-content:space-between;padding:.65rem 1rem;border-top:1px solid #303949;color:#aab4c3}.stage footer b{font-size:.65rem}.stage footer code{font-size:.62rem}aside{padding:.75rem 1rem;background:var(--accent-soft);color:var(--text-muted);font-size:.67rem}aside b{color:var(--text)}@media(max-width:720px){nav{grid-template-columns:1fr 1fr}.body{grid-template-columns:1fr}.body ol{border:0;border-bottom:1px solid var(--border)}header{align-items:flex-start}.bars{gap:.8rem}.pixel-grid{grid-template-columns:repeat(4,24px)}.pixel-grid i{width:24px;height:24px}}
</style>
