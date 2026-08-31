<script lang="ts">
  import { V } from '../../lib/autograd';

  let phase = 0;
  const phases = ['Track leaves', 'Run forward', 'Run backward', 'Clear gradients'];

  $: graph = build(phase >= 2);

  function build(run: boolean) {
    const x = V(2);
    const w = V(-1);
    const b = V(0.5);
    const m = x.mul(w);
    const s = m.add(b);
    const loss = s.pow(2);
    if (run) loss.backward();
    if (phase === 3) { w.grad = 0; b.grad = 0; }
    return { x, w, b, m, s, loss };
  }

  const f = (value: number) => value.toFixed(2);
  const next = () => { phase = (phase + 1) % phases.length; };
</script>

<section class="lab" aria-labelledby="autograd-title">
  <header>
    <div><span>Autograd lifecycle</span><h3 id="autograd-title">Watch PyTorch build and use the tape</h3></div>
    <button onclick={next}>{phase === 3 ? 'Start again ↻' : 'Next step →'}</button>
  </header>

  <nav aria-label="Autograd lifecycle steps">
    {#each phases as label, index}
      <button class:active={phase === index} class:done={phase > index} onclick={() => phase = index}>
        <b>{index + 1}</b><span>{label}</span>
      </button>
    {/each}
  </nav>

  <div class="workspace">
    <div class="program" aria-label="PyTorch code for the current lifecycle">
      <p class:active={phase === 0}><i>1</i><code>w = torch.tensor(-1., <mark>requires_grad=True</mark>)</code></p>
      <p class:active={phase === 0}><i>2</i><code>b = torch.tensor(.5, <mark>requires_grad=True</mark>)</code></p>
      <p class:active={phase === 1}><i>3</i><code>loss = (x * w + b) ** 2</code></p>
      <p class:active={phase === 2}><i>4</i><code>loss.backward()</code></p>
      <p class:active={phase === 3}><i>5</i><code>w.grad.zero_(); b.grad.zero_()</code></p>
    </div>

    <aside aria-live="polite">
      {#if phase === 0}
        <span>01 · choose leaves</span>
        <h4>Track only what may learn.</h4>
        <p><code>w</code> and <code>b</code> are user-created leaves. Turning on <code>requires_grad</code> tells PyTorch to remember operations that depend on them.</p>
      {:else if phase === 1}
        <span>02 · ordinary Python runs</span>
        <h4>The forward pass leaves a trail.</h4>
        <p>Each result stores the operation that created it. The final loss points backward through <code>Pow → Add → Mul</code>.</p>
      {:else if phase === 2}
        <span>03 · ask one question</span>
        <h4>How would each leaf change loss?</h4>
        <p><code>backward()</code> walks the saved trail in reverse. The answers land in <code>w.grad</code> and <code>b.grad</code>.</p>
      {:else}
        <span>04 · prepare the next batch</span>
        <h4>Clear the answers, not the parameters.</h4>
        <p>PyTorch adds new gradients to old ones. The training loop clears <code>.grad</code> before measuring the next batch.</p>
      {/if}
    </aside>
  </div>

  <div class="graph" class:backward={phase === 2} class:cleared={phase === 3} aria-label="Recorded computation graph">
    <div class="leaves">
      <article class="plain"><small>input · not tracked</small><b>x</b><strong>{f(graph.x.data)}</strong></article>
      <article class="tracked"><small>leaf · requires_grad</small><b>w</b><strong>{f(graph.w.data)}</strong>{#if phase >= 2}<em>.grad {f(graph.w.grad)}</em>{/if}</article>
      <article class="tracked"><small>leaf · requires_grad</small><b>b</b><strong>{f(graph.b.data)}</strong>{#if phase >= 2}<em>.grad {f(graph.b.grad)}</em>{/if}</article>
    </div>

    <div class="arrow"><i>→</i><small>{phase === 0 ? 'not run yet' : 'record'}</small></div>

    <div class="operations" class:hidden={phase === 0}>
      <article><small>MulBackward</small><b>m = x·w</b><strong>{f(graph.m.data)}</strong></article>
      <i>→</i>
      <article><small>AddBackward</small><b>s = m+b</b><strong>{f(graph.s.data)}</strong></article>
      <i>→</i>
      <article class="loss"><small>PowBackward</small><b>loss = s²</b><strong>{f(graph.loss.data)}</strong>{#if phase === 2}<em>seed 1.00</em>{/if}</article>
    </div>
  </div>

  {#if phase === 2}
    <div class="backward-strip" aria-label="Backward pass summary">
      <span><b>square</b><code>1 × 2s → −3</code></span>
      <i>←</i>
      <span><b>add</b><code>−3 × 1 → m:−3, b:−3</code></span>
      <i>←</i>
      <span><b>multiply</b><code>w.grad:−6 · x path:3</code></span>
    </div>
  {/if}

  <footer><b>{phase + 1} / 4</b><p>{phases[phase]}</p></footer>
</section>

<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}
  header{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem}
  header span,aside>span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.11em;color:var(--accent)}
  header h3{margin:.2rem 0 0}header button{border:0;border-radius:8px;background:var(--accent);color:white;padding:.58rem .8rem;font-weight:750;font-size:.75rem}
  nav{display:grid;grid-template-columns:repeat(4,1fr);border-block:1px solid var(--border);background:var(--bg)}
  nav button{display:flex;align-items:center;gap:.45rem;border:0;border-right:1px solid var(--border);padding:.65rem;background:transparent;color:var(--text-muted);font-size:.68rem;text-align:left}
  nav button:last-child{border-right:0}nav b{display:grid;place-items:center;width:1.25rem;height:1.25rem;border:1px solid var(--border);border-radius:50%;font:700 .57rem var(--font-mono)}
  nav button.active{background:var(--surface);color:var(--text);box-shadow:inset 0 -2px var(--accent)}nav button.active b,nav button.done b{background:var(--accent);border-color:var(--accent);color:white}
  .workspace{display:grid;grid-template-columns:1.4fr 1fr;border-bottom:1px solid var(--border)}
  .program{background:var(--code-bg);padding:.8rem 0;color:white;overflow:auto}.program p{display:flex;margin:0;padding:.42rem .8rem;opacity:.46;white-space:nowrap}.program p.active{opacity:1;background:#ffffff0d;box-shadow:inset 3px 0 var(--accent)}
  .program i{width:1.8rem;color:#647084;font:normal .62rem var(--font-mono)}.program code{font-size:.67rem;background:transparent;padding:0}.program mark{background:transparent;color:var(--accent-light)}
  aside{padding:1rem;background:var(--bg)}aside h4{margin:.25rem 0 .45rem;font-size:.95rem}aside p{margin:0;color:var(--text-muted);font-size:.75rem;line-height:1.55}
  .graph{display:flex;align-items:center;gap:.7rem;padding:1rem;background:#121925;color:white;overflow:auto}
  .leaves{display:grid;grid-template-columns:repeat(3,92px);gap:.4rem}.graph article{position:relative;display:grid;grid-template-columns:1fr auto;gap:.2rem;padding:.55rem;border:1px solid #354052;border-radius:8px;background:#1b2432;min-height:58px}
  .graph article small{grid-column:1/-1;color:#8d98aa;font:.53rem var(--font-mono)}.graph article b{font:700 .68rem var(--font-mono)}.graph article strong{font:800 .73rem var(--font-mono)}.graph article em{grid-column:1/-1;color:var(--accent-light);font:normal .58rem var(--font-mono)}
  .graph article.tracked{border-color:#6b4b40}.graph article.plain{opacity:.65}.arrow{display:grid;place-items:center;color:#7f8a9b;font-style:normal}.arrow small{font:.5rem var(--font-mono)}
  .operations{display:flex;align-items:center;gap:.35rem;transition:opacity .25s}.operations.hidden{opacity:.18}.operations article{min-width:104px}.operations>i{color:#707c8f;font-style:normal}.operations .loss{border-color:var(--accent);background:#34241f}
  .graph.backward .operations>i{color:var(--accent-light)}.graph.cleared .operations{opacity:.36}
  .backward-strip{display:grid;grid-template-columns:1fr auto 1.3fr auto 1fr;align-items:center;gap:.5rem;padding:.7rem;background:var(--accent-soft);border-top:1px solid var(--border)}
  .backward-strip span{display:grid;gap:.12rem}.backward-strip b{font-size:.65rem}.backward-strip code{font-size:.58rem;background:transparent;padding:0;color:var(--text-muted)}.backward-strip i{font-style:normal;color:var(--accent)}
  footer{display:flex;align-items:center;gap:.5rem;padding:.55rem 1rem;color:var(--text-muted);font-size:.65rem}footer b{color:var(--accent);font-family:var(--font-mono)}footer p{margin:0}
  @media(max-width:720px){nav{grid-template-columns:1fr 1fr}.workspace{grid-template-columns:1fr}.graph{display:block}.arrow{margin:.5rem}.operations{min-width:390px}.backward-strip{grid-template-columns:1fr}.backward-strip i{transform:rotate(90deg);justify-self:center}}
  @media(max-width:520px){header{align-items:flex-start}.leaves{grid-template-columns:repeat(3,82px)}nav button span{font-size:.6rem}}
</style>
