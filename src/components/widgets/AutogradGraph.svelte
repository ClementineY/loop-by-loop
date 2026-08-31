<script lang="ts">
  import { V } from '../../lib/autograd';

  let x = 2;
  let w = -1;
  let b = 0.5;
  let backward = false;

  $: graph = build(+x, +w, +b, backward);

  function build(xn: number, wn: number, bn: number, run: boolean) {
    const xv = V(xn);
    const wv = V(wn);
    const bv = V(bn);
    const m = xv.mul(wv);
    const s = m.add(bv);
    const y = s.pow(2);
    if (run) y.backward();
    return { x: xv, w: wv, b: bv, m, s, y };
  }

  const f = (value: number) => value.toFixed(2);
  function changed() { backward = false; }
</script>

<section class="lab" aria-labelledby="auto-title">
  <div class="head">
    <div>
      <span>Interactive lab</span>
      <h3 id="auto-title">Follow the gradient</h3>
      <code>m = x·w &nbsp;→&nbsp; s = m+b &nbsp;→&nbsp; y = s²</code>
    </div>
    <button onclick={() => backward = true}>Run backward →</button>
  </div>

  <div class="sliders">
    <label>x <b>{x}</b><input type="range" min="-3" max="3" step=".5" bind:value={x} oninput={changed}/></label>
    <label>w <b>{w}</b><input type="range" min="-3" max="3" step=".5" bind:value={w} oninput={changed}/></label>
    <label>b <b>{b}</b><input type="range" min="-2" max="2" step=".5" bind:value={b} oninput={changed}/></label>
  </div>

  <div class="forward" aria-label="Forward computation in three steps">
    <article>
      <div class="step-head"><b>1</b><span>multiply</span></div>
      <div class="equation">
        <span class="value">x <strong>{f(graph.x.data)}</strong></span>
        <i>×</i>
        <span class="value">w <strong>{f(graph.w.data)}</strong></span>
        <i>=</i>
        <span class="value result">m <strong>{f(graph.m.data)}</strong></span>
      </div>
      {#if backward}<p>grad m = {f(graph.m.grad)}</p>{/if}
    </article>

    <article>
      <div class="step-head"><b>2</b><span>add</span></div>
      <div class="equation">
        <span class="value">m <strong>{f(graph.m.data)}</strong></span>
        <i>+</i>
        <span class="value">b <strong>{f(graph.b.data)}</strong></span>
        <i>=</i>
        <span class="value result">s <strong>{f(graph.s.data)}</strong></span>
      </div>
      {#if backward}<p>grad s = {f(graph.s.grad)}</p>{/if}
    </article>

    <article class="output">
      <div class="step-head"><b>3</b><span>square</span></div>
      <div class="equation">
        <span class="value">s <strong>{f(graph.s.data)}</strong></span>
        <i>²</i>
        <i>=</i>
        <span class="value result">y <strong>{f(graph.y.data)}</strong></span>
      </div>
      {#if backward}<p>grad y = {f(graph.y.grad)} (the seed)</p>{/if}
    </article>
  </div>

  {#if backward}
    <div class="trace" aria-live="polite">
      <div class="trace-head">
        <div><span>Backward trace</span><h4>Right to left, one local rule at a time</h4></div>
        <p><code>grad(parent) = grad(result) × local derivative</code></p>
      </div>

      <ol>
        <li>
          <b>Seed the output</b>
          <p>Start with <code>grad y = ∂y/∂y = 1.00</code>.</p>
        </li>
        <li>
          <b>Undo the square</b>
          <p>The local derivative of <code>y = s²</code> is <code>2s = {f(2 * graph.s.data)}</code>.</p>
          <code>grad s = 1.00 × {f(2 * graph.s.data)} = {f(graph.s.grad)}</code>
        </li>
        <li class="branch">
          <b>Branch through the addition</b>
          <p>For <code>s = m + b</code>, changing either input by 1 changes <code>s</code> by 1. Both local derivatives are therefore 1.</p>
          <div>
            <code>grad m = {f(graph.s.grad)} × 1 = {f(graph.m.grad)}</code>
            <code>grad b = {f(graph.s.grad)} × 1 = {f(graph.b.grad)}</code>
          </div>
          <strong>The gradient branches; it does not divide. Gradients measure sensitivity, not a conserved quantity.</strong>
        </li>
        <li>
          <b>Undo the multiplication</b>
          <p>For <code>m = x·w</code>, the local derivative toward <code>x</code> is <code>w</code>, and toward <code>w</code> it is <code>x</code>.</p>
          <div>
            <code>grad x = {f(graph.m.grad)} × {f(graph.w.data)} = {f(graph.x.grad)}</code>
            <code>grad w = {f(graph.m.grad)} × {f(graph.x.data)} = {f(graph.w.grad)}</code>
          </div>
        </li>
      </ol>
    </div>
  {:else}
    <p class="caption">Read the three forward equations first. Then predict the signs of the gradients and run backward.</p>
  {/if}
</section>

<style>
  .lab{margin:2rem 0;padding:1rem;border:1px solid var(--border);border-radius:14px;background:var(--surface);overflow:hidden}
  .head{display:flex;justify-content:space-between;align-items:center;gap:1rem}
  .head span,.trace-head span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}
  .head h3{margin:.2rem 0}.head code{font-size:.72rem;color:var(--text-muted)}
  .head button{border:0;border-radius:8px;background:var(--accent);color:white;padding:.55rem .7rem;font-weight:750;font-size:.75rem}
  .sliders{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin:1rem 0}
  .sliders label{display:grid;grid-template-columns:1fr auto;font-size:.7rem;color:var(--text-muted)}
  .sliders input{grid-column:1/-1;accent-color:var(--accent)}
  .forward{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:#344052;border:1px solid #344052;border-radius:11px;overflow:hidden}
  .forward article{background:var(--ink);color:white;padding:.8rem;min-width:0}
  .forward article.output{background:#2d211e}
  .step-head{display:flex;align-items:center;gap:.45rem;color:#929dae;font:700 .6rem var(--font-mono);text-transform:uppercase;letter-spacing:.08em}
  .step-head b{display:grid;place-items:center;width:1.25rem;height:1.25rem;border-radius:50%;background:#2d3747;color:white}
  .equation{display:flex;align-items:center;justify-content:center;gap:.28rem;margin:.8rem 0 .35rem}
  .equation i{font-style:normal;color:#8994a5;font:700 .75rem var(--font-mono)}
  .value{display:grid;gap:.1rem;min-width:42px;padding:.35rem;border:1px solid #3a4556;border-radius:6px;color:#98a3b4;font:600 .57rem var(--font-mono)}
  .value strong{color:white;font-size:.72rem}.value.result{border-color:var(--accent);background:#35241f}
  .forward article>p{margin:.55rem 0 0;color:var(--accent-light);font:700 .65rem var(--font-mono);text-align:center}
  .trace{margin-top:1rem;border:1px solid var(--border);border-radius:11px;overflow:hidden}
  .trace-head{display:flex;justify-content:space-between;gap:1rem;align-items:end;padding:.85rem;background:var(--bg)}
  .trace-head h4{margin:.15rem 0 0;font-size:.9rem}.trace-head p{margin:0;color:var(--text-muted);font-size:.65rem}
  .trace ol{list-style:none;counter-reset:trace;margin:0;padding:0}
  .trace li{display:grid;grid-template-columns:145px 1fr;gap:.25rem 1rem;padding:.8rem;border-top:1px solid var(--border);font-size:.75rem}
  .trace li>b{grid-row:1/4;color:var(--text)}.trace li>p{margin:0;color:var(--text-muted)}
  .trace li>code,.trace li div{grid-column:2}.trace li div{display:flex;flex-wrap:wrap;gap:.5rem}
  .trace li code{font-size:.68rem}.trace li>strong{grid-column:2;color:var(--accent);font-size:.7rem}
  .caption{color:var(--text-muted);font-size:.75rem;margin:.7rem .2rem 0}
  @media(max-width:700px){.forward{grid-template-columns:1fr}.trace-head{display:block}.trace li{grid-template-columns:1fr}.trace li>b{grid-row:auto}.trace li>code,.trace li div,.trace li>strong{grid-column:1}}
  @media(max-width:520px){.sliders{grid-template-columns:1fr}.head{align-items:flex-start}.head code{display:none}}
</style>
