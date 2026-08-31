<script lang="ts">
  import { V } from '../../lib/autograd';

  let detached = true;
  let backward = false;
  let x = 3;

  $: graph = build(+x, detached, backward);

  function build(input: number, cut: boolean, run: boolean) {
    const teacherWeight = V(2);
    const studentWeight = V(1);
    const teacherPrediction = teacherWeight.mul(input);
    const target = cut ? V(teacherPrediction.data) : teacherPrediction;
    const studentPrediction = studentWeight.mul(input);
    const difference = studentPrediction.sub(target);
    const loss = difference.pow(2);
    if (run) loss.backward();
    return { teacherWeight, studentWeight, teacherPrediction, target, studentPrediction, loss };
  }

  const f = (value: number) => value.toFixed(1);
  function choose(value: boolean) { detached = value; backward = false; }
  function changed() { backward = false; }
</script>

<section class="lab" aria-labelledby="detach-title">
  <header>
    <div><span>Gradient-path lab</span><h3 id="detach-title">Detach changes the route, not the value</h3></div>
    <button onclick={() => backward = true}>Run backward →</button>
  </header>

  <div class="controls">
    <div class="modes" aria-label="Target graph mode">
      <button class:active={!detached} onclick={() => choose(false)}>Connected target</button>
      <button class:active={detached} onclick={() => choose(true)}>Detached target ✂</button>
    </div>
    <label>input x <b>{x}</b><input type="range" min="1" max="4" step="1" bind:value={x} oninput={changed}/></label>
  </div>

  <div class="story">
    <div class="lane teacher" class:cut={detached}>
      <span class="lane-label">Teacher branch</span>
      <article><small>parameter</small><b>wₜ</b><strong>2.0</strong>{#if backward}<em>grad {detached ? '—' : f(graph.teacherWeight.grad)}</em>{/if}</article>
      <i>× x →</i>
      <article><small>prediction</small><b>teacher(x)</b><strong>{f(graph.teacherPrediction.data)}</strong></article>
      <div class="bridge"><i>{detached ? '✂' : '→'}</i><small>{detached ? 'history stops' : 'history connected'}</small></div>
      <article class="target"><small>{detached ? 'plain tensor' : 'tracked tensor'}</small><b>target</b><strong>{f(graph.target.data)}</strong></article>
    </div>

    <div class="lane student">
      <span class="lane-label">Student branch</span>
      <article><small>parameter</small><b>wₛ</b><strong>1.0</strong>{#if backward}<em>grad {f(graph.studentWeight.grad)}</em>{/if}</article>
      <i>× x →</i>
      <article><small>prediction</small><b>student(x)</b><strong>{f(graph.studentPrediction.data)}</strong></article>
      <div class="bridge"><i>→</i><small>history connected</small></div>
      <article class="target"><small>tracked tensor</small><b>prediction</b><strong>{f(graph.studentPrediction.data)}</strong></article>
    </div>

    <div class="loss">
      <small>compare the two values</small>
      <code>(prediction − target)²</code>
      <b>loss {f(graph.loss.data)}</b>
      {#if backward}<em>backward starts here</em>{/if}
    </div>
  </div>

  <div class="result" class:visible={backward} aria-live="polite">
    <div><span>Forward values</span><b>identical in both modes</b><p>Detaching does not change <code>target={f(graph.target.data)}</code> or <code>loss={f(graph.loss.data)}</code>.</p></div>
    <div><span>Student gradient</span><b>{backward ? f(graph.studentWeight.grad) : 'run backward'}</b><p>The student stays connected and can learn from the target.</p></div>
    <div class:stopped={detached}><span>Teacher gradient</span><b>{backward ? (detached ? 'stopped ✂' : f(graph.teacherWeight.grad)) : 'run backward'}</b><p>{detached ? 'The target acts like a fixed snapshot for this loss.' : 'Without detach, this loss also changes the teacher.'}</p></div>
  </div>

  <footer>
    <code>{detached ? 'target = teacher(x).detach()' : 'target = teacher(x)'}</code>
    <p>{detached ? 'Same data. No route back to the teacher.' : 'The loss has routes to both models.'}</p>
  </footer>
</section>

<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}
  header{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem}header span,.result span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.11em;color:var(--accent)}header h3{margin:.2rem 0 0}header>button{border:0;border-radius:8px;background:var(--accent);color:white;padding:.58rem .8rem;font-weight:750;font-size:.75rem}
  .controls{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:.7rem 1rem;border-block:1px solid var(--border);background:var(--bg)}
  .modes{display:flex;border:1px solid var(--border);border-radius:8px;overflow:hidden}.modes button{border:0;border-right:1px solid var(--border);padding:.42rem .62rem;background:var(--surface);color:var(--text-muted);font-size:.68rem}.modes button:last-child{border:0}.modes button.active{background:var(--accent);color:white}
  .controls label{display:grid;grid-template-columns:1fr auto;width:180px;color:var(--text-muted);font-size:.67rem}.controls input{grid-column:1/-1;accent-color:var(--accent)}
  .story{padding:1rem;background:#121925;color:white}.lane{display:grid;grid-template-columns:100px 92px 34px 125px 72px 115px;align-items:center;gap:.35rem;margin-bottom:.7rem}.lane-label{font:700 .6rem var(--font-mono);color:#96a1b2}.lane article{display:grid;grid-template-columns:1fr auto;gap:.15rem;padding:.5rem;border:1px solid #384355;border-radius:8px;background:#1b2432}.lane article small{grid-column:1/-1;color:#8995a7;font:.52rem var(--font-mono)}.lane article b{font:.66rem var(--font-mono)}.lane article strong{font:.75rem var(--font-mono)}.lane article em{grid-column:1/-1;color:var(--accent-light);font:normal .57rem var(--font-mono)}
  .lane>i{font:normal .58rem var(--font-mono);color:#8894a6;text-align:center}.bridge{display:grid;place-items:center}.bridge i{font-style:normal;color:var(--accent-light);font-size:1rem}.bridge small{font:.48rem var(--font-mono);color:#8995a7}.teacher.cut .bridge{color:var(--accent)}.teacher.cut .target{border-style:dashed;opacity:.75}
  .loss{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:.7rem;margin-top:1rem;padding:.7rem;border:1px solid var(--accent);border-radius:9px;background:#32231f}.loss small{color:#a7b0bf;font:.55rem var(--font-mono)}.loss code{font-size:.68rem}.loss b{color:var(--accent-light);font:.75rem var(--font-mono)}.loss em{font:normal .55rem var(--font-mono);color:#a7b0bf}
  .result{display:grid;grid-template-columns:repeat(3,1fr);border-top:1px solid var(--border)}.result>div{padding:.8rem;border-right:1px solid var(--border)}.result>div:last-child{border:0}.result b{display:block;margin:.25rem 0;font-size:.88rem}.result p{margin:0;color:var(--text-muted);font-size:.67rem;line-height:1.45}.result .stopped{background:var(--accent-soft)}
  footer{display:flex;justify-content:space-between;gap:1rem;align-items:center;padding:.65rem 1rem;background:var(--bg);border-top:1px solid var(--border)}footer code{font-size:.68rem}footer p{margin:0;color:var(--text-muted);font-size:.68rem}
  @media(max-width:760px){.lane{grid-template-columns:1fr;gap:.45rem}.lane>i{transform:rotate(90deg)}.bridge{min-height:34px}.result{grid-template-columns:1fr}.result>div{border-right:0;border-bottom:1px solid var(--border)}}
  @media(max-width:520px){header,.controls,footer{align-items:flex-start;flex-direction:column}.controls label{width:100%}.modes{width:100%}.modes button{flex:1}}
</style>
