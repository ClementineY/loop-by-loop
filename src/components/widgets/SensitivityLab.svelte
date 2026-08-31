<script lang="ts">
  let w = 0.5;
  const x = 2;
  const target = 3;
  const delta = 0.1;
  const minW = -1;
  const maxW = 3;
  const maxLoss = 25;

  const lossAt = (weight: number) => (weight * x - target) ** 2;
  const gradientAt = (weight: number) => 2 * (weight * x - target) * x;
  const clamp = (value: number) => Math.max(minW, Math.min(maxW, value));
  const px = (weight: number) => 42 + ((weight - minW) / (maxW - minW)) * 476;
  const py = (loss: number) => 184 - (Math.min(maxLoss, Math.max(0, loss)) / maxLoss) * 148;
  const fmt = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}`;

  $: prediction = w * x;
  $: loss = lossAt(w);
  $: gradient = gradientAt(w);
  $: nextLoss = lossAt(w + delta);
  $: actualChange = nextLoss - loss;
  $: estimatedChange = gradient * delta;
  $: curve = Array.from({ length: 81 }, (_, index) => {
    const weight = minW + (index / 80) * (maxW - minW);
    return `${index ? 'L' : 'M'} ${px(weight).toFixed(1)} ${py(lossAt(weight)).toFixed(1)}`;
  }).join(' ');
  $: tangentLeft = clamp(w - 0.48);
  $: tangentRight = clamp(w + 0.48);
  $: interpretation = Math.abs(gradient) < 0.15
    ? 'Nearly flat: a small change in w barely changes the loss.'
    : gradient < 0
      ? 'Negative: increasing w a little should lower the loss.'
      : 'Positive: increasing w a little should raise the loss.';
</script>

<section class="lab" aria-labelledby="sensitivity-title">
  <header>
    <div><span>Sensitivity lab</span><h3 id="sensitivity-title">Nudge one value. Watch the loss respond.</h3></div>
    <code>loss = (2w − 3)²</code>
  </header>

  <div class="stage">
    <svg viewBox="0 0 560 220" role="img" aria-label="Loss curve with the current parameter and its local tangent">
      <line class="axis" x1="42" y1="184" x2="524" y2="184" />
      <line class="axis" x1="42" y1="28" x2="42" y2="184" />
      <text x="505" y="207">w</text><text x="10" y="30">loss</text>
      <path class="curve" d={curve} />
      <line class="tangent" x1={px(tangentLeft)} y1={py(loss + gradient * (tangentLeft - w))} x2={px(tangentRight)} y2={py(loss + gradient * (tangentRight - w))} />
      <line class="guide" x1={px(w)} y1={py(loss)} x2={px(w)} y2="184" />
      <circle class="point" cx={px(w)} cy={py(loss)} r="7" />
      <text class="point-label" x={Math.min(px(w) + 12, 465)} y={Math.max(py(loss) - 10, 24)}>gradient {fmt(gradient)}</text>
      <circle class="next" cx={px(w + delta)} cy={py(nextLoss)} r="4" />
    </svg>

    <div class="numbers">
      <article><small>parameter</small><b>w = {w.toFixed(2)}</b></article>
      <i>× {x}</i>
      <article><small>prediction</small><b>{prediction.toFixed(2)}</b></article>
      <i>compare with {target}</i>
      <article><small>loss</small><b>{loss.toFixed(2)}</b></article>
    </div>
  </div>

  <div class="control">
    <label><span>Move w</span><b>{w.toFixed(2)}</b><input aria-label="Parameter w" type="range" min={minW} max={maxW} step="0.05" bind:value={w} /></label>
    <button onclick={() => w = 1.5}>Move to the bottom</button>
  </div>

  <div class="readout">
    <article><span>If w increases by</span><b>+{delta.toFixed(2)}</b><p>a deliberately small nudge</p></article>
    <article><span>Gradient predicts</span><b>{fmt(estimatedChange)}</b><p>gradient × nudge</p></article>
    <article><span>Loss actually changes</span><b>{fmt(actualChange)}</b><p>close, but the curve bends</p></article>
  </div>

  <footer><strong>{interpretation}</strong><p>The gradient is the slope of the orange tangent at the current point—not the height of the loss.</p></footer>
</section>

<style>
  .lab{margin:2rem 0;border:1px solid var(--border);border-radius:16px;background:var(--surface);overflow:hidden}
  header{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:1rem}header span,.readout span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}header h3{margin:.2rem 0 0}header code{font-size:.72rem;color:var(--text-muted)}
  .stage{padding:1rem;background:#121925;color:white}.stage svg{display:block;width:100%;height:auto;max-height:300px}.axis{stroke:#465064;stroke-width:1}.stage text{fill:#8591a4;font:11px var(--font-mono)}.curve{fill:none;stroke:#68758a;stroke-width:3}.tangent{stroke:var(--accent-light);stroke-width:3}.guide{stroke:#5c687b;stroke-dasharray:4}.point{fill:var(--accent);stroke:#ffd0bf;stroke-width:3}.next{fill:#fff;stroke:var(--accent);stroke-width:2}.stage .point-label{fill:var(--accent-light);font-weight:700}
  .numbers{display:flex;align-items:center;justify-content:center;gap:.6rem;flex-wrap:wrap;margin-top:.5rem}.numbers article{min-width:100px;padding:.55rem .7rem;border:1px solid #394456;border-radius:8px;background:#1b2432}.numbers small{display:block;color:#8d99ab;font:.54rem var(--font-mono)}.numbers b{font:.75rem var(--font-mono)}.numbers i{color:#8d99ab;font:normal .58rem var(--font-mono)}
  .control{display:flex;align-items:center;gap:1rem;padding:.8rem 1rem;border-bottom:1px solid var(--border)}.control label{display:grid;grid-template-columns:1fr auto;flex:1;color:var(--text-muted);font-size:.68rem}.control label b{color:var(--text)}.control input{grid-column:1/-1;accent-color:var(--accent)}.control button{border:1px solid var(--border);border-radius:7px;background:var(--bg);color:var(--text);padding:.45rem .6rem;font-size:.68rem;font-weight:700}
  .readout{display:grid;grid-template-columns:repeat(3,1fr)}.readout article{padding:.9rem;border-right:1px solid var(--border)}.readout article:last-child{border:0}.readout b{display:block;margin:.25rem 0;font:750 1rem var(--font-mono)}.readout p,footer p{margin:0;color:var(--text-muted);font-size:.67rem}
  footer{padding:.8rem 1rem;background:var(--accent-soft);border-top:1px solid var(--border)}footer strong{display:block;font-size:.78rem;margin-bottom:.2rem}
  @media(max-width:620px){header,.control{align-items:flex-start;flex-direction:column}.control label{width:100%}.readout{grid-template-columns:1fr}.readout article{border-right:0;border-bottom:1px solid var(--border)}}
</style>
