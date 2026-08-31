<script lang="ts">
  import { broadcastShape, canReshape, numel } from '../../lib/tensor';
  let rows = 2;
  let cols = 3;
  let targetRows = 3;
  let targetCols = 2;
  let otherRows = 1;
  let otherCols = 3;
  $: source = [rows, cols];
  $: target = [targetRows, targetCols];
  $: valid = canReshape(source, target);
  $: count = numel(source);
  $: broadcast = broadcastShape(source, [otherRows, otherCols]);
</script>

<section class="lab" aria-labelledby="tensor-title">
  <div class="lab-head"><div><span>Interactive lab</span><h3 id="tensor-title">Shape studio</h3></div><code>{rows} × {cols} → {targetRows} × {targetCols}</code></div>
  <div class="controls">
    <fieldset><legend>Source shape</legend><label>Rows <input type="range" min="1" max="6" bind:value={rows} /></label><label>Columns <input type="range" min="1" max="6" bind:value={cols} /></label></fieldset>
    <fieldset><legend>Reshape to</legend><label>Rows <input type="range" min="1" max="6" bind:value={targetRows} /></label><label>Columns <input type="range" min="1" max="6" bind:value={targetCols} /></label></fieldset>
  </div>
  <div class="stage">
    <div><small>Original</small><div class="grid" style={`--cols:${cols}`}>{#each Array(count) as _, i}<i>{i}</i>{/each}</div></div>
    <b aria-hidden="true">→</b>
    <div class:invalid={!valid}><small>{valid ? 'Same values, new view' : 'Element count mismatch'}</small><div class="grid" style={`--cols:${targetCols}`}>{#each Array(valid ? count : Math.min(targetRows*targetCols,36)) as _, i}<i>{i}</i>{/each}</div></div>
  </div>
  <div class="broadcast"><div><b>Broadcast challenge</b><span>[{rows}, {cols}] + [{otherRows}, {otherCols}]</span></div><label>Other rows <input aria-label="Other tensor rows" type="range" min="1" max="4" bind:value={otherRows}/></label><label>Other cols <input aria-label="Other tensor columns" type="range" min="1" max="5" bind:value={otherCols}/></label><strong>{broadcast ? `→ [${broadcast.join(', ')}]` : 'incompatible'}</strong></div>
</section>

<style>
  .lab{margin:2rem 0;padding:1rem;border:1px solid var(--border);border-radius:14px;background:var(--surface);box-shadow:0 16px 45px var(--shadow)}.lab-head{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:.3rem .3rem 1rem;border-bottom:1px solid var(--border)}.lab-head span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}.lab-head h3{margin:.15rem 0 0;font-size:1.1rem}.lab-head code{color:var(--text-muted)}.controls{display:grid;grid-template-columns:1fr 1fr;gap:1rem;padding:1rem 0}.controls fieldset{border:1px solid var(--border);border-radius:9px;padding:.7rem}.controls legend{font-size:.72rem;font-weight:750}.controls label{display:grid;grid-template-columns:70px 1fr;align-items:center;font-size:.72rem;color:var(--text-muted)}input{accent-color:var(--accent)}.stage{display:grid;grid-template-columns:1fr auto 1fr;gap:1rem;align-items:center;background:var(--bg);border-radius:10px;padding:1rem;min-height:210px}.stage>div{min-width:0}.stage small{display:block;color:var(--text-muted);font-size:.7rem;margin-bottom:.5rem}.grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(18px,1fr));gap:3px;max-width:240px}.grid i{display:grid;place-items:center;aspect-ratio:1;border-radius:4px;background:var(--accent-soft);border:1px solid color-mix(in srgb,var(--accent) 28%,var(--border));font:650 .62rem var(--font-mono);font-style:normal}.invalid .grid{opacity:.28}.invalid small{color:#b13d2d}.broadcast{display:grid;grid-template-columns:1.1fr 1fr 1fr auto;align-items:center;gap:.8rem;margin-top:1rem;padding:.8rem;background:var(--ink);color:white;border-radius:10px}.broadcast div{display:flex;flex-direction:column}.broadcast b{font-size:.75rem}.broadcast span{font:.68rem var(--font-mono);color:#aab1c0}.broadcast label{font-size:.65rem;color:#aab1c0;display:flex;flex-direction:column}.broadcast strong{color:var(--accent-light);font:.75rem var(--font-mono)}@media(max-width:620px){.controls{grid-template-columns:1fr}.stage{grid-template-columns:1fr}.stage>b{transform:rotate(90deg);text-align:center}.broadcast{grid-template-columns:1fr}.grid{max-width:none}}
</style>
