<script lang="ts">
  import { convolve2d, KERNELS } from '../../lib/conv';
  const image=[[0,0,0,0,0,0],[0,1,1,1,1,0],[0,1,0,0,0,0],[0,1,1,1,0,0],[0,1,0,0,0,0],[0,0,0,0,0,0]];
  let kernelName='edge',row=0,col=0;
  $: kernel=KERNELS[kernelName];
  $: output=convolve2d(image,kernel);
  $: products=kernel.flatMap((r,a)=>r.map((v,b)=>v*image[row+a][col+b]));
  $: total=products.reduce((a,b)=>a+b,0);
  function next(){col++;if(col>=output[0].length){col=0;row=(row+1)%output.length}}
</script>
<section class="lab" aria-labelledby="conv-title">
  <div class="head"><div><span>Interactive lab</span><h3 id="conv-title">Kernel scanner</h3></div><select bind:value={kernelName} aria-label="Convolution kernel">{#each Object.keys(KERNELS) as name}<option value={name}>{name}</option>{/each}</select></div>
  <div class="stage">
    <div><small>6 × 6 image</small><div class="matrix image">{#each image as r,a}{#each r as v,b}<i class:hot={a>=row&&a<row+3&&b>=col&&b<col+3} style={`--shade:${v}`}>{v}</i>{/each}{/each}</div></div>
    <b>×</b><div><small>3 × 3 filter</small><div class="matrix kernel">{#each kernel.flat() as v}<i>{v.toFixed(2).replace('.00','')}</i>{/each}</div></div>
    <b>=</b><div><small>4 × 4 feature map</small><div class="matrix output">{#each output as r,a}{#each r as v,b}<i class:active={a===row&&b===col}>{v.toFixed(1)}</i>{/each}{/each}</div></div>
  </div>
  <div class="calculation"><code>{products.map(v=>v.toFixed(1)).join(' + ')}</code><strong>= {total.toFixed(1)}</strong><button onclick={next}>Next position →</button></div>
</section>
<style>
  .lab{margin:2rem 0;padding:1rem;border:1px solid var(--border);border-radius:14px;background:var(--surface)}.head{display:flex;justify-content:space-between;align-items:center}.head span{font:800 .61rem var(--font-mono);text-transform:uppercase;letter-spacing:.1em;color:var(--accent)}.head h3{margin:.2rem 0}.head select{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:8px;padding:.4rem}.stage{display:grid;grid-template-columns:1.2fr auto .8fr auto 1fr;gap:.8rem;align-items:center;margin:1rem 0;padding:1rem;background:var(--ink);border-radius:10px;color:white}.stage small{display:block;color:#8994a5;font-size:.62rem;margin-bottom:.4rem}.matrix{display:grid;gap:2px}.matrix.image{grid-template-columns:repeat(6,1fr)}.matrix.kernel{grid-template-columns:repeat(3,1fr)}.matrix.output{grid-template-columns:repeat(4,1fr)}.matrix i{display:grid;place-items:center;aspect-ratio:1;background:color-mix(in srgb,#fff calc(var(--shade,0)*55%),#222a38);border:1px solid #374154;border-radius:3px;font:500 .55rem var(--font-mono);font-style:normal}.matrix i.hot{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent)}.matrix i.active{background:var(--accent);border-color:var(--accent);color:white}.calculation{display:flex;align-items:center;gap:.8rem;overflow:hidden}.calculation code{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:.65rem;color:var(--text-muted)}.calculation strong{color:var(--accent);white-space:nowrap}.calculation button{margin-left:auto;white-space:nowrap;background:var(--accent);color:white;border:0;border-radius:7px;padding:.45rem .6rem;font-size:.7rem;font-weight:700}@media(max-width:650px){.stage{grid-template-columns:1fr}.stage>b{display:none}.matrix{max-width:260px}.calculation code{display:none}}
</style>
