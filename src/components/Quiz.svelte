<script lang="ts">
  let { question, answer }: { question: string; answer: string } = $props();
  let open = $state(false);
</script>

<section class="quiz">
  <div class="label"><span>?</span> Check your understanding</div>
  <h3>{question}</h3>
  <button type="button" aria-expanded={open} onclick={() => (open = !open)}>
    {open ? 'Hide explanation' : 'Reveal explanation'} <span aria-hidden="true">{open ? '−' : '+'}</span>
  </button>
  {#if open}<p class="answer">{answer}</p>{/if}
</section>

<style>
  .quiz{margin:2rem 0;padding:1.2rem 1.3rem;border:1px solid var(--border);border-radius:12px;background:var(--surface)}.label{display:flex;align-items:center;gap:.5rem;text-transform:uppercase;letter-spacing:.09em;color:var(--accent);font-size:.68rem;font-weight:800}.label span{display:grid;place-items:center;width:1.35rem;height:1.35rem;border-radius:50%;background:var(--accent);color:white}.quiz h3{margin:.8rem 0;font-size:1.05rem}.quiz button{border:0;background:transparent;color:var(--text-muted);padding:.25rem 0;font-weight:700;font-size:.82rem}.quiz button span{color:var(--accent);margin-left:.35rem}.answer{border-top:1px solid var(--border);margin:1rem 0 0;padding-top:1rem;color:var(--text-muted)}
</style>
