"""
Side-by-side comparison of standard vs performer attention.

Shows per-token:
  - Classic generation (live)
  - Performer's own greedy pick (live)
  - Probability the performer assigns to the classic model's chosen token
  - KL divergence at each step

Speed stats at the end.
"""
import sys
import os
import importlib.util
import time
import torch
import torch.nn.functional as F

# Load venv transformers FIRST so submodules are cached before local path takes over
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.activations

_base = os.path.join(os.path.dirname(__file__), '..', 'transformers', 'src')
sys.path.insert(0, _base)
import transformers.models
import transformers.models.llama

_module_path = os.path.join(_base, 'transformers', 'models', 'llama', 'modeling_llama_performer.py')
_spec = importlib.util.spec_from_file_location('transformers.models.llama.modeling_llama_performer', _module_path)
_module = importlib.util.module_from_spec(_spec)
sys.modules['transformers.models.llama.modeling_llama_performer'] = _module
_spec.loader.exec_module(_module)
PerformerLlamaForCausalLM = _module.LlamaForCausalLM

# ── Config ────────────────────────────────────────────────────────────────────
MODEL         = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT        = "<|user|>\nHow do I get a good night's sleep?</s>\n<|assistant|>\n"
MAX_NEW_TOKENS = 30
DTYPE         = torch.float32   # float32 for numerical stability
# ─────────────────────────────────────────────────────────────────────────────

print("Loading standard model...")
std_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cpu")
std_model.eval()

print("Loading performer model...")
perf_model = PerformerLlamaForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cpu")
perf_model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL)
prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]  # [1, T]

print(f"\nPrompt: {repr(PROMPT)}")
print(f"Prompt length: {prompt_ids.shape[1]} tokens\n")

# ── Per-token live comparison ─────────────────────────────────────────────────
W = 14  # column width for token display

header = (f"{'Step':>4}  {'Classic token':<{W}}  {'Perf token':<{W}}"
          f"  {'Classic p':>9}  {'Perf p(classic tok)':>19}  {'KL div':>7}")
print(header)
print("─" * len(header))

current_ids = prompt_ids.clone()
classic_tokens, perf_tokens = [], []
per_token_kl, perf_p_of_classic = [], []
std_times, perf_times = [], []

eos_id = tokenizer.eos_token_id

with torch.no_grad():
    for step in range(1, MAX_NEW_TOKENS + 1):
        # ── Standard forward ──
        t0 = time.perf_counter()
        std_out = std_model(input_ids=current_ids, use_cache=False)
        std_times.append(time.perf_counter() - t0)

        # ── Performer forward ──
        t0 = time.perf_counter()
        perf_out = perf_model(input_ids=current_ids, use_cache=False)
        perf_times.append(time.perf_counter() - t0)

        std_logits  = std_out.logits[0, -1].float()   # [vocab]
        perf_logits = perf_out.logits[0, -1].float()

        std_probs  = F.softmax(std_logits,  dim=-1)
        perf_probs = F.softmax(perf_logits, dim=-1)

        # Classic picks its token (greedy)
        classic_id = std_logits.argmax().item()
        # Performer picks its own greedy token
        perf_id    = perf_logits.argmax().item()

        classic_p   = std_probs[classic_id].item()
        perf_p_cls  = perf_probs[classic_id].item()   # performer prob for classic's choice
        kl          = F.kl_div(perf_probs.log(), std_probs, reduction='sum').item()

        classic_tok = repr(tokenizer.decode([classic_id]))[1:-1]  # strip outer quotes
        perf_tok    = repr(tokenizer.decode([perf_id]))[1:-1]

        print(f"{step:>4}  {classic_tok:<{W}}  {perf_tok:<{W}}"
              f"  {classic_p:>8.2%}  {perf_p_cls:>19.2%}  {kl:>7.3f}")

        classic_tokens.append(classic_id)
        perf_tokens.append(perf_id)
        per_token_kl.append(kl)
        perf_p_of_classic.append(perf_p_cls)

        # Advance sequence with classic model's choice
        next_id = torch.tensor([[classic_id]])
        current_ids = torch.cat([current_ids, next_id], dim=-1)

        if classic_id == eos_id:
            break

# ── Summary ───────────────────────────────────────────────────────────────────
print()
classic_text  = tokenizer.decode(classic_tokens, skip_special_tokens=True)
perf_text     = tokenizer.decode(perf_tokens,    skip_special_tokens=True)
token_overlap = sum(c == p for c, p in zip(classic_tokens, perf_tokens))
n             = len(classic_tokens)

avg_std_ms   = sum(std_times)  / n * 1000
avg_perf_ms  = sum(perf_times) / n * 1000
std_tps      = n / sum(std_times)
perf_tps     = n / sum(perf_times)

print("══ Generations ═══════════════════════════════════════")
print(f"  Classic:   {classic_text}")
print(f"  Performer: {perf_text}")
print()
print("══ Alignment ═════════════════════════════════════════")
print(f"  Token match rate:          {token_overlap}/{n} ({token_overlap/n:.1%})")
print(f"  Avg KL divergence/step:    {sum(per_token_kl)/n:.4f}")
print(f"  Avg performer p(classic):  {sum(perf_p_of_classic)/n:.2%}")
print()
print("══ Speed (full-sequence forward pass per token) ══════")
print(f"  Standard:   {avg_std_ms:6.1f} ms/tok  ({std_tps:.2f} tok/s)")
print(f"  Performer:  {avg_perf_ms:6.1f} ms/tok  ({perf_tps:.2f} tok/s)")
print(f"  Speedup:    {avg_std_ms/avg_perf_ms:.2f}x")
