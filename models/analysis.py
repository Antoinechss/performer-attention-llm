"""
Three-section comparison of standard vs performer attention.

  A — Per-token live generation: quality side-by-side + probability alignment
  B — Prefill speed scaling: O(N²) vs O(N·M) across sequence lengths
  C — Mixed-head quality sweep: KL / top-5 overlap as num_performer_heads grows
"""
import sys
import os
import importlib.util
import time
import torch
import torch.nn.functional as F

# ── Config ───────────────────────────────────────────────────────────────────
MODEL          = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT         = "<|user|>\nHow do I get a good night's sleep?</s>\n<|assistant|>\n"
MAX_NEW_TOKENS = 20
DTYPE          = torch.float32

RUN_A = False   # per-token live generation (slow — needs full model, token by token)
RUN_B = True    # attention kernel speed benchmark (fast — no model load needed)
RUN_C = False   # mixed-head quality sweep (moderate — one forward pass per K value)
# ─────────────────────────────────────────────────────────────────────────────

# Models only needed for sections A and C
if RUN_A or RUN_C:
    # ── Import order: venv transformers FIRST, then local path ───────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers.activations   # cache in sys.modules before local path shadows it

    _base = os.path.join(os.path.dirname(__file__), '..', 'transformers', 'src')
    sys.path.insert(0, _base)
    import transformers.models
    import transformers.models.llama

    def _load_performer_module():
        path = os.path.join(_base, 'transformers', 'models', 'llama', 'modeling_llama_performer.py')
        name = 'transformers.models.llama.modeling_llama_performer'
        spec = importlib.util.spec_from_file_location(name, path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _perf_mod = _load_performer_module()
    PerformerLlamaForCausalLM = _perf_mod.LlamaForCausalLM
    print("Loading standard model...")
    std_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cpu")
    std_model.eval()

    # Change SECTION_A_PERFORMER_HEADS to try different K values in the live comparison
    # 32 = all performer (worst quality) | 0 = pure softmax | 1,2,4... = mixed
    SECTION_A_PERFORMER_HEADS = 4

    print("Loading performer model (all heads, will patch for Section A)...")
    perf_model = PerformerLlamaForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cpu")
    perf_model.eval()

    tokenizer  = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    num_heads  = std_model.config.num_attention_heads  # 32 for TinyLlama

# ════════════════════════════════════════════════════════════════════════════
# SECTION A — per-token live comparison
# ════════════════════════════════════════════════════════════════════════════
if RUN_A:
    for layer in perf_model.model.layers:
        layer.self_attn.num_performer_heads = SECTION_A_PERFORMER_HEADS
        layer.self_attn.num_standard_heads  = num_heads - SECTION_A_PERFORMER_HEADS

    print(f"\n{'═'*70}")
    print(f"SECTION A — Per-token generation  [{SECTION_A_PERFORMER_HEADS}/{num_heads} performer heads]")
    print(f"{'═'*70}\n")

    W = 14
    header = (f"{'Step':>4}  {'Classic':.<{W}}  {'Performer':.<{W}}"
              f"  {'Classic p':>9}  {'Perf p(classic)':>15}  {'KL':>7}")
    print(header)
    print("─" * len(header))

    current_ids   = prompt_ids.clone()
    classic_tokens, perf_tokens = [], []
    kl_per_step, perf_p_track  = [], []

    with torch.no_grad():
        for step in range(1, MAX_NEW_TOKENS + 1):
            std_out  = std_model(input_ids=current_ids,  use_cache=False)
            perf_out = perf_model(input_ids=current_ids, use_cache=False)

            std_logits  = std_out.logits[0, -1].float()
            perf_logits = perf_out.logits[0, -1].float()

            std_probs  = F.softmax(std_logits,  dim=-1)
            perf_probs = F.softmax(perf_logits, dim=-1)

            classic_id = std_logits.argmax().item()
            perf_id    = perf_logits.argmax().item()

            classic_p  = std_probs[classic_id].item()
            perf_p_cls = perf_probs[classic_id].item()
            kl         = F.kl_div(perf_probs.log(), std_probs, reduction='sum').item()

            c_tok = repr(tokenizer.decode([classic_id]))[1:-1]
            p_tok = repr(tokenizer.decode([perf_id]))[1:-1]

            print(f"{step:>4}  {c_tok:<{W}}  {p_tok:<{W}}"
                  f"  {classic_p:>8.2%}  {perf_p_cls:>15.2%}  {kl:>7.3f}")

            classic_tokens.append(classic_id)
            perf_tokens.append(perf_id)
            kl_per_step.append(kl)
            perf_p_track.append(perf_p_cls)

            current_ids = torch.cat([current_ids, torch.tensor([[classic_id]])], dim=-1)
            if classic_id == tokenizer.eos_token_id:
                break

    n = len(classic_tokens)
    print(f"\n  Classic:   {tokenizer.decode(classic_tokens, skip_special_tokens=True)}")
    print(f"  Performer: {tokenizer.decode(perf_tokens,    skip_special_tokens=True)}")
    match = sum(c == p for c, p in zip(classic_tokens, perf_tokens))
    print(f"\n  Token match: {match}/{n}  |  Avg KL: {sum(kl_per_step)/n:.4f}"
          f"  |  Avg perf p(classic): {sum(perf_p_track)/n:.2%}")

# ════════════════════════════════════════════════════════════════════════════
# SECTION B — Attention-kernel-only speed scaling
#
# Why isolate the kernel?
#   Full model forward is dominated by MLP + LayerNorm at small N.
#   To see O(N²) vs O(N·M) we must time ONLY the attention operation.
#
# Two sub-benchmarks:
#   B1 — Prefill: process N tokens at once  (performer: O(N·M), std: O(N²))
#   B2 — Decoding step: one new token, growing KV cache
#          Standard:           O(N·D) per step  (grows with history)
#          Performer (naive):  O(N·M·D) per step (recomputes kv_sum — SLOW)
#          Performer (state):  O(M·D) per step   (incremental update — FAST)
# ════════════════════════════════════════════════════════════════════════════
import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'performer'))
from performer_attention import PerformerAttentionCore, _HAS_TRITON, _python_scan

try:
    from triton_scan import triton_scan_forward  as _triton_scan_raw
    from triton_scan import triton_decode_forward as _triton_decode_raw
except ImportError:
    _triton_scan_raw   = None
    _triton_decode_raw = None

_CUDA = torch.cuda.is_available()
_dev  = torch.device("cuda" if _CUDA else "cpu")
_TRITON_BENCH = _HAS_TRITON and _CUDA   # True only on a CUDA machine with triton installed

H, D    = 32, 64                    # heads, head_dim
M_VALS  = [64, 128, 256, 512]       # sweep: random feature counts
REPEATS = 3
performer_cores = {
    m: PerformerAttentionCore(head_dim=D, num_features=m).to(_dev)
    for m in M_VALS
}
scale = D ** -0.25
_CW = 14   # column width for each M value

def time_fn(fn, repeats=REPEATS):
    if _CUDA:
        torch.cuda.synchronize()
    fn()  # warmup
    if _CUDA:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if _CUDA:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000

# ── B1: Prefill scaling ───────────────────────────────────────────────────
print(f"\n{'═'*70}")
print("SECTION B1 — Prefill  O(N²·D) std  vs  O(N·M·D) Triton scan")
print(f"             Crossover at N = M  |  H={H}, D={D}")
print(f"             Triton: {'ACTIVE' if _TRITON_BENCH else 'INACTIVE'}")
print(f"{'═'*70}\n")

SEQ_LENS = [256, 512, 1024, 2048, 4096]

# End-to-end comparison: std attention vs performer (phi + scan)
# Using fp16 to fit N=4096 with H=32 on GPU
_bench_dtype = torch.float16 if _CUDA else torch.float32
_E2E_VALS = [128, 256]   # M values for end-to-end columns

_hdr_b1 = (f"{'N':>6}  {'Std (ms)':>10}"
           + "".join(f"  {'e2e M='+str(m):>{_CW}}" for m in _E2E_VALS)
           + "".join(f"  {'Scan M='+str(m):>{_CW}}" for m in M_VALS)
           + f"  {'best speedup':>13}")
print(_hdr_b1)
print("─" * len(_hdr_b1))

with torch.no_grad():
    for N in SEQ_LENS:
        q = torch.randn(1, H, N, D, device=_dev, dtype=_bench_dtype)
        k = torch.randn(1, H, N, D, device=_dev, dtype=_bench_dtype)
        v = torch.randn(1, H, N, D, device=_dev, dtype=_bench_dtype)

        def std_attn():
            scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
            w = torch.softmax(scores, dim=-1)
            return torch.matmul(w, v)

        std_ms = time_fn(std_attn)
        row = f"{N:>6}  {std_ms:>10.2f}"

        # End-to-end: phi + scan via forward()
        e2e_times = {}
        for m in _E2E_VALS:
            core = performer_cores[m]
            fn = lambda c=core: c(q, k, v)
            e2e_times[m] = time_fn(fn)
            row += f"  {e2e_times[m]:>{_CW}.2f}"

        # Scan-only timing for each M (phi pre-computed)
        for m in M_VALS:
            if _TRITON_BENCH and _triton_scan_raw is not None:
                core    = performer_cores[m]
                phi_q_b = core.phi(q * scale, is_query=True).float()
                phi_k_b = core.phi(k * scale, is_query=False).float()
                v_b     = v.float()
                fn = lambda pq=phi_q_b, pk=phi_k_b, vv=v_b: _triton_scan_raw(pq, pk, vv)
                t_ms = time_fn(fn)
            else:
                t_ms = time_fn(lambda c=performer_cores[m]: c(q, k, v))
            row += f"  {t_ms:>{_CW}.2f}"

        best_e2e = min(e2e_times.values())
        best_speedup = std_ms / best_e2e
        row += f"  {best_speedup:>12.2f}x"
        print(row)

# ── B2: Decoding step scaling ─────────────────────────────────────────────
print(f"\n{'═'*70}")
print("SECTION B2 — Decode step  O(N·D) std  vs  O(M·D) Triton fused state")
print(f"             Triton cost is constant in N, grows with M")
print(f"{'═'*70}\n")

CACHE_SIZES = [64, 128, 256, 512, 1024]

_hdr_b2 = (f"{'Cache N':>8}  {'Std (ms)':>10}"
           + "".join(f"  {'Triton M='+str(m):>{_CW}}" for m in M_VALS))
print(_hdr_b2)
print("─" * len(_hdr_b2))

with torch.no_grad():
    for N in CACHE_SIZES:
        q_new = torch.randn(1, H, 1, D, device=_dev)
        k_all = torch.randn(1, H, N, D, device=_dev)
        v_all = torch.randn(1, H, N, D, device=_dev)

        def std_decode():
            scores = torch.matmul(q_new, k_all.transpose(-2, -1)) * (D ** -0.5)
            w      = torch.softmax(scores, dim=-1)
            return torch.matmul(w, v_all)

        std_ms = time_fn(std_decode)
        row = f"{N:>8}  {std_ms:>10.3f}"

        for m in M_VALS:
            core       = performer_cores[m]
            phi_k_all  = core.phi(k_all * scale, is_query=False)
            kv_state_m = torch.einsum("bhnm,bhnd->bhmd", phi_k_all, v_all).float()
            k_state_m  = phi_k_all.sum(dim=2).float()
            omega_m    = core.omega.float()

            if _TRITON_BENCH and _triton_decode_raw is not None:
                fn = lambda kv=kv_state_m, ks=k_state_m, om=omega_m: \
                    _triton_decode_raw((q_new * scale).float(), om, kv, ks)
            else:
                def fn(kv=kv_state_m, ks=k_state_m, c=core):
                    phi_q = c.phi(q_new * scale, is_query=True)
                    out   = torch.einsum("bhnm,bhmd->bhnd", phi_q, kv)
                    denom = torch.einsum("bhnm,bhm->bhn", phi_q, ks) + 1e-6
                    return out / denom.unsqueeze(-1)
            row += f"  {time_fn(fn):>{_CW}.3f}"

        print(row)

# ════════════════════════════════════════════════════════════════════════════
# SECTION B3 — CPU prefill: true O(N²) vs O(N·M) scaling
#
# Why CPU?
#   GPU kernel launch overhead (~70µs) masks compute differences at small N.
#   On CPU the measured time IS the math — no latency floor.
#
# Why _python_scan instead of cumsum?
#   _python_scan maintains only S[B,H,M,D] and z[B,H,M] — O(M·D) memory.
#   It never materialises [B,H,N,M,D], so the true crossover is at N=M.
#   The cumsum approach materialises [B,H,N,M,D] (O(N·M·D) memory), which
#   shifts the crossover to N=M·D — an implementation artefact, not theory.
#
# Python loop overhead caveat:
#   Each of the N iterations dispatches ~5 small PyTorch calls (~10µs each).
#   This O(N × dispatch_overhead) term inflates times but preserves O(N) shape.
#   Crossover appears slightly above N=M due to this constant factor.
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("SECTION B3 — CPU prefill  O(N²·D) std  vs  O(N·M·D) performer scan")
print(f"             Performer uses _python_scan: O(M·D) memory, no [N,M,D] tensor")
print(f"             Note: Python dispatch overhead inflates scan constant ~5-10×")
print(f"{'═'*70}\n")

_H_CPU   = 4
_M_CPU   = [64, 128, 256, 512]
_N_CPU   = [64, 128, 256, 512, 1024, 2048, 4096]
_cpu_cores = {
    m: PerformerAttentionCore(head_dim=D, num_features=m)
    for m in _M_CPU
}

_hdr_b3 = (f"{'N':>6}  {'Std (ms)':>10}"
           + "".join(f"  {'Scan M='+str(m):>{_CW}}" for m in _M_CPU))
print(_hdr_b3)
print("─" * len(_hdr_b3))

for N in _N_CPU:
    q_c = torch.randn(1, _H_CPU, N, D)
    k_c = torch.randn(1, _H_CPU, N, D)
    v_c = torch.randn(1, _H_CPU, N, D)

    def std_cpu():
        w = torch.softmax(
            torch.matmul(q_c, k_c.transpose(-2, -1)) * (D ** -0.5), dim=-1)
        return torch.matmul(w, v_c)

    std_ms = time_fn(std_cpu)
    row = f"{N:>6}  {std_ms:>10.1f}"

    for m in _M_CPU:
        core  = _cpu_cores[m]
        phi_q = core.phi(q_c * scale, is_query=True).float()
        phi_k = core.phi(k_c * scale, is_query=False).float()

        fn = lambda pq=phi_q, pk=phi_k, vv=v_c: _python_scan(pq, pk, vv.float())
        row += f"  {time_fn(fn):>{_CW}.1f}"

    print(row)

# ════════════════════════════════════════════════════════════════════════════
# SECTION C — Mixed-head quality sweep
# Load model once, change num_performer_heads at runtime (no weight reload)
# ════════════════════════════════════════════════════════════════════════════
if RUN_C:
    print(f"\n{'═'*70}")
    print("SECTION C — Mixed-head sweep  [quality as performer heads increase]")
    print(f"{'═'*70}\n")

    with torch.no_grad():
        std_ref = std_model(input_ids=prompt_ids, use_cache=False)
    std_logits_ref = std_ref.logits[0, -1].float()
    std_probs_ref  = F.softmax(std_logits_ref, dim=-1)
    std_top5_ref   = set(std_logits_ref.topk(5).indices.tolist())

    SWEEP = [0, 1, 2, 4, 8, 16, 32]

    print(f"{'K heads':>8}  {'KL div':>8}  {'Top-5 overlap':>14}  {'Perf p(std top1)':>17}  {'Note'}")
    print("─" * 65)

    for k in SWEEP:
        for layer in perf_model.model.layers:
            layer.self_attn.num_performer_heads = k
            layer.self_attn.num_standard_heads  = num_heads - k

        with torch.no_grad():
            out = perf_model(input_ids=prompt_ids, use_cache=False)

        logits  = out.logits[0, -1].float()
        probs   = F.softmax(logits, dim=-1)
        kl      = F.kl_div(probs.log(), std_probs_ref, reduction='sum').item()
        top5    = set(logits.topk(5).indices.tolist())
        overlap = len(top5 & std_top5_ref)
        p_top1  = probs[std_logits_ref.argmax().item()].item()

        note = ""
        if k == 0:       note = "← pure softmax (baseline)"
        elif k == num_heads: note = "← all performer"

        print(f"{k:>8}  {kl:>8.4f}  {overlap:>14}/5  {p_top1:>17.2%}  {note}")
