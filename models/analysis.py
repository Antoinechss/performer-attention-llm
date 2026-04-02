"""
Performer vs standard attention analysis.

  A — Per-token generation: quality comparison (requires model load)
  B — Speed benchmarks: B1 prefill scaling, B2 decode step
  C — Mixed-head quality sweep (requires model load)
"""
import sys, os, time, importlib.util
import torch
import torch.nn.functional as F

# ── Config ──────────────────────────────────────────────────────────────────
MODEL          = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT         = "<|user|>\nHow do I get a good night's sleep?</s>\n<|assistant|>\n"
MAX_NEW_TOKENS = 20
DTYPE          = torch.float32

RUN_A = False
RUN_B = True
RUN_C = False
# ────────────────────────────────────────────────────────────────────────────

# ── Model loading (Sections A & C only) ─────────────────────────────────────
if RUN_A or RUN_C:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers.activations

    _base = os.path.join(os.path.dirname(__file__), '..', 'transformers', 'src')
    sys.path.insert(0, _base)
    import transformers.models
    import transformers.models.llama

    def _load_performer_module():
        path = os.path.join(_base, 'transformers', 'models', 'llama', 'modeling_llama_performer.py')
        spec = importlib.util.spec_from_file_location(
            'transformers.models.llama.modeling_llama_performer', path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    _perf_mod = _load_performer_module()
    PerformerLlamaForCausalLM = _perf_mod.LlamaForCausalLM

    print("Loading standard model...")
    std_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cpu")
    std_model.eval()

    SECTION_A_PERFORMER_HEADS = 4

    print("Loading performer model...")
    perf_model = PerformerLlamaForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cpu")
    perf_model.eval()

    tokenizer  = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
    num_heads  = std_model.config.num_attention_heads


# ═══════════════════════════════════════════════════════════════════════════
# SECTION A — Per-token generation comparison
# ═══════════════════════════════════════════════════════════════════════════
if RUN_A:
    for layer in perf_model.model.layers:
        layer.self_attn.num_performer_heads = SECTION_A_PERFORMER_HEADS
        layer.self_attn.num_standard_heads  = num_heads - SECTION_A_PERFORMER_HEADS

    print(f"\n{'═'*70}")
    print(f"SECTION A — Generation  [{SECTION_A_PERFORMER_HEADS}/{num_heads} performer heads]")
    print(f"{'═'*70}\n")

    W = 14
    hdr = f"{'Step':>4}  {'Classic':.<{W}}  {'Performer':.<{W}}  {'p(cls)':>7}  {'p_perf(cls)':>11}  {'KL':>6}"
    print(hdr)
    print("─" * len(hdr))

    current_ids = prompt_ids.clone()
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
            print(f"{step:>4}  {c_tok:<{W}}  {p_tok:<{W}}  {classic_p:>6.1%}  {perf_p_cls:>11.1%}  {kl:>6.2f}")

            classic_tokens.append(classic_id)
            perf_tokens.append(perf_id)
            kl_per_step.append(kl)
            perf_p_track.append(perf_p_cls)

            current_ids = torch.cat([current_ids, torch.tensor([[classic_id]])], dim=-1)
            if classic_id == tokenizer.eos_token_id:
                break

    n = len(classic_tokens)
    print(f"\n  Classic:   {tokenizer.decode(classic_tokens, skip_special_tokens=True)}")
    print(f"  Performer: {tokenizer.decode(perf_tokens, skip_special_tokens=True)}")
    match = sum(c == p for c, p in zip(classic_tokens, perf_tokens))
    print(f"  Match: {match}/{n}  |  Avg KL: {sum(kl_per_step)/n:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION B — Speed benchmarks (kernel-level, no model load needed)
# ═══════════════════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'performer'))
from performer_attention import PerformerAttentionCore, _HAS_TRITON

try:
    from triton_scan import triton_scan_forward as _triton_scan_raw
    from triton_scan import triton_decode_forward as _triton_decode_raw
except ImportError:
    _triton_scan_raw = _triton_decode_raw = None

_CUDA = torch.cuda.is_available()
_dev  = torch.device("cuda" if _CUDA else "cpu")
_TRITON = _HAS_TRITON and _CUDA

H, D     = 32, 64
M_VALS   = [128, 256]
REPEATS  = 3
scale    = D ** -0.25

performer_cores = {m: PerformerAttentionCore(head_dim=D, num_features=m).to(_dev) for m in M_VALS}


def time_fn(fn, repeats=REPEATS):
    if _CUDA: torch.cuda.synchronize()
    fn()
    if _CUDA: torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    if _CUDA: torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


# ── B1: Prefill ─────────────────────────────────────────────────────────────
if RUN_B:
    print(f"\n{'═'*70}")
    print(f"B1 — Prefill  O(N²) std vs O(N·M) performer  |  H={H} D={D}")
    print(f"{'═'*70}\n")

    SEQ_LENS = [256, 512, 1024, 2048, 4096]
    _dtype = torch.float16 if _CUDA else torch.float32
    _CW = 12

    hdr = f"{'N':>6}  {'Std':>8}" + "".join(f"  {'e2e M='+str(m):>{_CW}}" for m in M_VALS) + f"  {'speedup':>8}"
    print(hdr)
    print("─" * len(hdr))

    with torch.no_grad():
        for N in SEQ_LENS:
            q = torch.randn(1, H, N, D, device=_dev, dtype=_dtype)
            k = torch.randn(1, H, N, D, device=_dev, dtype=_dtype)
            v = torch.randn(1, H, N, D, device=_dev, dtype=_dtype)

            def std_attn():
                w = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5), dim=-1)
                return torch.matmul(w, v)

            std_ms = time_fn(std_attn)
            row = f"{N:>6}  {std_ms:>7.2f}"

            e2e_times = {}
            for m in M_VALS:
                core = performer_cores[m]
                fn = lambda c=core: c(q, k, v)
                e2e_times[m] = time_fn(fn)
                row += f"  {e2e_times[m]:>{_CW}.2f}"

            best = min(e2e_times.values())
            row += f"  {std_ms/best:>7.2f}x"
            print(row)

    # ── B2: Decode ──────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"B2 — Decode step  O(N) std vs O(M) performer state")
    print(f"{'═'*70}\n")

    CACHE_SIZES = [64, 256, 1024]

    hdr2 = f"{'Cache':>6}  {'Std':>8}" + "".join(f"  {'M='+str(m):>{_CW}}" for m in M_VALS)
    print(hdr2)
    print("─" * len(hdr2))

    with torch.no_grad():
        for N in CACHE_SIZES:
            q_new = torch.randn(1, H, 1, D, device=_dev)
            k_all = torch.randn(1, H, N, D, device=_dev)
            v_all = torch.randn(1, H, N, D, device=_dev)

            def std_decode():
                w = torch.softmax(torch.matmul(q_new, k_all.transpose(-2, -1)) * (D ** -0.5), dim=-1)
                return torch.matmul(w, v_all)

            std_ms = time_fn(std_decode)
            row = f"{N:>6}  {std_ms:>7.3f}"

            for m in M_VALS:
                core      = performer_cores[m]
                phi_k_all = core.phi(k_all * scale, is_query=False)
                kv_state  = torch.einsum("bhnm,bhnd->bhmd", phi_k_all, v_all).float()
                k_state   = phi_k_all.sum(dim=2).float()
                omega_m   = core.omega.float()

                if _TRITON and _triton_decode_raw is not None:
                    fn = lambda kv=kv_state, ks=k_state, om=omega_m: \
                        _triton_decode_raw((q_new * scale).float(), om, kv, ks)
                else:
                    def fn(kv=kv_state, ks=k_state, c=core):
                        phi_q = c.phi(q_new * scale, is_query=True)
                        out   = torch.einsum("bhnm,bhmd->bhnd", phi_q, kv)
                        denom = torch.einsum("bhnm,bhm->bhn", phi_q, ks) + 1e-6
                        return out / denom.unsqueeze(-1)
                row += f"  {time_fn(fn):>{_CW}.3f}"

            print(row)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION C — Mixed-head quality sweep
# ═══════════════════════════════════════════════════════════════════════════
if RUN_C:
    print(f"\n{'═'*70}")
    print("SECTION C — Quality vs performer head count")
    print(f"{'═'*70}\n")

    with torch.no_grad():
        std_ref = std_model(input_ids=prompt_ids, use_cache=False)
    std_logits_ref = std_ref.logits[0, -1].float()
    std_probs_ref  = F.softmax(std_logits_ref, dim=-1)
    std_top5_ref   = set(std_logits_ref.topk(5).indices.tolist())

    print(f"{'K':>4}  {'KL':>7}  {'Top5':>5}  {'p(top1)':>8}")
    print("─" * 30)

    for k in [0, 1, 2, 4, 8, 16, 32]:
        for layer in perf_model.model.layers:
            layer.self_attn.num_performer_heads = k
            layer.self_attn.num_standard_heads  = num_heads - k

        with torch.no_grad():
            out = perf_model(input_ids=prompt_ids, use_cache=False)

        logits  = out.logits[0, -1].float()
        probs   = F.softmax(logits, dim=-1)
        kl      = F.kl_div(probs.log(), std_probs_ref, reduction='sum').item()
        overlap = len(set(logits.topk(5).indices.tolist()) & std_top5_ref)
        p_top1  = probs[std_logits_ref.argmax().item()].item()

        print(f"{k:>4}  {kl:>7.3f}  {overlap:>4}/5  {p_top1:>8.1%}")
