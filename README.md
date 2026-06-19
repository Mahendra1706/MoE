# MoE from scratch — building the router that actually decides

This is a from-scratch Mixture-of-Experts implementation where the only part I borrowed from PyTorch is `nn.Linear`. Everything that actually makes it an MoE — the router, the dispatch logic, the load balancing — is written and derived by hand first, then coded.

I'm not writing this as a tutorial. I'm writing this the way I actually arrived at it — including the decision I almost got wrong, the formula I had to re-derive twice, and the result that surprised me when I actually ran it.

---

## Why route at all

A normal feed-forward layer is one big function every token has to pass through. An MoE layer replaces that one big function with N smaller functions (experts), and adds a router whose only job is: for this specific token, which experts should see it.

The promise of MoE is that you get the capacity of a huge model (more total parameters) without the compute cost of a huge model (only a few experts run per token). Mixtral 8x7B is the proof of this — 47B total parameters, but only 13B active per token because only 2 of 8 experts fire per layer. That's the whole appeal: stay cheap at inference while still having a lot of room to specialize.

But none of that works if the router is dumb. The router is the part that decides who learns what. Everything else — the experts, the loss, the load balancing — only matters in relation to what the router chooses. So that's where I spent almost all my time.

---

## Token-choice vs expert-choice — and why I didn't default to the popular one

Before I wrote a single line of router code, I went back to the original Shazeer et al. MoE paper, because I wanted to know if "router picks experts for each token" was the only way to do this. It isn't.

There are two fundamentally different ways to frame routing:

**Token-choice routing** — each token independently asks "which experts do I want?" This is what I built. The router computes a score for every (token, expert) pair, and each token picks its own top-k.

**Expert-choice routing** — flip the direction. Each expert independently asks "which tokens do I want?" Every expert has a fixed budget of tokens it's willing to process, and it picks the top tokens that best match it.

These sound symmetric but they are not the same problem at all once you look at what each one optimizes for.

Token-choice optimizes for **per-token quality** — every token gets to choose its best-matching experts, no matter what. But this means load balance is never guaranteed. A token might genuinely want expert 0 the most, and if every token in the batch agrees, expert 0 gets flooded while expert 7 starves. You only fix this indirectly, by adding a penalty (the aux loss) that nudges the router away from doing this — but the router is still free to ignore the penalty if the CE loss gain is big enough.

Expert-choice optimizes for **per-expert balance** — by construction, no expert can ever be overloaded because each expert has a hard capacity. Load balance is structurally guaranteed, not learned. But the cost is that some tokens might not get picked by any expert at all if they don't score well with anyone, which means those tokens get dropped or fall back to a default — a real quality hit for tokens that are unusual or rare.

I chose token-choice for one main reason: it's the dominant paradigm in every production MoE I actually study — Mixtral, DeepSeek, Switch Transformer. If I'm trying to eventually map onto DeepSeek's MoE split-2 architecture later (mHC, CSA, HCA), I need to understand token-choice routing at the implementation level first, because that's the substrate everything else sits on. Expert-choice is a legitimate and interesting alternative, but it solves a different problem (guaranteed balance) at a different cost (dropped tokens), and that tradeoff isn't the one the architectures I'm targeting made.

The real consequence of this choice: **I now have to fight for load balance instead of getting it for free.** That fight is the aux loss.

---

## The router itself

For each token, the router does:

```
logits = x @ W_g          # (B*T, D) @ (D, E) → (B*T, E)
top_vals, top_idx = topk(logits, k)
masked = scatter -inf everywhere except top_idx, keep top_vals there
weights = softmax(masked)  # (B*T, E), only k non-zero entries per row
```

The thing I had to be precise about here: **selection and weighting are two different operations using the same numbers.** `topk` decides *which* experts a token uses — this part is discrete, no gradient flows through it. `softmax` after masking decides *how much* each selected expert contributes — this part is continuous, and this is the only path gradient has back into `W_g`.

If you skip the masking step and just softmax the raw logits over all 8 experts, you get gradient signal for experts that weren't even selected, which doesn't match what's happening structurally. The mask-then-softmax order is what makes the "selected experts get weighted, ignored experts get exactly zero gradient contribution" property hold.

---

## Dispatch — where the actual computation happens

```
for each expert e:
    mask = tokens that picked expert e
    tokens_for_e = x[mask]
    output_for_e = Expert_e(tokens_for_e)
    out[mask] += weight[mask, e] * output_for_e
```

The `+=` here is not a stylistic choice — it's required because a token that picked experts 0 and 2 needs both contributions to land in the same output row. If this were `=` instead of `+=`, whichever expert processed the token last would silently overwrite the other expert's contribution, and you'd lose half of every token's signal without any error being raised. This is the kind of bug that doesn't crash — it just quietly makes your model worse, which is worse than crashing.

---

## The two losses, and why they're not redundant

This is the part of my notebook that took the longest to actually get right in words, not just in math.

**Importance loss** measures the gate **weights** — literally how much trust, summed over the whole batch, does each expert receive.

```
importance_e = Σ over all tokens of weight[token, e]
imp_loss = CV²(importance)
```

**Load loss** measures the **token count** — how many tokens, regardless of how much weight, physically got dispatched to each expert.

```
load_e = count of tokens where e is in their top_idx
load_loss = CV²(load)
```

CV² is coefficient of variation squared — `variance / mean²`. It's scale-invariant, which matters because the absolute scale of "weight" and "count" are completely different things, but CV² gives you a comparable unevenness score for both.

The non-obvious thing I had to sit with: **equal importance does NOT imply equal load**, and vice versa. Concretely — say expert 0 receives 1 token with weight 1.0, and expert 1 receives 5 tokens each with weight 0.2. Total importance for both experts is the same: 1.0. But the load is wildly different: 1 token vs 5 tokens. If you only optimized importance, the router could learn to dump a lot of low-confidence tokens onto one expert while keeping importance numerically balanced — that's a real failure mode that importance loss alone wouldn't catch, but load loss would.

This is why both losses exist together, not as redundant safety nets, but because they are penalizing two genuinely different and decoupled quantities. You need both because the router has two separate ways to misbehave.

---

## What I actually got when I ran this

I didn't just want the model to run — I wanted to know if my router was behaving the way an MoE router is supposed to. So I trained `TinyMoEModel` (2 stacked MoE layers, 8 experts each, top-2) on TinyShakespeare, character-level, `d_model=128`, 1000 steps.

**Did it learn anything real?**
Random-guess baseline for a 65-character vocabulary is `ln(65) ≈ 4.17`. My final validation loss landed at **2.50**. That's a real gap below random, which tells me the router and experts are jointly doing something meaningful — not just memorizing noise, since this is measured on held-out validation data the model never trained on.

**Did load balancing actually work?**
Here's the part I didn't expect going in. No expert ever went fully dead — every expert in both layers received tokens throughout training, so the load loss did its baseline job of preventing total collapse. But it did not produce a *perfectly* even split. Looking at total tokens routed per expert across training, `moe1`'s least-used expert got roughly half the tokens of its most-used expert. That's a meaningful skew, not a rounding error.

**The more interesting finding — the two MoE layers specialized differently.**
I checked routing entropy on a fresh validation batch. For top-2 routing, max possible entropy is `ln(2) ≈ 0.693` — meaning the router is maximally unsure between its two picks. `moe1`'s entropy sat at **0.62**, close to max — its router is still routing fairly close to uniform between its top-2 choices. `moe2`'s entropy was **0.46**, noticeably sharper — its router has developed real preference, picking one expert much more confidently than the other.

I didn't expect the two layers to diverge like that. My read on why: by the time tokens reach `moe2`, they've already been transformed by `moe1` and carry more specialized information, so `moe2`'s router has an easier time finding a clear "this expert fits better" signal. `moe1` sees raw embeddings, which are closer to undifferentiated, so its routing decisions stay closer to a coin flip between the top-2.

This matches something Mixtral's own analysis touches on — routing specialization isn't uniform across layers, and deeper or later layers tend to show more confident expert assignment. I'm seeing a small-scale version of that same pattern in a 2-layer toy model, which is the kind of thing that makes me trust the implementation is mechanically sound, not just numerically lucky.

---

## The honest tradeoff I'm sitting with

The aux loss prevents catastrophe (no dead experts) but doesn't force fairness. That's not a bug — it's the literal tradeoff token-choice routing makes. The router is still free to prefer some experts over others as long as the CE loss gain from that preference outweighs the aux loss penalty I'm charging it. At `λ=0.01` for both losses, the model is clearly choosing to pay a little balance penalty in exchange for routing decisions it considers worth it.

The open question I'm carrying into the next iteration: is that λ too small, letting real specialization patterns get amplified into unwanted concentration, or is some unevenness actually fine — maybe even desirable, since it could mean experts are genuinely starting to specialize rather than being artificially flattened into sameness? I don't think CV² alone can answer that. It can tell me *how uneven* the routing is, not whether that unevenness is healthy specialization or unhealthy collapse. That distinction needs a different diagnostic than the one I currently have, and that's the next thing worth building.

not guarnteed updation: baseline comparison against vanilla top-k routing and larger-scale training, blocked currently by local compute constraints