# Product matcher

Customer photo in → product identified → price, available sizes and purchase
link out. Or: a confirmation question, or a handover to a person.

This is Phase 1 of `docs/proposal.md`. It has no WhatsApp dependency and no
Shopify dependency: the catalogue is a plain data structure, so the matcher can
be developed and measured before either integration exists.

## Try it

```bash
python3 -m pytest -q                               # 75 tests, no network needed
python3 -m matcher.cli demo --audit                # synthetic Instagram screenshot
python3 -m matcher.cli match photo.jpg --text "pp" --catalogue cat.json --images ./imgs
python3 eval/run_eval.py --simulate-verification 0.90
```

## How a match is decided

```
customer image ─┬─► crop ensemble ──► perceptual hashes ─┐
                │   (full, trimmed,                      │
                │    chrome-stripped,                    ├─► fused score
                │    centre, upper, lower)               │   per product
                └─► embeddings ──────────────────────────┤
  message text ─────► tokens ───────────────────────────┘
  OCR (optional) ──►
                                   │
                                   ▼
                      top-4 shortlist ──► Claude verification
                                   │      (skipped when the hash
                                   │       is already unambiguous)
                                   ▼
                              confidence gate
                                   │
            ┌──────────────────────┼──────────────────────┐
            ▼                      ▼                      ▼
         ANSWER                 CONFIRM                ESCALATE
    price + sizes + link   "is this the one?"      hand to a person
```

Three retrieval signals, because each covers the others' failure mode:

| Signal | Strong at | Blind to |
|---|---|---|
| Perceptual hash | screenshots of our own photos | different angle; near-twin garments |
| Embedding | angle, crop, lighting | fine trim differences |
| Text (OCR + message) | named products in screenshots | photos with no text |

## Modules

| File | Responsibility |
|---|---|
| `catalogue.py` | Products, size variants, stock, price, URL. No Shopify coupling. |
| `images.py` | Loading, border stripping, the crop ensemble. |
| `fingerprint.py` | dHash, pHash, colour histogram, edge orientation, colour family. |
| `embed.py` | `ClassicalEmbedder` (no weights) and `SiglipOnnxEmbedder`. |
| `index.py` | Hybrid retrieval, scored per product; similar-in-other-colours. |
| `intent.py` | "pp", "size 42?", "another colour?", and what must reach a human. |
| `verify.py` | Claude vision verification with a structured-output schema. |
| `gate.py` | **The only module that decides what a customer sees.** |
| `compose.py` | Reply text. Every fact read from the catalogue, never generated. |
| `pipeline.py` | Orchestration + the audit record. |

## Two safety properties, by construction

**Prices cannot be hallucinated.** No model is ever asked for a price, a size
or a URL. `compose.py` formats them from catalogue rows; the model only ever
chooses *which product*, never *what is true about it*.

**Retrieval scores alone never answer a customer.** Without a verification
verdict the best outcome is a confirmation question — because a retrieval score
has no notion of "these are two different garments that happen to look alike".
The sole exception is a near-duplicate that is *unambiguous* (see below).

## Embedders

`ClassicalEmbedder` is the default and needs no model weights, so everything
here runs and is testable immediately. It is genuinely good at near-duplicates
and weak at generalising across angle and lighting, and its cosine scores sit
in a narrow band — which is why the gate uses a *relative* margin rather than
an absolute score threshold.

`SiglipOnnxEmbedder` is the production encoder. Export the weights where
Hugging Face is reachable, then point `SIGLIP_ONNX_PATH` at the file:

```bash
optimum-cli export onnx --model google/siglip-base-patch16-384 \
    --task feature-extraction siglip-onnx/
```

`embedder.name` is written to every audit record, so a silent downgrade to the
classical embedder can never be mistaken for production accuracy.

## What the eval harness found

The harness exists to stop accuracy being a matter of opinion. Running it
caught two real bugs that code review had not:

**1. A perceptual hash proves similarity, not identity.** The first version
auto-answered on any hash within 8 bits, on the reasoning that a customer
sending a screenshot of our own photo is the strongest evidence available.
Measured: **15 confident-and-wrong answers, a 10.2% error rate** — every one of
them a case where the correct product was a near-tied runner-up (chose dup=0
when the truth was dup=1; chose dup=4 when the truth was dup=4). One picked a
red garment for a black one at distance 0. Fixed by requiring the runner-up to
be at least `dup_unique_margin` bits worse, plus a colour-agreement check.

**2. The pipeline and the gate disagreed about the same condition.** The
pipeline skipped the verification call whenever `dup_distance <= 8`, but after
fix 1 the gate only answered when that near-duplicate was *also* unique. So on
contested twins verification had already been skipped, and **145 of 160
enquiries fell through to "verification unavailable"**. Fixed by giving both
one shared definition: `gate.unambiguous_near_duplicate()`.

A third change came from measurement rather than a bug: when the verifier picks
something other than retrieval's top hit, the two signals disagree, and that is
where the remaining errors concentrated. An override is now allowed to inform
the answer but never to trigger one — it goes out as a confirmation.

Current state on synthetic fixtures, with a simulated verifier:

| Verifier accuracy | Auto-answer | Error among auto-answers | Foreign garments escalated |
|---|---|---|---|
| 0.90 | 14.4% | **0%** | 87.5% (0 auto-answered) |
| 0.75 | 11.9% | **0%** | 81.2% (0 auto-answered) |
| none (safe mode) | 1.2% | **0%** | 43.8% (0 auto-answered) |

The property worth having is in the shape of that table: **verifier quality
moves coverage, not correctness.** A weaker verifier asks more questions; it
does not produce more wrong answers.

### Read the coverage figures with care

The 14.4% auto-answer rate is **pessimistic by construction and is not a
forecast.** The fixtures are flat-colour synthetic garments including four
deliberate near-twin pairs differing only in trim, matched with the
no-weights classical embedder — close to a worst case for margin-based
decisions. Real photographs carry far more texture, and SigLIP separates
garments much more widely.

**The honest summary: this demonstrates correctness, not coverage.** Coverage
can only be measured on Panache's real catalogue with the labelled enquiry
images (`docs/proposal.md` §8.3), and that measurement is the go-live gate.

## Tuning

Every threshold is in `config.py`, and `assistant_setting` in
`schema/001_init.sql` holds the live values so they can be changed from the
admin panel without a deploy.

| Threshold | Default | Meaning |
|---|---|---|
| `dup_strong_max` | 8 | Hamming distance counted as a near-duplicate |
| `dup_unique_margin` | 6 | Bits by which the runner-up must be worse |
| `candidate_floor` | 0.35 | Below this, nothing is plausible → escalate |
| `margin_clear_ratio` | 0.10 | Relative gap for an uncontested match |
| `verify_top_k` | 4 | Candidates sent to verification |

`candidate_floor` is **not yet calibrated against real data.** On fixtures,
garments we do not stock score 0.32–0.51, which overlaps the current floor;
they are caught by the verification stage instead. Recalibrate it with
`eval/run_eval.py --sweep` once the real catalogue exists.

## Not built yet

WhatsApp transport, the Shopify sync, conversation state across turns, and the
admin panel. The schema has tables for all four; see `docs/proposal.md`.
