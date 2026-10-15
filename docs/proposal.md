# Panache by Divya — WhatsApp AI Sales Assistant
## Feasibility assessment & solution proposal

**Status:** awaiting approval — nothing built yet
**Date:** 14 October 2026
**Prepared for:** Panache by Divya (Melbourne) — panachebydivya.com

---

## 1. Verdict

**Feasible, and buildable almost entirely on infrastructure you already pay for.** Expected
recurring cost is roughly **A$10–100/month**, dominated by optional WhatsApp provider fees
rather than by AI usage.

Nine of your thirteen requirements are routine engineering. Four need honest discussion before
you approve, because they are where this kind of project usually fails:

| # | Requirement | Assessment |
|---|---|---|
| 2, 3, 4 | Price, sizes, product link | **Straightforward.** Direct Shopify data lookups. |
| 6 | Text enquiries ("pp", "size 42?") | **Straightforward.** Well within current LLM capability. |
| 9 | Shopify sync | **Straightforward.** Webhooks plus a nightly reconcile. |
| 10, 12 | Tone, conversation context | **Straightforward.** |
| 5 | Similar products | **Straightforward**, and genuinely valuable for sold-out lines. |
| 13 | Admin controls | **Straightforward.** |
| **1, 7** | **Image recognition & accuracy** | **The hard part.** Feasible, but not at 100%. See §4. |
| **11** | **Existing WhatsApp number** | **Conditional.** Depends on Meta "Coexistence" eligibility. See §3. |
| **8** | **Human handover** | Easy to build; its *design* is what makes §7 safe. |

The single most important design decision in this project: **the assistant must be allowed to
not know.** A boutique's reputation survives "let me check that for you and come straight
back" — it does not survive confidently quoting the wrong price for the wrong lehenga. Every
accuracy mechanism below exists to serve that.

---

## 2. What we can reuse (and what it costs)

| Need | Using | Extra cost |
|---|---|---|
| Orchestration, webhooks, retries | Your self-hosted **n8n** | A$0 |
| Catalogue, inventory, conversations, vectors | Your **PostgreSQL** (+ `pgvector` extension) | A$0 |
| Read API for the admin panel | Your **PostgREST** | A$0 |
| Compute for matcher + image embedding | Your **Hostinger VPS** (Docker) | A$0 |
| Product data, prices, stock | **Shopify Admin API** | A$0 (included in your plan) |
| Code, deploys | **GitHub** | A$0 |
| Image embeddings | **SigLIP**, self-hosted on the VPS via ONNX | A$0 |
| Reasoning + image verification | **Claude API** (Haiku 5.5 / Sonnet 5.5) | ~A$5–30/mo |
| WhatsApp messaging | **Meta WhatsApp Cloud API** | ~A$0–15/mo |
| WhatsApp provider (only if Coexistence) | A BSP | A$0–80/mo |

**Nothing new needs to be rented.** The only genuinely new recurring line is Claude API usage,
and at your likely volume it is small change.

### Why AI cost is not the worry

A typical image enquiry sends the customer's photo plus 3–5 candidate product photos to Claude
for verification: roughly 6,000 input tokens. On **Claude Haiku 5.5** (US$0.10 per million input
tokens) that is about **US$0.0006 per enquiry** — around **A$1 per 1,000 image enquiries**. Even
routing every match through **Claude Sonnet 5.5** for higher accuracy lands near **A$18 per
1,000**. Text-only enquiries ("price please") cost a fraction of that.

We will start on Sonnet 5.5 for verification (accuracy first), measure, and step down to Haiku
5.5 on the easy cases once the audit log shows it is safe. Budget **A$30/month** for AI and
expect to spend less.

### WhatsApp messaging cost

Meta charges per message, by category. Replies you send **inside the 24-hour window after a
customer messages you are "service" messages**, and each number gets **1,000 free service
messages per month**; beyond that they are roughly **A$0.015 each** (rates vary by source —
confirm against Meta's own rate card before we budget). Because this assistant only ever
*replies* to inbound enquiries, almost all its traffic is free or near-free. We will avoid
paid marketing templates entirely in v1.

---

## 3. WhatsApp integration — the one real constraint

Your requirement is "use our existing WhatsApp Business number without disrupting normal
business conversations." Historically this was impossible: registering a number on the Cloud
API **de-registered it from the WhatsApp Business app**, taking chat history, status, groups and
calling with it. Staff would have lost the phone app they actually use.

Meta's **Coexistence** (launched 2025) fixes exactly this: the same number runs on the Business
app *and* the Cloud API simultaneously, with messages mirrored both ways via webhooks. That is
precisely what you want.

Three caveats, and one is a genuine unknown:

1. **Eligibility is not guaranteed.** It rolls out by country, number and provider. Numbers
   previously attached to a different WhatsApp API account are commonly rejected.
2. **It is one-way.** Once a number moves to Coexistence you cannot move it back.
3. **Onboarding appears to require a provider.** Every source describes Coexistence as a
   sub-flow of Meta's Embedded Signup, run from an approved Tech Provider or BSP. Whether a
   single business can self-serve this with its own Meta app is something **I could not verify** —
   the network policy here blocked Meta's developer docs. This needs confirming before we commit.

### Three paths

| | Path A — Coexistence via BSP | Path B — Dedicated API number | Path C — Migrate number + self-hosted inbox |
|---|---|---|---|
| Your number | Unchanged, keeps working in the app | New second number for the bot | Existing number, **leaves** the Business app |
| Staff workflow | Untouched | Untouched | Must move to a web inbox |
| Recurring cost | A$0–80/mo | A$0 | A$0 |
| Risk | Eligibility; provider lock-in | Customers message the old number | Irreversible; disruptive |
| Speed to live | Fast | Fastest | Medium |

**Recommendation: attempt Path A. If the number is ineligible, fall back to Path B** for the
pilot — put the bot on a second number advertised on Instagram and the website, prove the
accuracy numbers with zero risk to your main line, then revisit Coexistence. I would avoid
Path C: migrating your live number is irreversible and changes how your team works, which is a
poor trade for a system that has not yet earned trust.

A pre-flight eligibility check is item 4 in §8.

---

## 4. Image recognition — how we make it trustworthy

This is the requirement that decides whether the project succeeds, so it gets the most design.

### Why naive image matching fails for fashion

Your catalogue is full of garments that are genuinely hard to tell apart: the same silhouette in
six colourways, the same lehenga with different embroidery density, a kurta set photographed
flat versus on a model. Customers send cropped Instagram screenshots, reel stills, photos of a
phone screen, mirror selfies, and pictures of competitors' products. A single
nearest-neighbour lookup will return a confident answer in all of those cases — including the
wrong one.

### A four-stage pipeline

**Stage 1 — Preprocess.** Strip Instagram/browser chrome from screenshots, correct orientation,
generate the full image plus 2–3 crops (centre, upper body, lower body). This is what makes
cropped screenshots work: we match every crop and keep the best score.

**Stage 2 — Hybrid retrieval.** Three independent signals, fused:
- **Visual:** SigLIP image embeddings over *every* product image you have (model shot, flat lay,
  detail shots), compared with `pgvector` cosine similarity. SigLIP consistently outperforms
  CLIP on product-retrieval benchmarks, which is why it's the choice here.
- **Text in the image:** OCR. Website screenshots usually contain the product name and price;
  Instagram screenshots contain your handle and caption. This single signal resolves a large
  share of otherwise-ambiguous screenshots.
- **Message text:** whatever the customer typed alongside the photo.

Fusing all three beats any one of them, especially on near-identical outfits.

**Stage 3 — Verification by Claude.** The top 3–5 candidates and the customer's image go to
Claude with a structured-output schema, and a deliberately sceptical instruction: identify the
match, cite the *distinguishing* visual evidence (neckline, embroidery motif, dupatta border,
colour), and return `none_of_these` when appropriate. A retrieval score says "these look
similar"; this stage says *why*, and is far better at refusing.

**Stage 4 — Confidence gate.** The verdict, the retrieval margin between the top two candidates,
and the verification confidence decide the action:

| Confidence | Action |
|---|---|
| **High** — clear verdict, strong margin | Answer directly: price, available sizes, link |
| **Medium** — plausible but contested | **Send the candidate's photo and name back and ask "is this the one you meant?"** then answer on confirmation |
| **Low / none / empty catalogue match** | Warm holding reply + escalate to your team |

That middle row is the most important row in this document. It converts most would-be errors
into a natural, friendly, one-tap exchange that customers read as attentiveness rather than
failure — and it is why "never guess when uncertain" is achievable without the assistant
becoming useless.

### What accuracy to actually expect

I will not quote a number before measuring on your catalogue — anyone who does is guessing.
Realistic expectations from comparable systems:

- Clean screenshots of **your own** product photography: **high auto-resolve rate.**
- Cropped Instagram/reel screenshots: **materially lower**, which the confirmation step absorbs.
- Customer-worn photos, competitor products, items never in your catalogue: **mostly escalation**
  — which is the correct outcome, not a failure.

**This is why §8 item 3 (a labelled set of real customer images) is the most valuable thing you
can give me.** With 150–300 historical enquiry images and their correct products, we can measure
auto-resolve rate and — critically — **error rate**, tune the thresholds against real data, and
set a go-live gate: *the assistant does not answer customers automatically until measured error
rate on high-confidence matches is below an agreed threshold.* Without that set, every accuracy
claim in this project is an opinion.

### Every decision is logged

Each match writes an audit row: customer image, candidates with scores, Claude's verdict and
reasoning, the action taken, and later whether a human corrected it. That log is how we tune
thresholds, prove accuracy to you monthly, and catch drift when new collections land.

---

## 4a. Consequences of "no colour variations" *(added 15 Oct 2026)*

Confirmed: one product = one garment in one colourway; variants are sizes only (38/40/42/44/46).
Two real design consequences.

**Good news for accuracy.** Colour becomes a *discriminating* signal rather than a trap. The
classic failure mode for fashion matching — same product, six colourways, retrieval can't tell
them apart because they share a SKU family — does not exist in your catalogue. Colour is also
one of the strongest and most stable signals in an image embedding. So where §4 warned about
near-identical colourways, the residual risk narrows to genuinely different products that happen
to look alike (same silhouette, similar palette, different embroidery). That is a smaller and
more tractable problem.

**Requirement 6 needs a different answer than first assumed.** "Do you have this in another
colour?" cannot be served by a variant lookup, because there are no colour variants. Answering
it correctly is a *retrieval* task: find visually similar garments in a different colour family.
So the flow becomes:

1. Extract the dominant colour family of the matched product.
2. Search for visually similar products (same embedding neighbourhood, similar silhouette) whose
   dominant colour family differs.
3. Reply honestly — each outfit is a single colourway — and offer the closest alternatives with
   photos and links. In stock first.

This is the same machinery as requirement 5 (similar products for sold-out items), so it is
cheap to add — but it must be built deliberately, or the assistant will answer "no" to a very
common and very commercial question. It also means **colour family should be a stored,
queryable attribute**, derived once per product at sync time, not inferred per enquiry.

**Size answers get simpler.** A fixed, small size set means "available in 42?" is a direct
inventory lookup on a known option value, and the assistant can list the full in-stock range
unprompted ("42 is available — we also have 40 and 44 in this one"), which tends to convert
better than a bare yes/no.

---

## 5. Architecture

```
  Customer                                    Your team
     │                                            ▲
     │ photo + "price?"                           │ escalation (WhatsApp/email)
     ▼                                            │
┌─────────────────────┐                  ┌────────┴──────────┐
│  WhatsApp Cloud API │                  │   Admin panel     │
│  (your number)      │                  │  kill switch      │
└──────────┬──────────┘                  │  take over chat   │
           │ webhook                     │  review + audit   │
           ▼                             └────────┬──────────┘
┌──────────────────────────────────────┐          │ PostgREST
│  n8n  (orchestration)                │          │
│  verify sig · dedupe · route · reply │          │
└───┬──────────────────────┬───────────┘          │
    │                      │                      │
    ▼                      ▼                      ▼
┌─────────────────┐  ┌──────────────┐   ┌────────────────────────┐
│ Matcher service │  │  Claude API  │   │ PostgreSQL + pgvector  │
│ (FastAPI,Docker)│  │ verify ·     │   │ products · variants    │
│ preprocess      │  │ intent ·     │   │ images · embeddings    │
│ SigLIP embed    │  │ compose      │   │ conversations · audit  │
│ OCR · kNN       │  └──────────────┘   └───────────┬────────────┘
└─────────────────┘                                 ▲
                                                    │ webhooks + nightly reconcile
                                        ┌───────────┴────────────┐
                                        │  Shopify Admin API     │
                                        │  products · inventory  │
                                        └────────────────────────┘
```

**Division of labour:** n8n does what it is good at — webhooks, routing, retries, visible
workflow state. Image preprocessing, embedding and vector search go into a **small FastAPI
service in Docker** on the VPS, because that work is painful to express as n8n nodes and needs
to be unit-testable. Postgres is the single source of truth for everything conversational.

### Components

1. **Shopify sync.** Webhooks (`products/create|update|delete`, `inventory_levels/update`) with
   HMAC verification and idempotent handlers, treating each webhook as a *trigger to re-read* the
   Admin API rather than trusting its payload — `inventory_levels/update` is known to arrive with
   most fields null. Plus a **nightly full reconcile**, because webhooks get missed and silent
   catalogue drift is how a bot ends up quoting last month's price. New product images are
   embedded automatically, so new arrivals become matchable without intervention.

2. **Matcher service.** The four-stage pipeline from §4.

3. **Conversation state.** Each chat carries an active product context, so follow-ups work
   naturally: photo → "price please" → "available in 42?" → "in blue?" with no re-sending.
   Context expires on a timer and resets when a new photo arrives.

4. **Response composer.** Facts (price in AUD, in-stock sizes from live inventory,
   `panachebydivya.com/products/{handle}`) are pulled from the database, never invented by the
   model; Claude only chooses the wording. This matters — it structurally prevents hallucinated
   prices, which is the failure mode most likely to cost you a customer.

5. **Handover.** Triggers on low confidence, explicit requests for a person, and sensitive
   intents (refunds, alterations, complaints, bulk/wholesale, delivery disputes, bargaining).
   Notifies your team, flags the conversation, and **pauses automation for that chat** so the
   assistant never talks over a human mid-conversation.

6. **Admin panel.** Global automation on/off, per-chat takeover, conversation review, and the
   match audit log with thumbnails and accuracy stats. Built on PostgREST — modest scope. If you
   would rather have a full-featured shared team inbox, **Chatwoot** (open source, self-hosted on
   your VPS, A$0) is the better answer; it is a bigger install but a far nicer daily tool for
   several staff. Your call — see §9.

7. **Guardrails.** No discounting or price negotiation, no stock promises beyond live inventory,
   no delivery-date commitments, no medical/sizing-fit guarantees. Volume caps per customer per
   hour. A one-time, one-line disclosure that customers are talking to Panache's assistant with
   a person available on request — good practice, increasingly expected, and it measurably
   *reduces* complaints when the bot does get something wrong.

---

## 6. Delivery plan

Each phase ends in something you can see and judge. No phase depends on trusting the next.

**Phase 1 — Catalogue & accuracy, no WhatsApp (the gate).**
Shopify sync into Postgres, embeddings for every product image, matcher service, and an
**offline accuracy report** against your labelled image set. Deliverable: a measured number for
auto-resolve rate and error rate, plus a tuned set of thresholds. *We do not proceed to
customers until this passes an agreed bar.*

**Phase 2 — WhatsApp in shadow mode.**
Integration live, text enquiries handled, but **every reply is a draft to your team, nothing
auto-sends.** You see exactly what it would have said, on real traffic, at zero risk. Usually
the most informative week of a project like this.

**Phase 3 — Go live, graduated.**
Auto-send for high-confidence image matches and text follow-ups. Confirmation cards for medium.
Everything else escalates. Kill switch from day one.

**Phase 4 — Polish.**
Admin panel, similar-product suggestions for sold-out lines, weekly accuracy reporting,
threshold tuning from the audit log.

---

## 7. Risks, stated plainly

| Risk | Mitigation |
|---|---|
| **Wrong product → wrong price to a customer** | Facts from database not model; confidence gate; confirmation card; audit log; measured go-live bar |
| **Coexistence ineligible** | Path B pilot on a second number (§3) |
| **Near-identical colourways confused** | Hybrid retrieval + Claude verification citing distinguishing features; colourway pairs explicitly included in the eval set |
| **Customer sends a competitor's outfit** | `none_of_these` → "we don't carry this one, but here's something similar" → escalate |
| **Catalogue drift (prices, stock)** | Nightly reconcile + webhook idempotency; staleness alarm if sync stops |
| **Bot talks over a staff member** | Per-chat automation pause on handover |
| **VPS too weak for embeddings** | Needs spec check (§8.5). Fallback: hosted embedding API, a few dollars a month |
| **Over-automation damages brand** | Shadow mode first; narrow scope; everything non-sales escalates |

**What I am deliberately not promising:** 100% image accuracy, correct identification of
garments not in your catalogue, or useful answers to styling and fit questions that need
Divya's judgement. Those escalate, by design.

---

## 8. What I need from you

Items 1–5 block the build. Items 6–10 are needed before go-live.

1. **Shopify Admin API access.** A custom app token with `read_products`, `read_inventory`,
   `read_locations`, plus your `*.myshopify.com` domain. Also: roughly how many products and
   variants, and do all products have clean studio images?
2. ~~**Size and colour conventions.**~~ **Resolved 15 Oct 2026:** no colour variations — one
   product is one garment in one colourway. Sizes are numeric: **38, 40, 42, 44, 46**, as variant
   option values. Design consequences in §4a. *Still needed:* confirmation that these are the
   variant option values in Shopify as spelled (e.g. `42`, not `Size 42` or `42 (M)`), and what
   happens for free-size / unstitched items if you carry any.
3. **A labelled image set — the highest-value item.** 150–300 real customer enquiry images from
   your WhatsApp history, each tagged with the correct product. Include the hard cases: cropped
   screenshots, reel stills, near-identical colourways, and items you don't stock. Without this
   there is no honest accuracy measurement.
4. **WhatsApp account facts.** The number; whether it's on the WhatsApp Business app today;
   whether you have a Meta Business Manager / existing WhatsApp Business Account; whether the
   business is Meta-verified; and who controls the number. I'll run the Coexistence eligibility
   check from this.
5. **VPS and stack details.** vCPU / RAM / disk, Docker available, n8n version and whether it's
   reachable on a public HTTPS hostname (needed for webhooks), Postgres version, and whether we
   can install `pgvector`.
6. **Brand voice.** 10–20 real examples of how you and your team reply — greetings, sign-offs,
   emoji habits, whether Hinglish is used, and how you currently answer "price please". The
   assistant should sound like Panache, not like a chatbot.
7. **Escalation preferences.** Who gets notified, by what channel, business hours, and what the
   after-hours holding message should say.
8. **Policy answers.** Do we quote prices on sold-out items? How should discount requests,
   bargaining and wholesale enquiries be handled? Shipping and returns lines we may quote
   verbatim.
9. **Privacy decisions.** Customer-sent images are personal information under Australian privacy
   law. I need a retention period (my suggestion: 90 days for audit, then purge), and agreement
   on the AI-assistant disclosure line.
10. **Repo decision.** Build in this `dscore` repository, or a new dedicated one?

**One environment note:** this cloud session's network policy currently blocks all outbound
hosts, so I could not inspect your live catalogue, Meta's docs or Shopify's docs directly. For
the build I'll need `panachebydivya.com`, `*.myshopify.com`, `graph.facebook.com`,
`api.anthropic.com` and `huggingface.co` allowed under the environment's **Network access**
settings — or we develop against the VPS instead.

---

## 9. Decisions — approved 14 October 2026

| Decision | Chosen |
|---|---|
| **WhatsApp path** | **Attempt Coexistence** on the existing number. Eligibility check is the first task; Path B (second-number pilot) is the agreed fallback if Meta rejects the number. |
| **Admin surface** | **Lightweight custom panel** on PostgREST — kill switch, per-chat takeover, conversation review, match audit log. |
| **Embeddings** | **Self-host SigLIP on the VPS.** Conditional on the spec check in §8.5; hosted API is the fallback if the VPS can't carry it. |
| **Auto-send at launch** | **Auto-answer on high confidence**, photo confirmation card on medium, escalation on low — gated on Phase 1's measured error rate. |

### Immediate next steps

1. Run the WhatsApp Coexistence eligibility check (needs §8.4).
2. VPS spec check and SigLIP benchmark (needs §8.5).
3. Shopify sync + schema, once the Admin API token exists (§8.1). Size/colour model is
   settled (§4a) — schema can be written now.
4. Assemble the labelled image set (§8.3) — this gates Phase 1's accuracy report, which in
   turn gates any customer-facing reply.

## Sources

WhatsApp Coexistence: [Whautomate](https://whautomate.com/whatsapp-coexistence) ·
[YCloud](https://www.ycloud.com/blog/whatsapp-business-app-coexistence-meta-update) ·
[LeadSquared](https://www.leadsquared.com/learn/marketing/whatsapp-coexistence/) ·
[Converzap](https://converzap.com/en/blog/whatsapp-coexistence-vs-cloud-api/) ·
[respond.io](https://respond.io/help/whatsapp/whatsapp-coexistence) ·
[Clientify](https://clientify.com/en/blog/communication/whatsapp-coexistence)
WhatsApp pricing: [Zian AI](https://zian.ai/whatsapp-business-api-cost-per-message-ai-agents/) ·
[OnCloudAPI](https://oncloudapi.com/blogs/whatsapp-business-api-pricing-2026-country-wise-rates) ·
[Whautomate](https://whautomate.com/whatsapp-business-api-pricing)
Shopify webhooks: [mgroupweb](https://mgroupweb.com/blogs/shopify-webhooks-developer-guide/) ·
[InventiveHQ](https://inventivehq.com/blog/shopify-webhooks-guide) ·
[Shopify community](https://community.shopify.com/t/tracking-product-quantity-via-rest-webhooks/322189)
Image matching: [PRISM](https://arxiv.org/pdf/2509.14985) ·
[Grocery VLM retrieval](https://arxiv.org/pdf/2605.18029) ·
[LookSync](https://arxiv.org/pdf/2511.00072) ·
[Mercari SigLIP](https://engineering.mercari.com/en/blog/entry/20241104-similar-looks-recommendation-via-vision-language-model/) ·
[Elastic SigLIP-2](https://www.elastic.co/search-labs/blog/multimodal-search-siglip-2-elasticsearch)

*All third-party pricing and policy figures should be confirmed against Meta's and Shopify's own
documentation before we commit budget — vendor blogs disagree with each other, and I could not
reach the primary sources from this environment.*
