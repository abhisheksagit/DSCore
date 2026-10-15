-- Panache WhatsApp sales assistant - durable store.
--
-- Runs on the existing PostgreSQL instance. Requires the pgvector extension.
-- The in-memory index in matcher/index.py is the query path for a catalogue of
-- this size; these tables are the source of truth it is built from, and the
-- place every decision is recorded.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- fuzzy title match for OCR'd text

-- ---------------------------------------------------------------- catalogue

CREATE TABLE IF NOT EXISTS product (
    product_id      text PRIMARY KEY,              -- Shopify product id
    handle          text NOT NULL UNIQUE,
    title           text NOT NULL,
    price_cents     integer NOT NULL CHECK (price_cents >= 0),
    status          text NOT NULL DEFAULT 'active',
    tags            text[] NOT NULL DEFAULT '{}',
    -- Derived once at sync time. Panache stocks one colourway per product, so
    -- "do you have this in another colour?" is a retrieval question and needs
    -- this to be queryable rather than recomputed per enquiry.
    colour_family   text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now(),
    -- Set when a Shopify read last confirmed this row. A stale value here is
    -- the alarm that the sync has silently stopped.
    synced_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS product_title_trgm ON product USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS product_colour_family ON product (colour_family);
CREATE INDEX IF NOT EXISTS product_synced_at ON product (synced_at);

-- Sizes only: 38, 40, 42, 44, 46. size_raw keeps whatever Shopify holds so a
-- parse failure is visible and fixable rather than silently dropped.
CREATE TABLE IF NOT EXISTS variant (
    variant_id      text PRIMARY KEY,
    product_id      text NOT NULL REFERENCES product(product_id) ON DELETE CASCADE,
    size_raw        text NOT NULL,
    size            text,                          -- normalised, NULL if unparseable
    inventory       integer NOT NULL DEFAULT 0,
    price_cents     integer,                        -- NULL = inherit product price
    updated_at      timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT variant_size_known CHECK (size IS NULL OR size IN ('38','40','42','44','46'))
);

CREATE INDEX IF NOT EXISTS variant_product ON variant (product_id);
CREATE INDEX IF NOT EXISTS variant_in_stock ON variant (product_id, size) WHERE inventory > 0;

CREATE TABLE IF NOT EXISTS product_image (
    image_id        text PRIMARY KEY,
    product_id      text NOT NULL REFERENCES product(product_id) ON DELETE CASCADE,
    src_url         text NOT NULL,
    local_path      text,
    position        integer NOT NULL DEFAULT 0,
    -- Perceptual hashes as bigint so Postgres can do the near-duplicate
    -- prefilter itself (xor + bit_count) without loading vectors.
    dhash           bigint,
    phash           bigint,
    fetched_at      timestamptz
);

CREATE INDEX IF NOT EXISTS product_image_product ON product_image (product_id);

-- One row per (image, crop view). Several views per photo, mirroring what the
-- query side does, so a customer's tight crop is compared against a
-- comparably tight catalogue crop.
CREATE TABLE IF NOT EXISTS image_embedding (
    image_id        text NOT NULL REFERENCES product_image(image_id) ON DELETE CASCADE,
    view            text NOT NULL DEFAULT 'full',
    model           text NOT NULL,                 -- e.g. 'siglip-onnx', 'classical-v1'
    embedding       vector(768) NOT NULL,
    dhash           bigint,
    phash           bigint,
    colour_family   text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (image_id, view, model)
);

-- Cosine distance, matching the L2-normalised vectors the embedders emit.
-- Lists tuned for a boutique catalogue; revisit past ~50k rows.
CREATE INDEX IF NOT EXISTS image_embedding_cosine
    ON image_embedding USING ivfflat (embedding vector_cosine_ops) WITH (lists = 32);

-- ------------------------------------------------------------ conversations

CREATE TABLE IF NOT EXISTS conversation (
    conversation_id   bigserial PRIMARY KEY,
    wa_contact        text NOT NULL,               -- customer's WhatsApp id
    -- Active product context, so follow-ups work without re-sending a photo:
    -- photo -> "price please" -> "available in 42?" -> "another colour?"
    active_product_id text REFERENCES product(product_id) ON DELETE SET NULL,
    context_set_at    timestamptz,
    -- Automation is paused per chat the moment a human takes over, so the
    -- assistant can never talk over a staff member mid-conversation.
    automation_paused boolean NOT NULL DEFAULT false,
    paused_reason     text,
    paused_until      timestamptz,
    disclosed_at      timestamptz,                 -- AI disclosure sent once
    last_message_at   timestamptz NOT NULL DEFAULT now(),
    created_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (wa_contact)
);

CREATE INDEX IF NOT EXISTS conversation_needs_human
    ON conversation (automation_paused, last_message_at) WHERE automation_paused;

CREATE TABLE IF NOT EXISTS message (
    message_id      bigserial PRIMARY KEY,
    conversation_id bigint NOT NULL REFERENCES conversation(conversation_id) ON DELETE CASCADE,
    wa_message_id   text UNIQUE,                   -- idempotency: webhooks retry
    direction       text NOT NULL CHECK (direction IN ('inbound','outbound')),
    sender          text NOT NULL CHECK (sender IN ('customer','assistant','staff')),
    body            text,
    media_path      text,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS message_conversation ON message (conversation_id, created_at DESC);

-- -------------------------------------------------------------- audit trail

-- One row per match decision. This is how thresholds get tuned, how accuracy
-- is reported, and how any single answer can be explained after the fact.
CREATE TABLE IF NOT EXISTS match_audit (
    audit_id           bigserial PRIMARY KEY,
    conversation_id    bigint REFERENCES conversation(conversation_id) ON DELETE SET NULL,
    message_id         bigint REFERENCES message(message_id) ON DELETE SET NULL,
    query_image_path   text,
    intent             text,
    requested_size     text,
    action             text NOT NULL CHECK (action IN ('answer','confirm','escalate','no_candidates')),
    chosen_product_id  text REFERENCES product(product_id) ON DELETE SET NULL,
    top_score          real,
    margin_ratio       real,
    dup_distance       integer,
    embedder           text,
    verification_model text,
    verification_conf  text,
    verification_note  text,
    reasons            text[] NOT NULL DEFAULT '{}',
    candidates         jsonb NOT NULL DEFAULT '[]',
    ocr_text           text,
    elapsed_ms         real,
    -- Filled in later by staff review. This column is what turns the log into
    -- a measured error rate rather than an opinion.
    human_verdict      text CHECK (human_verdict IN ('correct','wrong','unclear')),
    human_product_id   text REFERENCES product(product_id) ON DELETE SET NULL,
    reviewed_at        timestamptz,
    reviewed_by        text,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS match_audit_created ON match_audit (created_at DESC);
CREATE INDEX IF NOT EXISTS match_audit_unreviewed
    ON match_audit (created_at DESC) WHERE human_verdict IS NULL;
CREATE INDEX IF NOT EXISTS match_audit_action ON match_audit (action, created_at DESC);

-- ------------------------------------------------------------- admin config

-- Global kill switch and tunables, so automation can be stopped from the
-- admin panel without a deploy.
CREATE TABLE IF NOT EXISTS assistant_setting (
    key        text PRIMARY KEY,
    value      jsonb NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    updated_by text
);

INSERT INTO assistant_setting (key, value) VALUES
    ('automation_enabled',   'false'::jsonb),      -- off until the accuracy gate passes
    ('auto_answer_enabled',  'false'::jsonb),      -- shadow mode: draft, do not send
    ('candidate_floor',      '0.35'::jsonb),
    ('margin_clear_ratio',   '0.10'::jsonb),
    ('dup_strong_max',       '8'::jsonb),
    ('verification_model',   '"claude-sonnet-5-5"'::jsonb)
ON CONFLICT (key) DO NOTHING;

-- Accuracy over reviewed decisions. The headline number for weekly reporting:
-- of the answers sent automatically, how many were wrong?
CREATE OR REPLACE VIEW match_accuracy AS
SELECT
    action,
    count(*)                                                      AS reviewed,
    count(*) FILTER (WHERE human_verdict = 'correct')             AS correct,
    count(*) FILTER (WHERE human_verdict = 'wrong')               AS wrong,
    round(100.0 * count(*) FILTER (WHERE human_verdict = 'wrong')
          / NULLIF(count(*) FILTER (WHERE human_verdict IN ('correct','wrong')), 0), 2)
                                                                  AS error_rate_pct
FROM match_audit
WHERE human_verdict IS NOT NULL
GROUP BY action;
