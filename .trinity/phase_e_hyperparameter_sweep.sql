-- Phase E.HYPER — Hyperparameter Sweep (32 experiments)
-- Anchor: phi^2 + phi^-2 = 3 · TRINITY · NEVER STOP
-- Champion baseline: IGLA-TRAIN_V2-FP32-E0059-H2048-rng43 BPB=1.75
--
-- Methodology (frozen unless noted):
--   model=train_v2, number_format=fp32, variant=WT+resid
--   steps_budget=2000, loss=BPB (NTP CE / ln(2))
--   seeds=F17/F18/F19 = 1597/2584/4181
--
-- Distribution: acc0(8) acc1(8) acc2(7) acc3(9) = 32 total
-- Phases:
--   E.LR   (priority 85-90) — Learning Rate Ladder          acc0(3) acc1(3)
--   E.OPT  (priority 75-83) — Alternative Optimizers         acc2(3) acc3(3)
--   E.DIM  (priority 70-82) — d_model Capacity Sweep         acc0(2) acc1(2) acc2(2)
--   E.CTX  (priority 65-75) — Context Length Sweep           acc1(1) acc2(1) acc3(3)
--   E.NGRAM(priority 60-72) — N-gram Sweep                   acc0(2) acc1(1) acc2(1) acc3(1)
--   E.VAR  (priority 70-80) — Architecture Variants          acc0(1) acc1(1) acc3(2)

INSERT INTO experiment_queue
    (canon_name, config_json, priority, seed, steps_budget, account, status, created_by)
VALUES

-- ═══════════════════════════════════════════════════════════════
-- Phase E.LR — Learning Rate Ladder (6 experiments)
-- Hypothesis: champion uses lr=0.002; phi-optimal may be different
-- phi^3=4.236 → alpha_phi/phi^3 ladder: 0.0015, 0.003, 0.004, 0.0025, 0.005, 0.006
-- acc0(3), acc1(3)
-- ═══════════════════════════════════════════════════════════════

    ('IGLA-TRAIN_V2-FP32-E0110-LR0015-H2048-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.0015,"note":"E.LR: lr=phi^-2*0.004; below champion lr"}'::jsonb,
     90, 1597, 2000, 'acc0', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0111-LR0030-H2048-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.003,"note":"E.LR: lr=phi*0.002; above champion lr"}'::jsonb,
     89, 2584, 2000, 'acc0', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0112-LR0040-H2048-rng4181',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.004,"note":"E.LR: lr=alpha_phi/phi^3; Fibonacci-optimal"}'::jsonb,
     88, 4181, 2000, 'acc0', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0113-LR0025-H2048-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.0025,"note":"E.LR: lr midpoint 0.002..0.003"}'::jsonb,
     87, 1597, 2000, 'acc1', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0114-LR0050-H2048-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.005,"note":"E.LR: lr=2.5x champion; aggressive upper bound"}'::jsonb,
     86, 2584, 2000, 'acc1', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0115-LR0060-H2048-rng4181',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.006,"note":"E.LR: lr=3x champion; expect instability"}'::jsonb,
     85, 4181, 2000, 'acc1', 'pending', 'human'),

-- ═══════════════════════════════════════════════════════════════
-- Phase E.OPT — Alternative Optimizers (6 experiments)
-- Hypothesis: AdamW may not be optimal; SGD/RMSprop may generalize better
-- acc2(3), acc3(3)
-- ═══════════════════════════════════════════════════════════════

    ('IGLA-TRAIN_V2-FP32-E0120-SGD-H2048-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"SGD","lr":0.01,"note":"E.OPT: vanilla SGD; lr=5x AdamW (standard scaling)"}'::jsonb,
     83, 1597, 2000, 'acc2', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0121-SGDM-H2048-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"SGD","lr":0.01,"momentum":0.9,"note":"E.OPT: SGD+Momentum=0.9; classical deep learning"}'::jsonb,
     82, 2584, 2000, 'acc2', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0122-RMSPROP-H2048-rng4181',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"RMSprop","lr":0.001,"note":"E.OPT: RMSprop; adaptive lr without momentum bias"}'::jsonb,
     81, 4181, 2000, 'acc2', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0123-ADAM-H2048-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"Adam","lr":0.002,"weight_decay":0,"note":"E.OPT: Adam no weight_decay; compare vs AdamW"}'::jsonb,
     80, 1597, 2000, 'acc3', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0124-ADAMW-WD001-H2048-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"weight_decay":0.001,"note":"E.OPT: AdamW wd=0.001 (light decay)"}'::jsonb,
     79, 2584, 2000, 'acc3', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0125-ADAMW-WD005-H2048-rng4181',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"weight_decay":0.05,"note":"E.OPT: AdamW wd=0.05 (strong decay)"}'::jsonb,
     78, 4181, 2000, 'acc3', 'pending', 'human'),

-- ═══════════════════════════════════════════════════════════════
-- Phase E.DIM — d_model Capacity Sweep (6 experiments)
-- Hypothesis: champion at 2048; Fibonacci/Lucas sizes may hit sweet spot
-- Lucas: L4=7,L5=11,L6=18 → sizes: 512, 768, 1280, 1536(confirmed), 2560, 3072
-- acc0(2), acc1(2), acc2(2)
-- ═══════════════════════════════════════════════════════════════

    ('IGLA-TRAIN_V2-FP32-E0130-H512-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":512,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.DIM: d_model=512; minimum viable for GF16 L-R9 guard"}'::jsonb,
     72, 1597, 2000, 'acc0', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0131-H768-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":768,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.DIM: d_model=768; GPT-2 small width"}'::jsonb,
     71, 2584, 2000, 'acc0', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0132-H1280-rng4181',
     '{"model":"train_v2","number_format":"fp32","d_model":1280,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.DIM: d_model=1280; between 1024 and 1536"}'::jsonb,
     70, 4181, 2000, 'acc1', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0133-H1536-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":1536,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.DIM: d_model=1536; confirmed BPB=1.78 in H1536 lane"}'::jsonb,
     82, 1597, 2000, 'acc1', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0134-H2560-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":2560,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.DIM: d_model=2560; between champion 2048 and H3072"}'::jsonb,
     76, 2584, 2000, 'acc2', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0135-H3072-rng4181',
     '{"model":"train_v2","number_format":"fp32","d_model":3072,"ctx_len":12,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.DIM: d_model=3072; confirmed BPB=1.964 in E0087"}'::jsonb,
     77, 4181, 2000, 'acc2', 'pending', 'human'),

-- ═══════════════════════════════════════════════════════════════
-- Phase E.CTX — Context Length Sweep (5 experiments)
-- Hypothesis: ctx_len=12 is champion; shorter/longer may expose pattern
-- acc1(1), acc2(1), acc3(3)
-- ═══════════════════════════════════════════════════════════════

    ('IGLA-TRAIN_V2-FP32-E0140-CTX08-H2048-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":8,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.CTX: ctx=8; minimal context; expect higher BPB"}'::jsonb,
     65, 1597, 2000, 'acc3', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0141-CTX10-H2048-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":10,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.CTX: ctx=10; F5=5*2; Lucas proximity"}'::jsonb,
     67, 2584, 2000, 'acc3', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0142-CTX14-H2048-rng4181',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":14,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.CTX: ctx=14; ctx=n_gram; exact alignment hypothesis"}'::jsonb,
     68, 4181, 2000, 'acc3', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0143-CTX16-H2048-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":16,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.CTX: ctx=16; one tested before; confirm pattern"}'::jsonb,
     69, 1597, 2000, 'acc1', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0144-CTX20-H2048-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":20,"n_gram":14,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.CTX: ctx=20; F7=13 neighborhood; wider context"}'::jsonb,
     66, 2584, 2000, 'acc2', 'pending', 'human'),

-- ═══════════════════════════════════════════════════════════════
-- Phase E.NGRAM — N-gram Order Sweep (5 experiments)
-- Hypothesis: n_gram=14 fixed; test 12,13,15,16,18
-- Lucas: L6=18 → n_gram=18 may be natural closure
-- acc0(2), acc1(1), acc2(1), acc3(1)
-- ═══════════════════════════════════════════════════════════════

    ('IGLA-TRAIN_V2-FP32-E0150-NG12-H2048-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":12,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.NGRAM: n_gram=12=ctx_len; exact alignment"}'::jsonb,
     62, 1597, 2000, 'acc0', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0151-NG13-H2048-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":13,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.NGRAM: n_gram=13; F7=13 Fibonacci"}'::jsonb,
     61, 2584, 2000, 'acc0', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0152-NG15-H2048-rng4181',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":15,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.NGRAM: n_gram=15; one above champion"}'::jsonb,
     63, 4181, 2000, 'acc1', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0153-NG16-H2048-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":16,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.NGRAM: n_gram=16; power-of-2 alignment"}'::jsonb,
     64, 1597, 2000, 'acc2', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0154-NG18-H2048-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":18,"variant":"WT+resid","optimizer":"AdamW","lr":0.002,"note":"E.NGRAM: n_gram=18=L6=phi^6+phi^-6; Lucas closure"}'::jsonb,
     72, 2584, 2000, 'acc3', 'pending', 'human'),

-- ═══════════════════════════════════════════════════════════════
-- Phase E.VAR — Architecture Variants (4 experiments)
-- Hypothesis: WT+resid is champion; ablate components
-- acc0(1), acc1(1), acc3(2)
-- ═══════════════════════════════════════════════════════════════

    ('IGLA-TRAIN_V2-FP32-E0160-WT-ONLY-H2048-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT-only","optimizer":"AdamW","lr":0.002,"note":"E.VAR: WT without residual; ablate resid contribution"}'::jsonb,
     75, 1597, 2000, 'acc0', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0161-RESID-ONLY-H2048-rng2584',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"resid-only","optimizer":"AdamW","lr":0.002,"note":"E.VAR: resid without WT; ablate WT contribution"}'::jsonb,
     74, 2584, 2000, 'acc1', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0162-NO-RESID-H2048-rng4181',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"no-resid","optimizer":"AdamW","lr":0.002,"note":"E.VAR: no residual connections; baseline MLP"}'::jsonb,
     73, 4181, 2000, 'acc3', 'pending', 'human'),

    ('IGLA-TRAIN_V2-FP32-E0163-WT-RESID-D6-H2048-rng1597',
     '{"model":"train_v2","number_format":"fp32","d_model":2048,"ctx_len":12,"n_gram":14,"variant":"WT+resid","attn_layers":6,"optimizer":"AdamW","lr":0.002,"note":"E.VAR: WT+resid depth=6 attn layers; vs default depth"}'::jsonb,
     80, 1597, 2000, 'acc3', 'pending', 'human')

ON CONFLICT (canon_name, seed, account) DO NOTHING;

-- ═══════════════════════════════════════════════════════════════
-- L7 audit row
-- ═══════════════════════════════════════════════════════════════
INSERT INTO gardener_decisions (ts, action, affected_exp_ids, reason, snapshot)
SELECT
    now(),
    'enqueue',
    array_agg(id),
    'Phase E.HYPER — 32-experiment hyperparameter sweep. Champion frozen at BPB=1.75 (H2048 FP32). Phases: E.LR(6) E.OPT(6) E.DIM(6) E.CTX(5) E.NGRAM(5) E.VAR(4). acc0-acc3 balanced. Anchor: phi^2+phi^-2=3.',
    jsonb_build_object(
        'phase', 'E.HYPER',
        'champion', 'IGLA-TRAIN_V2-FP32-E0059-H2048-rng43 BPB=1.75',
        'total_experiments', 32,
        'sub_phases', jsonb_build_array('E.LR','E.OPT','E.DIM','E.CTX','E.NGRAM','E.VAR'),
        'steps_budget', 2000,
        'seeds', jsonb_build_array(1597, 2584, 4181),
        'trinity', 'phi^2+phi^-2=3'
    )
FROM experiment_queue
WHERE canon_name LIKE 'IGLA-TRAIN_V2-FP32-E01%-%H2048%'
  AND status = 'pending';
