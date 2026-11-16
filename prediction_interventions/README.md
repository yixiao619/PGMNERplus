# PPIB-focused three-arm study, 2026-11-06

Fixed reference /data/jyx/PGMNER_functional_2x2_20261106: PPIB partial90.362%, VIB partial90.573% best DEV. No cherry-picking across configurations; these are one-seed development results, not established significance.

Only PPIB changes. Three arms: keep80 (g(1-g), target.8), entropy60 (binary entropy normalized by4ln2, target.6), entropy_keep80 (normalized entropy,target.8). Reference quadratic target.6 reused. Both polarization shapes have maximum.25. Identical budget weight, delayed polarization schedule, mu gate, posterior noise, KL, frozen vision, common/local split, partial OT, contrastive weight.03, no local entity auxiliary, deterministic settings and separate gradient clipping. Seed42, eight epochs, DEV-best, no TEST. No manuscript edits.

Tests passed: unchanged default forward output and polarization loss exactly match prior implementation with identical weights; entropy maximum scaling; finite penalty and gradients at logits-100,-20,0,20,100; finite module backward and nonzero gate gradient for each new arm. Existing same-GPU20-step deterministic verification inherited; no claim of full cross-device determinism.

Pipeline: three real GPU smoke jobs, then three parallel training jobs, then functional audits including equal-count keep-high/random/low and image/matching perturbations. All commands/configs/checkpoints/results retained. No automatic favorable-score retries. Compare candidate against both baseline PPIB and VIB, and require selection utility rather than merely extreme gates. Partial-OT benefits remain configuration-dependent and do not alone prove semantically correct rejection.

Output /data/jyx/PGMNER_ppib_focus_20261106. Status pipeline_status.json; final summary.json; source snapshot and SHA256. Expected1.5–2.5 hours depending on GPU sharing, possibly longer.
