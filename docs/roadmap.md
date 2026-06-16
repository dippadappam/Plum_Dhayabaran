# Document-Intelligence Roadmap

The phased plan for the full document-intelligence stack on top of the
seven-stage pipeline. Each phase is independently shippable, keeps all 167
existing tests green, and reports an eval delta from the Phase-1 harness.

| Phase | Ships | Status |
|---|---|---|
| 1 | Evaluation harness: synthetic labeled dataset, degradation pipeline, field-level P/R/F1 eval with hallucination rate, confidence-reliability report, e2e decision eval, measured baseline | **Built** (see `docs/eval_baseline.md`) |
| 2 | Ingestion & preprocessing: EXIF orient, deskew, perspective, denoise, multi-document split, multi-page PDF, HEIC | Planned |
| 3 | High-precision extraction: per-field confidence, blind dual-pass on decision-critical fields + verification pass, handwriting-focused re-pass | Planned |
| 4 | Medical normalization: RxNorm (local Prescribable Content) + Indian brand table, ICD-10 tables, OCR-confusable correction + fuzzy matching | Planned |
| 5 | Validation rings: internal consistency, cross-document, entity validation (drug exists, diagnosis exists, RxClass drug↔diagnosis plausibility) | Planned |
| 6 | Calibrated confidence + tiered routing + reviewer UI (uncertain-field flags, corrections, audit) | Planned |
| 7 | Feedback loop: correction store → few-shot retrieval, table growth, eval growth, calibration | Planned |
| 8 | Document-level fraud: cross-claim perceptual-hash reuse detection, metadata checks, entity-based signals | Planned |

## Recorded future-work items (do not lose)

1. **In-browser guided camera capture** (roadmap, alongside Phase 2): frame
   guide, document edge detection, auto-snap when the image is sharp.
   Flash/torch control is browser-limited and unreliable on iOS — do not
   depend on it; the framing guide plus auto-capture-when-sharp is the
   value.

2. **Hindi and regional-language medical-term normalization** (Layer-3 /
   Phase-4 strengthening): Claude vision already reads Hindi and Devanagari
   documents; the gap is mapping Hindi drug and diagnosis names to the
   English RxNorm and ICD-10 dictionaries, which needs a transliteration or
   translation step in front of the matching cascade.

3. **Provider-identity authenticity** (required Layer-4 / Phase-5 finding):
   a bill with no provider name, no letterhead, or an invalid doctor
   registration or GSTIN format must be a red flag that lowers confidence
   and routes to a human — not a free pass that quietly proceeds without
   the network discount. Until Phase 5 ships this is a **current gap**
   (also listed in architecture.md Known Limitations).
