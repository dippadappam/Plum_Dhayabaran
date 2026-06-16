# Extraction Eval Baseline

Generated: 2026-06-13T02:43:10.541400+00:00  
Documents: 51 | Extraction failures: 0

## Overall

- Precision: **84.7%**
- Recall: **92.2%**
- F1: **88.3%**
- **Hallucination rate: 15.3%** (wrong values + spurious fields / all predictions; a wrong value is graded worse than an abstained null)
- Quality score: 83.8% (correct=1, abstained=0.5, wrong=0)
- Outcome counts: {'correct': 411, 'wrong': 22, 'missed': 13, 'spurious': 52}

### Per document type

| Group | P | R | F1 | Halluc. | Quality | Fields |
|---|---|---|---|---|---|---|
| hospital_bill |  88.6% |  89.4% |  89.0% |  11.4% |  87.2% | 109 |
| lab_report |  46.2% |  80.0% |  58.5% |  53.8% |  46.2% | 104 |
| pharmacy_bill |  98.4% |  94.8% |  96.6% |   1.6% |  96.6% | 134 |
| prescription |  96.6% |  95.8% |  96.2% |   3.4% |  95.1% | 123 |
| prescription_hw | 100.0% | 100.0% | 100.0% |   0.0% | 100.0% | 28 |

### Per degradation profile

| Group | P | R | F1 | Halluc. | Quality | Fields |
|---|---|---|---|---|---|---|
| blur |  86.5% |  97.0% |  91.4% |  13.5% |  86.5% | 37 |
| clean |  91.7% |  97.1% |  94.3% |   8.3% |  91.7% | 36 |
| combined_heavy |  52.0% |  38.2% |  44.1% |  48.0% |  51.3% | 38 |
| combined_moderate |  88.1% | 100.0% |  93.7% |  11.9% |  88.1% | 42 |
| crumple |  86.4% |  97.4% |  91.6% |  13.6% |  86.4% | 44 |
| glare |  91.2% |  93.9% |  92.5% |   8.8% |  91.2% | 34 |
| handwritten | 100.0% | 100.0% | 100.0% |   0.0% | 100.0% | 9 |
| handwritten_blur | 100.0% | 100.0% | 100.0% |   0.0% | 100.0% | 10 |
| handwritten_skew | 100.0% | 100.0% | 100.0% |   0.0% | 100.0% | 9 |
| lowres |  82.1% |  97.0% |  88.9% |  17.9% |  82.1% | 39 |
| noise |  80.0% |  94.1% |  86.5% |  20.0% |  80.0% | 40 |
| rotate90 |  82.1% |  97.0% |  88.9% |  17.9% |  82.1% | 39 |
| shadow |  90.2% |  94.9% |  92.5% |   9.8% |  90.2% | 41 |
| skew |  87.2% |  97.1% |  91.9% |  12.8% |  87.2% | 39 |
| stamp |  78.0% |  94.1% |  85.3% |  22.0% |  78.0% | 41 |

### Per field

| Group | P | R | F1 | Halluc. | Quality | Fields |
|---|---|---|---|---|---|---|
| date |  96.0% |  94.1% |  95.0% |   4.0% |  95.1% | 51 |
| diagnosis | 100.0% | 100.0% | 100.0% |   0.0% | 100.0% | 15 |
| doctor_name |  78.0% |  76.5% |  77.2% |  22.0% |  77.5% | 51 |
| doctor_registration |  56.0% |  93.3% |  70.0% |  44.0% |  55.8% | 26 |
| document_type |  90.2% |  90.2% |  90.2% |   9.8% |  90.2% | 51 |
| hospital_name | 100.0% | 100.0% | 100.0% |   0.0% | 100.0% | 51 |
| line_items[] |  62.1% |  93.7% |  74.7% |  37.9% |  61.9% | 97 |
| medicines[] |  95.8% |  93.2% |  94.5% |   4.2% |  92.9% | 77 |
| patient_name |  98.0% |  94.1% |  96.0% |   2.0% |  96.1% | 51 |
| total |  95.7% |  91.7% |  93.6% |   4.3% |  93.8% | 24 |
| treatment |   0.0% |     -- |     -- | 100.0% |   0.0% | 4 |

### Confidence reliability (stated vs measured)

| Stated confidence | Documents | Mean stated | Field accuracy |
|---|---|---|---|
| 0.0-0.5 | 3 |  30.0% |  29.2% |
| 0.5-0.7 | 1 |  52.0% |  42.9% |
| 0.7-0.8 | 1 |  78.0% |  40.0% |
| 0.8-0.9 | 4 |  84.2% | 100.0% |
| 0.9-1.0 | 42 |  97.3% |  86.3% |

### Regression vs previous run

- precision: 81.1% -> 84.7% (+3.6 pts)
- recall: 92.4% -> 92.2% (-0.2 pts)
- f1: 86.4% -> 88.3% (+1.9 pts)
- hallucination_rate: 18.9% -> 15.3% (-3.6 pts)

## End-to-end decision accuracy

Generated: 2026-06-13T02:47:04.593579+00:00  
Result: **4 of 6 bundles decided as expected.**

| Bundle | Outcome | Amount | Category | Conf. | Result |
|---|---|---|---|---|---|
| clean_consult | APPROVED | 1350.0 | CONSULTATION | 0.92 | PASS |
| dental_partial | NEEDS_RESUBMISSION | None | DENTAL | 0.93 | FAIL |
| excluded_obesity | REJECTED | 0 | CONSULTATION | 0.93 | PASS |
| network_discount | APPROVED | 2160.0 | CONSULTATION | 0.94 | PASS |
| unreadable_bill | APPROVED | 1242.0 | CONSULTATION | 0.66 | FAIL |
| waiting_diabetes | REJECTED | 0 | CONSULTATION | 0.92 | PASS |

## End-to-end decision accuracy

Generated: 2026-06-13T03:09:16.195559+00:00  
Result: **5 of 6 bundles decided as expected.**

| Bundle | Outcome | Amount | Category | Conf. | Result |
|---|---|---|---|---|---|
| clean_consult | APPROVED | 1350.0 | CONSULTATION | 0.92 | PASS |
| dental_partial | NEEDS_RESUBMISSION | None | DENTAL | 0.93 | FAIL |
| excluded_obesity | REJECTED | 0 | CONSULTATION | 0.93 | PASS |
| network_discount | APPROVED | 2160.0 | CONSULTATION | 0.94 | PASS |
| unreadable_bill | MANUAL_REVIEW | 1350.0 | CONSULTATION | 0.67 | PASS |
| waiting_diabetes | REJECTED | 0 | CONSULTATION | 0.92 | PASS |

## End-to-end decision accuracy

Generated: 2026-06-13T11:39:46.095511+00:00  
Result: **5 of 6 bundles decided as expected.**

| Bundle | Outcome | Amount | Category | Conf. | Result |
|---|---|---|---|---|---|
| clean_consult | APPROVED | 1350.0 | CONSULTATION | 0.93 | PASS |
| dental_partial | NEEDS_RESUBMISSION | None | DENTAL | 0.93 | FAIL |
| excluded_obesity | REJECTED | 0 | CONSULTATION | 0.93 | PASS |
| network_discount | APPROVED | 2160.0 | CONSULTATION | 0.94 | PASS |
| unreadable_bill | MANUAL_REVIEW | 1242.0 | CONSULTATION | 0.66 | PASS |
| waiting_diabetes | REJECTED | 0 | CONSULTATION | 0.92 | PASS |

## End-to-end decision accuracy

Generated: 2026-06-13T14:30:00.930527+00:00  
Result: **5 of 6 bundles decided as expected.**

| Bundle | Outcome | Amount | Category | Conf. | Result |
|---|---|---|---|---|---|
| clean_consult | APPROVED | 1350.0 | CONSULTATION | 0.92 | PASS |
| dental_partial | NEEDS_RESUBMISSION | None | DENTAL | 0.93 | FAIL |
| excluded_obesity | REJECTED | 0 | CONSULTATION | 0.93 | PASS |
| network_discount | APPROVED | 2160.0 | CONSULTATION | 0.94 | PASS |
| unreadable_bill | NEEDS_RESUBMISSION | None | CONSULTATION | 0.66 | PASS |
| waiting_diabetes | REJECTED | 0 | CONSULTATION | 0.92 | PASS |

## End-to-end decision accuracy

Generated: 2026-06-13T15:24:27.373571+00:00  
Result: **5 of 6 bundles decided as expected.**

| Bundle | Outcome | Amount | Category | Conf. | Result |
|---|---|---|---|---|---|
| clean_consult | APPROVED | 1350.0 | CONSULTATION | 0.93 | PASS |
| dental_partial | NEEDS_RESUBMISSION | None | DENTAL | 0.93 | FAIL |
| excluded_obesity | REJECTED | 0 | CONSULTATION | 0.93 | PASS |
| network_discount | APPROVED | 2160.0 | CONSULTATION | 0.94 | PASS |
| unreadable_bill | NEEDS_RESUBMISSION | None | CONSULTATION | 0.66 | PASS |
| waiting_diabetes | REJECTED | 0 | CONSULTATION | 0.92 | PASS |
