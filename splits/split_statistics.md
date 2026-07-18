# Dataset Split Statistics

> Generated: 2026-07-13T15:42:25
> Seed: 42  |  Ratios: 0.7/0.15/0.15
> Subject column: `record_id`

## 1. Subject Split

| Split | Subjects |
|---|---|
| train | 89 |
| val | 19 |
| test | 20 |

## 2. Subject Overlap Check

| Check | Overlap |
|---|---|
| Train ∩ Val | 0 |
| Train ∩ Test | 0 |
| Val ∩ Test | 0 |
| **Status** | ✅ PASS |

## 3. Window Split

| Split | Windows | % |
|---|---|---|
| train | 250,686 | 69.6% |
| val | 53,576 | 14.9% |
| test | 56,103 | 15.6% |
| **Total** | **360,365** | **100%** |

## 4. Label Distribution per Split

| Label | Train | Train% | Val | Val% | Test | Test% |
|---|---|---|---|---|---|---|
| AF | 40,732 | 16.2% | 10,406 | 19.4% | 5,972 | 10.6% |
| Normal | 148,690 | 59.3% | 31,551 | 58.9% | 35,273 | 62.9% |
| Mixed | 1,110 | 0.4% | 105 | 0.2% | 44 | 0.1% |
| Other | 476 | 0.2% | 25 | 0.0% | 470 | 0.8% |
| Unlabeled | 59,678 | 23.8% | 11,489 | 21.4% | 14,344 | 25.6% |

## 5. Subject Assignment

**train** (89 subjects): `001, 002, 005, 006, 007, 008, 009, 010, 011, 012, 013, 015, 017, 018, 019, 020, 021, 025, 026, 027...`

**val** (19 subjects): `003, 004, 014, 022, 023, 024, 033, 046, 052, 063, 064, 066, 068, 070, 087, 108, 112, 116, 124`

**test** (20 subjects): `032, 049, 053, 058, 060, 077, 086, 090, 099, 103, 105, 110, 111, 113, 115, 117, 122, 131, 133, 139`


> For the full per-subject assignment, see `subject_split.json`.

---
*Generated on 2026-07-13T15:42:26*
