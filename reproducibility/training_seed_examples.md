# Seed sensitivity at 1,000 training epochs

These GPU runs use the archived model inputs and the paper workflow settings for each tissue, with training extended to 1,000 epochs. For each dataset we evaluated its original training seed and two additional seeds (17 and 42). The comparison criterion was the absolute difference from the archived receiver score–response Spearman correlation. Seed 17 was selected **after comparing these three runs** as the closest example in each tissue. The [A1](tonsil_a1/README.md) and [P2CRC](crc_p2/README.md) end-to-end workflows default to this 1,000-epoch example. Both accept `--paper` to reproduce the original 200-epoch configuration.

## Tonsil A1 · CXCL13–CXCR5

Across 2,500 receivers the archived correlation was 0.6216. In the top 300 receivers the archived germinal-centre enrichment was 2.3395-fold and the response lift over the remaining receivers was 0.8067.

| Training seed | Receiver–response ρ | Absolute difference from archive | Germinal-centre enrichment (fold) | Response lift |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.4141 | 0.2074 | 1.9528 | 0.6126 |
| 17 | 0.6548 | 0.0333 | 2.2912 | 0.7810 |
| 42 | 0.5758 | 0.0457 | 2.1655 | 0.7449 |

## P2CRC · TGFB1–TGFBR2

Across all 6,000 bins the archived correlation was 0.1431 and the ROC AUC for the 200 response hotspots was 0.6074. The final column compares the top 5% lists among the 4,280 receivers with candidate events. Despite the close correlation and AUC for seed 17, just 29 of its 214 highest-ranked receivers were also in the archived top list.

| Training seed | Receiver–response ρ | Absolute difference from archive | Hotspot ROC AUC | Shared top 5% with archive |
| ---: | ---: | ---: | ---: | ---: |
| 20260624 | 0.1176 | 0.0256 | 0.5828 | 23/214 |
| 17 | 0.1366 | 0.0066 | 0.6065 | 29/214 |
| 42 | 0.0947 | 0.0484 | 0.5908 | 26/214 |
