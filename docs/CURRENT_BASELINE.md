# Current Earth baseline

Generated from the tracked deterministic real-terrain validation report. Historical plans and audits provide investigation context; this page is the current regression contract.

## Configuration

- Grid: 64×128
- Time scale: MONTHLY
- Spin-up/evaluation: 1.0 / 1.0 years

## Headline skill

- Köppen group accuracy: 0.674
- Köppen class accuracy: 0.388
- Coldest-month threshold accuracy: 0.900
- Warmest-month threshold accuracy: 0.677

## Climate state

- Global temperature: 17.96 °C
- Global precipitation: 2.996 mm/day
- Cloud fraction: 0.161

## Regional precipitation targets

Annual regional means from the validation fixture. These are broad climatological guardrails, not station-level targets.

| Region | Model (mm/year) | Target (mm/year) | Status |
| --- | ---: | ---: | --- |
| Sahara | 173 | < 200 | within target |
| Kalahari | 139 | < 200 | within target |
| Atacama | 64 | < 50 | outside target |
| Canadian Prairies | 480 | 400–500 | within target |
| US Midwest | 937 | 800–1000 | within target |
| Central Europe | 923 | 550–750 | outside target |
| SE US | 986 | 1100–1500 | outside target |
| East China | 663 | 1300–1800 | outside target |
| S Japan | 960 | 1600–2200 | outside target |

Run `scripts/run_real_terrain_validation.py --compare` to reproduce this baseline, and regenerate this page after an intentional baseline update.
