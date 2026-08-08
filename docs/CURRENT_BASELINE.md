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

Run `scripts/run_real_terrain_validation.py --compare` to reproduce this baseline, and regenerate this page after an intentional baseline update.
