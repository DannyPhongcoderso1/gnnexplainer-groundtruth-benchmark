# Elliptic PGExplainer Step03 Pilot And Parity

## Scope

This is a train/validation-only pilot for Task28. It uses IDs from the frozen common train and validation pools only. It does not access final targets, does not start full training, and does not produce Task28 final PGExplainer results.

## Timing

| Backbone | n | mean | std | median | min | max | estimated 30 epoch seconds | peak GPU memory bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | 10 | 0.133849 | 0.002751 | 0.133376 | 0.130552 | 0.139056 | 8030.96 | 2933682176 |
| GraphSAGE | 10 | 0.113923 | 0.001376 | 0.113695 | 0.112454 | 0.117366 | 6835.36 | 2549965312 |

## Integrity

- Target semantics: frozen backbone original predicted class.
- PGExplainer parameters changed: `True`
- Backbone weights unchanged: `True`
- Optimizer isolation: `True`
- Reload parity: `True`
- Interrupted/resumed fixture parity: `True`
- Final targets accessed: `False`

## Recommendation

`freeze implementation for Step04 compute plan`

## Stop Boundary

STOP FOR REVIEW. Step04 is the next gate and full Task28 training is not started here.
