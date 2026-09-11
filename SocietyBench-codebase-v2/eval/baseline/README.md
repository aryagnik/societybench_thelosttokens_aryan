# Non-LLM Baselines (M4, paper Table 8)

`baseline_freq_momentum.py` — **frequency + momentum**, two "pseudo-models"; pure code, no API calls, zero cost.
They produce probabilities only, no dates (time axis fixed at 50). Run on **all 5 events**.

## Launch (run once per event's final workspace; a shared run-id collects results in one place)
```bash
for E in event1_library event2_trump_tariff event3_tiktok event4_us_iran event5_smci; do
  python3 eval/baseline/baseline_freq_momentum.py \
    --workspace runs_new/$E/final --event-name $E --run-id M4
done
# Results: runs_new/<E>/final/results/run_M4/brier/{frequency,momentum}/
# Read by predict_step4_scorecard as two "models"; the time axis is separately recorded as 50.
```

## Baseline definitions (tunable, see `_rate` in the script)
- **frequency**: λ = pre-cutoff historical event density (nodes / span in days); P(occurs within W days) = clamp(λ·W, 0.02, 0.98).
- **momentum**: same, but λ is estimated from only the **last 7 days** before the cutoff (recent trend continuation).
- Note: the plan did not pin an exact definition for the "similar-event empirical base rate"; this is a reasonable default — to change it, edit `_rate` only.
