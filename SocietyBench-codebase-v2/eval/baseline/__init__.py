"""SocietyBench non-LLM baselines (class D, paper Table 8).

baseline_freq_momentum.py  M4 frequency + momentum, two "pseudo-models"; pure
code, no API calls; they produce probabilities only, no dates (time axis fixed
at 50). Results go to results/run_<id>/brier/{frequency,momentum}/ and are read
by predict_step4_scorecard as two "models".
"""
