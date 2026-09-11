# Scoring

Every model is scored along two **independent** axes per event, on a 0–100 scale where higher is better.

## 1. Calibration score $S_{\text{cal}}$

For each calibration question $q$ the model returns a probability $\hat p_q \in [0,1]$ against the binary ground-truth outcome $y_q \in \{0,1\}$.

Each question carries a weight $w_q = w^{\text{time}}_q \cdot w^{\text{win}}_q$ with two factors:

- **Time factor** $w^{\text{time}}_q = 1 / (1 + 0.04 \cdot \Delta t_q)$ — the further the resolution from the cutoff (in days $\Delta t_q$), the lower the weight.
- **Window factor** $w^{\text{win}}_q = W_q / (W_q + 4)$ — the shorter the question window $W_q$, the lower the weight.

The weighted MAE is normalized against a uniform 50 % predictor:

$$
\mathrm{wMAE}_{\text{cal}} = \frac{\sum_q w_q \, \lvert \hat p_q - y_q \rvert}{\sum_q w_q}
\qquad
S_{\text{cal}} = 100 \cdot \max\!\bigl(0,\; 1 - \mathrm{wMAE}_{\text{cal}} / 0.50\bigr)
$$

- A perfect predictor scores **100**.
- The uniform 50 % baseline scores **0**.

## 2. Temporal accuracy score $S_{\text{time}}$

For each ground-truth event $e$ the model predicts a date $\hat d_e$, with error $\Delta_e = |\hat d_e - d^{\text{GT}}_e|$ in days.

The weighted MAE in days is normalized against a **bucket-midpoint baseline** $\mathrm{wMAE}^{\text{base}}_{\text{time}}$ that places every event at the middle of its 30-day bucket (1–30 / 31–60 / 61–90, i.e. days 15.5, 45.5, 75.5):

$$
\mathrm{wMAE}^{\text{days}}_{\text{time}} = \frac{\sum_e w^{\text{time}}_e \, \Delta_e}{\sum_e w^{\text{time}}_e}
\qquad
S_{\text{time}} = \frac{100}{1 + \mathrm{wMAE}^{\text{days}}_{\text{time}} / \mathrm{wMAE}^{\text{base}}_{\text{time}}}
$$

Reference points:

| Performance | $S_{\text{time}}$ |
|---|---|
| Matches the baseline | 50 |
| Two-fold improvement over baseline | 66.7 |
| Two-fold regression over baseline | 33.3 |

## 3. Aggregation across events

Both $S_{\text{cal}}$ and $S_{\text{time}}$ are computed independently per event, giving a per-event score-pair $(S_{\text{cal}}, S_{\text{time}})$. The **cross-event mean** (arithmetic average of the five per-event scores) is reported as the headline number, alongside per-event scorecards so per-event variance remains visible.

## 4. Why two axes?

A model can be **well-calibrated but date-blind** (good probabilities, bad when-it-happens) or vice versa. Reporting only one number would conflate the two failure modes. Keeping them orthogonal lets researchers diagnose models more precisely.
