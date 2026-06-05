# P10-P90 Confidence Bands

## What are P10-P90 Bands?

The **P10-P90 confidence band** is a range that captures where the true value is expected to fall with approximately **80% confidence**.

- **P10 (10th percentile)**: The lower bound. In 10% of historical cases, the actual volume was *below* this value relative to the prediction.
- **P90 (90th percentile)**: The upper bound. In 90% of historical cases, the actual volume was *below* this value relative to the prediction.

Together, the range **[P10, P90]** covers the central 80% of observed prediction errors.

---

## How is it calculated?

For each plant, we look at the **historical errors** on the test set:

```
error = actual_volume - predicted_volume
```

Then we compute:
- `p10_err = percentile(errors, 10)` — the 10th percentile of errors
- `p90_err = percentile(errors, 90)` — the 90th percentile of errors

Finally, we apply these to the forecast:

```
lower_bound = predicted_volume + p10_err
upper_bound = predicted_volume + p90_err
```

Because errors can be negative (under-prediction) or positive (over-prediction), the band naturally widens or narrows based on the plant's historical volatility.

---

## Why use P10-P90 instead of fixed intervals?

| Approach | Problem |
|----------|---------|
| Fixed ±10% | Ignores that some plants are more volatile than others |
| Standard deviation | Assumes normal distribution of errors (often not true) |
| **P10-P90 (percentile-based)** | Captures the *actual empirical distribution* of errors per plant |

**Example:**
- Plant 710 has very regular demand → narrow P10-P90 band
- Plant 515 has erratic demand → wide P10-P90 band

This gives **plant-specific uncertainty** rather than a one-size-fits-all margin.

---

## What does the band mean in practice?

When you see a prediction with a P10-P90 band:

> "Predicted: 15 m³/hr | Band: [8, 26] m³/hr"

This means:
- Based on historical performance, the actual volume will fall between **8 and 26 m³/hr** approximately **80% of the time**.
- There is still a 10% chance it falls below 8, and a 10% chance it exceeds 26.

---

## Limitations

1. **Past performance ≠ future guarantee**: If demand patterns shift (new clients, new schedules), the historical error distribution may no longer apply.
2. **Sparse data**: Plants with very few operating hours may have unstable percentile estimates.
3. **Extreme events**: The P10-P90 band does *not* capture black-swan events (mega-projects, strikes, natural disasters).

---

## Related Concepts

| Term | Meaning | Difference from P10-P90 |
|------|---------|------------------------|
| **P50** | Median error | Single point, not a range |
| **MAE** | Mean Absolute Error | Average error magnitude, not a range |
| **StdDev band** | Mean ± 1.96σ | Assumes normal distribution |
| **Prediction Interval** | Model-derived uncertainty | Requires probabilistic model; P10-P90 is empirical |

---

## Implementation in C.O.R.P.

In the C.O.R.P. v2 application, P10-P90 bands are computed **per plant** from the validation errors stored in `validation_hourly.csv`. The band is drawn as a translucent shaded region around the forecast line in the hourly prediction chart.
