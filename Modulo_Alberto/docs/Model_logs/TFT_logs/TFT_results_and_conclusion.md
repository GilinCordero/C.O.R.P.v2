# TFT Results and Conclusion

## Temporal Fusion Transformer Experiment

During the experimentation phase, a **Temporal Fusion Transformer (TFT)** model was trained and evaluated as a candidate for hourly demand forecasting across GCC plants. This document records the results and the rationale for its eventual discard in favor of LightGBM.

---

## Results: Mean Absolute Error (MAE) per Plant

The following table summarizes the hourly MAE achieved by the TFT model on the validation set:

| Plant | TFT MAE (m³/hr) |
|-------|-----------------|
| 510   | 5.83            |
| 511   | 4.82            |
| 512   | 5.44            |
| 514   | 4.50            |
| 515   | 5.25            |
| 710   | 3.96            |
| **Average** | **~4.97**  |

---

## Comparison with Current Model

By comparison, the current **LightGBM native hourly model** achieves:

| Metric | LightGBM | TFT | Difference |
|--------|----------|-----|------------|
| **Global MAE** | **1.64 m³/hr** | ~4.97 m³/hr | **3× lower** |
| **RMSE** | 3.82 m³/hr | — | Significantly lower |
| **Training time** | ~3–5 min (CPU) | Hours (GPU required) | Orders of magnitude faster |
| **Inference time** | Milliseconds per prediction | Seconds per batch | Real-time capable |

---

## Reasons for Discarding TFT

### 1. Inferior Accuracy
Despite being a state-of-the-art deep learning architecture for time-series, TFT underperformed LightGBM significantly on this dataset. The tree-based model better captured the **sparse, irregular hourly demand patterns** where approximately **70% of hours have zero volume**.

### 2. Architectural Complexity
TFT requires:
- 3D tensor inputs `(samples, timesteps, features)`
- Categorical embedding configurations
- Attention mechanism tuning
- Specialized libraries (`pytorch-forecasting`)

This introduced unnecessary complexity for a problem that is fundamentally **tabular with temporal structure**.

### 3. Hardware and Training Constraints
- TFT training **required GPU acceleration** and took **hours per experiment**.
- LightGBM trains to convergence in **under 5 minutes on CPU**, enabling rapid iteration and A/B testing.

### 4. Operational Fragility
The TFT struggled with:
- High frequency of **zero-demand hours**
- **Varying operational schedules** across plants
- Need for strict regular time-indexing (gaps in data caused issues)

LightGBM handled these sparsity patterns natively through its **leaf-wise splits** and **zero-tolerant objective**.

### 5. Production Maintenance Burden
Retraining TFT in production would demand:
- Persistent **GPU infrastructure**
- Complex dependency management (**PyTorch + CUDA**)
- Lengthy deployment cycles

LightGBM runs on a **standard Python environment** with **no GPU dependencies**.

### 6. Interpretability Gap
While TFT provides attention weights, explaining *why* it predicted a specific hourly volume to a business stakeholder proved far more difficult than reading **LightGBM feature importance** or **SHAP values**.

---

## Conclusion

The Temporal Fusion Transformer was a **theoretically sound candidate** but was **discarded in favor of LightGBM** due to:
- **Substantially lower accuracy** (~3× worse MAE)
- **Excessive operational overhead** (GPU, complex dependencies)
- **Poor fit for sparse hourly demand patterns** observed in the GCC dataset

The experiment validated that for this specific dataset — characterized by sparsity, multiple plants, and rich engineered features — a **gradient-boosted tree approach outperforms deep learning** in both accuracy and operational practicality.

---

## Related Documents

- [Main Model Justification](../model_justification.md) — LightGBM justification and feature analysis
- [Week 8 Experiment Results](../week8_experiment_results.md) — Additional experiment logs
