# Feature & Target Variable Document
---
### Target variable definition
- Name: `volume_m3` por hora 
- Type: `continuous` 
- Distribution: `

### Feature set definition (first_iteration)

| Feature Name | Original Column | Type | Encoding | Relevance | Included |
|--------------|-----------------|------|----------|-----------|----------|
| `time_idx`   | N/A            | `integer`| N/A    | N/A 
| `month` (sin and cos as separate features)  | `typed_time`   | `real (double)` | Cyclical Encoding | Critical 
| `day` (sin and cos as separate features)    | `date` | `integer` | `real (double)` | Cyclical Encoding | Critical 
| `day_of_week` (sin and cos as separate features) | `date` | `integer` | Extracted from date | Medium 

### Feature selection rationale





