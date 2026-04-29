# Cleaning Report
---
### Inconsistencies, duplicates and missing values

|  *What was done* |  *Description* |
|---|---|
|  "Dropped  `ship_addr_, map_page, name` rows with null values" |  Dropped  `ship_addr_, map_page, name` with null values due to there null values represented less than a 0.20% of the remmisions combined" |
|  "Dropped duplicates where `start_time, at_plant_time and truck_code` where the same value" |  Dropped this specific coincidences due to the possibility of having identical start_time and at_plant_times , but the impossibility of having the same tru_idg |

### Cleaning Log
---
|  *Field* |  *Description* |
|---|---|
|  Columns |   |
|     Issue found     |                                                            Some rows had duplicate                                                              |
|           Action taken          |                                                                                                                                      |