# Cleaning Report
---
### Inconsistencies, duplicates and missing values

|  *What was done* |  *Description* |
|---|---|
|  "Dropped `ship_addr_, map_page, name` rows with null values" |  Dropped `ship_addr_, map_page, name` with null values due to there null values represented less than a 0.20% of the remmisions combined" |
|  "Dropped duplicates where `start_time, at_plant_time and truck_code` where the same value" |  Dropped this specific coincidences due to the possibility of having identical start_time and at_plant_times , but the impossibility of having the same tru_idg |
|                                               ''Data types casting for all calendar-based variables'                                              |                                                                                 Casting from string to datetime                                                                                |

### Cleaning Log
---
#### Duplicates and Null Values
|  *Field* |  *Description* |
|---|---|
|  Columns |  `ship_addr,map_pae, name, start_time, at_plant_time, truck_code` |
|     Issue found     |                                                            Some rows had duplicate values and some others had null values                                                              |
|           Action taken          |                                                                   Dropped those rows, as the % of error was extremely low                                                                   |

#### Formatting Errors

| *Field* | *Description* |
|---|---|
|     Columns     |    `typed_time`, `start_time`, `at_plant_time`   |
|         Issue found        |                         The data-type was `string` but it needs to be changed to `datetime`                         |
|              Action taken              |                                                           Changed the dates data-types                                                          |

#### Inconsistencies

| *Field* | *Description* |
|---|---|
|     Columns     |    `ship_plant_code`   |
|         Issue found        |                         Plant `717` had no remissions registered since 2022                         |
|              Action taken              |                                                           Dropped all rows that had that plant as `ship_plant_code`                                                          |
                                                                                                                                                                  

