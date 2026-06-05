# Cleaning Report

### Project Phase 1

The `Cleaning Process` is one of our approach major sections as it consists of many iterative sub-processes regarding early analysis of multiple features of the raw dataset. This is how the documentation and desicion-making for this process works in order to guarantee consistency and full data integrity:

!\[UML_Process_subsection_cleaning]\(../Project_Design/Cleaning/cleaning_process.png)

---
#### TL;DR

1. Plant `710` had missing values for about 7 months (May/2023 - Feb/2024), **we will apply MultiOutput Donor Imputation from similar plants (`515`)**
2. Bad formatted values and duplicates, as well as impossible conbinations (e.g. same `truck_code` delivering at the same time for different plants)
3. Less than a `0.2%` of the values where missing, so they were dropped.
4. All plants missing data during just two winter periods, could be attributed to vacations or, considering non-consistency throughout the years, it could be something related to extreme weather conditions.
5. All plants were missing without apparent reason data from `April 2024` to `May 2024`, **so we will apply MultiOutput Imputation**
6. Some `u_Cicle` with value of one, actually had `typed_time` - `at_plant_time` = 1, so they had to be deleted
---
### Inconsistencies, duplicates and missing values

|  *What was done* |  *Description* |
|---|---|
|  "Dropped `ship_addr_, map_page, name` rows with null values" |  Dropped `ship_addr_, map_page, name` with null values due to there null values represented less than a 0.20% of the remmisions combined" |
|  "Dropped duplicates where `start_time, at_plant_time and truck_code` where the same value" |  Dropped this specific coincidences due to the possibility of having identical start_time and at_plant_times , but the impossibility of having the same tru_idg |
|                                               ''Data types casting for all calendar-based variables'                                              |                                                                                 Casting from string to datetime                                                                                |

---

#### Cleaning Log:

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
|         Issue found        |                         The data-type was `string` but it needs to be changed to `datetime`. Also date was being ignored for the Q2 of 2025 as it had a dummy value                         |
|              Action taken              |                                                           Changed the dates data-types and transformed the date format of this features in Q2 2025 to have the correct day, month and year                                                          |

#### Inconsistencies

| *Field* | *Description* |
|---|---|
|     Columns     |    `ship_plant_code`   |
|         Issue found        |                         Plant `717` had no remissions registered since 2022                         |
|              Action taken              |                                                           Dropped all rows that had that plant as `ship_plant_code`                                                          |
                    
| *Field* | *Description* |
|---|---|
|     Columns     |    All   |
|         Issue found        |                         Some rows had a 1 minute `u_Cicle` with the difference between `at_plant_time` and `typed_time` confirming an apparent 1 minute time of service.                          |
|              Action taken              |                                                           Droppes rows (83) that had the same truck deliver time as u_Cicle when u_Cicle == 1                                                          |
                                                  
| *Field* | *Description* |
|---|---|
|     Columns     |    `order_code`   |
|         Issue found        |                          This column had just a 794 different values but dates ranging from 2020-2026, no correlation between same order_codes gives no information at all                          |
|              Action taken              |                                                           Dropped column due to unusefulness                                                   |
                       
#### Data Imputation

1.
| *Field* | *Description* |
|---|---|
|     Columns     |    `All`   |
|         Issue found        |                         Plant `710` as mentioned earlier                         |
|              _Imputation Technique_              |                                                           Donor Imputation, Auto-regressive Imputation                                                          |
|                         Other Teqniques used                         |                                                                   MultiOutputRegressor (To predict remissions and volume for the missing data filling), RandomForestRegressor (To capture complex non-linear patterns for imputation reliability)                                                                  |
|                         _Issue Description_                         |                                                           For this plant only, there were about 7 whole months (crossing 2023 and 2024) without any data whatsoever.                                                          |

2. 
| *Field* | *Description* |
|---|---|
|     Columns     |    `All`   |
|         Issue found        |                         All plants were missing data for a period of 5 months (January to May)                          |
|              _Imputation Technique_              |                                                           Auto-regressive Imputation                                                           |
|                         Other Teqniques used                         |                                                                   MultiOutputRegressor (To predict remissions and volume for the missing data filling), RandomForestRegressor (To capture complex non-linear patterns for imputation reliability)                                                                  |
|                         _Description_                         |                                                           For this plant only, there were about 7 whole months (crossing 2023 and 2024) without any data whatsoever.                                                          |

