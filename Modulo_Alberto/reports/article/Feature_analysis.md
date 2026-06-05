#### First Observations of our database:

* 356332 rows (remissions) in total
* 6 active plants
* Plant 514 active since 2022, the rest since 2020
* 15 default columns

#### Our features

##### - Current Features

| _variable name_ | _meaning/description_ | _early-importance-rating (0-5)_ | _importance-rating-justification_ | _addition-verdict-justification_ | _early-training-set-addition-verdict_ | repeatable? | unique_count |
|---|---|---|---|---|---|---|---|
| **tkt_code** | The id (number) assigned to each remission, meaning every time an appointment is created and completed successfully. | 0 | NO addition to training | It is just a number to identify each order; it doesn't mean anything to us. | NO addition | Yes (confirmed during EDA) | 356332 |
| **order_date** | The date when the order was placed. Contains only the date component without hour precision. | 3 | Medium importance for temporal grouping | Useful for daily-level aggregations and seasonality detection, but lacks hourly resolution. | YES — as reference only | Yes | 357 |
| **start_time** | The exact date and time when the truck started loading concrete. This is the core timestamp representing real demand. | 5 | Critical temporal variable | Represents the precise moment of demand. Essential for hourly aggregation, lags, rolling windows, and all cyclic encodings. | YES — core feature | Yes | 12721 |
| **truck_code** | Identifier code assigned to the concrete mixer truck that fulfilled the remission. | 1 | Low predictive value | Individual truck identity does not explain demand patterns. May carry minor operational noise. | NO | Yes | 77 |
| **ship_plant_code** | The originating concrete plant code (510, 511, 512, 514, 515, 710). | 4 | Essential for multi-plant grouping | Each plant has distinct demand patterns, schedules, and capacities. Required for plant-level time series modeling. | YES — categorical grouping key | Yes | 5 |
| **u_Volumen** | The volume of concrete delivered in cubic meters (m³) for this remission. | 5 | Target variable | This is the variable we aim to predict. All features serve to estimate this quantity at the hourly level. | YES — as target (y) only | Yes | 16 |
| **typed_time** | The date and time when the remission was typed/recorded into the system. Represents when the truck departs. | 3 | Operational temporal reference | Correlated with start_time but represents dispatch, not demand onset. Less relevant than start_time for forecasting. | NO for demand prediction | Yes | 49890 |
| **at_plant_time** | The date and time when the truck arrived at the plant to be loaded. | 2 | Operational lag indicator | Represents logistical lead time. Could explain minor operational delays but not core demand patterns. | NO for demand prediction | Yes | 49901 |
| **u_Cicle** | Cycle number associated with the remission order in the operational workflow. | 2 | Operational sequencing | May reflect batching or queue order. Weak direct relationship with hourly demand volume. | NO | Yes | 209 |
| **name** | The name of the client who placed the order. | 1 | Highly granular categorical | Too many unique values for direct modeling. Could be used for client segmentation but not as raw feature. | NO as raw feature | Yes | 780 |
| **Nombre del proyecto** | The name of the construction project associated with the order. | 1 | Project-level granularity | Similar to client name — too granular. Project types could be grouped, but raw name is not useful. | NO as raw feature | Yes | 4916 |
| **ship_addr_line** | The delivery address line for the concrete shipment. | 0 | Geographical noise | Address text has no predictive value for time series. Could be geocoded to distance, but not used as-is. | NO | Yes | 11372 |
| **map_page** | A reference page or code from a map system for delivery location. | 0 | Obsolete geographical reference | Not relevant for temporal demand forecasting. Potential legacy field. | NO | Yes | 375 |


