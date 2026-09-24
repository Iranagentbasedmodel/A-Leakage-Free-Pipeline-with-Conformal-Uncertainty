# Data Sources & Feature Descriptions

> Pipeline: Physics-Informed Energy-Conserving Pipeline v8.1
> Purpose: Document each dataset used in the pipeline, its features, direct download link, and file size.

---

## 1. RECS 2020 (Residential Energy Consumption Survey)

### 1.1 Description
The 2020 RECS is a national sample survey conducted by the U.S. Energy Information Administration (EIA). It collects energy-related data for housing units occupied as primary residences and the households that live in them. The survey represents approximately 123.5 million housing units in the United States. The public-use microdata file contains household characteristics, energy consumption, expenditures, and end-use estimates[reference:0].

### 1.2 Features
The dataset contains 799 columns. Key variable groups include:

| Variable Group | Example Variables | Description |
|---|---|---|
| Identification | `DOEID` | Unique housing unit identifier |
| Geography | `BA_climate`, `STATE_FIPS` | Climate zone and state |
| Housing Characteristics | `TYPEHUQ`, `YEARBUILT`, `TOTROOMS` | Building type, vintage, size |
| Energy Consumption | `TOTALBTU`, `TOTALBTUSPH`, `BTUELCOL`, `TOTALBTUWTH`, `BTUELLGT` | Total and end-use energy consumption (heating, cooling, DHW, lighting) |
| Energy Expenditures | `TOTALDOL`, `DOLLAREL`, `DOLLARGAS` | Energy cost variables |
| Appliances & Equipment | `FUELHEAT`, `FUELH2O`, `TEMPHOME` | Heating fuel, water heating fuel, thermostat settings |
| Survey Weights | `NWEIGHT` | Population weighting variable |
| Climate | `HDD65`, `CDD65` | Heating and cooling degree days |

### 1.3 Download Link
- **CSV (v7):** [https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv](https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv) [reference:1]
- **SAS (v7):** [https://www.eia.gov/consumption/residential/data/2020/sas/recs2020_public_v7.zip](https://www.eia.gov/consumption/residential/data/2020/sas/recs2020_public_v7.zip) [reference:2]
- **Codebook (XLSX):** [https://www.eia.gov/consumption/residential/data/2020/xls/RECS%202020%20Codebook%20for%20Public%20File%20-%20v7.xlsx](https://www.eia.gov/consumption/residential/data/2020/xls/RECS%202020%20Codebook%20for%20Public%20File%20-%20v7.xlsx) [reference:3]

### 1.4 File Size
- **CSV:** 18,496 rows × 799 columns → approximately 45–55 MB (uncompressed)
- **SAS ZIP:** approximately 20–30 MB (compressed)

---

## 2. ResStock 2024.2 (Upgrade Files)

### 2.1 Description
ResStock is a highly granular, bottom-up building stock model developed by the National Renewable Energy Laboratory (NREL). It uses multiple data sources, statistical sampling, and advanced building energy simulations to estimate subhourly energy consumption of the U.S. housing stock[reference:4]. The 2024.2 release includes 16 measure packages (upgrades) across two weather years[reference:5].

The pipeline uses the following upgrade files:

| Upgrade ID | Measure Package | Description |
|---|---|---|
| `upgrade0` | Baseline | No upgrade (reference building stock) |
| `upgrade3` | Reference HVAC 2025 | Standard HVAC system update |
| `upgrade4` | Cold Climate ASHP | Cold-climate air-source heat pump |
| `upgrade10` | Air Sealing | Envelope air sealing measures |
| `upgrade11` | Attic Insulation | Attic insulation upgrade |
| `upgrade15` | ENERGY STAR Windows | High-performance window replacement |

### 2.2 Features
Each ResStock upgrade file contains 40 mapped columns, including:

| Feature Group | Example Columns | Description |
|---|---|---|
| Building ID | `bldg_id` | Unique building identifier |
| Climate | `ba_climate` | Building America climate zone |
| Building Characteristics | `typehuq`, `fuelheat`, `infiltration`, `wh_eff`, `wh_fuel` | Housing type, heating fuel, envelope, water heater |
| Energy Consumption | Annual and timeseries energy use | Electricity, natural gas, propane, fuel oil consumption |
| Upgrade Delta | Savings percentage | Energy savings relative to baseline |
| Geography | `county` | County-level location |

### 2.3 Download Link
- **OEDI Data Lake (2024.2 Release):** [https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_amy2018_release_2/](https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_amy2018_release_2/)
- **Upgrade lookup table:** `upgrades_lookup.json` in the same directory[reference:6]

### 2.4 File Size
- **Per upgrade file (CSV.gz):** 300–550 MB compressed (550,000 rows × 40 columns)
- **Total for 6 upgrades:** approximately 2.5–3.5 GB compressed
- **Full 2024.2 dataset:** 6.24 GB (as reported for the complete ResStock repository)[reference:7]

---

## 3. TMY3 Weather Data

### 3.1 Description
Typical Meteorological Year (TMY3) data provides representative hourly weather data for building energy simulations. The dataset covers 1,020 weather stations across the United States and includes 8,935,200 hourly records. TMY3 files are in EPW format and are used to derive climate features such as HDD (Heating Degree Days) and CDD (Cooling Degree Days)[reference:8].

### 3.2 Features
The TMY3 data provides 19 climate features, including:

| Feature Group | Example Variables | Description |
|---|---|---|
| Temperature | Dry bulb temperature, dew point | Hourly temperature measurements |
| Solar Radiation | Global horizontal, direct normal, diffuse horizontal | Solar irradiance components |
| Wind | Wind speed, wind direction | Wind characteristics |
| Humidity | Relative humidity | Moisture content |
| Derived Features | `TMY3_HDD65`, `TMY3_CDD65` | Heating and cooling degree days (base 65°F) |

### 3.3 Download Link
- **TMY3 Weather Data for ComStock and ResStock (BuildStock_TMY3_FIPS.zip):** [https://data.nrel.gov/system/files/156/Buildstock_TMY3_FIPS-1678817889.zip](https://data.nrel.gov/system/files/156/Buildstock_TMY3_FIPS-1678817889.zip) [reference:9]
- **Dataset landing page:** [https://data.nrel.gov/submissions/156](https://data.nrel.gov/submissions/156) [reference:10]

### 3.4 File Size
- **BuildStock_TMY3_FIPS.zip:** 760.6 MB[reference:11]

---

## 4. Summary Table

| Dataset | Source | Direct Download Link | File Size |
|---|---|---|---|
| RECS 2020 (CSV) | EIA | [recs2020_public_v7.csv](https://www.eia.gov/consumption/residential/data/2020/csv/recs2020_public_v7.csv) | ~50 MB |
| RECS 2020 (SAS) | EIA | [recs2020_public_v7.zip](https://www.eia.gov/consumption/residential/data/2020/sas/recs2020_public_v7.zip) | ~25 MB |
| ResStock 2024.2 (6 upgrades) | NREL / OEDI | [OEDI Data Lake](https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_amy2018_release_2/) | ~2.5–3.5 GB (compressed) |
| TMY3 Weather Data | NREL | [BuildStock_TMY3_FIPS.zip](https://data.nrel.gov/system/files/156/Buildstock_TMY3_FIPS-1678817889.zip) | 760.6 MB |

---

## 5. Data Lineage

```text
+---------------------------------------------------------------------+
| RECS 2020 (EIA)                                                     |
| 18,496 rows | 799 columns | Survey microdata                        |
+---------------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------+
| TMY3 Weather Data (NREL)                                            |
| 1,020 stations | 8,935,200 hourly records | 19 climate features    |
+---------------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------+
| ResStock 2024.2 (NREL/OEDI)                                         |
| 6 upgrade files | ~550K rows each | 40 mapped columns               |
+---------------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------+
| Physics-Informed Pipeline v8.1                                      |
| Feature engineering | Transfer learning | Multi-task modeling       |
+---------------------------------------------------------------------+
