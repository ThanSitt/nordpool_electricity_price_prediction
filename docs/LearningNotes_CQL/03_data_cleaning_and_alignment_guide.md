# Data Cleaning and Alignment Guide

This file is the most important one for your project.

## 1. Main rule

Before modeling, all data must speak the same time language.

That means:

1. Same time zone (时区).
2. Same time resolution (时间分辨率).
3. Same column names.
4. Same missing value rules.

If this part is wrong, the model will also be wrong.

## 2. Your current situation

You have different data types:

1. Electricity price data.
2. Temperature data from Helsinki-Vantaa airport.
3. Wind speed and wind direction data from Oulu.

Your files do not match yet.

One file may be hourly.
One file may be 10-minute data.
One file may use a different column name.

This is normal.

## 3. Best first decision: choose one common resolution

For your first project version, do this:

1. Keep the target price resolution as it is in the price file.
2. Convert weather data to the same resolution.

If the price file is hourly, make the weather hourly.

If later you move to 15-minute market data, then make all sources 15-minute.

For now, hourly is easier and safer.

## 4. Data cleaning order

Clean in this order:

1. Price data.
2. Temperature data.
3. Wind data.
4. Merge them.
5. Make final feature table.

Do not merge dirty files.

## 5. Clean each file alone

### Step 1. Read the file

Load the CSV file.

Check:

1. Column names.
2. Time column.
3. Value column.
4. Missing values.

### Step 2. Rename columns

Use one clean naming style.

Examples:

1. timestamp
2. price
3. temperature
4. wind_speed
5. wind_direction

Use the same names in all files where possible.

### Step 3. Convert time column

Convert the time column to datetime (日期时间).

Then:

1. Sort by time.
2. Remove duplicates.
3. Check for gaps.

### Step 4. Check time zone

This is very important.

Ask:

1. Is the data in local Finnish time?
2. Is it in UTC?
3. Does it contain summer time / winter time changes?

All sources should finally use the same time zone.

## 6. How to handle 10-minute weather data

Your wind and temperature data are 10-minute data.

That is fine.

You do not need to keep it 10-minute for the first model.

If you want hourly data:

1. Convert 10-minute rows into hourly rows.
2. Use mean for temperature.
3. Use mean for wind speed.
4. Use special handling for wind direction.

## 7. How to handle wind direction

Wind direction (风向) is circular.

This means:

1. 0 degrees and 360 degrees are almost the same.
2. 10 degrees and 350 degrees are close in reality.
3. Plain averaging of degrees can give wrong results.

So do not simply average raw direction numbers without thinking.

Better methods:

### Method A. sin and cos encoding

Convert direction into two columns:

1. wind_dir_sin
2. wind_dir_cos

This is simple and useful.

### Method B. u and v vector components

Use wind speed plus direction to create vector features.

This can also work well.

For your beginner project, sin and cos is usually easier to understand.

## 8. How to convert 10-minute data to hourly data

Use resampling (重采样).

For normal numeric columns:

1. Temperature -> hourly mean.
2. Wind speed -> hourly mean.

For wind direction:

1. First convert it to sin and cos.
2. Then resample those values.
3. After that, keep the hourly result.

This is safer than averaging direction degrees directly.

## 9. What about 15-minute data

You asked about 15-minute resolution.

Important idea:

1. Do not create fake price values too early.
2. Do not force the whole project into 15-minute data before the basic pipeline works.

For your first model, it is better to match the price file you already have.

Later, if you get true 15-minute price data, then you can rebuild the whole pipeline.

## 10. Merging rule

After cleaning and resampling, merge by time.

Final merged table should look like this:

1. timestamp
2. price
3. temperature
4. wind_speed
5. wind_dir_sin
6. wind_dir_cos
7. time features

Use the timestamp as the key.

## 11. Missing value handling

Missing data will happen.

Use this logic:

1. Small gap: fill carefully.
2. Medium gap: inspect manually.
3. Large gap: maybe drop that period or mark it.

Good choices:

1. Forward fill (前向填充) only when it makes sense.
2. Interpolation (插值) for smooth weather values.
3. Do not invent values without understanding the source.

## 12. Data validation checklist

Before moving to training, check this list:

1. All timestamps are sorted.
2. All timestamps use the same time zone.
3. All files use the same resolution.
4. No duplicated timestamps.
5. No weird negative values where they do not belong.
6. Wind direction is encoded correctly.
7. Merged table has the expected number of rows.

## 13. Visual checks you should always do

Make these plots:

1. Original 10-minute weather data.
2. Resampled hourly weather data.
3. Price over time.
4. Temperature over time.
5. Wind speed over time.
6. Price against temperature.
7. Price against wind speed.

These plots help you see mistakes faster than reading numbers.

## 14. Suggested final cleaned files

I suggest saving cleaned data like this:

1. data/processed/prices_hourly.csv
2. data/processed/temperature_hourly.csv
3. data/processed/wind_hourly.csv
4. data/processed/final_master_table.csv

This makes your project organized and repeatable.

## 15. Step-by-step cleaning order for your current files

### A. Electricity price file

1. Read the file.
2. Find the timestamp column.
3. Convert timestamp to datetime.
4. Set it as index.
5. Check if it is hourly.
6. Remove duplicates.
7. Check missing hours.

### B. Temperature file

1. Read each temperature CSV.
2. Convert timestamp.
3. Merge all temperature chunks if needed.
4. Sort by time.
5. Resample to hourly.
6. Check values for outliers.

### C. Wind file

1. Read each wind CSV.
2. Convert timestamp.
3. Clean wind speed.
4. Clean wind direction.
5. Convert direction to sin and cos.
6. Resample to hourly.
7. Check if any gaps exist.

### D. Final merge

1. Use timestamp as the join key.
2. Join price, temperature, and wind.
3. Remove rows with too many missing fields.
4. Save the final dataset.

## 16. What you should understand before next step

You should be able to explain:

1. Why wind direction cannot be treated like a normal number.
2. Why all data must share one time resolution.
3. Why the first model should be simple.
4. Why cleaning comes before training.
5. Why plots are part of data cleaning.

If you understand these five points, you are ready to start the real cleaning work.
