# Machine Learning Summer Project: Nord Pool Price Prediction

Project Information
• Name of the project: Machine Learning Summer Project: Nord Pool Price Prediction
• Goals of the project: To build, compare, and automate predictive models (XGBoost and LightGBM) for day-ahead electricity prices in Finland over a 7-day horizon.
• Duration: 2026 05-08
• Coordinator: Lasse Haverinen
• Other implementors: Chenqi Li, Sitt Min
• Degree program: DIN23SP
Report's Main Field of Education
☑ Information Technology
Summary
This report details a summer project that uses machine learning to forecast Nord Pool electricity prices in Finland over a 7-day period. The study compares two gradient-boosting algorithms, XGBoost and LightGBM, across six different dataset versions. These datasets range from a basic two-feature weather model to a complex 68 features set that includes cross-border grid flows and nuclear power output.
A major finding of the project is that careful feature engineering is much more important than simply adding raw data variables. By focusing on how the data is shaped, the best model—a well-tuned LightGBM using supply-side features—achieved a mean absolute error of 2.64 EUR/MWh. This result is ten times better than the initial weather-only model. Finally, the system was successfully deployed using GitHub Actions. It now operates as a fully automated daily MLOps pipeline that makes live predictions and continually tests its own accuracy against real market prices.
Keywords
artificial intelligence, data engineering, feature engineering, LightGBM, machine learning, MLOps, Nord Pool, predictive modeling, time-series forecasting, XGBoost

Lead
The electricity prices of Finland exhibit significant volatility compared to other European markets, with values fluctuating substantially in response to weather conditions, cross-border grid dynamics, and generation availability. Prices can range from near zero to hundreds of euros per megawatt-hour within a single day. This project was initiated to investigate the feasibility of utilizing machine learning algorithms to forecast such unpredictable dynamics, serving both as a practical application of these techniques and as a rigorous technical challenge. Through a collaborative effort evaluating XGBoost and LightGBM, the findings indicate that predictive performance is primarily driven by the quality of feature engineering rather than the specific choice of algorithm. The most effective model developed during this period achieved a mean absolute error of 2.64 euros per megawatt-hour and is currently deployed in a daily automated pipeline, with its robustness to be further evaluated during the winter season. Furthermore, the Nord Pool electricity market constitutes a highly complex forecasting environment due to these same fluctuating factors. This study aimed to determine if contemporary machine learning methodologies could generate reliable seven-day price forecasts for the Finnish market, while simultaneously providing comprehensive experience across the entire data science pipeline, from data acquisition to deployment. A comparative analysis of XGBoost and LightGBM across six dataset iterations confirmed that deliberate feature engineering is the primary contributor to model accuracy. The optimal model, a fine-tuned LightGBM that integrates features related to weather, grid status, and nuclear supply, achieved a ten-fold reduction in error compared to a baseline model utilizing only weather data.
1 Introduction
The Nord Pool day-ahead electricity market is characterized by significant volatility, driven by a complex interplay of weather conditions, fluctuating demand, and dynamic grid operations. Accurate price prediction is essential for stakeholders across the energy sector, enabling effective trading strategies, optimal consumption planning, and enhanced grid stability. However, the inherent noise and non-linear dependencies within market data pose substantial challenges to traditional forecasting methods.
This summer project addresses these challenges by developing a machine learning-based forecasting system for Finnish (FI) spot prices. Working collaboratively to benchmark XGBoost and LightGBM algorithms, our objective was to achieve reliable 7-day predictions while minimizing the Mean Absolute Error (MAE). Through an iterative process of data exploration, feature engineering, and hyperparameter tuning, we evolved our models from simple weather-based baselines to a fully automated MLOps pipeline. This report details our methodology, key findings regarding model evaluation, and the practical challenges overcome in deploying a live forecasting tool.
2 Project Timeline
Our project progressed through distinct phases, characterized by an iterative refinement of our dataset and modeling approach.
Table 1.
Project Timeline and Key Milestones
Phase Activity Outcome
Early summer Set up data pipeline; trained first weather-only models MAE ~33 EUR/MWh — baseline established
Mid-summer Added price lag and time features; rebuilt dataset MAE dropped to 2.82 EUR/MWh — major breakthrough
Late summer Automatic model tuning; added grid and nuclear data MAE improved to 2.64 EUR/MWh (LightGBM)
End of summer Deployed automated daily forecasting pipeline 10 models running live, self-evaluating each day
Ongoing Collecting live predictions; winter not yet tested Results pending for cold-weather months

3. Dataset Evolution and Feature Engineering
   3.1 Data Acquisition and the Initial Baseline (V1 to V1.5)
   The project commenced with a rudimentary dataset comprising two primary weather variables: hourly temperature readings from Helsinki-Vantaa Airport and wind speed measurements from Oulu. These specific features were selected to represent the fundamental weather conditions most directly impacting electricity demand and renewable energy generation across Finland. However, an initial XGBoost model (V1) trained exclusively on this raw weather data achieved a surprisingly low R^2 coefficient of 0.107 and a Mean Absolute Error (MAE) of 33.13 EUR/MWh. In practical terms, this indicated that the model could only explain roughly 10% of the variance in prices, missing the actual price by an average of 33 euros—a margin nearly equivalent to the cost of a megawatt-hour on a typical day. It became starkly evident that weather conditions alone capture almost none of the electricity market's short-term pricing dynamics.
   Subsequent attempts to refine the dataset simply by increasing the temporal resolution from hourly to 15-minute intervals (V1.5) yielded negligible statistical improvements. The R^2 coefficient increased merely from 0.107 to 0.125. This empirical outcome underscored a fundamental paradigm in predictive modeling: artificially expanding the volume of weak predictors does not inherently enhance predictive capability.
   3.2 Exploratory Data Analysis (EDA) and Visualization
   To address this severe predictive limitation, a comprehensive Exploratory Data Analysis (EDA) was conducted. By employing robust data visualization techniques, several critical market behaviors were elucidated. For instance, box plots delineating price distributions by the hour of the day starkly highlighted the impact of morning and evening peak hours, while bar charts comparing weekdays to weekends emphasized predictable, demand-driven price drops. Furthermore, scatter plots mapping temperature against price visually confirmed the non-linear relationship between extreme winter cold and sudden price surges.

Figure 1:boxplot for Price by Hour of Day

Figure 2: Bar Charts for Average Price by weekday

Figure 3:scatter plots for temperature vs price

Most crucially, the generation of correlation heatmaps revealed the strong autoregressive characteristics of electricity prices. The visual evidence clearly indicated that historical price fluctuations and temporal rhythms—specifically the price from exactly 24 hours or seven days prior—act as the strongest determinants of future market behavior, necessitating a complete shift in the data processing strategy.

Figure 4:Correlation Heatmap
3.3 Targeted Feature Engineering (V2.5 Baseline)
Guided directly by the insights derived from the EDA, the methodology pivoted toward sophisticated, targeted feature engineering. Acknowledging that electricity prices possess an inherent memory, 49 distinct engineered features were introduced to construct the V2.5 baseline. The most impactful additions were historical price lags (capturing prices from 15 minutes, one hour, 24 hours, and seven days prior) and rolling statistical averages over various time windows to capture recent market volatility.
Furthermore, to address the cyclic nature of time identified in the visual analysis, hours and months were encoded using sine and cosine transformations, allowing the model to recognize temporal circularity (e.g., treating 11 PM and midnight as temporally proximate). Categorical variables such as Finnish public holiday indicators and peak-hour flags were also integrated. Training these 49 engineered features on 15-minute resolution data across roughly 105,000 rows produced an exceptional $R^2$ of 0.972 and plummeted the MAE to 2.82 EUR/MWh. This ten-fold improvement over the weather baseline definitively proved that models require structured access to historical market memory to forecast accurately.
3.4 The High-Volatility Feature Experiment (V2.5.1)
Although the V2.5 baseline was formidable, the inherent complexities of feature expansion were highlighted during a controlled experiment (V2.5.1). To explore the model's predictive limits, a 'high-volatility probability' feature was conceptualized. The hypothesis posited that sudden supply and demand changes, typically driven by weather, precipitate sharp price swings. To quantify this, a secondary XGBClassifier was trained using solely weather and temporal features to output a probability between 0 and 1, indicating the likelihood of extreme volatility (defined as the top 15% of a 6-hour rolling price standard deviation) in the upcoming 15-minute interval.
It is imperative to note that this specific engineered feature was tested exclusively on the XGBoost architecture. The subsequent evaluation revealed that integrating this probability actively degraded predictive accuracy, increasing the MAE by approximately 1%. Detailed analysis attributed this regression to the secondary classifier's weak performance—specifically, a recall score of merely 0.24 for high-volatility moments. Consequently, it injected a weak signal combined with substantial noise into an already optimal feature space. The high-volatility feature was completely discarded. This experimental failure reinforced a vital lesson: incorporating extraneous features to artificially inflate variance explanation frequently diminishes model robustness, and rigorous hyperparameter tuning must always precede assumptions of data inadequacy.
3.5 Final Dataset Integration: Supply-Side Dynamics (V3 and V3.1)
Following the resolution to prioritize model tuning over noisy feature generation, the dataset underwent its final evolutions (V3 and V3.1) through the systematic integration of supply-side metrics. Version 3 incorporated real-time measurements of cross-border electricity flows between Finland and neighboring grids (Estonia, Norway, and Sweden), expanding the model's perspective to encompass wider Nordic market dynamics. Version 3.1 finalized the dataset by assimilating Finnish nuclear power output, a crucial metric given that unexpected nuclear maintenance directly reduces base-load supply and drives prices upward.

Table 2.
Model Performance Comparison Across All Dataset Versions
Model Features Test MAE RMSE R² Description
V1
(XGBoost) Weather only (temp, wind) 33.13 46.34 0.107 Hourly weather-only baseline
V1.5
(XGBoost) Weather only (temp, wind, direction) 32.19 45.78 0.125 15-min weather-only baseline
V2
(XGBoost) Full engineered (lags, rolling, calendar, holiday) 7.22 14.62 0.911 Hourly + feature engineering
V2
(LightGBM) Full engineered (41) 7.10 14.65 0.911 Hourly + feature engineering
V2.5
(LightGBM) Full engineered (49) + Optuna tuning 2.64 7.92 0.974 15-min + engineered + tuned LightGBM
V2.5
(XGBoost) Full engineered (49) 2.82 8.22 0.972 15-min + engineered (default XGBoost)
V2.5.3
(XGBoost) Full engineered (49), Optuna-tuned 2.7236 8.1642 0.9722 Best production XGBoost
V3 (XGBoost) + grid flows (62), default params 2.847 8.368 0.971 Grid hurt at default
V3.1 (XGBoost) + grid flows (62), Optuna-tuned 2.6982 7.9724 0.9735 Grid helps under tuned model — live
V4 (XGBoost) + grid + nuclear (68), Optuna-tuned 2.7020 8.0376 0.9730 Best XGBoost — live
V3.1
(LightGBM) + grid + nuclear (68), V2.5 params 2.6390 7.8957 0.9740 Best model overall — live

Figure 2. Top 30 feature importances for LightGBM V3.1. Red bars indicate the new grid and nuclear features.

4. Model Benchmarking and Iterative Evaluation
   This section details the evolutionary trajectory of the predictive system, transitioning from rudimentary baselines to highly optimized gradient boosting ensembles. Rather than arbitrarily selecting algorithms, this project adopted a systematic, hypothesis-driven approach to model iteration (Versions 1 through V3.1), rigorously evaluating the individual impacts of temporal resolution, feature engineering, and hyperparameter tuning.
   4.1 Baseline Establishment and the Role of Temporal Resolution (V1 & V1.5)
   The initial phase of modeling aimed to establish a foundational performance baseline utilizing only meteorological data, specifically temperature, wind speed, and wind direction. Version 1 (V1) operated at an hourly resolution and yielded a suboptimal R-squared ($R^2$) value of $0.107$, indicating that weather features alone were insufficient to capture the variance in electricity prices.
   To determine whether data granularity was the limiting factor, Version 1.5 (V1.5) was developed by increasing the temporal resolution to 15-minute intervals. However, this adjustment resulted in only a marginal gain, achieving an $R^2$ of $0.125$. This empirical finding confirmed a critical insight from the exploratory data analysis (EDA): market prices are not solely dictated by concurrent weather conditions. Consequently, increasing data frequency without enriching the underlying information space provides negligible predictive benefits.
   4.2 Feature Engineering Breakthrough: Exploiting Autoregression (V2 & V2.5)
   Recognizing the limitations of the meteorological baseline, the focus shifted toward feature engineering in Version 2 (V2). EDA of the Nord Pool market revealed that electricity prices possess a strongly autoregressive nature; historical prices are highly indicative of future trends due to cyclical daily and weekly demand patterns.
   To exploit this, we introduced complex engineered features, including cyclic time encodings utilizing sine and cosine transformations, categorical calendar flags to identify weekends and holidays, and crucial price lag and rolling statistics. This paradigm shift resulted in a massive performance breakthrough, elevating the $R^2$ to $0.911$. By unifying these engineered features with the finer 15-minute granularity, Version 2.5 (V2.5) successfully captured intraday market dynamics, substantially reducing the Root Mean Squared Error (RMSE) from $14.62$ to $8.22$ EUR/MWh.
   4.3 Investigating Extreme Spikes: The High-Volatility Probability Experiment (V2.5.1)
   Despite the success of V2.5, predicting extreme price spikes remained inherently challenging. To address this, a controlled experiment (V2.5.1) was conducted using XGBoost to test a novel, custom-engineered risk feature: high_volatility_prob.
   • Origin of the Feature: A secondary, independent XGBClassifier was trained utilizing only forward-known variables, namely weather forecasts and calendar time, to output a probability between $0$ and $1$. The binary target label for this classifier was defined as $1$ if the 6-hour rolling standard deviation of the price exceeded the 85th percentile of historical data, and $0$ otherwise.
   • Rationale for the Experiment: The rationale was to provide the primary XGBoost regression model with an advance "storm warning." We hypothesized that supplying a localized probability of market turbulence would allow the regression trees to dynamically adjust their splitting behavior during anticipated supply-demand shocks.
   • Experimental Results: Contrary to our hypothesis, the integration of this feature slightly degraded overall model performance, increasing the Mean Absolute Error (MAE) by approximately $1\%$. Although histogram analysis showed some signal separation between stable and volatile periods, the secondary classifier suffered from severe class imbalance, achieving a recall of only $0.24$ for actual high-volatility events.
   • Final Action: Ultimately, the high_volatility_prob feature injected spurious noise into the dataset, interfering with the existing, highly effective autoregressive features. It was subsequently discarded from the production pipeline in favor of optimizing the model's internal hyperparameters.
   4.4 Algorithmic Optimization and Loss Function Shift (V2.5.2 & V2.5.3)
   Both XGBoost and LightGBM operate on the principles of gradient boosting, sequentially constructing decision trees to minimize residual errors. Initially, our out-of-the-box implementation of XGBoost produced misleadingly poor results when exposed to complex feature spaces. Because the default parameters provided too few trees, the model lacked the capacity required to extract meaningful signals from the expanded dataset.
   To conduct a scientifically rigorous and fair comparison (V2.5.2), both algorithms were subjected to automated hyperparameter tuning via the Optuna framework. Furthermore, we deliberately shifted our loss function from Mean Squared Error (MSE) to Mean Absolute Error (MAE). Because MAE is mathematically less sensitive to the extreme, transient price spikes characteristic of the Finnish Nord Pool market, it prevented the models from overly penalizing rare outliers at the expense of general accuracy. Under these optimized conditions, tuning proved to be a far more significant driver of accuracy than arbitrarily adding new features.
   4.5 Incorporating Supply-Side Dynamics (V3 & V3.1)
   The final iterations sought to determine if integrating real-time supply-side constraints could further refine the forecast. Version 3 introduced lagged cross-border grid transmission flows, while Version 3.1 incorporated nuclear power generation metrics. A comparative analysis confirmed that, provided the hyperparameters are carefully tuned to handle the expanded dimensionality, these physical supply-side signals effectively lower the overall error rate.
   4.6 Comprehensive Model Evaluation
   The ultimate evaluation consisted of a head-to-head comparison between the optimal configurations: LightGBM V3.1 and XGBoost V4, both utilizing the shared 70-feature dataset containing weather, grid, and nuclear data. To ensure scientific validity and prevent future data leakage, both models were evaluated on the final $20\%$ of the dataset using a strict chronological split.
   • Overall Metrics: LightGBM demonstrated a slight quantitative edge, achieving the highest predictive accuracy of the project with an MAE of $2.639$ EUR/MWh, an RMSE of $7.896$, and an $R^2$ of $0.974$, marginally outperforming XGBoost's MAE of $2.702$ EUR/MWh.
   • Visual and Residual Analysis: Time-series line charts confirmed that both models track daily cyclical patterns exceptionally well but consistently underestimate the amplitude of extreme peaks. This observation was corroborated by residual histograms; while the residuals for both models centered precisely around zero (mean $\approx 0.07$, indicating an absence of systematic over-prediction or under-prediction bias), they exhibited pronounced long tails. These tails confirm that rare, volatile market shocks constitute the primary source of remaining error.
   • Feature Importance: An analysis of feature importance revealed intriguing algorithmic differences in how the models learned. LightGBM relied heavily on the newly introduced supply-side features, prioritizing cross-border grid lags (fi_se_total_lag_96) and nuclear fluctuations (nuclear_change_1d). Conversely, XGBoost remained predominantly dependent on its autoregressive price history.
   • Recursive Forecasting: Finally, transitioning from offline single-step evaluation to live 7-day recursive forecasting illuminated a significant operational challenge. Because each predicted 15-minute interval must be fed back into the rolling buffer to serve as the "historical" input for the subsequent step, minor prediction errors inevitably compound recursively over the prolonged 168-hour forecast horizon, highlighting the natural discrepancy between offline test metrics and live operational accuracy.

4 Automating the Pipeline with MLOps
To transition our work from a static academic exercise to a functional prototype, we engineered an automated MLOps pipeline orchestrated by a GitHub Actions cron job that executes daily. The system fetches live data from public APIs — electricity prices from Elering, weather forecasts from FMI and Open-Meteo, and cross-border grid measurements from Fingrid — and passes it through a feature engineering module that mirrors the exact transformation steps used during model training. It then loads the saved model artifacts and generates updated seven-day recursive forecasts for all ten trained models, committing the results back to the shared repository automatically.
What makes this architecture meaningful is its capacity for self-evaluation. As actual market prices arrive each day, the pipeline matches them against prior predictions and calculates the error automatically — making the project a living system rather than a one-time exercise. In practice, live forecasting accuracy is lower than the offline test results, which is expected: the offline evaluation tests one prediction at a time against real data, while the live system must forecast an entire week recursively, where each step's output feeds into the next and small errors accumulate over the horizon. How much this gap narrows as the system collects data across different seasons — and whether enriched data sources can close it — remains the central open question for the project going into winter.

Figure 4. Seven-day forecast output from all ten live models (August 2026).
5 Summary and Conclusions
This summer project successfully developed a robust, automated system for predicting Nord Pool electricity prices. Our final benchmark revealed that an Optuna-tuned LightGBM model utilizing engineered weather, grid, and nuclear features (V3.1) provided the most accurate forecasts.
The project provided invaluable practical experience. We learned that data preparation — specifically EDA and thoughtful feature engineering — is paramount. We discovered the vital practice of creating backup datasets before manipulating data to preserve raw information. We realized that indiscriminately adding new features often introduces noise, and that rigorous hyperparameter tuning is essential before expanding the feature set. Finally, managing the live deployment illuminated the challenges of recursive forecasting and the practical gap between offline metrics and live production accuracy. Future work should focus on mitigating error accumulation over the 7-day horizon and evaluating performance during the highly volatile winter months.

References
Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2623–2631. https://doi.org/10.1145/3292500.3330701

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.
https://doi.org/10.1145/2939672.2939785
Elering. (2025). NPS price API. https://dashboard.elering.ee/api/nps/price

Fingrid. (2025). Fingrid open data.
https://data.fingrid.fi/en/

Finnish Meteorological Institute. (2025). FMI open data.
https://en.ilmatieteenlaitos.fi/open-data

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146–3154.
https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

Open-Meteo. (2025). Open-Meteo weather forecast API.
https://open-meteo.com/
