# AI_Based-Soil-Moisture-Forecasting

Code, result tables and figure for the paper "AI-Based Soil Moisture Forecasting for Smart Irrigation: A Comparative Study of Boosting, ConvLSTM, and Transformer Forecasting Configurations".

The study compares forecasting configurations for a 24-step-ahead soil-moisture target (48 input steps, 24 target steps, hourly data). It is a model-based forecasting experiment. It does not evaluate irrigation scheduling, water savings or a field deployment.

Read this before using the results. The data have properties that limit what the comparison shows. They are described under Data and limitations.

Repository contents
notebooks/
  01_arima_hybrid_diagnostics.ipynb   Diagnostic plots for the saved ARIMA+MLP hybrid predictions
  02_lightgbm_xgboost.ipynb           XGBoost / LightGBM predictions and diagnostics
  03_convlstm_transformer.ipynb       ConvLSTM and Transformer evaluation
  04_revision_analysis.ipynb          Revision analysis: horizon-wise error, threshold proxies,
                                      regime breakdowns, block bootstrap, GNN check,
                                      ConvLSTM-24 retraining
results/
  breakdown_MAE.csv                          MAE by regime / node / season (Tables IV, IVb)
  block_bootstrap_CI.csv                     95% week-block bootstrap intervals (Table II)
  block_bootstrap_paired_differences.csv     Paired differences between configurations
figures/
  fig_horizon_error.png                      MAE and RMSE versus forecast hour (1 to 24)
requirements.txt
Main results

All values are on the min–max normalized target scale (not mm or %), over the same 141,123 test windows and all 24 target steps. Intervals and paired differences are in results/.

Configuration	MAE	RMSE	R²
Persistence	0.00949	0.01724	0.96415
Linear regression	0.00878	0.01660	0.96676
XGBoost	0.00866	0.01440	0.97499
LightGBM	0.00868	0.01479	0.97364
Transformer	0.01092	0.01405	0.97620
ConvLSTM-24	0.00875	0.01538	0.97149

Persistence is already at R² = 0.964, and the differences among the learned configurations are small relative to the block-bootstrap intervals. The table does not support a claim that one model family is better. The model inputs also differ: persistence, linear regression, XGBoost and LightGBM use the final-step inputs (11 features), while the Transformer and ConvLSTM-24 use the full 48 × 9 input sequence.

ARIMA+MLP, the original single-output ConvLSTM and the graph neural network are not part of this ranking. Their evaluation differs from the main task (see the paper's "Original Results Excluded from the Comparative Ranking" table).

Data and limitations
Target. The soil-moisture target is the Visual Crossing Timeline Weather API element soilmoisture01, an API-provided estimate of the 0–0.1 m layer. It is not an in-situ sensor measurement.
The 15 nodes are not independent. They were constructed from a single-location API series by applying node multipliers (0.90 to 1.10) and Gaussian noise (std 0.1 mm), with clipping to [0, 100]. All weather inputs are identical across nodes. The multipliers and noise were not calibrated against field measurements, and the generation code did not set a random seed, so the noise cannot be reproduced exactly.
Split. Nodes 0–11 are used for training and nodes 12–14 for testing. This is a node-based split within the same 2015–2025 period, not a chronological hold-out. Test-period weather and the underlying soil-moisture trajectory are therefore present in the training data through the other nodes. The test nodes also have the largest multipliers (1.071–1.100, versus at most 1.057 in training). Gains over persistence may partly reflect this. A chronological split was not tested.
Scaling. Scaling was fitted on the complete dataset (see the paper's Limitations).
Windows. Windows that span gaps between seasons are not removed.
Uncertainty. Week-block bootstrap intervals (870 node-week blocks, 2,000 resamples, seed 42) remain optimistic because the nodes share the same weeks.
Threshold and regime analyses are descriptive proxies. They are not verified rain or irrigation events.
<!-- TODO (authors): state how the soil-moisture values for 2015 to March 2022 were obtained. Visual Crossing documents hourly soil-moisture history only from 1 April 2022. -->
Reproducing the notebooks

The notebooks were written for Google Colab and read data and models from Google Drive paths such as /content/drive/MyDrive/Datasets/Models_Final_V5 (arrays, scaler) and .../comparison_models (trained models). To run them:

Put the arrays (X_seq_*.npy, X_phys_*.npy, y_*.npy), the scaler and the trained models in the Drive folders referenced at the top of each notebook, or edit those paths.
Use a GPU runtime for the Transformer and ConvLSTM parts.
Run 04_revision_analysis.ipynb top to bottom. Predictions are cached to Drive after the first run, and outputs go to Datasets/revision_outputs/.

Two points that are easy to trip over:

The model file named lightgbm_baseline.pkl actually holds the XGBoost model, and actuallightgbm_baseline.pkl holds LightGBM.
XGBoost may print a warning about a model saved with an older version. This does not change the predictions, but pinned versions in requirements.txt would avoid it.
