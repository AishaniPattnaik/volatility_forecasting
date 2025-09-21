# volatility_forecasting
Built a model using S&amp;P 500 returns +GARCH3 volatility as inputs to an LSTM4 to forecast next-day realised volatility.  Benchmarked against standalone GARCH and LSTM; hybrid achieved ~12% lower RMSE on out-of-sample forecasts.  Hybrid model captured more high-volatility events vs only GARCH and LSTM, improving extreme risk identification. 

======================================================

Files:
- train_hybrid_vol_forecast.py  : main script. Downloads data, prepares features, trains models, evaluates, saves dataset + models.
- requirements.txt              : pip dependencies.

How to run:
1. Create and activate a python env (recommended).
2. pip install -r requirements.txt
3. python train_hybrid_vol_forecast.py

Outputs (in ./output):
- sp500_data.csv           : dataset built by the script (features + target).
- models/lstm_hybrid.h5    : saved Keras hybrid model.
- models/lstm_baseline.h5  : saved Keras LSTM-only model.
- figures/*.png            : evaluation and diagnostic plots.
- metrics_summary.txt      : RMSEs and summary.

To push to GitHub:
- git init; git add .; git commit -m "initial"
- create GitHub repo and push per usual steps.


