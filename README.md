# volatility_forecasting
Built a model using S&amp;P 500 returns +GARCH3 volatility as inputs to an LSTM4 to forecast next-day realised volatility.  Benchmarked against standalone GARCH and LSTM; hybrid achieved ~12% lower RMSE on out-of-sample forecasts.  Hybrid model captured more high-volatility events vs only GARCH and LSTM, improving extreme risk identification. 
