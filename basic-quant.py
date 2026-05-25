conn = sqlite3.connect('clash_data.db', timeout=20)
    

    conn.execute('PRAGMA journal_mode=WAL;')
    

    cards = pd.read_sql("SELECT DISTINCT card_id FROM daily_card_stats", conn)['card_id']
    

    daily_market = pd.read_sql("""
        SELECT date, SUM(num_wins) * 1.0 / SUM(num_games) as market_avg
        FROM daily_card_stats
        GROUP BY date
        ORDER BY date ASC
    """, conn)
    daily_market_mvmnt = daily_market["market_avg"].pct_change().dropna()
    conn.commit()
    conn.close()
    for card_id in cards:
        #model = tf.keras.models.load_model('models'+cardid+'.keras')
        conn = sqlite3.connect('clash_data.db', timeout=20)
        
        df = pd.read_sql(f"""
            SELECT date, (CAST(num_wins AS FLOAT)/num_games) as win_rate, num_games, num_wins, card_name
            FROM daily_card_stats 
            WHERE card_id = '{card_id}' 
            AND date >= '20260226'
            AND date <= {until_date}
            ORDER BY date ASC""", conn)
        usage_stat = pd.read_sql(f"""
SELECT 
    -- Calculate numerator per card, denominator for total
    SUM(dcs.num_games) * 1.0 / (
        SELECT SUM(dtg.total_games) * 2 
        FROM daily_total_games dtg
        WHERE dtg.date >= 20260326 AND dtg.date <= {last_day}
    ) as usage_rate,
    dcs.card_id, dcs.date
FROM daily_card_stats dcs
WHERE dcs.date >= 20260326 AND dcs.date <= {last_day}
AND card_id = '{card_id}' 
GROUP BY dcs.card_id, dcs.date
ORDER BY dcs.date ASC
""", conn)
            
        if df.empty:
            continue
            
        print(f"Processing: {df['card_name'].iloc[-1]} on {df['date'].iloc[-1]}| Games: {df['num_games'].iloc[-1]}")
        
        df = df.merge(daily_market, on='date')
        df = df.merge(usage_stat, on='date')
        
       
        df['final_winrate'] = df['num_wins']/df['num_games']
        df['final_winrate'] = (df['num_wins'] + confidence/2) / (df['num_games'] + confidence) 
        df['final_winrate'] *= (0.5 / df['market_avg'])
        
        # ---------------------------------------------------------
        # INDICATORS (Vectorized for Speed & Low-Data Compatibility)
        # ---------------------------------------------------------
        
        # Bollinger Bands (20-day)
        df['MA20'] = df['final_winrate'].rolling(window=20, min_periods=1).mean()
        df['STD20'] = df['final_winrate'].rolling(window=20, min_periods=1).std().fillna(0)
        df['upper'] = df['MA20'] + (df['STD20'] * 2)
        df['lower'] = df['MA20'] - (df['STD20'] * 2)

        # Sharpe Ratio
        df['rolling_mean_14'] = df['final_winrate'].rolling(window=14, min_periods=1).mean()
        df['rolling_std_14'] = df['final_winrate'].rolling(window=14, min_periods=1).std().replace(0, 0.001).fillna(0.001)
        df['sharpe'] = (df['rolling_mean_14'] - 0.5) / df['rolling_std_14']

        # RSI (14-day Wilder's Smoothing)
        delta = df['final_winrate'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan) # Prevent division by zero
        df['rsi'] = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))

        # MACD (12-day EMA - 26-day EMA)
        df['EMA12'] = df['final_winrate'].ewm(span=12, adjust=False, min_periods=1).mean()
        df['EMA26'] = df['final_winrate'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Hist_Slope'] = df['MACD_Hist'].diff(periods=3)
        
        # Drawdown Analysis
        df['peak_winrate'] = df['final_winrate'].cummax()
        df['drawdown'] = (df['final_winrate'] - df['peak_winrate']) / df['peak_winrate']
        # 5 day ROC
        df['roc5'] = None
        df['usage_roc5'] = None
        if len(df) >= 6:
            df.loc[df.index[-1], 'roc5'] = (df['final_winrate'].iloc[-1] - df['final_winrate'].iloc[-6])/df['final_winrate'].iloc[-6]
            df.loc[df.index[-1], 'usage_roc5'] = (df['usage_rate'].iloc[-1] - df['usage_rate'].iloc[-6])/df['usage_rate'].iloc[-6]

        #Beta ANALYSIS Calculate covariance matrix and market variance Calculate Beta
        df['card_pct_change'] = df['final_winrate'].pct_change()
        df['market_pct_change'] = df['market_avg'].pct_change()
        covariance = df['card_pct_change'].cov(df['market_pct_change'])
        market_variance = df['market_pct_change'].var()
        beta = covariance / market_variance
        df.loc[df.index[-1], 'beta'] = beta
        
        # ---------------------------------------------------------
        # PREDICTIVE ANALYSIS: Geometric Brownian Motion (7-day)
        # ---------------------------------------------------------
        df['GBM7'] = None
        df['OU_Prediction'] = None
        if len(df) > 5: # Need at least a few days of variance
            log_returns = np.log(df['final_winrate'] / df['final_winrate'].shift(1)).dropna()
        
            
            mu_daily = log_returns.mean()
            sigma_daily = log_returns.std()
            
            if np.isnan(sigma_daily) or sigma_daily == 0:
                sigma_daily = 0.0001
            
            s0 = df['final_winrate'].iloc[-1]
            T_days = 7
            num_simulations = 200
            
            simulated_prices = np.zeros((T_days + 1, num_simulations))
            simulated_prices[0] = s0
            
            for t in range(1, T_days + 1):
                Z = np.random.normal(0, 1, num_simulations)
                # dt = 1 day, so it's omitted from the standard formula for simplicity
                simulated_prices[t] = simulated_prices[t - 1] * np.exp((mu_daily - 0.5 * sigma_daily**2) + sigma_daily * Z)
            
            mean_final_winrate = round(np.mean(simulated_prices[-1]), 4)
            df.loc[df.index[-1], 'GBM7'] = mean_final_winrate
            
        # Ornstead Unblock Process, mean reverting OUP7
        if len(df) > 5:
            y = df['final_winrate'].values
            X_current = y[:-1].reshape(-1, 1)
            X_next = y[1:]
            
            model = LinearRegression()
            model.fit(X_current, X_next)
            
           
            lambda_hat = max(1e-5, min(0.999, model.coef_[0])) 
            alpha_hat = model.intercept_
            residuals = X_next - model.predict(X_current)
            
           
            dt = 1 
            theta_est = -np.log(lambda_hat) / dt
            mu_est = alpha_hat / (1 - lambda_hat)
            
           
            mu_est = max(0.1, min(0.9, mu_est))
            
            sigma_est = np.std(residuals) * np.sqrt(-2 * np.log(lambda_hat) / (dt * (1 - lambda_hat**2)))
            
          
            n_steps = 7 
            x = np.zeros(n_steps)
            x[0] = y[-1]
            
            for t in range(1, n_steps):
                noise = np.random.normal()
                dx = theta_est * (mu_est - x[t-1]) * dt + sigma_est * np.sqrt(dt) * noise
                
              
                x[t] = np.clip(x[t-1] + dx, 0.0, 1.0) 
            
            
        df.loc[df.index[-1], 'OU_Prediction'] = x[-1]
                        
        
       
        latest = df.iloc[-1]
        
       
        conn.execute('''
            INSERT INTO quant_signals (date, card_id, rsi, upper_bollinger, lower_bollinger, sharpe, macd, macd_slope, drawdown, gbm7, oup7, beta, roc5, usage_roc5)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, card_id) DO UPDATE SET
                rsi=excluded.rsi, 
                upper_bollinger=excluded.upper_bollinger,
                lower_bollinger=excluded.lower_bollinger,
                sharpe=excluded.sharpe,
                macd=excluded.macd,
                macd_slope=excluded.macd_slope,
                drawdown=excluded.drawdown,
                gbm7=excluded.gbm7,
                oup7=excluded.oup7,
                beta=excluded.beta,
                roc5=excluded.roc5,
                usage_roc5=excluded.usage_roc5
        ''', (
            latest['date'], card_id, latest['rsi'], latest['upper'], latest['lower'], 
            latest['sharpe'], latest['MACD_Hist'], latest['MACD_Hist_Slope'], latest['drawdown'], latest['GBM7'], latest['OU_Prediction'], latest['beta'], latest['roc5'], latest['usage_roc5']
        ))
        conn.commit()
        conn.close()
