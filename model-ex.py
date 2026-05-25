def XGBoostpred(last_day):
    xgb_model = xgboost.Booster()
    xgb_model.load_model("models/xgb.ubj")
    conn = sqlite3.connect('clash_data.db', timeout=20)
    
    daily_market = pd.read_sql("""
            SELECT date, SUM(num_wins) * 1.0 / SUM(num_games) as market_avg
            FROM daily_card_stats
            GROUP BY date
            ORDER BY date ASC
        """, conn)
    cards = pd.read_sql("SELECT DISTINCT card_id FROM daily_card_stats", conn)['card_id']
    all_data = pd.read_sql(f"""
            SELECT date, (CAST(num_wins AS FLOAT)/num_games) as win_rate, num_games, num_wins, card_name, card_id
            FROM daily_card_stats
            WHERE date >= 20260326 AND date <= {last_day}
            ORDER BY date ASC""", conn)
    all_quant = pd.read_sql(f"""
                    SELECT date, rsi, upper_bollinger, lower_bollinger, sharpe, macd, drawdown, gbm7, oup7, macd_slope, beta, roc5, card_id, pair_divergence, sector_lag, neg_divergence, usage_roc5
                    FROM quant_signals 
                    WHERE date = {last_day}
                """, conn)
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
GROUP BY dcs.card_id, dcs.date
""", conn)
    conn.close()
    all_data = all_data.merge(daily_market, on='date')
    all_data = all_data.merge(usage_stat, on=['card_id', 'date'], how='left')
    all_data['final_winrate'] = (all_data['num_wins'] + 5) / (all_data['num_games'] + 10) 
    all_data['final_winrate'] *= (0.5 / all_data['market_avg'])
    
    all_data = all_data.merge(all_quant, on=['card_id', 'date'], how='left')
    
    all_data['winrate_lag_1'] = all_data.groupby('card_name')['final_winrate'].shift(1)
    all_data['winrate_lag_2'] = all_data.groupby('card_name')['final_winrate'].shift(2)
    
    all_data['date'] = pd.to_datetime(all_data['date'])
    
   
    all_data['day_of_week'] = all_data['date'].dt.dayofweek
    all_data['day_sin'] = np.sin(2 * np.pi * all_data['day_of_week']/7)
    start_date = date(2019, 7, 1)
    end_date = date.today()
    months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    first_mondays = []
    for m in months:
       
        first_day_of_month = m.dayofweek
       
        days_to_monday = (7 - first_day_of_month) % 7
        first_mondays.append(m + pd.Timedelta(days=days_to_monday))
    
    fm_df = pd.DataFrame({'first_monday': first_mondays})
    fm_df['year_month'] = fm_df['first_monday'].dt.to_period('M')
    
    
  
    all_data['date'] = pd.to_datetime(all_data['date'])
    all_data['year_month'] = all_data['date'].dt.to_period('M')
    
   
    all_data = all_data.merge(fm_df, on='year_month', how='left')
 
    all_data['days_since_1st_monday'] = (all_data['date'] - all_data['first_monday']).dt.days
    all_data = all_data.drop(columns=['first_monday', 'year_month'])
    
    grouped_cards = all_data.groupby('card_name')
    alpha = 0.5 
    
    all_data['tw_winrate_7d'] = grouped_cards['final_winrate'].transform(
        lambda x: x.rolling(window=7, min_periods=1).apply(time_weighted_average, raw=True)
    )
    all_data['tw_usage_7d'] = grouped_cards['usage_rate'].transform(
        lambda x: x.rolling(window=7, min_periods=1).apply(time_weighted_average, raw=True)
    )
    
   
    
   
    ml_df = all_data.dropna(subset=[ 'macd', 
    'rsi', 
    'sharpe', 
    'macd_slope', 
    'roc5', 
    'pair_divergence', 
    'sector_lag', 
    'neg_divergence', 
    'usage_roc5', 
    'day_sin', 
    'days_since_1st_monday',
    'tw_winrate_7d',  
    'tw_usage_7d' ])
    
  
    features = [
    'macd', 
    'rsi', 
    'sharpe', 
    'macd_slope', 
    'roc5', 
    'pair_divergence', 
    'sector_lag', 
    'neg_divergence', 
    'usage_roc5', 
    'day_sin', 
    'days_since_1st_monday',
    'tw_winrate_7d',  
    'tw_usage_7d'    
]
    
    unique_cards = ml_df['card_id'].unique()

    for card in unique_cards:
   
        card_data = ml_df[ml_df['card_id'] == card]
        
        if card_data.empty:
            continue
            
        X_card = card_data[features]
        
    
        dmatrix_card = xgboost.DMatrix(X_card)
        
     
        preds = xgb_model.predict(dmatrix_card)
        print(card + " | " + str(preds[0]))
        conn = sqlite3.connect('clash_data.db', timeout=20)
        conn.execute('''
            INSERT INTO quant_signals (date, card_id, xgb_up)
            VALUES (?, ?, ?)
            ON CONFLICT(date, card_id) DO UPDATE SET
                xgb_up=excluded.xgb_up
        ''', (
            last_day, card, float(preds[0])
        ))
        conn.commit()
        conn.close()

def time_weighted_average(x):
    alpha = 0.5 
    weights = (1 - alpha) ** np.arange(len(x))[::-1]
    return np.average(x, weights=weights)
def LSTMpred(last_day):
    conn = sqlite3.connect('clash_data.db', timeout=20)

    daily_market = pd.read_sql("""
            SELECT date, SUM(num_wins) * 1.0 / SUM(num_games) as market_avg
            FROM daily_card_stats
            GROUP BY date
            ORDER BY date ASC
        """, conn)
    cards = pd.read_sql("SELECT DISTINCT card_id FROM daily_card_stats", conn)['card_id']
    all_data = pd.read_sql(f"""
            SELECT date, (CAST(num_wins AS FLOAT)/num_games) as win_rate, num_games, num_wins, card_name, card_id
            FROM daily_card_stats
            WHERE date >= '20260326'
            AND date<= {last_day}
            ORDER BY date ASC""", conn)
    all_quant = pd.read_sql(f"""
                    SELECT date, rsi, upper_bollinger, lower_bollinger, sharpe, macd, drawdown, gbm7, oup7, macd_slope, beta, roc5, card_id, pair_divergence, sector_lag, neg_divergence, usage_roc5
                    FROM quant_signals 
                    WHERE date >= '20260326'
                    AND date<= {last_day}
                """, conn)
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
GROUP BY dcs.card_id, dcs.date
""", conn)
    conn.close()
    all_data = all_data.merge(daily_market, on='date')
    all_data = all_data.merge(usage_stat, on=['card_id', 'date'], how='left')
    all_data['final_winrate'] = (all_data['num_wins'] + 5) / (all_data['num_games'] + 10) 
    all_data['final_winrate'] *= (0.5 / all_data['market_avg'])
    
    all_data = all_data.merge(all_quant, on=['card_id', 'date'], how='left')
    
    pivot_df = all_data.pivot(index='date', columns='card_id', values='final_winrate')
    #pivot_cur = current_data.pivot(index='date', columns='card_id', values='final_winrate')
    
    
        
    all_data['date'] = pd.to_datetime(all_data['date'])
    

    all_data['day_of_week'] = all_data['date'].dt.dayofweek
    start_date = date(2019, 7, 1)
    end_date = date.today()
    months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    first_mondays = []
    for m in months:
  
        first_day_of_month = m.dayofweek
    
        days_to_monday = (7 - first_day_of_month) % 7
        first_mondays.append(m + pd.Timedelta(days=days_to_monday))
    
    fm_df = pd.DataFrame({'first_monday': first_mondays})
    fm_df['year_month'] = fm_df['first_monday'].dt.to_period('M')
    
 

    all_data['date'] = pd.to_datetime(all_data['date'])
    all_data['year_month'] = all_data['date'].dt.to_period('M')
    

    all_data = all_data.merge(fm_df, on='year_month', how='left')
    

    all_data['days_since_1st_monday'] = (all_data['date'] - all_data['first_monday']).dt.days
    

    all_data = all_data.drop(columns=['first_monday', 'year_month'])
    
    all_data['day_sin'] = np.sin(2 * np.pi * all_data['day_of_week'] / 7)
    all_data['day_cos'] = np.cos(2 * np.pi * all_data['day_of_week'] / 7)


    feature_cols = [
    'macd', 'rsi', 'sharpe', 'macd_slope', 'roc5', 
    'pair_divergence', 'sector_lag', 'usage_rate', 'neg_divergence',
    'usage_roc5', 'day_sin', 'day_cos', "days_since_1st_monday"
]
    

    for card_id in cards:
        if not os.path.exists('models/' + card_id + ".keras"):
            continue
        model = load_model('models/' + card_id + ".keras")
        scaler_x_path = f'scalers/{card_id}_x.pkl' 
        scaler_y_path = f'scalers/{card_id}_y.pkl'
  
        target_card_df = all_data[all_data['card_id'] == card_id].copy()
        target_card_df = target_card_df.sort_values('date')
        target_card_df[feature_cols] = target_card_df[feature_cols].replace([np.inf, -np.inf], np.nan)

        target_card_df[feature_cols] = target_card_df[feature_cols].fillna(0)
        
        if len(target_card_df) < 20: 
            continue
        
       
        scaler_x = joblib.load(scaler_x_path) 
        scaler_y = joblib.load(scaler_y_path)
    
    
        target_card_df = all_data[all_data['card_id'] == card_id].copy()
        target_card_df = target_card_df.sort_values('date')
        
        window_size = 7
        if len(target_card_df) < (window_size + 1): 
            continue
    
        target_card_df[feature_cols] = target_card_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
  
        scaled_x = scaler_x.transform(target_card_df[feature_cols])
    
     
        X = []
        for i in range(window_size, len(scaled_x)):
            X.append(scaled_x[i - window_size:i, :])
        X = np.array(X)
    
  
        predictions_scaled = model.predict(X)
        predictions = scaler_y.inverse_transform(predictions_scaled)
    
     
        latest_prediction = float(predictions[-1][0])

        print(card_id + " | " + str(latest_prediction))
        print(latest_prediction)
        conn = sqlite3.connect('clash_data.db', timeout=20)
        conn.execute('''
            INSERT INTO quant_signals (date, card_id, lstm)
            VALUES (?, ?, ?)
            ON CONFLICT(date, card_id) DO UPDATE SET
                lstm=excluded.lstm
        ''', (
            last_day, card_id, float(latest_prediction)
        ))
        conn.commit()
        conn.close()
