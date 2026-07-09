import db
import numpy as np
import pandas as pd
from datetime import timedelta, datetime, date
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
confidence = 100
def TodayAndYesterday():
    now = datetime.now()
    yest = now - timedelta(days=1)
    sevenago = now - timedelta(days=7)
    
    cur_day = now.strftime("%Y%m%d")
    last_day = yest.strftime("%Y%m%d")
    seven_ago_str = sevenago.strftime("%Y%m%d")
    
    return cur_day, last_day, seven_ago_str

cur_day, last_day, seven_ago_str = TodayAndYesterday()
conn = db.get_connection()

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
        AND date<= '{last_day}'
        ORDER BY date ASC""", conn)
all_quant = pd.read_sql(f"""
                SELECT date, rsi, upper_bollinger, lower_bollinger, sharpe, macd, drawdown, gbm7, oup7, macd_slope, beta, roc5, card_id, pair_divergence, sector_lag, neg_divergence, usage_roc5
                FROM quant_signals 
                WHERE date >= '20260326'
                AND date<= '{last_day}'
            """, conn)
usage_stat = pd.read_sql(f"""
SELECT 
    SUM(dcs.num_games) * 1.0 / (
        SELECT SUM(dtg.total_games) * 2 
        FROM daily_total_games dtg
        WHERE dtg.date >= '20260326' AND dtg.date <= '{last_day}'
    ) as usage_rate,
    dcs.card_id, dcs.date
FROM daily_card_stats dcs
WHERE dcs.date >= '20260326' AND dcs.date <= '{last_day}'
GROUP BY dcs.card_id, dcs.date
""", conn)
conn.close()

all_data = all_data.merge(daily_market, on='date')
all_data = all_data.merge(usage_stat, on=['card_id', 'date'], how='left')
all_data['final_winrate'] = (all_data['num_wins'] + confidence/2) / (all_data['num_games'] + confidence) 
all_data['final_winrate'] *= (0.5 / all_data['market_avg'])

all_data = all_data.merge(all_quant, on=['card_id', 'date'], how='left')
all_data = all_data.sort_values(['card_name', 'date']).reset_index(drop=True)

# 1. Update Target: Is the winrate going up TOMORROW compared to TODAY?
all_data['future_winrate'] = all_data.groupby('card_name')['final_winrate'].shift(-1) # Shift -1 for tomorrow

all_data['3day_forward_winrate'] = all_data.groupby('card_name')['final_winrate'].transform(
    lambda x: x.rolling(3, min_periods=1).mean().shift(-3)
)


# Only assign labels where we actually HAVE a tomorrow. This prevents the last day from being labeled 0 and causing the model to learn nonsense.
all_data.loc[all_data['future_winrate'].notna(), 'target_is_up'] = (all_data['future_winrate'] > all_data['final_winrate']).astype(int)

#all_data.loc[all_data['future_winrate'].notna(), 'target_is_up'] = (all_data['3day_forward_winrate'] > all_data['final_winrate']).astype(int)
# 2. Time-Weighted Decay Setup
alpha = 0.5 
def time_weighted_average(x):
    weights = (1 - alpha) ** np.arange(len(x))[::-1]
    return np.average(x, weights=weights)

grouped_cards = all_data.groupby('card_name')
all_data['tw_winrate_7d'] = grouped_cards['final_winrate'].transform(lambda x: x.rolling(window=7, min_periods=1).apply(time_weighted_average, raw=True))
all_data['tw_usage_7d'] = grouped_cards['usage_rate'].transform(lambda x: x.rolling(window=7, min_periods=1).apply(time_weighted_average, raw=True))
all_data['winrate_std_7d'] = grouped_cards['final_winrate'].transform(lambda x: x.rolling(window=7, min_periods=1).std().fillna(0))

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

features = [
    'macd', 'rsi', 'sharpe', 'macd_slope', 'roc5', 'pair_divergence', 
    'sector_lag', 'neg_divergence', 'usage_roc5', 'day_sin', 
    'days_since_1st_monday', 'tw_winrate_7d', 'tw_usage_7d', 'winrate_std_7d'
]

# Drop rows missing features or TARGET (This naturally removes the inference day from training)
ml_df = all_data.dropna(subset=['target_is_up'] + features)

X = ml_df[features]
y = ml_df['target_is_up']

# CRITICAL FIX: shuffle=False to prevent temporal data leakage!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, shuffle=False)

ratio = (y_train_sub == 0).sum() / (y_train_sub == 1).sum()

best_xgb_model = XGBClassifier(
    scale_pos_weight=ratio,
    random_state=42,
    eval_metric="auc",
    early_stopping_rounds=50,
    max_depth=6,
    learning_rate=0.015,
    n_estimators=500,
    subsample=0.7,
    colsample_bytree=0.9
)

print("Training model...")
best_xgb_model.fit(X_train_sub, y_train_sub, eval_set=[(X_val, y_val)], verbose=False)
importances = best_xgb_model.feature_importances_

print("\n--- Feature Importances ---")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance*100:.2f}%")

predictions = best_xgb_model.predict(X_test)
print("\n--- Classification Report (Test Set) ---")
print(classification_report(y_test, predictions))

best_xgb_model.save_model("models/xgb.ubj")
print("Saved models/xgb.ubj")
