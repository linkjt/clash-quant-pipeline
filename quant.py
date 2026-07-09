import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import db
import psycopg2
from datetime import datetime, timedelta, date
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import xgboost
import weasyprint
import os
from weasyprint import HTML
import joblib

confidence = 100
db_path = "/mnt/typeshyt/local_db/clash_data.db"



def TodayAndYesterday():
    now = datetime.now()
    yest = now - timedelta(days=1)
    sevenago = now - timedelta(days=7)
    
    cur_day = now.strftime("%Y%m%d")
    last_day = yest.strftime("%Y%m%d") # Fixed to actually be yesterday
    seven_ago_str = sevenago.strftime("%Y%m%d") # Formatted for SQL
    
    return cur_day, last_day, seven_ago_str
def calculate_conviction(row):
    score = 0

    # +1 Point: XGBoost predicts UP
    if not(row['xgb_up']) == None and row['xgb_up'] > .75: score += 1
    if not(row['xgb_up']) == None and row['xgb_up'] < .25: score -= 1
    
    # +1 Point: LSTM predicts UP (assuming LSTM column is binary)
    if not pd.isna(row['lstm']) and row['lstm'] is not None:
        if row['lstm'] > row['final_winrate'] + .005: score += 1
        if row['lstm'] < row['final_winrate'] - .005: score -= 1
    # +1 Point: Card has positive Pair Divergence (lagging)
    if not pd.isna(row['pair_divergence']) and row['pair_divergence'] is not None:
        if row['pair_divergence'] < -.15: score += 1
        if row['pair_divergence'] > .15: score -= 1
    
    # -1 Point: RSI is over 70 (Overbought)
    if row['rsi'] > 70: score -= 1
    if row['rsi'] < 30: score += 1
    if ((row['xgb_up']) == None): score = 0
    return score
def categorize_card(score):
    if score >= 4: return "High Conviction Buy"
    elif score == 3: return "Weak Buy"
    elif score == -3: return "Weak Sell"
    elif score <= -4: return "High Conviction Sell"
    else: return "Neutral"

    #72.3
def ClashReportPDF(last_day):
    conn = db.get_connection()
    daily_market = pd.read_sql(f"""
        SELECT SUM(num_wins) * 1.0 / SUM(num_games) as market_avg
        FROM daily_card_stats
        WHERE date = '{last_day}'
    """, conn)
    
    market_avg = daily_market['market_avg'].iloc[0] if not daily_market.empty else 0.5
    
    
    all_data = pd.read_sql(f"""
        SELECT (CAST(num_wins AS FLOAT)/num_games) as win_rate, num_games, num_wins, card_name, card_id
        FROM daily_card_stats
        WHERE date = '{last_day}'
        """, conn)
        
    corX = pd.read_sql(f"""
                SELECT top_cor, top_cor_val, next_cor, next_cor_val, neg_cor, neg_cor_val, cluster_num sector_lag, card_id
                FROM card_corx 
            """, conn)
        
    all_quant = pd.read_sql(f"""
                    SELECT rsi, upper_bollinger, lower_bollinger, sharpe, macd, drawdown, gbm7, oup7, macd_slope, beta, roc5, card_id, pair_divergence, sector_lag, lstm, rforest_up, xgb_up
                    FROM quant_signals 
                    WHERE date = '{last_day}'
                """, conn)
    usage_stat = pd.read_sql(f"""
    SELECT 
        -- Calculate numerator per card, denominator for total
        SUM(dcs.num_games) * 1.0 / (
            SELECT SUM(dtg.total_games) * 2 
            FROM daily_total_games dtg
            WHERE dtg.date = '{last_day}'
        ) as usage_rate,
        dcs.card_id
    FROM daily_card_stats dcs
    WHERE dcs.date = '{last_day}'
    GROUP BY dcs.card_id
    """, conn)
                
    conn.close()
    all_data = all_data.merge(all_quant, on='card_id', validate='many_to_one')
    all_data = all_data.merge(usage_stat, on='card_id', validate='many_to_one')
    all_data = all_data.merge(corX, on='card_id', validate='many_to_one')
    all_data['final_winrate'] = (all_data['num_wins'] + confidence/2) / (all_data['num_games'] + confidence) 
    all_data['final_winrate'] *= (0.5 / market_avg)
    
    all_data['conviction_score'] = all_data.apply(calculate_conviction, axis=1)
    
    all_data['conviction_label'] = all_data['conviction_score'].apply(categorize_card)
    

    all_data = all_data[(all_data['final_winrate'] > .5) | (all_data['lstm'] > .5)]
    # Lowest Scoring (Including Inverse)
    # Sorting by score ascending to get worst performers first
    all_data['xgb_up'] = pd.to_numeric(all_data['xgb_up'], errors='coerce')
    top_sells = all_data.nsmallest(5, 'xgb_up')

    # Largest 5
    top_buys = all_data.nlargest(5, 'xgb_up')
    date_str = datetime.now().strftime("%B %d, %Y")
    
    # Generate the dynamic rows for the Buys
    buys_html = ""
    for index, buy in top_buys.iterrows():
        buys_html += f"""
        <tr>
            <td><strong>{buy['card_name']}</strong></td>
            <td><span class="buy">Bullish ({(round(abs(buy['xgb_up']),3))}%)</span></td>
            <td>TENSORFLOW DOWN</td>
            <td>{round(buy['rsi'],2)}</td>
            <td>{round(buy['win_rate'], 3)}%</td>
            <td>{round(buy['usage_rate'], 3)}%</td>
            <td><span class="buy">STRONG BUY</span></td>
        </tr>
        
        """

    # Generate the dynamic rows for the Sells
    sells_html = ""
    for index, sell in top_sells.iterrows():
        sells_html += f"""
        <tr>
            <td><strong>{sell['card_name']}</strong></td>
            <td><span class="sell">Bearish ({round(abs(1-sell['xgb_up']),4)}%)</span></td>
            <td>TENSORFLOW DOWN</td>
            <td>{round(sell['rsi'],2)}</td>
            <td>{round(sell['win_rate'],4)}%</td>
            <td>{round(sell['usage_rate'],4)}%</td>
            <td><span class="sell">STRONG SELL</span></td>
        </tr>
        
        """

    # Assemble the HTML (Using the CSS from the generated PDF)
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <style>
        @page {{ size: A4; margin: 15mm; background-color: #f0f2f5; }}
        body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #2c3e50; font-size: 10.5pt; }}
        .header {{ background-color: #1a252f; color: #ecf0f1; padding: 20px; border-radius: 4px; position: relative; }}
        .header h1 {{ margin: 0; font-size: 24pt; text-transform: uppercase; }}
        .date-badge {{ position: absolute; top: 20px; right: 20px; background-color: #3498db; padding: 5px 12px; border-radius: 15px; font-weight: bold; font-size: 10pt; color: white; }}
        h2 {{ border-left: 5px solid #3498db; padding-left: 10px; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; background-color: #ffffff; margin-bottom: 25px; }}
        th, td {{ padding: 10px; border-bottom: 1px solid #ecf0f1; text-align: left; }}
        th {{ background-color: #ecf0f1; font-weight: bold; font-size: 9pt; text-transform: uppercase; }}
        .buy {{ color: #27ae60; font-weight: bold; }}
        .sell {{ color: #e74c3c; font-weight: bold; }}
        .thesis-text {{ font-size: 9.5pt; color: #555; font-style: italic; }}
        .stats-bar {{ background-color: #ffffff; padding: 15px; border: 1px solid #bdc3c7; text-align: center; margin-top: 20px; }}
        .stat-item {{ display: inline-block; width: 30%; }}
        .stat-item strong {{ display: block; font-size: 16pt; color: #1a252f; }}
        .stat-item span {{ font-size: 9pt; color: #7f8c8d; text-transform: uppercase; }}
    </style>
    </head>
    <body>
        <div class="header">
            <h1>Meta Arbitrage Report</h1>
            <p>Quantitative Prediction & Asset Allocation</p>
            <div class="date-badge">{date_str}</div>
        </div>
        <div class="stats-bar">
            <div class="stat-item"><strong>{round(market_avg,4)}%</strong><span>Global Market Average</span></div>
            <div class="stat-item"><strong>{72.3}%</strong><span>XGBoost 30D Accuracy</span></div>
            <div class="stat-item"><strong>{6}</strong><span>Active Sector Shifts</span></div>
        </div>
        <h2>Top High-Conviction "Buys"</h2>
        <table>
            <tr><th>Card</th><th>XGBoost</th><th>LSTM Target</th><th>RSI</th><th>Win Rate</th><th>Usage Rate</th><th>Conviction</th></tr>
            {buys_html}
        </table>
        <h2>Top Imminent "Sells"</h2>
        <table>
            <tr><th>Card</th><th>XGBoost</th><th>LSTM Target</th><th>RSI</th><th>Win Rate</th><th>Usage Rate</th><th>Conviction</th></tr>
            {sells_html}
        </table>
    </body>
    </html>
    """
    
    HTML(string=html_content).write_pdf(f"pdf_reports/ClashReport-{last_day}.pdf")


def categorize_card(score):
    if score >= 3: return "High Conviction Buy"
    elif score == 1: return "Weak Buy"
    elif score <= -3: return "Inverse/Sell"
    else: return "Neutral"
def ClashReport():
    cur_day, last_day, seven_ago_str = TodayAndYesterday()
    
    conn = db.get_connection()
    cards = pd.read_sql("SELECT DISTINCT card_id FROM daily_card_stats", conn)['card_id']
    
    # Get Market Bias for the target day (yesterday is usually safest for complete data)
    daily_market = pd.read_sql(f"""
        SELECT SUM(num_wins) * 1.0 / SUM(num_games) as market_avg
        FROM daily_card_stats
        WHERE date = '{last_day}'
    """, conn)
    
    market_avg = daily_market['market_avg'].iloc[0] if not daily_market.empty else 0.5
    

    # Open the file ONCE for the entire process (Massive speed increase)
    with open(f"text_reports/ClashReport-{cur_day}.txt", "w") as f:
        f.write(f"Clash Report for {cur_day} (Analyzing {last_day} data).\n")
        f.write("##############################################\n\n")
        f.write(f"Current Win Rate Average Across All Cards = {round(market_avg, 4)}\n")
        f.write(f"The Average Win Rate for cards is adjusted in calculations using Bayesian smoothing and normalization to be 50%, the number above is what it was prior to adjustment\n\n")
        
        for card_id in cards:
            # 1. Get Quant Signals for Yesterday
            quant = pd.read_sql(f"""
                SELECT rsi, upper_bollinger, lower_bollinger, sharpe, macd, drawdown, gbm7, oup7, macd_slope, beta, roc5, pair_divergence, neg_divergence, sector_lag, lstm, rforest_up, xgb_up
                FROM quant_signals 
                WHERE card_id = '{card_id}' 
                AND date = '{last_day}'
            """, conn)
            
            corX = pd.read_sql(f"""
                SELECT top_cor, top_cor_val, next_cor, next_cor_val, neg_cor, neg_cor_val, cluster_num sector_lag
                FROM card_corx 
                WHERE card_id = '{card_id}' 
            """, conn)
            
            # 2. Get YESTERDAY'S exact win rate (for indicators)
            daily_stat = pd.read_sql(f"""
                SELECT (CAST(num_wins AS FLOAT)/num_games) as win_rate, num_games, num_wins, card_name
                FROM daily_card_stats 
                WHERE card_id = '{card_id}'
                AND date = '{last_day}'
            """, conn)

            # 3. Get 7-DAY Usage Rate (for human context)
            usage_stat = pd.read_sql(f"""
                SELECT SUM(num_games) * 1.0 / (
                    SELECT SUM(total_games) * 2 
                    FROM daily_total_games 
                    WHERE date BETWEEN '{seven_ago_str}' AND '{last_day}'
                ) as usage_rate
                FROM daily_card_stats 
                WHERE card_id = '{card_id}'
                AND date BETWEEN '{seven_ago_str}' AND '{last_day}'
            """, conn)

            if not quant.empty and not daily_stat.empty and not usage_stat.empty and not corX.empty:
                card_name = daily_stat["card_name"].iloc[0]
                wins = daily_stat["num_wins"].iloc[0]
                games = daily_stat["num_games"].iloc[0]
                usage_rate = usage_stat["usage_rate"].iloc[0]
                
                # Adjusted Bayesian Smoothing (Lighter weight)
                final_winrate = (wins + 5) / (games + 10)
                final_winrate *= (0.5 / market_avg) # Normalize
                
                macd_cur_diff = quant["macd"].iloc[0]
                macd_slope = quant['macd_slope'].iloc[0]
                
                beta = quant["beta"].iloc[0]
                
                gbm7 = quant["gbm7"].iloc[0]
                oup7 = quant["oup7"].iloc[0]
                lstm = quant["lstm"].iloc[0]
                xgb_up = quant["xgb_up"].iloc[0]
                
                
                pair_divergence = quant["pair_divergence"].iloc[0]
                pair = corX["top_cor"].iloc[0]
                
                sector_lag = quant["sector_lag"].iloc[0]
                
                drawdown = quant["drawdown"].iloc[0]
                
                upper_band = quant["upper_bollinger"].iloc[0]
                lower_band = quant["lower_bollinger"].iloc[0]
                rsi = quant["rsi"].iloc[0]
                sharpe = quant["sharpe"].iloc[0]
                
                roc5 = quant["roc5"].iloc[0]
                
                # Flag to check if we should print the header for this card
                sig = False 
                report_lines = []

                if upper_band is not None and final_winrate > upper_band:
                    sig = True
                    report_lines.append(f" - OVERPERFORMING: Win rate ({round(final_winrate, 3)}) breached Upper Bollinger ({round(upper_band, 3)})\n")
                
                if lower_band is not None and final_winrate < lower_band:
                    sig = True
                    report_lines.append(f" - UNDERPERFORMING: Win rate ({round(final_winrate, 3)}) breached Lower Bollinger ({round(lower_band, 3)})\n")
                
                if rsi is not None and rsi > 60:
                    sig = True
                    report_lines.append(f" - OVERPLAYED: RSI is high at {round(rsi, 1)} (Strong uptrend or overplayed)\n")
                
                if rsi is not None and rsi < 40:
                    sig = True
                    report_lines.append(f" - UNDERPLAYED: RSI is low at {round(rsi, 1)} (Strong downtrend or underplayed)\n")
                    
                if macd_cur_diff is not None and macd_cur_diff > 0.008:
                    sig = True
                    report_lines.append(f" - STRONG UPWARD MOEMENTUM: MACD vs. Signal Line Diffrence ({round(macd_cur_diff, 3)}) (winrate is going up)\n")
                    
                    if macd_slope is not None and macd_slope >= 0:
                        report_lines.append(f" - GROWING STRONGER: The MACD Histogram Slope shows that the above trend is growing ({round(macd_slope, 3)})\n")
                    if macd_slope is not None and macd_slope <= 0:
                        report_lines.append(f" - GROWING WEAKER: The MACD Histogram Slope shows that the above trend is weakening ({round(macd_slope, 3)})\n")
                if macd_cur_diff is not None and macd_cur_diff < -0.008:
                    sig = True
                    report_lines.append(f" - STRONG DOWNWARD MOEMENTUM: MACD vs. Signal Line Diffrence ({round(macd_cur_diff, 3)}) (winrate is going down)\n")
                    if macd_slope is not None and macd_slope >= 0:
                        report_lines.append(f" - GROWING STRONGER: The MACD Histogram Slope shows that the above trend is growing ({round(macd_slope, 3)})\n")
                    if macd_slope is not None and macd_slope <= 0:
                        report_lines.append(f" - GROWING WEAKER: The MACD Histogram Slope shows that the above trend is weakening ({round(macd_slope, 3)})\n")
                     
                        
                if pair_divergence is not None and pair_divergence > .15:
                    
                    report_lines.append(f" - PAIR REVERTING: This card is overused based on its highest corralated card, {pair}\n")
                    
                if pair_divergence is not None and pair_divergence < -.15:
                    
                    report_lines.append(f" - PAIR UNDERUSED: This card is underused based on its highest corralated card, {pair}\n")
            
                
                if gbm7 is not None and gbm7 > .54:
                    sig = True
                    report_lines.append(f" - HIGH PROSPECTIVE WINRATE: Predictied Win Rate in 7 Days using 150 simulations for Geometric Brownian Motion (GBM) ({round(gbm7, 3)}) (less trust worthy)\n")
                if oup7 is not None and oup7 > .54:
                    sig = True
                    report_lines.append(f" - HIGH PROSPECTIVE WINRATE: Predictied Win Rate in 7 Days using ornstein uhlenbeck process, a better mean-reverting predictor compared to GBM ({round(oup7, 3)})\n")
                
                if roc5 is not None and roc5 > .10:
                    sig = True
                    report_lines.append(f" - RAPIDLY WINING CARD: Over the past 5 days, the winrate has increased by {round(drawdown*100, 3)}% of itslef\n")
                if roc5 is not None and roc5 < -.10:
                    sig = True
                    report_lines.append(f" - RAPIDLY LOSING CARD: Over the past 5 days, the winrate has decrased by {-round(roc5*100, 3)}% of itsself\n")
                
                if sharpe is not None and sharpe > 2:
                    sig = True
                    report_lines.append(f" - GREAT INVESTMENT: High risk-adjusted return (Sharpe: {round(sharpe, 2)})\n")
                elif sharpe is not None and sharpe < 1:
                    # Only print bad sharpe if it's already doing something else notable
                    if sig: 
                        report_lines.append(f" - VOLATILE/WEAK: Poor risk-adjusted return (Sharpe: {round(sharpe, 2)})\n")
                if sig:
                    if beta is not None and beta == 1:
                        
                        report_lines.append(f" - META MOVING CARD: This card has a win rate that typically moves with the meta\n")    
                    
                    if beta is not None and beta > 1.10:
                        
                        report_lines.append(f" - SPECALIZED CARD: This card has a win rate that is hyper dependedt on a meta that favors it\n")
                        
                    if beta is not None and beta < .90:
                        
                        report_lines.append(f" - IMMUNE TO META: This card has a win rate that is meta independent\n")
                    
                    if drawdown is not None and drawdown > -.10:
                        
                        report_lines.append(f" - STRONG CARD STABILITY: Drawdown level ({round(drawdown, 3)}) (winrate is not likely to tank during a nerf)\n")
                        
                    if drawdown is not None and drawdown < -.30:
                        
                        report_lines.append(f" - WEAK CARD STABILITY: Drawdown level ({round(drawdown, 3)}) (winrate is very likely to tank during a nerf)\n")
                    # If signals triggered, write to file
                if sig:
                    f.write(f"[{card_name}] | WinRate: {round(final_winrate, 3)} | "
        f"7Day-Usage: {round(usage_rate * 100, 2)}% | Daily Games: {games} | "
        f"Real Win Rate Right Now {lstm if lstm is not None else 'N/A'} | "
        f"Winrate Going Up Tommorow {xgb_up > 0.5 if xgb_up is not None else 'N/A'}\n")
                    for line in report_lines:
                        f.write(line)
                    f.write("\n")

    conn.close()
            
            
def normalize_and_smooth(df, global_market_avg, confidence=50):
    """
    df: DataFrame for ONE card
    global_market_avg: The average winrate of ALL cards in the DB for the period
    """
    # 1. Bayesian Smoothing
    df['smoothed_wins'] = df['num_wins'] + (confidence * 0.5)
    df['smoothed_games'] = df['num_games'] + confidence
    df['smoothed_rate'] = df['smoothed_wins'] / df['smoothed_games']
    
    # 2. Global Shift 
    # Use the market average passed in, not the card's own average
    shift_factor = 0.5 / global_market_avg
    
    # 3. Apply shift
    df['final_winrate'] = df['smoothed_rate'] * shift_factor
    
    return df
    
def run_quant_analysis(until_date):
    target_date_str = str(until_date).replace('-', '')
    print(target_date_str)
    conn = db.get_connection()
    cursor = conn.cursor()
    
    
    cards = pd.read_sql("SELECT DISTINCT card_id FROM daily_card_stats", conn)['card_id']
    daily_market = pd.read_sql("""
        SELECT date, SUM(num_wins) * 1.0 / SUM(num_games) as market_avg
        FROM daily_card_stats
        GROUP BY date ORDER BY date ASC
    """, conn)
    
    # FIXED: Replaced outer scope global leak leak with normalized loop parameter target matching
    usage_stat = pd.read_sql(f"""
    SELECT 
        SUM(dcs.num_games) * 1.0 / (
            SELECT SUM(dtg.total_games) * 2 
            FROM daily_total_games dtg
            WHERE dtg.date >= '20260326' AND dtg.date <= '{target_date_str}'
        ) as usage_rate,
        dcs.card_id, dcs.date
    FROM daily_card_stats dcs
    WHERE dcs.date >= '20260326' AND dcs.date <= '{target_date_str}'
    GROUP BY dcs.card_id, dcs.date 
    ORDER BY dcs.date ASC
""", conn)
    
    daily_market['date'] = daily_market['date'].astype(str)
    usage_stat['date'] = usage_stat['date'].astype(str)

    for card_id in cards:
        df = pd.read_sql(f"""
            SELECT date, (CAST(num_wins AS FLOAT)/num_games) as win_rate, num_games, num_wins, card_name, card_id
            FROM daily_card_stats 
            WHERE card_id = '{card_id}' AND date >= '20260226' AND date <= '{target_date_str}'
            ORDER BY date ASC""", conn)
            
        if df.empty:
            continue
            
        df['date'] = df['date'].astype(str)
        df = df.merge(daily_market, on='date', how='left')
        df = df.merge(usage_stat, on=['card_id', 'date'], how='left')
        
        # Safe Normalization Equations
        df['final_winrate'] = (df['num_wins'] + confidence/2) / (df['num_games'] + confidence) 
        df['final_winrate'] *= (0.5 / df['market_avg'])
        
        # TECHNICAL INDICATOR ARRAY MATH (Full Vector Tracking)
        df['MA20'] = df['final_winrate'].rolling(window=20, min_periods=1).mean()
        df['STD20'] = df['final_winrate'].rolling(window=20, min_periods=1).std().fillna(0)
        df['upper'] = df['MA20'] + (df['STD20'] * 2)
        df['lower'] = df['MA20'] - (df['STD20'] * 2)

        df['rolling_mean_14'] = df['final_winrate'].rolling(window=14, min_periods=1).mean()
        df['rolling_std_14'] = df['final_winrate'].rolling(window=14, min_periods=1).std().replace(0, 0.001).fillna(0.001)
        df['sharpe'] = (df['rolling_mean_14'] - 0.5) / df['rolling_std_14']

        delta = df['final_winrate'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, min_periods=1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=1, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))

        df['EMA12'] = df['final_winrate'].ewm(span=12, adjust=False, min_periods=1).mean()
        df['EMA26'] = df['final_winrate'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Hist_Slope'] = df['MACD_Hist'].diff(periods=3).fillna(0)
        
        df['peak_winrate'] = df['final_winrate'].cummax()
        df['drawdown'] = (df['final_winrate'] - df['peak_winrate']) / df['peak_winrate'].replace(0, 1)
        
        # FIXED: Vectorized the 5-day rate of change instead of editing row indexes manually
        df['roc5'] = df['final_winrate'].pct_change(periods=5).fillna(0)
        df['usage_roc5'] = df['usage_rate'].pct_change(periods=5).fillna(0)

        df['card_pct_change'] = df['final_winrate'].pct_change()
        df['market_pct_change'] = df['market_avg'].pct_change()
        
        # Rolling covariance analysis mapping
        rolling_cov = df['card_pct_change'].rolling(window=14).cov(df['market_pct_change'])
        rolling_var = df['market_pct_change'].rolling(window=14).var()
        df['beta'] = (rolling_cov / rolling_var).fillna(1.0)
        
        # TIME-SERIES PREDICTIVE FORECAST ENGINE
        df['GBM7'] = df['final_winrate'] # Fallback initialization values
        df['OU_Prediction'] = df['final_winrate']
        
        if len(df) > 10:
            # Vectorized local window parameters for forecasting functions
            log_returns = np.log(df['final_winrate'] / df['final_winrate'].shift(1)).dropna()
            mu_daily = log_returns.mean()
            sigma_daily = max(0.0001, log_returns.std())
            
            s0 = df['final_winrate'].iloc[-1]
            Z = np.random.normal(0, 1, 200)
            df.loc[df.index[-1], 'GBM7'] = s0 * np.exp((mu_daily - 0.5 * sigma_daily**2)*7 + sigma_daily * np.sqrt(7) * Z.mean())
            
            try:
                y = df['final_winrate'].values
                X_current = y[:-1].reshape(-1, 1)
                X_next = y[1:]
                lr = LinearRegression().fit(X_current, X_next)
                lambda_hat = max(1e-5, min(0.999, lr.coef_[0]))
                theta_est = -np.log(lambda_hat)
                mu_est = max(0.1, min(0.9, lr.intercept_ / (1 - lambda_hat)))
                df.loc[df.index[-1], 'OU_Prediction'] = np.clip(y[-1] + theta_est * (mu_est - y[-1]), 0.0, 1.0)
            except:
                pass

        latest = df.iloc[-1]
        cursor.execute('''
            INSERT INTO quant_signals (date, card_id, rsi, upper_bollinger, lower_bollinger, sharpe, macd, macd_slope, drawdown, gbm7, oup7, beta, roc5, usage_roc5)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(date, card_id) DO UPDATE SET
                rsi=excluded.rsi, upper_bollinger=excluded.upper_bollinger, lower_bollinger=excluded.lower_bollinger,
                sharpe=excluded.sharpe, macd=excluded.macd, macd_slope=excluded.macd_slope, drawdown=excluded.drawdown,
                gbm7=excluded.gbm7, oup7=excluded.oup7, beta=excluded.beta, roc5=excluded.roc5, usage_roc5=excluded.usage_roc5
        ''', (
            str(latest['date']), str(card_id), float(latest['rsi']), float(latest['upper']), float(latest['lower']), 
            float(latest['sharpe']), float(latest['MACD_Hist']), float(latest['MACD_Hist_Slope']), float(latest['drawdown']), 
            float(latest['GBM7']), float(latest['OU_Prediction']), float(latest['beta']), float(latest['roc5']), float(latest['usage_roc5'])
        ))
    conn.commit()
    cursor.close()
    conn.close()    
    
    
    
    
def CorralationFinding(until_date):
    conn = db.get_connection()
    cursor = conn.cursor()
    # ---------------------------------------------------------
    # CORRALTION FINDING
    # ---------------------------------------------------------
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
        AND date <= '{until_date}'
        ORDER BY date ASC""", conn)
    all_data = all_data.merge(daily_market, on='date')
    all_data['final_winrate'] = (all_data['num_wins'] + confidence/2) / (all_data['num_games'] + confidence) 
    all_data['final_winrate'] *= (0.5 / all_data['market_avg'])
    pivot_df = all_data.pivot(index='date', columns='card_id', values='final_winrate')
    
    # --- ADD THIS LINE TO DROP NANS ---
    # Drops columns (cards) that have ANY NaNs. 
    # Use axis=0 to drop rows (dates) instead, depending on your needs.
    pivot_df_clean = pivot_df.dropna(axis=1) 
    # ----------------------------------
    
    # Generate the correlation matrix from cleaned data
    correlation_matrix = pivot_df_clean.corr()
    
    # Handle potential remaining NaNs in correlation if a card has 0 variance
    correlation_matrix = correlation_matrix.fillna(0)
    
    distance_matrix = 1 - correlation_matrix
    
    # 2. Run K-Means (Grouping into 6 distinct Archetypes/Sectors)
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    # 3. Create a clean DataFrame of what cards belong to what cluster
    card_clusters = pd.DataFrame({
        'Card': correlation_matrix.columns,
        'Cluster_ID': cluster_labels
    })

# Print to see your sub-markets!
    print(card_clusters.sort_values('Cluster_ID'))
    
    valid_cards = [card_id for card_id in cards if card_id in correlation_matrix.columns]
    
    for card_id in valid_cards:
        
        top_2pos = correlation_matrix[card_id].nlargest(3).iloc[1:]
        top_neg = correlation_matrix[card_id].nsmallest(1)
        most_cor = pd.concat([top_2pos, top_neg])
        
        
        top_pos_id = top_2pos.index[0]
        sec_pos_id = top_2pos.index[1]
        neg_id = top_neg.index[0]
        top_pos_val = int(top_2pos.iloc[0])
        sec_pos_val = int(top_2pos.iloc[1])
        neg_val = int(top_neg.iloc[0])
        
        
        pair_strings = []
        
        
        cluster_id = card_clusters.loc[card_clusters['Card'] == card_id, 'Cluster_ID']
        print(cluster_id)
        print(cluster_id.iloc[0])
        cursor.execute('''
            INSERT INTO card_corx (card_id, last_update, top_cor, top_cor_val, next_cor, next_cor_val, neg_cor, neg_cor_val, cluster_num)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(card_id) DO UPDATE SET
                last_update=excluded.last_update, 
                top_cor=excluded.top_cor,
                top_cor_val=excluded.top_cor_val,
                next_cor=excluded.next_cor,
                next_cor_val=excluded.next_cor_val,
                neg_cor=excluded.neg_cor,
                neg_cor_val=excluded.neg_cor_val,
                cluster_num=excluded.cluster_num
                
        ''', (
            card_id, until_date, top_pos_id, top_pos_val, sec_pos_id, sec_pos_val, neg_id, neg_val, int(cluster_id.iloc[0])
        ))
        
    conn.commit()
    cursor.close()
    conn.close()
def time_weighted_average(x):
    alpha = 0.5 
    weights = (1 - alpha) ** np.arange(len(x))[::-1]
    return np.average(x, weights=weights)
def XGBoostpred(last_day):
    xgb_model = XGBClassifier()
    xgb_model.load_model("models/xgb.ubj")
    
    target_date_str = str(last_day).replace('-', '')
    conn = db.get_connection()
    
    daily_market = pd.read_sql("SELECT date, SUM(num_wins)*1.0/SUM(num_games) as market_avg FROM daily_card_stats GROUP BY date", conn)
    all_data = pd.read_sql(f"SELECT date, num_games, num_wins, card_name, card_id FROM daily_card_stats WHERE date >= '20260326' AND date <= '{target_date_str}'", conn)
    all_quant = pd.read_sql(f"SELECT date, rsi, upper_bollinger, lower_bollinger, sharpe, macd, drawdown, gbm7, oup7, macd_slope, beta, roc5, card_id, pair_divergence, sector_lag, neg_divergence, usage_roc5 FROM quant_signals WHERE date >= '20260326' AND date <= '{target_date_str}'", conn)
    
    usage_stat = pd.read_sql(f"""
    SELECT 
        SUM(dcs.num_games) * 1.0 / (
            SELECT SUM(dtg.total_games) * 2 
            FROM daily_total_games dtg
            WHERE dtg.date >= '20260326' AND dtg.date <= '{target_date_str}'
        ) as usage_rate,
        dcs.card_id, dcs.date
    FROM daily_card_stats dcs
    WHERE dcs.date >= '20260326' AND dcs.date <= '{target_date_str}'
    GROUP BY dcs.card_id, dcs.date 
    ORDER BY dcs.date ASC
""", conn)
    conn.close()

    if all_data.empty: 
        return

    # Convert all tracking parameters to plain strings to clean up silent Pandas index drops
    for d in [all_data, all_quant, usage_stat, daily_market]:
        d['date'] = d['date'].astype(str).str.replace('-', '')

    all_data = all_data.merge(daily_market, on='date', how='left')
    all_data = all_data.merge(usage_stat, on=['card_id', 'date'], how='left')
    
    all_data['final_winrate'] = (all_data['num_wins'] + confidence/2) / (all_data['num_games'] + confidence)
    all_data['final_winrate'] *= (0.5 / all_data['market_avg'].replace(0, 0.5))
    
    all_data = all_data.merge(all_quant, on=['card_id', 'date'], how='left')
    all_data = all_data.sort_values(['card_name', 'date']).reset_index(drop=True)
    
    # Generate engineered window series
    gc = all_data.groupby('card_name')
    all_data['tw_winrate_7d'] = gc['final_winrate'].transform(lambda x: x.rolling(7, min_periods=1).apply(time_weighted_average, raw=True))
    all_data['tw_usage_7d'] = gc['usage_rate'].transform(lambda x: x.rolling(7, min_periods=1).apply(time_weighted_average, raw=True))
    all_data['winrate_std_7d'] = gc['final_winrate'].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))
    
    all_data['datetime_obj'] = pd.to_datetime(all_data['date'], format='%Y%m%d')
    all_data['day_sin'] = np.sin(2 * np.pi * all_data['datetime_obj'].dt.dayofweek / 7)
    
    # Calculate uniform calendar metrics
    fm = [m + pd.Timedelta(days=(7 - m.dayofweek) % 7) for m in pd.date_range('2019-07-01', date.today(), freq='MS')]
    fm_df = pd.DataFrame({'first_monday': fm})
    fm_df['year_month'] = fm_df['first_monday'].dt.to_period('M')
    all_data['year_month'] = all_data['datetime_obj'].dt.to_period('M')
    all_data = all_data.merge(fm_df, on='year_month', how='left')
    all_data['days_since_1st_monday'] = (all_data['datetime_obj'] - all_data['first_monday']).dt.days.fillna(0)
    
    target_rows = all_data[all_data['date'] == target_date_str].copy()
    if target_rows.empty: 
        return

    features = ['macd', 'rsi', 'sharpe', 'macd_slope', 'roc5', 'pair_divergence', 'sector_lag', 'neg_divergence', 'usage_roc5', 'day_sin', 'days_since_1st_monday', 'tw_winrate_7d', 'tw_usage_7d', 'winrate_std_7d']
    
    # Ensure every single cell is forced to a raw numeric matrix before passing to load_model parameters
    inference_df = target_rows.fillna(0)
    X_matrix = inference_df[features]
    
    probabilities = xgb_model.predict_proba(X_matrix)[:, 1]
    inference_df['xgb_up'] = probabilities
    
    conn = db.get_connection()
    cursor = conn.cursor()
    data_to_write = list(zip(inference_df['date'], inference_df['card_id'].astype(str), inference_df['xgb_up'].astype(float)))
    cursor.executemany('''
        INSERT INTO quant_signals (date, card_id, xgb_up) VALUES (%s, %s, %s)
        ON CONFLICT(date, card_id) DO UPDATE SET xgb_up=excluded.xgb_up
    ''', data_to_write)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"-> Saved Predictions for: {target_date_str}")
def LSTMpred(last_day):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except ImportError:
        print("TensorFlow is not installed. Skipping LSTM predictions...")
        return

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
    -- Calculate numerator per card, denominator for total
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
    
    pivot_df = all_data.pivot(index='date', columns='card_id', values='final_winrate')
    #pivot_cur = current_data.pivot(index='date', columns='card_id', values='final_winrate')
    
    
        
    all_data['date'] = pd.to_datetime(all_data['date'])
    
    # 2. Extract the day of the week (0=Monday, 6=Sunday)
    all_data['day_of_week'] = all_data['date'].dt.dayofweek
    start_date = date(2019, 7, 1)
    end_date = date.today()
    months = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    first_mondays = []
    for m in months:
        # First day of month is Monday (0) to Sunday (6)
        first_day_of_month = m.dayofweek
        # Calculate days to add to get to the first Monday
        days_to_monday = (7 - first_day_of_month) % 7
        first_mondays.append(m + pd.Timedelta(days=days_to_monday))
    
    fm_df = pd.DataFrame({'first_monday': first_mondays})
    fm_df['year_month'] = fm_df['first_monday'].dt.to_period('M')
    
    # --- 2. Add to your existing all_data DataFrame ---
    # Ensure all_data date is datetime
    all_data['date'] = pd.to_datetime(all_data['date'])
    all_data['year_month'] = all_data['date'].dt.to_period('M')
    
    # Merge first Monday data
    all_data = all_data.merge(fm_df, on='year_month', how='left')
    
    # Calculate days since first Monday
    all_data['days_since_1st_monday'] = (all_data['date'] - all_data['first_monday']).dt.days
    
    # Drop helper columns
    all_data = all_data.drop(columns=['first_monday', 'year_month'])
    
    all_data['day_sin'] = np.sin(2 * np.pi * all_data['day_of_week'] / 7)
    all_data['day_cos'] = np.cos(2 * np.pi * all_data['day_of_week'] / 7)

    # Define your feature columns (including the target winrate)
    feature_cols = [
    'macd', 'rsi', 'sharpe', 'macd_slope', 'roc5', 
    'pair_divergence', 'sector_lag', 'usage_rate', 'neg_divergence',
    'usage_roc5', 'day_sin', 'day_cos', "days_since_1st_monday"
]
    
    # --- 2. MODELING LOOP ---
    for card_id in cards:
        if not os.path.exists('models/' + card_id + ".keras"):
            continue
        model = load_model('models/' + card_id + ".keras")
        scaler_x_path = f'scalers/{card_id}_x.pkl'  # Assumes you saved scalers during training
        scaler_y_path = f'scalers/{card_id}_y.pkl'
        # Filter by card
        target_card_df = all_data[all_data['card_id'] == card_id].copy()
        target_card_df = target_card_df.sort_values('date')
        target_card_df[feature_cols] = target_card_df[feature_cols].replace([np.inf, -np.inf], np.nan)
        # 2. Fill NaN with 0 (or use .ffill() for forward fill, or .mean())
        target_card_df[feature_cols] = target_card_df[feature_cols].fillna(0)
        
        if len(target_card_df) < 20: # Skip cards with insufficient data
            continue
        
       
        scaler_x = joblib.load(scaler_x_path) 
        scaler_y = joblib.load(scaler_y_path)
    
        # Filter and clean
        target_card_df = all_data[all_data['card_id'] == card_id].copy()
        target_card_df = target_card_df.sort_values('date')
        
        window_size = 14
        if len(target_card_df) < (window_size + 1): 
            continue
    
        target_card_df[feature_cols] = target_card_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
        # 2. CRITICAL: Use .transform(), NOT .fit_transform()
        scaled_x = scaler_x.transform(target_card_df[feature_cols])
    
        # 3. Create windows
        X = []
        for i in range(window_size, len(scaled_x)):
            X.append(scaled_x[i - window_size:i, :])
        # Add the window for tomorrow's prediction (using features up to today)
        X.append(scaled_x[-window_size:, :])
        X = np.array(X)
    
        # Generate all predictions
        predictions_scaled = model.predict(X)
        predictions = scaler_y.inverse_transform(predictions_scaled)
    
        # 4. CRITICAL: Get the LATEST prediction (-1) which represents tomorrow's forecast
        latest_prediction = float(predictions[-1][0])

        print(card_id + " | " + str(latest_prediction))
        print(latest_prediction)
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO quant_signals (date, card_id, lstm)
            VALUES (%s, %s, %s)
            ON CONFLICT(date, card_id) DO UPDATE SET
                lstm=excluded.lstm
        ''', (
            last_day, card_id, float(latest_prediction)
        ))
        conn.commit()
        cursor.close()
        conn.close()
        



def CorExacute(target_date):
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # 1. Load Yesterday's Win Rates for ALL cards into a dictionary for instant lookup
    daily_stats = pd.read_sql(f"""
        SELECT card_id, (CAST(num_wins AS FLOAT)/num_games) as winrate, num_games
        FROM daily_card_stats
        WHERE date = '{target_date}'
    """, conn)
    # Create a quick lookup dictionary: {'card_id': 0.51, ...}
    winrate_dict = daily_stats.set_index('card_id')['winrate'].to_dict()
    
    # 2. Load the Correlations Table
    correlations = pd.read_sql("SELECT * FROM card_corx", conn)
    
    # 3. Calculate Sector (Cluster) Momentum
    # Merge daily stats with clusters to get the average winrate of each cluster today
    cluster_df = pd.merge(daily_stats, correlations[['card_id', 'cluster_num']], on='card_id')
    cluster_momentum = cluster_df.groupby('cluster_num')['winrate'].mean().to_dict()
    
    
    
    for _, row in correlations.iterrows():
        card_id = row['card_id']
        current_wr = winrate_dict.get(card_id)
        
        if current_wr is None: continue # Skip if no data for this card today
            
        # --- PAIR DIVERGENCE ---
        top_cor_id = row['top_cor']
        top_cor_wr = winrate_dict.get(top_cor_id)
        top_cor_val = row['top_cor_val'] # The historical confidence (e.g., 0.85)
        
        #+-.15 is signifigent, + means its going down, - menas its going up
        
        pair_divergence = 0
        if top_cor_wr is not None:
            # If correlation is strong (e.g., 0.85), the spread should be close to 0. 
            # We multiply by the correlation value so weak correlations don't trigger false alarms.
            pair_divergence = (current_wr - top_cor_wr) * top_cor_val
            
        # --- NEGATIVE DIVERGENCE ---
        neg_cor_id = row['neg_cor']
        neg_cor_wr = winrate_dict.get(neg_cor_id)
        neg_cor_val = row['neg_cor_val'] # E.g., -0.75
        
        neg_divergence = 0
        if neg_cor_wr is not None:
            # If they are negatively correlated, a high spread is NORMAL.
            # We are looking for when they are moving in the SAME direction unexpectedly.
            neg_divergence = (current_wr + neg_cor_wr - 1.0) * abs(neg_cor_val)

        # --- SECTOR MOMENTUM ---
        cluster = row['cluster_num']
        sector_wr = cluster_momentum.get(cluster, 0.5)
        sector_lag = current_wr - sector_wr # Negative means this card is lagging its sector (Buy signal)
        
        cursor.execute('''
            INSERT INTO quant_signals (date, card_id, pair_divergence, neg_divergence, sector_lag, sector_wr)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT(date, card_id) DO UPDATE SET
                pair_divergence=excluded.pair_divergence, 
                neg_divergence=excluded.neg_divergence,
                sector_lag=excluded.sector_lag,
                sector_wr=excluded.sector_wr
        ''', (
            target_date, card_id, pair_divergence,neg_divergence, sector_lag, sector_wr
        ))
        
    conn.commit()
    cursor.close()
    conn.close()
        
        
        
        
    
cur_day, last_day, _ = TodayAndYesterday()

# Default to running the last 3 days to ensure no gaps, but can be adjusted for full backfill
start_date = datetime.strptime(last_day, "%Y%m%d") - timedelta(days=16)
# Ensure we don't start before the beginning of the data recording period (May 30, 2026)
earliest_date = datetime(2026, 3, 30)
if start_date < earliest_date:
    start_date = earliest_date

end_date = datetime.strptime(last_day, "%Y%m%d")
current_date = start_date

while current_date <= end_date:
    loop_day_str = current_date.strftime("%Y%m%d")
    print(f"\n================ Running Engine Day: {loop_day_str} ================")
    
    run_quant_analysis(loop_day_str)
    CorralationFinding(loop_day_str)
    CorExacute(loop_day_str)
    LSTMpred(loop_day_str)
    XGBoostpred(loop_day_str)
    ClashReportPDF(loop_day_str)
    current_date += timedelta(days=1)
