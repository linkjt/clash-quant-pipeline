import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import sqlite3
import psycopg2
from datetime import datetime, timedelta, date
db_path = "/mnt/typeshyt/local_db/clash_data.db"
st.title('Clash Royale Quant')
confidence = 100
def TodayAndYesterday():
    now = datetime.now()
    yest = now - timedelta(days=1)
    sevenago = now - timedelta(days=7)
    
    cur_day = now.strftime("%Y%m%d")
    last_day = yest.strftime("%Y%m%d") # Fixed to actually be yesterday
    seven_ago_str = sevenago.strftime("%Y%m%d") # Formatted for SQL
    
    return cur_day, last_day, seven_ago_str
@st.cache_data
def grabdata():
    cur_day, last_day, seven_ago_str = TodayAndYesterday()
    conn = psycopg2.connect(
    dbname="thehub_db",
    user="linkjt",
    password="qwertYdada0",
    host="192.168.4.60",
    port="5432"
)
    
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
                    SELECT date, rsi, upper_bollinger, lower_bollinger, sharpe, macd, drawdown, gbm7, oup7, macd_slope, beta, roc5, card_id, pair_divergence, sector_lag, neg_divergence, usage_roc5, lstm, rforest_up, xgb_up
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
    all_data['date'] = pd.to_datetime(all_data['date'])
    
    all_data['sma_20'] = all_data.groupby('card_id')['final_winrate'].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    
    # 12-day Exponential Moving Average grouped by card_id
    all_data['ema_12'] = all_data.groupby('card_id')['final_winrate'].transform(
        lambda x: x.ewm(span=12, adjust=False).mean()
    )
    
    return all_data
data = grabdata()
st.subheader('Raw data')
st.write(data)
st.subheader('Card Analysis Dashboards')
card_list = data['card_name'].unique()
for card in card_list:
    card_df = data[data['card_name']==card].sort_values('date')
    with st.expander(f"Card Chart: {card}",expanded=False):
        available_metrics = ['final_winrate', "rsi", "upper_bollinger", "lower_bollinger", "sharpe", "macd", "drawdown", "gbm7", "oup7", "macd_slope", "beta", "roc5", "card_id", "pair_divergence", "sector_lag", "neg_divergence", "usage_roc5", "sma_20", 'ema_12']
        selected_metrics = st.multiselect(
            f"select metric to toggle",
            options=available_metrics,
            default=['final_winrate'],
            key=f"metric_select_{card}")
        if selected_metrics:
            fig = px.line(
                card_df,
                x='date',
                y=selected_metrics,
                title=f"trends for {card}",
                labels={'date':'Date','value':'Metric Value'},
                template='plotly_dark')
            fig.update_traces(mode='lines+markers')
            fig.update_layout(hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("select at least one metric")
"""
st.line_chart(
    data,
    x="date",
    y=["card_id", "final_winrate"],
    color=["#FF0000", "#0000FF"],
)
"""
