
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import psycopg2
import json
import os
import db
from datetime import datetime, timedelta
from gensim.models import Word2Vec

st.set_page_config(page_title="Clash Bot", page_icon="\u2694\ufe0f", layout="wide",
                   initial_sidebar_state="collapsed")
CONFIDENCE = 100
SHORT_HORIZON_DAYS = 1     # xgb_up / lstm: forecast is for stored_date + 1
LONG_HORIZON_DAYS = 7      # gbm7 / oup7: forecast is for stored_date + 7
META_HORIZON_DAYS = 16     # xgb_meta_up: forecast concerns ~stored_date + 16


# ---------------------------------------------------------------------------
# Config loading (thresholds + DB creds) -- fail soft, not hard
# ---------------------------------------------------------------------------
def load_threshold(path, default=0.5):
    try:
        with open(path) as f:
            return json.load(f)["threshold"], True
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return default, False


XGB_THRESHOLD, xgb_thresh_found = .5, True#load_threshold("xgb_threshold.json")
META_THRESHOLD, meta_thresh_found = load_threshold("xgb_meta_threshold.json")



WIN_CONDITIONS = [
    "Hog Rider", "Balloon", "Giant", "Golem", "Lava Hound", "Royal Giant", 
    "Battle Ram", "Ram Rider", "Wall Breakers", "Goblin Giant", "Elixir Golem", 
    "Skeleton Barrel", "Electro Giant", "X-Bow", "Mortar", "Goblin Drill", 
    "Goblin Barrel", "Graveyard", "Three Musketeers", "Miner", "Sparky", "Rocket", "Royal Hogs"
]
CHAMPIONS = ["Archer Queen", "Golden Knight", "Skeleton King", "Mighty Miner", "Monk", "Little Prince"]

# --- CACHED DATA & MODEL LOADING ---
@st.cache_resource
def load_word2vec_model():
    """Loads and caches the Gensim Word2Vec model into memory."""
    try:
        model = Word2Vec.load("models/deck2vec.model")
        return model
    except FileNotFoundError:
        st.warning("Deck2Vec model not found at models/deck2vec.model -- "
                   "deck recommendations are disabled.")
        return None

def calculate_deck_synergy(model, deck, candidate):
    """
    Calculates the average cosine similarity in Deck2Vec space.
    """
    if candidate not in model.wv:
        return 0.0
    
    similarities = []
    for card in deck:
        if card != candidate and card in model.wv:
            similarities.append(model.wv.similarity(candidate, card))
            
    return np.mean(similarities) if similarities else 0.0
def is_valid_deck(proposed_deck):
    """Returns True only if the deck obeys Clash Royale's physical rules."""
    if len(proposed_deck) != 8:
        return False, "Invalid Deck"
    # Rule 1: Max 2 Evolutions
    evo_count = sum(1 for card in proposed_deck if "Evolution" in card)
    if evo_count > 2:
        return False, "Invalid Deck"
        
    # Rule 2
    champ_count = sum(1 for card in proposed_deck if "Hero" in card or card in CHAMPIONS)
    if champ_count > 2:
        return False, "Invalid Deck"
    if champ_count + evo_count > 3:
        return False, "Invalid Deck"
        
    # Rule 3: No Base/Evolution Duplicates (e.g., Ice Spirit AND Evolution Ice Spirit)
    # Strip prefixes to check base identities
    base_cards = [card.replace("Evolution ", "").replace("Hero ", "") for card in proposed_deck]
    if len(set(base_cards)) != len(base_cards):
        return False, "Invalid Deck" # A duplicate base card exists
        
    return True, "Valid Deck"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def yesterday_str():
    return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")

def optimize_archetype(seed_deck, blended_scores, d2v_model, synergy_weight=0.03, max_iterations=20):
    
    print(f"Seed Deck: {', '.join(seed_deck)}")
    
    current_deck = seed_deck.copy()
    iteration = 0
    # HILL-CLIMBING LOOP: Keep optimizing until we hit a mathematical peak
    while iteration < max_iterations:
        iteration += 1
        
        # 1. Identify weakest non-wincon card
        deck_scores = {card: blended_scores.get(card, 0.5) for card in current_deck}
        non_wincons = [c for c in current_deck if c not in WIN_CONDITIONS]
        
        if not non_wincons:
            non_wincons = current_deck # Fallback
            
        weakest_card = min(non_wincons, key=lambda c: deck_scores.get(c, 0.5))
        weakest_score = deck_scores[weakest_card]
        
        remaining_deck = [c for c in current_deck if c != weakest_card]
        weakest_synergy = calculate_deck_synergy(d2v_model, remaining_deck, weakest_card)
        weakest_utility = weakest_score + (synergy_weight * weakest_synergy)
        
        # 2. Query candidates
        if weakest_card not in d2v_model.wv:
            print(f"[{iteration}] Halt: '{weakest_card}' lacks embedding data.")
            break
            
        similar_candidates = d2v_model.wv.most_similar(weakest_card, topn=20)
        
        best_candidate = None
        best_utility = weakest_utility
        best_improvement = 0
        
        # 3. Score and Filter candidates
        for candidate, similarity in similar_candidates:
            # FILTER: Prevent duplicate cards and prevent adding a NEW win condition
            if candidate in current_deck:
                continue
            if candidate in WIN_CONDITIONS:
                continue 
                
            candidate_score = blended_scores.get(candidate, 0.45)
            candidate_synergy = calculate_deck_synergy(d2v_model, remaining_deck, candidate)
            candidate_utility = candidate_score + (synergy_weight * candidate_synergy)
            
            improvement = candidate_utility - weakest_utility
            
            if candidate_utility > best_utility:
                best_utility = candidate_utility
                best_candidate = candidate
                best_improvement = improvement
                
        # 4. Perform Swap or Break
        if best_candidate and best_improvement > 0.002:
            proposed_deck = remaining_deck + [best_candidate]
            val, mess= is_valid_deck(proposed_deck)
            if not(val):
                continue
            # 0.1% minimum improvement to prevent endless micro-swaps
            print(f"[{iteration}] Swap: [{weakest_card}] -> [{best_candidate}] (Utility Gain: +{best_improvement:.4f})")
            current_deck = proposed_deck
        else:
            print(f"[{iteration}] Peak Reached: Archetype is mathematically optimized.")
            break # Exit the while loop
            
    print(f"Final Optimized Deck: {', '.join(current_deck)}")
    return current_deck
def calculate_utility_scores(df, confidence=100):
    if df.empty:
        return {}
    
    # 1. Bayesian Winrate calculation
    df['market_avg'] = df['market_avg'].replace(0, 0.5)
    df['final_winrate'] = (df['num_wins'] + confidence / 2) / (df['num_games'] + confidence)
    df['final_winrate'] *= (0.5 / df['market_avg'])
    
    # 2. Extract Card-Specific Volatility (STD20) from Bollinger Bands
    # Standard deviation is exactly 1/4th of the distance between the upper and lower bands
    df['card_volatility'] = (df['upper_bollinger'] - df['lower_bollinger']) / 4.0
    
    # Safety catch: If volatility is 0 or NaN (brand new card), give it a baseline 1.5% variance
    df['card_volatility'] = df['card_volatility'].fillna(0.015).replace(0, 0.015)
    
    # 3. Dynamic Blended Score
    # (xgb_up - 0.5) * 2 scales the 0 to 1 probability into a -1.0 to 1.0 multiplier
    # We multiply that by the card's actual historical variance constraint
    df['momentum_impact'] = ((df['xgb_up'] - 0.5) * 2) * df['card_volatility']
    
    df['blended_score'] = df['final_winrate'] + df['momentum_impact']
    
    return df.set_index('card_name')['blended_score'].to_dict()
@st.cache_data(ttl=3600)
def load_data(confidence_val):
    last_day = yesterday_str()
    conn = db.get_connection()
    try:
        daily_market = pd.read_sql("""
            SELECT date, SUM(num_wins) * 1.0 / SUM(num_games) as market_avg
            FROM daily_card_stats GROUP BY date ORDER BY date ASC
        """, conn)

        all_data = pd.read_sql("""
            SELECT date, (CAST(num_wins AS FLOAT)/num_games) as win_rate,
                   num_games, num_wins, card_name, card_id
            FROM daily_card_stats
            WHERE date >= %(start)s AND date <= %(end)s
            ORDER BY date ASC
        """, conn, params={"start": "20260326", "end": last_day})

        all_quant = pd.read_sql("""
            SELECT date, rsi, upper_bollinger, lower_bollinger, sharpe, macd, drawdown,
                   gbm7, oup7, macd_slope, beta, roc5, card_id, pair_divergence, sector_lag,
                   neg_divergence, usage_roc5, lstm, rforest_up, xgb_up, pair_zscore, xgb_meta_up
            FROM quant_signals
            WHERE date >= %(start)s AND date <= %(end)s
        """, conn, params={"start": "20260326", "end": last_day})

        # BUGFIX: same non-correlated usage denominator bug as the training
        # scripts had -- this version correlates the denominator to each
        # row's own date instead of reusing one constant for the whole
        # multi-month window.
        usage_stat = pd.read_sql("""
            SELECT dcs.card_id, dcs.date,
                   SUM(dcs.num_games) * 1.0 / (
                       SELECT SUM(dtg.total_games) * 2
                       FROM daily_total_games dtg
                       WHERE dtg.date = dcs.date
                   ) as usage_rate
            FROM daily_card_stats dcs
            WHERE dcs.date >= %(start)s AND dcs.date <= %(end)s
            GROUP BY dcs.card_id, dcs.date
        """, conn, params={"start": "20260326", "end": last_day})
    finally:
        conn.close()

    all_data = all_data.merge(daily_market, on='date', how='left')
    all_data = all_data.merge(usage_stat, on=['card_id', 'date'], how='left')
    all_data['market_avg'] = all_data['market_avg'].replace(0, np.nan).fillna(0.5)
    all_data['final_winrate'] = (all_data['num_wins'] + confidence_val / 2) / (all_data['num_games'] + confidence_val)
    all_data['final_winrate'] *= (0.5 / all_data['market_avg'])

    all_data = all_data.merge(all_quant, on=['card_id', 'date'], how='left')
    all_data['date'] = pd.to_datetime(all_data['date'])
    all_data = all_data.sort_values(['card_id', 'date']).reset_index(drop=True)

    all_data['sma_20'] = all_data.groupby('card_id')['final_winrate'].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    all_data['ema_12'] = all_data.groupby('card_id')['final_winrate'].transform(
        lambda x: x.ewm(span=12, adjust=False).mean())

    # --- Time-shift every forecast column to its actual target date -------
    # Each of these columns is stored under the date whose DATA was used to
    # make the prediction, not the date being predicted about. Build a
    # parallel "target date" column for each so charts/tables can plot or
    # filter on when the forecast is actually FOR.
    all_data['lstm_target_date'] = all_data['date'] + pd.Timedelta(days=SHORT_HORIZON_DAYS)
    all_data['gbm7_target_date'] = all_data['date'] + pd.Timedelta(days=LONG_HORIZON_DAYS)
    all_data['oup7_target_date'] = all_data['date'] + pd.Timedelta(days=LONG_HORIZON_DAYS)
    all_data['xgb_up_target_date'] = all_data['date'] + pd.Timedelta(days=SHORT_HORIZON_DAYS)
    all_data['xgb_meta_target_date'] = all_data['date'] + pd.Timedelta(days=META_HORIZON_DAYS)

    return all_data


data = load_data(CONFIDENCE)

latest_date = data['date'].max()
latest_df = data[data['date'] == latest_date].copy()
latest_df['is_predicted_meta'] = latest_df['xgb_meta_up'] >= META_THRESHOLD


# ---------------------------------------------------------------------------
# Header + explainer
# ---------------------------------------------------------------------------
st.title("\U0001F6E1\ufe0f Clash Royale Quantitative Meta Engine")

if not xgb_thresh_found or not meta_thresh_found:
    st.warning(
        "Couldn't find a saved decision threshold for "
        + (", ".join(filter(None, [
            None if xgb_thresh_found else "the short-term model (xgb_threshold.json)",
            None if meta_thresh_found else "the meta model (xgb_meta_threshold.json)",
        ])))
        + " -- falling back to 0.5. Run the training scripts to generate these; "
          "until then, 'predicted riser' / 'predicted meta' counts below use an untuned cutoff."
    )

with st.expander("\u2139\ufe0f What is this and how do I read it?", expanded=False):
    st.markdown(f"""
This dashboard applies stock-market-style quantitative analysis to Clash Royale
card win rates. Every card's daily win rate is treated like a security's price:
it gets technical indicators (RSI, Bollinger Bands, MACD, Sharpe ratio, etc.)
and a set of forecasting models trained on top of them.

**Why the latest win rate here is yesterday's.** Today's win rate is not
actually knowable yet. Some of today's games are in, but not enough of them to
be worth anything until the day closes out. So instead of guessing from a thin
sample, the models use the past several days of movement to work out whether
today will be higher or lower than yesterday, and to estimate what the number
itself will land on.

**Short-term XGBoost (classification)** predicts whether a card's win rate today
will be higher than yesterday's.

**Meta XGBoost (classification)** predicts whether a card's win rate will be in
the top movers about {META_HORIZON_DAYS} days out, give or take a couple of days
on either side. That horizon is what lets you look at the start of a season and
see roughly where it ends up, or look at the end of one and see what the next
one opens like.

**LSTM (regression)** predicts the actual numeric win rate a card will have
today, from the past 7 days of indicators.

On top of the win rate models there is a deck model. It has seen millions of
decks and learned how cards relate to each other functionally, by what they get
played alongside. Combined with the models above, it runs a daily deck
optimization: find the cards in your deck with the weakest expected win rate
today and swap them for cards with a better expected value that still do the
same job in the deck.
""")

st.markdown(f"**Latest Complete Day:** `{latest_date.strftime('%Y-%m-%d')}`  "
            f"&nbsp;|&nbsp; **Today's forecast date:** `{(latest_date + timedelta(days=1)).strftime('%Y-%m-%d')}`")

# ---------------------------------------------------------------------------
# Today's Key Stats
# ---------------------------------------------------------------------------
st.subheader("Today's Key Stats")
col1, col2, col3, col4 = st.columns(4)

with col1:
    risers_count = int((latest_df['xgb_up'] > XGB_THRESHOLD).sum())
    st.metric("Predicted Risers (Today)", f"{risers_count} cards",
              help=f"Cards with xgb_up above the tuned threshold ({XGB_THRESHOLD:.2f}), "
                   f"forecasting {(latest_date + timedelta(days=1)).strftime('%b %d')}.")

with col2:
    meta_count = int(latest_df['is_predicted_meta'].sum())
    st.metric(f"Predicted Next-Meta Cards (\u2265{META_THRESHOLD:.2f})", f"{meta_count} cards",
              help=f"Forecasting the top-15%-mover window around "
                   f"{(latest_date + timedelta(days=META_HORIZON_DAYS)).strftime('%b %d')}.")

with col3:
    avg_lstm = latest_df['lstm'].mean()
    st.metric(f"Avg Predicted Win Rate ({(latest_date + timedelta(days=1)).strftime('%b %d')})",
              f"{avg_lstm:.2%}" if pd.notna(avg_lstm) else "N/A")

with col4:
    if not latest_df.empty and latest_df['lstm'].notna().any():
        top_row = latest_df.sort_values('lstm', ascending=False).iloc[0]
        st.metric("Top Predicted Card (Today)", top_row['card_name'],
                  delta=f"{top_row['lstm']:.1%} projected WR")
    else:
        st.metric("Top Predicted Card (Today)", "N/A")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "\U0001F52E Today & Meta Predictions",
    "\U0001F4C8 Card Technical Analysis",
    "\U0001F5C4\ufe0f Raw Data Feed",
    "\U0001F4CB Model Reports",
    "\U0001F0CF Deck Optimizations"
])

# --- TAB 1: PREDICTIONS DASHBOARD ------------------------------------------
with tab1:
    st.subheader("Daily ML Forecasts & Next-Meta Predictions")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown(f"### \U0001F4CA Forecast Table \u2014 predictions for "
                     f"{(latest_date + timedelta(days=1)).strftime('%b %d')} "
                     f"(short-term) / {(latest_date + timedelta(days=META_HORIZON_DAYS)).strftime('%b %d')} (meta)")

        display_df = latest_df[[
            'card_name', 'final_winrate', 'lstm', 'xgb_up', 'xgb_meta_up', 'rsi', 'sharpe'
        ]].copy()
        display_df.columns = [
            'Card Name', 'Current WR', 'LSTM Projected WR (Today)',
            'XGB Up Prob (Today)', 'Meta Prob (~16d out)', 'RSI', 'Sharpe Ratio',
        ]

        sort_by = st.selectbox("Sort Table By:",
                                ['LSTM Projected WR (Today)', 'Meta Prob (~16d out)',
                                 'Current WR', 'Sharpe Ratio', 'XGB Up Prob (Today)'], index=0)
        display_df = display_df.sort_values(sort_by, ascending=False)

        st.dataframe(
            display_df.style.format({
                'Current WR': '{:.2%}',
                'LSTM Projected WR (Today)': '{:.2%}',
                'XGB Up Prob (Today)': '{:.2f}',
                'Meta Prob (~16d out)': '{:.2%}',
                'RSI': '{:.1f}',
                'Sharpe Ratio': '{:.2f}',
                
            }, na_rep="-").background_gradient(
                subset=['LSTM Projected WR (Today)', 'Meta Prob (~16d out)'],
                cmap='Greens', vmin=0, vmax=1).background_gradient(
                subset=['XGB Up Prob (Today)'],
                cmap='Greens', vmin=.5, vmax=1).background_gradient(
                subset=['XGB Up Prob (Today)'], 
                cmap='RdYlGn',  # Diverging colormap: Red for low, Green for high
                vmin=0, 
                vmax=1
            ),
                
            use_container_width=True, height=450,
        )

    with col_b:
        st.markdown("### \U0001F525 Top Projected Meta Cards")
        st.caption(f"Cards passing the `{META_THRESHOLD:.2f}` threshold for the "
                    f"top-15%-mover window around "
                    f"{(latest_date + timedelta(days=META_HORIZON_DAYS)).strftime('%b %d')}")

        meta_cards = latest_df[latest_df['is_predicted_meta']].sort_values('xgb_meta_up', ascending=False)
        if not meta_cards.empty:
            for _, row in meta_cards.iterrows():
                st.write(f"**{row['card_name']}**")
                st.progress(float(row['xgb_meta_up']))
                lstm_str = f"{row['lstm']:.1%}" if pd.notna(row['lstm']) else "N/A"
                st.caption(f"Meta Prob: {row['xgb_meta_up']:.1%} | Today's LSTM WR: {lstm_str}")
        else:
            st.info("No cards currently meet or exceed the selected meta threshold.")

# --- TAB 2: INDIVIDUAL CARD DEEP DIVE --------------------------------------
with tab2:
    st.subheader("Technical & Quantitative Card Analytics")
    selected_card = st.selectbox("Select Card to Analyze:", sorted(data['card_name'].unique()))
    card_df = data[data['card_name'] == selected_card].sort_values('date')

    col_c1, col_c2 = st.columns([3, 1])

    with col_c2:
        st.markdown("#### Chart Indicators")
        show_bollinger = st.checkbox("Show Bollinger Bands", value=True)
        show_lstm = st.checkbox("Show LSTM Forecast", value=True)
        show_sma = st.checkbox("Show SMA 20", value=True)
        show_gbm_oup = st.checkbox("Show 7-Day GBM/OU Forecast", value=True)

        latest_card_row = card_df.iloc[-1]
        st.markdown("---")
        st.markdown("#### Latest Metrics")
        rsi_val = latest_card_row.get('rsi', pd.NA)
        sharpe_val = latest_card_row.get('sharpe', pd.NA)
        pair_div_val = latest_card_row.get('pair_divergence', pd.NA)
        xgb_val = latest_card_row.get('xgb_up', pd.NA)

        st.write(f"**RSI:** {rsi_val:.1f}" if pd.notna(rsi_val) else "**RSI:** N/A")
        st.write(f"**Sharpe Ratio:** {sharpe_val:.2f}" if pd.notna(sharpe_val) else "**Sharpe Ratio:** N/A")
        st.write(f"**Pair Divergence:** {pair_div_val:.4f}" if pd.notna(pair_div_val) else "**Pair Divergence:** N/A")

        # BUGFIX: xgb_up is a probability in [0, 1], not a 0/1 label.
        # `xgb_val == 1` was true almost never, so this badge showed
        # "DOWN" for nearly every card regardless of the real prediction --
        # compare against the tuned threshold instead, consistent with how
        # the top-of-page metric already did it.
        if pd.notna(xgb_val):
            direction = "\U0001F7E2 UP" if xgb_val >= XGB_THRESHOLD else "\U0001F534 DOWN"
            st.write(f"**XGB Signal (Today):** {direction} ({xgb_val:.1%})")
        else:
            st.write("**XGB Signal (Today):** N/A")

    with col_c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=card_df['date'], y=card_df['final_winrate'],
            mode='lines+markers', name='Actual Win Rate',
            line=dict(color='#00F0FF', width=2),
        ))

        if show_lstm and 'lstm' in card_df.columns:
            # Shifted to lstm_target_date -- the value stored under date t
            # is the model's forecast FOR t+1, made using data through t.
            fig.add_trace(go.Scatter(
                x=card_df['lstm_target_date'], y=card_df['lstm'],
                mode='lines', name='LSTM Forecast (shifted +1d to target date)',
                line=dict(color='#FF007F', width=2, dash='dash'),
            ))

        if show_gbm_oup:
            if 'gbm7' in card_df.columns:
                fig.add_trace(go.Scatter(
                    x=card_df['gbm7_target_date'], y=card_df['gbm7'],
                    mode='lines', name='GBM 7-Day Forecast (shifted +7d)',
                    line=dict(color='#7CFC00', width=1.5, dash='dot'),
                ))
            if 'oup7' in card_df.columns:
                fig.add_trace(go.Scatter(
                    x=card_df['oup7_target_date'], y=card_df['oup7'],
                    mode='lines', name='OU 7-Day Forecast (shifted +7d)',
                    line=dict(color='#FFA500', width=1.5, dash='dot'),
                ))

        if show_bollinger and 'upper_bollinger' in card_df.columns:
            fig.add_trace(go.Scatter(
                x=card_df['date'], y=card_df['upper_bollinger'],
                mode='lines', name='Upper Bollinger',
                line=dict(color='rgba(255,255,255,0.3)', width=1),
            ))
            fig.add_trace(go.Scatter(
                x=card_df['date'], y=card_df['lower_bollinger'],
                mode='lines', name='Lower Bollinger',
                line=dict(color='rgba(255,255,255,0.3)', width=1),
                fill='tonexty', fillcolor='rgba(255,255,255,0.05)',
            ))

        if show_sma and 'sma_20' in card_df.columns:
            fig.add_trace(go.Scatter(
                x=card_df['date'], y=card_df['sma_20'],
                mode='lines', name='SMA 20',
                line=dict(color='#FFD700', width=1.5),
            ))

        # Vertical marker at "latest complete day" so it's visually obvious
        # where actual data ends and forecast-only territory begins.
        fig.add_vline(x=latest_date, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                       annotation_text="latest complete day", annotation_position="top")

        fig.update_layout(
            title=f"{selected_card} Win Rate Trend & Forecasts (forecast series plotted at their target date, not the date they were computed)",
            template='plotly_dark', xaxis_title="Date", yaxis_title="Win Rate",
            hovermode="x unified", height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: RAW DATA --------------------------------------------------------
with tab3:
    st.subheader("Database Query Output")
    st.write(f"Total Rows: `{len(data)}` across `{data['card_name'].nunique()}` unique cards, "
             f"`{data['date'].nunique()}` days.")
    st.caption("Forecast columns (lstm, gbm7, oup7, xgb_up, xgb_meta_up) are shown here under "
               "their STORAGE date, i.e. the date whose data produced them -- not their target "
               "date. Use the `*_target_date` columns to see what date each one is actually "
               "forecasting.")
    st.dataframe(data, use_container_width=True)

# --- TAB 4: MODEL REPORTS ---------------------------------------------------
with tab4:
    st.subheader("Training Reports")
    st.caption("Generated by the training scripts each time they run -- not live queries, "
               "so these reflect whenever the models were last retrained.")

    report_col1, report_col2 = st.columns(2)

    def show_report(col, title, report_path, cm_path, roc_path):
        with col:
            st.markdown(f"#### {title}")
            if os.path.exists(report_path):
                with open(report_path) as f:
                    st.code(f.read(), language=None)
            else:
                st.info(f"No report found at `{report_path}` yet -- run the training script.")
            img_cols = st.columns(2)
            if os.path.exists(cm_path):
                img_cols[0].image(cm_path, caption="Confusion Matrix")
            if os.path.exists(roc_path):
                img_cols[1].image(roc_path, caption="ROC Curve")

    show_report(report_col1, "Short-Term XGBoost Model",
                "model_reports/xgb_report.txt",
                "model_reports/confusion_matrix.png",
                "model_reports/roc_curve.png")

    show_report(report_col2, "Meta (16-Day) XGBoost Model",
                "model_reports/xgb_meta_report.txt",
                "model_reports/confusion_matrix_meta.png",
                "model_reports/roc_curve_meta.png")

    st.markdown("#### LSTM Training Log")
    lstm_log_path = "model_reports/LSTM.log"
    if os.path.exists(lstm_log_path):
        with open(lstm_log_path) as f:
            lines = f.readlines()
        st.code("".join(lines[-200:]), language=None)
        st.caption(f"Showing the last {min(200, len(lines))} lines of {len(lines)} total.")
    else:
        st.info(f"No LSTM log found at `{lstm_log_path}` yet.")
with tab5:
    last_day = yesterday_str()
    st.write("Build a deck, check it against the slot rules, and find functionally similar "
             "cards using the Deck2Vec embeddings.")
    
    # Load resources
    w2v_model = load_word2vec_model()
    df_cards = data
    all_card_names = list(set(df_cards["card_name"].tolist()))
    
    # Sidebar: User Inputs
    st.sidebar.header("🔧 Current Deck Selection")
    selected_cards = st.sidebar.multiselect(
        "Select up to 8 cards for your deck:",
        options=all_card_names,
        max_selections=8
    )
    
    # Main Dashboard Grid
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Current Deck Breakdown (Most Recent Stats)")
        # is_valid_deck returns a (bool, message) tuple -- unpack it, because the
        # tuple itself is always truthy and would make the check below useless.
        is_valid, message = is_valid_deck(selected_cards)
        if selected_cards:
            current_deck_stats = df_cards[df_cards["card_name"].isin(selected_cards)]
            current_deck_stats = current_deck_stats[current_deck_stats['date']==last_day]
            st.dataframe(current_deck_stats[["card_name", "win_rate", "xgb_up", "lstm", "xgb_meta_up"]], use_container_width=True)

            if is_valid:
                st.success(message)
            else:
                st.error(message)
        else:
            st.info("Pick 8 valid cards in the sidebar to begin optimization.")
    
    with col2:
        st.subheader("🔮 Deck2Vec Synergistic (Functional) Recommendations")
        if selected_cards:
            if w2v_model is not None:
                # Filter cards to only those present in the Word2Vec vocabulary
                valid_vocab_cards = [card for card in selected_cards if card in w2v_model.wv]
                
                if valid_vocab_cards:
                    # Find most similar cards according to the neural network
                    recommendations = w2v_model.wv.most_similar(positive=valid_vocab_cards, topn=10)
                    
                    rec_df = pd.DataFrame(recommendations, columns=["Card Name", "Synergy Score"])
                    # Filter out cards already in the deck
                    rec_df = rec_df[~rec_df["Card Name"].isin(selected_cards)]
                    
                    st.write("Cards that functionally pair well with your choices:")
                    st.dataframe(rec_df.style.format({"Synergy Score": "{:.2%}"}), use_container_width=True)
                else:
                    st.warning("Selected cards aren't indexed in the current Deck2Vec vocabulary.")
            else:
                st.info("Deck2Vec model failed to load, so recommendations are unavailable.")
        else:
            st.write("Add cards (via sidebar) to see what aligns with your deck composition.")
    st.subheader("\u2699\ufe0f Daily Deck Optimization")
    if selected_cards:
        if w2v_model is not None:
            # Filter cards to only those present in the Word2Vec vocabulary
            valid_vocab_cards = [card for card in selected_cards if card in w2v_model.wv]
            
            if valid_vocab_cards:
                df_today = df_cards[df_cards["date"]==last_day]
                blended_scores = calculate_utility_scores(df_today)
                optimized_deck = optimize_archetype(valid_vocab_cards, blended_scores, w2v_model, synergy_weight=0.03, max_iterations=20)
                # Find most similar cards according to the neural network
                rec_df = pd.DataFrame(optimized_deck, columns=["Card Name"])
                # Filter out cards already in the deck
                rec_df = rec_df[~rec_df["Card Name"].isin(selected_cards)]
                
                st.write("Optimal swaps for your deck today:")
                st.write(optimized_deck)
            else:
                st.warning("Selected cards aren't indexed in the current Deck2Vec vocabulary.")
        else:
            st.info("Deck2Vec model failed to load, so recommendations are unavailable.")
    else:
        st.write("Add cards in the sidebar to see today's suggested swaps.")
