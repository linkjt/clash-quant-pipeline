import db
import pandas as pd
def evaluate_predictions(days_ago=1):
    conn = db.get_connection()
    

    from datetime import datetime, timedelta
    now = datetime.now()
    target_date = (now - timedelta(days=1)).strftime("%Y%m%d") 
    past_date = (now - timedelta(days=1 + days_ago)).strftime("%Y%m%d")
    
    print(f"Evaluating predictions made on {past_date} for the target date {target_date}...\n")
    
  
    predictions_df = pd.read_sql(f"""
        SELECT card_id, xgb_up as predicted_direction
        FROM quant_signals
        WHERE date = '{past_date}'
    """, conn)
    
    past_win = pd.read_sql(f"""
        SELECT card_id, card_name, (CAST(num_wins AS FLOAT)/num_games) as past_winrate
        FROM daily_card_stats
        WHERE date = '{past_date}'
    """, conn)
    

    actuals_df = pd.read_sql(f"""
        SELECT card_id, card_name, (CAST(num_wins AS FLOAT)/num_games) as actual_winrate
        FROM daily_card_stats
        WHERE date = '{target_date}'
    """, conn)
    
    if predictions_df.empty or actuals_df.empty:
        print("Not enough historical data to evaluate yet.")
        return
        
    
    temp_df = pd.merge(predictions_df, past_win[['card_id', 'card_name', 'past_winrate']], on='card_id')
    eval_df = pd.merge(temp_df, actuals_df[['card_id', 'actual_winrate']], on='card_id')
    
    
 
    eval_df['actual_direction'] = (eval_df['actual_winrate'] > eval_df['past_winrate'])
    

    eval_df['correct_prediction'] = ((eval_df['predicted_direction']>.5) == eval_df['actual_direction'])
    
    accuracy = eval_df['correct_prediction'].mean() * 100
    total_cards = len(eval_df)
    
    print(f"--- EVALUATION RESULTS ---")
    print(f"Total Cards Evaluated: {total_cards}")
    print(f"Model Accuracy: {accuracy:.2f}%\n")
    
  
    eval_df['delta'] = eval_df['actual_winrate'] - eval_df['past_winrate']
    print("Top 3 Biggest Moves (Did we catch them?):")
    top_moves = eval_df.reindex(eval_df['delta'].abs().sort_values(ascending=False).index).head(5)
    top_pos_moves = eval_df.reindex(eval_df['delta'].sort_values(ascending=False).index).head(5)
    for _, row in top_moves.iterrows():
        pred_text = "UP" if row['predicted_direction'] >.5 else "DOWN"
        act_text = "UP" if row['actual_direction'] == True else "DOWN"
        match =( "✅" if row['correct_prediction'] == True else "❌")
        print(f"{row['card_name']}: Moved {act_text} ({row['delta']:.3f}). We predicted {pred_text}. {match}")
    print("Top 3 Biggest Positive Moves (Did we catch them?):")
    for _, row in top_pos_moves.iterrows():
        pred_text = "UP" if row['predicted_direction'] >.5 else "DOWN"
        act_text = "UP" if row['actual_direction'] == True else "DOWN"
        match =( "✅" if row['correct_prediction'] == True else "❌")
        print(f"{row['card_name']}: Moved {act_text} ({row['delta']:.3f}). We predicted {pred_text}. {match}")

    conn.close()


evaluate_predictions(days_ago=1)
