import db
import pandas as pd
import numpy as np
import os
from gensim.models import Word2Vec


WIN_CONDITIONS = [
    "Hog Rider", "Balloon", "Giant", "Golem", "Lava Hound", "Royal Giant", 
    "Battle Ram", "Ram Rider", "Wall Breakers", "Goblin Giant", "Elixir Golem", 
    "Skeleton Barrel", "Electro Giant", "X-Bow", "Mortar", "Goblin Drill", 
    "Goblin Barrel", "Graveyard", "Three Musketeers", "Miner", "Sparky", "Rocket", "Royal Hogs"
]
CHAMPIONS = ["Archer Queen", "Golden Knight", "Skeleton King", "Mighty Miner", "Monk", "Little Prince"]
def get_latest_data():
    conn = db.get_connection()
    

    query = """
        SELECT q.card_id, q.xgb_up, q.upper_bollinger, q.lower_bollinger, 
               c.cluster_num, d.card_name, d.num_wins, d.num_games,
               (SELECT SUM(num_wins) * 1.0 / SUM(num_games) 
                FROM daily_card_stats 
                WHERE date = (SELECT MAX(date) FROM quant_signals)) as market_avg
        FROM quant_signals q
        JOIN card_corx c ON q.card_id = c.card_id
        JOIN daily_card_stats d ON q.card_id = d.card_id
        WHERE q.date = (SELECT MAX(date) FROM quant_signals)
        AND d.date = q.date
    """
    preds = pd.read_sql(query, conn)
    conn.close()
    return preds
def is_valid_deck(proposed_deck):
    """Returns True only if the deck obeys Clash Royale's physical rules."""
    
    # Rule 1: Max 2 Evolutions
    evo_count = sum(1 for card in proposed_deck if "Evolution" in card)
    if evo_count > 2:
        return False
        
    # Rule 2: Max 1 Champion (Handle your 'Hero' naming convention or official names)
    champ_count = sum(1 for card in proposed_deck if "Hero" in card or card in CHAMPIONS)
    if champ_count > 2:
        return False
    if champ_count + evo_count > 3:
        return False
        
    # Rule 3: No Base/Evolution Duplicates (e.g., Ice Spirit AND Evolution Ice Spirit)
    # Strip prefixes to check base identities
    base_cards = [card.replace("Evolution ", "").replace("Hero ", "") for card in proposed_deck]
    if len(set(base_cards)) != len(base_cards):
        return False # A duplicate base card exists
        
    return True
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
def optimize_archetype(name, seed_deck, blended_scores, d2v_model, synergy_weight=0.03, max_iterations=20):
    print(f"\n================ OPTIMIZING: {name} ================")
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
            if not(is_valid_deck(proposed_deck)):
                continue
            # 0.1% minimum improvement to prevent endless micro-swaps
            print(f"[{iteration}] Swap: [{weakest_card}] -> [{best_candidate}] (Utility Gain: +{best_improvement:.4f})")
            current_deck = proposed_deck
        else:
            print(f"[{iteration}] Peak Reached: Archetype is mathematically optimized.")
            break # Exit the while loop
            
    print(f"Final Optimized Deck: {', '.join(current_deck)}")
    return current_deck
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
def generate_decks():
    preds = get_latest_data()
    if preds.empty:
        print("Error: No data found in the database. Run your pipeline first.")
        return
        
    blended_scores = calculate_utility_scores(preds)
    
    d2v_path = "models/deck2vec.model"
    if not os.path.exists(d2v_path):
        print(f"Error: Deck2Vec model not found at {d2v_path}.")
        return
        
    d2v_model = Word2Vec.load(d2v_path)
    
    archetypes = {
        "2.6 Hog Cycle": ["Hog Rider", "Musketeer", "Ice Golem", "Ice Spirit", "Skeletons", "The Log", "Fireball", "Cannon"],
        "Pekka Bridge Spam": ["P.E.K.K.A", "Battle Ram", "Electro Wizard", "Royal Ghost", "Bandit", "Zap", "Poison", "Magic Archer"],
        "Golem Beatdown": ["Golem", "Baby Dragon", "Night Witch", "Mega Minion", "Lightning", "The Log", "Lumberjack", "Tornado"],
        "RHogs EvoRR FM Zappies" : ["Arrows", "Electro Spirit","Flying Machine", "Evolution Goblin Cage", "Hero Barbarian Barrel", "Royal Hogs", "Evolution Royal Recruits", "Zappies"]
    }
    
    for name, deck in archetypes.items():
        optimize_archetype(name, deck, blended_scores, d2v_model)

if __name__ == '__main__':
    generate_decks()
