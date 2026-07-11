import json
from datetime import datetime, timedelta, date
import time
import psycopg2
import hashlib
import db



# --- 2. INITIALIZE TABLES ---


def DateToWeek(dayofbattle):
    year = dayofbattle[0:4]
    month = dayofbattle[4:6]
    day = dayofbattle[6:8]
    date_obj = datetime(int(year), int(month), int(day))
    weekstart = date_obj - timedelta(days=date_obj.weekday())
    return weekstart.strftime("%Y-%m-%d")

def CreateAvgAllTimeSQL():
    conn = db.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT SUM(total_games) FROM daily_total_games")
    total_games_all_time = cursor.fetchone()[0] or 1

    cursor.execute('''
        INSERT INTO card_metadata (card_id, avg_winrate, avg_userate)
        SELECT 
            card_id,
            SUM(num_wins * 1.0) / SUM(num_games) as avg_winrate,
            SUM(num_games * 1.0) / %s as avg_userate
        FROM daily_card_stats
        GROUP BY card_id
        ON CONFLICT(card_id) DO UPDATE SET
            avg_winrate = excluded.avg_winrate,
            avg_userate = excluded.avg_userate
    ''', (total_games_all_time,))

    conn.commit()
    conn.close()

def AnalyzeMatchesSQL(cursor, m_id, match):
    if not match:
        return False

    date_str = match["battleTime"][:8] 
    team_win = 1 if match["team"][0].get("trophyChange", 0) > 0 else 0
    opponent_win = 1 if match["opponent"][0].get("trophyChange", 0) > 0 else 0

    cursor.execute('''
        INSERT INTO daily_total_games (date, total_games) 
        VALUES (%s, 1) 
        ON CONFLICT (date) 
        DO UPDATE SET total_games = daily_total_games.total_games + 1
    ''', (date_str,))

    all_participants = [
        (match["team"][0]["cards"], team_win),
        (match["opponent"][0]["cards"], opponent_win)
    ]

    for cards, is_win in all_participants:
        seen_cards = set()
        deck = []
        
        for card in cards:
            card_id = str(card["id"])
            if card_id in seen_cards:
                continue
            seen_cards.add(card_id)

            card_name = str(card["name"])
            
            evo = card.get("evolutionLevel", 0)
            if evo == 1: 
                card_id = "E" + card_id
                card_name = "Evolution " + card_name
            elif evo == 2: 
                card_id = "H" + card_id
                card_name = "Hero " + card_name
                
            deck.append(card_name)
            
            # Insert standard daily stats
            cursor.execute('''
                INSERT INTO daily_card_stats (date, card_id, card_name, num_wins, num_games)
                VALUES (%s, %s, %s, %s, 1)
                ON CONFLICT(date, card_id) DO UPDATE SET
                    num_wins = daily_card_stats.num_wins + %s,
                    num_games = daily_card_stats.num_games + 1
            ''', (date_str, card_id, card_name, is_win, is_win))
            
        # Sort, Hash, and Insert the complete Deck
        deck.sort()
        stringdeck = "|".join(deck).encode('utf-8')
        deckhash = hashlib.md5(stringdeck).hexdigest()
        
        cursor.execute('''
    INSERT INTO decks (deckhash, deck, games_played) 
    VALUES (%s, %s, 1) 
    ON CONFLICT(deckhash) 
    DO UPDATE SET games_played = decks.games_played + 1
''', (deckhash, deck))

            
    # Mark as completely processed (Stats + Decks)
    cursor.execute("UPDATE raw_matches SET processed = 3 WHERE match_id = %s", (m_id,))
    return True

# --- BACKFILL TOOL: Extracts decks from previously processed matches ---
def deck_fix(cursor, m_id, match):
    if not match:
        return False

    all_participants = [
        (match["team"][0]["cards"]),
        (match["opponent"][0]["cards"])
    ]

    for cards in all_participants:
        seen_cards = set()
        deck = []
        for card in cards:
            card_id = str(card["id"])
            if card_id in seen_cards:
                continue
            seen_cards.add(card_id)

            card_name = str(card["name"])
            evo = card.get("evolutionLevel", 0)
            if evo == 1: 
                card_name = "Evolution " + card_name
            elif evo == 2: 
                card_name = "Hero " + card_name
                
            deck.append(card_name)
            
        deck.sort()
        stringdeck = "|".join(deck).encode('utf-8')
        deckhash = hashlib.md5(stringdeck).hexdigest()
        
        cursor.execute('''
    INSERT INTO decks (deckhash, deck, games_played) 
    VALUES (%s, %s, 1) 
    ON CONFLICT(deckhash) 
    DO UPDATE SET games_played = decks.games_played + 1
''', (deckhash, deck))
            
    # Upgrade status to 2 so it doesn't get scanned again
    cursor.execute("UPDATE raw_matches SET processed = 3 WHERE match_id = %s", (m_id,))
    return True
    
def deck_fix_cycle():
    while True:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Grab matches that have stats (1) but NO decks extracted yet
        cursor.execute("SELECT match_id, match_data FROM raw_matches WHERE processed = 2 OR processed = 1 LIMIT 100")
        unprocessed = cursor.fetchall()
        
        if not unprocessed:
            conn.close()
            print("All historical matches have been backfilled with decks. You can stop this cycle.")
            time.sleep(15) 
            continue

        for m_id, m_json in unprocessed:
            try:
                match = json.loads(m_json)
                deck_fix(cursor, m_id, match) # Call the fix, NOT the main analyzer
            except Exception as e:
                print(f"Error processing backfill for {m_id}: {e}")
                cursor.execute("UPDATE raw_matches SET processed = -1 WHERE match_id = %s", (m_id,))

        conn.commit()
        conn.close()

def cleanup_old_data(days_to_keep=30):
    conn = get_db_connection()
    target_date = (datetime.now() - timedelta(days=days_to_keep)).strftime("%Y%m%d")
    
    with conn.cursor() as cur:
        # Added mandatory parentheses around the OR statement
        cur.execute("""
            DELETE FROM raw_matches 
            WHERE (processed = 1 OR processed = 2  OR processed = 3)
              AND substring(match_id FROM 1 FOR 8) < %s
        """, (target_date,))
    
    conn.commit()
    conn.close()

def run_analysis_cycle():
    while True:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Grab entirely raw matches
        cursor.execute("SELECT match_id, match_data FROM raw_matches WHERE processed = 0 LIMIT 100")
        unprocessed = cursor.fetchall()
        
        if not unprocessed:
            conn.close()
            cleanup_old_data(30)
            time.sleep(15) 
            continue

        for m_id, m_json in unprocessed:
            try:
                match = json.loads(m_json)
                AnalyzeMatchesSQL(cursor, m_id, match)
            except Exception as e:
                print(f"Error processing match {m_id}: {e}")
                cursor.execute("UPDATE raw_matches SET processed = -1 WHERE match_id = %s", (m_id,))

        conn.commit()
        conn.close()

cleanup_old_data()

# Only run deck_fix_cycle() if you have historical data you want to extract decks from.
#deck_fix_cycle() 
run_analysis_cycle()
