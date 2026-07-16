import requests
import random
import json
import time
import psycopg2
import sqlite3
import os
import db
from collections import deque
db_path = "/mnt/typeshyt/local_db/clash_data.db"
API_KEY = os.getenv("Clash_Key")
base_url = "https://api.clashroyale.com/v1"
headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

def urlencode(thing):
    return "%23" + str(thing).lstrip('#')

def gatherMatch(user):
    encodeduser = urlencode(user)
    endpoint = f"{base_url}/players/{encodeduser}/battlelog"
    backoff = 10
    max_backoff = 300
    while True:
        try:
            response = requests.get(endpoint, headers=headers, timeout=10)
            if response.status_code == 429:
                print(f"Rate limited (429) on player {user}. Sleeping for {backoff} seconds...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue
            response.raise_for_status()
            data = response.json()
            return [m for m in data if m.get("gameMode", {}).get("name") == "Ladder"]
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                print(f"Rate limited (429 HTTPError) on player {user}. Sleeping for {backoff} seconds...")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue
            if e.response is not None and e.response.status_code == 403:
                print(f"ERROR 403: Forbidden for player {user}. YOUR API KEY IS LIKELY INVALID OR YOUR IP HAS CHANGED. Please check Supercell developer portal.")
            else:
                print(f"HTTP Error scraping player {user}: {e}")
            return []
        except Exception as e:
            print(f"Error scraping player {user}: {e}")
            return []

def processMatchData(matchs):
    if not matchs:
        return
        
  
    conn = db.get_connection()
    cursor = conn.cursor()
    
    for match in matchs:
        if match != -1 and match is not None:
            try:
                tag1 = match['team'][0]['tag']
                tag2 = match['opponent'][0]['tag']
                sortted_tags = sorted([tag1, tag2])
                match_id = f"{match['battleTime']}_{sortted_tags[0]}"
                match_json = json.dumps(match)
                
                cursor.execute('''
                INSERT INTO raw_matches (match_id, match_data, processed) 
                VALUES (%s, %s, 0)
                ON CONFLICT (match_id) DO NOTHING
            ''', (match_id, match_json))
            except KeyError:
                continue
                
    conn.commit()
    conn.close()

print("Beginning non-blocking collection engine...")
start_player = "#2GVVL2UY0"

player_queue = deque([start_player])
seen_players = {start_player}

while True:
    time.sleep(2.5)
    
    if not player_queue:
        player_queue.append(start_player)
        seen_players = {start_player}
        
    current_player = player_queue.popleft()
    print(f"Scraping Tag: {current_player} | Active Queue Track: {len(player_queue)}")
    
    ladder_matches = gatherMatch(current_player)
    if not ladder_matches:
        continue
        
    processMatchData(ladder_matches)
    
    for match in ladder_matches:
        try:
            opp_tag = match["opponent"][0]["tag"]
            if opp_tag not in seen_players:
                seen_players.add(opp_tag)
                player_queue.append(opp_tag)
                
                if len(seen_players) > 10000:
                    seen_players.clear()
                    seen_players.add(start_player)
        except KeyError:
            continue
