import db
import os
from gensim.models import Word2Vec

class DeckCorpus:
    def __init__(self, min_games=2):
        self.min_games = min_games

    def __iter__(self):
        conn = db.get_connection()
        cursor = conn.cursor()
        # Select deck array and games_played from PostgreSQL
        cursor.execute("SELECT deck, games_played FROM decks WHERE games_played >= %s", (self.min_games,))
        for row in cursor:
            deck, games_played = row
            # Yield the deck list games_played times to model relative meta frequencies
            for _ in range(games_played):
                yield deck
        cursor.close()
        conn.close()

def train_deck2vec():
    print("Loading deck corpus from PostgreSQL...")
    corpus = DeckCorpus(min_games=2)
    
    print("Training Deck2Vec model via Gensim (Pure CPU)...")
    # vector_size = 32 (32-dimensional embeddings)
    # window = 8 (Looks at the whole 8-card deck context)
    # min_count = 15 (Ignores cards appearing fewer than 15 times overall)
    # sg = 1 (Skip-gram model: captures deep synergies better than CBOW)
    model = Word2Vec(
        sentences=corpus,
        vector_size=32,
        window=8,
        min_count=15,
        sg=1,
        workers=4
    )
    

    model.save("models/deck2vec.model")
    print("Deck2Vec model saved to models/deck2vec.model")

if __name__ == '__main__':
    train_deck2vec()
