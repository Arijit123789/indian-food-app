from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def preprocess_features(df):
    # Basic feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X_ing = vectorizer.fit_transform(df['ingredients'])
    return X_ing, vectorizer

def train_classifier(df, X):
    le = pd.factorize(df['state'])[0]
    X_train, X_test, y_train, y_test = train_test_split(X, le, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, acc

def recommend_dishes(df, query, vectorizer, top_n=5):
    query_vec = vectorizer.transform([query])
    all_vecs = vectorizer.transform(df['ingredients'])
    sims = cosine_similarity(query_vec, all_vecs).flatten()
    idx = sims.argsort()[-top_n:][::-1]
    return df.iloc[idx][['name', 'state', 'ingredients']]
