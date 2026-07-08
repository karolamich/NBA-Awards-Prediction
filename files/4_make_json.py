import pandas as pd
import joblib
import json
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

df_allnba_raw = pd.read_csv(os.path.join(PROJECT_DIR, 'data/final/final_model_df_allnba.csv'))
df_rookie_raw = pd.read_csv(os.path.join(PROJECT_DIR, 'data/final/final_model_df_rookie.csv'))

df_allnba = df_allnba_raw.copy()
df_rookie = df_rookie_raw.copy()

meta_cols = ['PLAYER_NAME', 'PLAYER_ID','TEAM_ID', 'TEAM_ABBREVIATION', 'SEASON', 'PTS_WON', 'PTS_MAX']

# AllNBA
test_allnba  = df_allnba[df_allnba['SEASON'] == '2025-26']
X_test_allnba  = test_allnba.drop(columns=meta_cols + ['TARGET'])

# ROOKIE
test_rookie  = df_rookie[df_rookie['SEASON'] == '2025-26']
X_test_rookie  = test_rookie.drop(columns=meta_cols + ['TARGET'])


path_model_allnba = os.path.join(PROJECT_DIR, 'models/model_allnba.pkl')
path_model_rookie = os.path.join(PROJECT_DIR, 'models/model_rookie.pkl')

model_allnba = joblib.load(path_model_allnba)
model_rookie = joblib.load(path_model_rookie)

def predict_teams_with_filter(model, X, df_copy, n_teams=3, team_size=5, min_games=0):
    df = df_copy.copy()
    df['PRED'] = model.predict(X)
    
    df.loc[df['GP'] < min_games, 'PRED'] = -999
    
    df_sorted = df.sort_values('PRED', ascending=False)
    teams = {}
    for i in range(1, n_teams + 1):
        start = (i - 1) * team_size
        end = i * team_size
        teams[i] = df_sorted.iloc[start:end]['PLAYER_NAME'].tolist()
    return teams

teams_allnba = predict_teams_with_filter(model_allnba, X_test_allnba, test_allnba, min_games=63)
teams_rookie = predict_teams_with_filter(model_rookie, X_test_rookie, test_rookie, n_teams=2)

output = {
    "first all-nba team":   teams_allnba[1],
    "second all-nba team":  teams_allnba[2],
    "third all-nba team":   teams_allnba[3],
    "first rookie all-nba team":  teams_rookie[1],
    "second rookie all-nba team": teams_rookie[2]
}


output_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_DIR, 'Michalak_Karolina.json')

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Zapisano predykcje do {output_path}")
print(json.dumps(output, ensure_ascii=False, indent=2))