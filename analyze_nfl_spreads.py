import pandas as pd
import nfl_data_py as nfl
import requests
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from dotenv import load_dotenv

load_dotenv()

# === CONFIG ===
API_KEY = os.getenv('ODDS_API_KEY')
SPORT = 'americanfootball_nfl'
REGIONS = 'us'
MARKETS = 'h2h_spreads'
MODEL_PATH = 'nfl_spread_model.pkl'

# === YOUR SPREADS ===
my_df = pd.read_csv('my_spreads.csv')
my_df['week'] = my_df['week'].astype(int)

# === FETCH HARD ROCK SPREADS ===
def fetch_hardrock_spreads(week):
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds'
    params = {
        'apiKey': API_KEY,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': 'decimal',
        'dateFormat': 'iso'
    }
    try:
        r = requests.get(url, params=params)
        data = r.json()
        spreads = []
        for game in data:
            home = game['home_team']
            away = game['away_team']
            for book in game['bookmakers']:
                if 'hardrock' in book['key'].lower():
                    for outcome in book['markets'][0]['outcomes']:
                        if outcome['name'] == home:
                            spreads.append({
                                'week': week,
                                'home_team': home,
                                'away_team': away,
                                'hr_spread': outcome['point']
                            })
        return pd.DataFrame(spreads)
    except Exception as e:
        print(f"Warning: Could not fetch Hard Rock data for Week {week}: {e}")
        return pd.DataFrame()

# === TRAIN ML MODEL ===
def train_ml_model():
    print("Training ML model on 2020-2024 data...")
    years = [2020, 2021, 2022, 2023, 2024]
    sched = nfl.import_schedules(years)
    df = sched[['week', 'home_team', 'away_team', 'home_score', 'away_score']].copy()
    df['actual_margin'] = df['home_score'] - df['away_score']
    df = df.dropna()
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    team_map = {t: i for i, t in enumerate(teams)}
    df['home_id'] = df['home_team'].map(team_map)
    df['away_id'] = df['away_team'].map(team_map)
    X = df[['week', 'home_id', 'away_id']]
    y = df['actual_margin']
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(team_map, 'team_map.pkl')
    print("ML Model trained and saved.")
    return model, team_map

# Load or train
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    team_map = joblib.load('team_map.pkl')
else:
    model, team_map = train_ml_model()

def predict_spread(home, away, week):
    if home not in team_map or away not in team_map:
        return 0.0
    X = pd.DataFrame({
        'week': [week],
        'home_id': [team_map[home]],
        'away_id': [team_map[away]]
    })
    return model.predict(X)[0]

# === RUN ANALYSIS ===
def run_analysis():
    results = []
    for week in my_df['week'].unique():
        print(f"\n=== Week {week} ===")
        hr_df = fetch_hardrock_spreads(week)
        week_my = my_df[my_df['week'] == week].copy()
        sched = nfl.import_schedules([2025])
        sched = sched[sched['week'] == week]
        sched['actual_margin'] = sched['home_score'] - sched['away_score']
        merged = week_my.merge(hr_df, on=['home_team', 'away_team'], how='left')
        merged = merged.merge(sched[['home_team', 'away_team', 'actual_margin']], on=['home_team', 'away_team'], how='left')
        merged['ml_projection'] = merged.apply(lambda r: predict_spread(r['home_team'], r['away_team'], week), axis=1)
        merged['spread_diff'] = abs(merged['my_spread'] - merged['hr_spread'].fillna(0))
        merged['home_covers_my'] = merged['actual_margin'] > merged['my_spread']
        merged['home_covers_hr'] = merged['actual_margin'] > merged['hr_spread']
        results.append(merged)
    df = pd.concat(results, ignore_index=True)
    df.to_csv('full_analysis.csv', index=False)
    print("\nAnalysis saved to full_analysis.csv")

    # === PATTERNS ===
    print("\n" + "="*50)
    print("PATTERN ANALYSIS")
    print("="*50)
    match = df[df['spread_diff'] == 0]
    if len(match) > 0:
        print(f"EXACT MATCH (n={len(match)}): {match['home_covers_my'].mean()*100:.1f}% cover")
    diff2 = df[abs(df['spread_diff'] - 2) < 0.1]
    if len(diff2) > 0:
        print(f"2-PT DIFF (n={len(diff2)}): Avg margin = {diff2['actual_margin'].mean():+.1f}")
    print(f"\nYOUR COVER: {df['home_covers_my'].mean()*100:.1f}%")
    print(f"HARD ROCK: {df['home_covers_hr'].mean()*100:.1f}%")

    # Plot
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=df, x='spread_diff', y='actual_margin', hue='home_covers_my', s=100)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Your Spread Diff vs Actual Margin')
    plt.xlabel('|Your - Hard Rock|')
    plt.ylabel('Home Margin')
    plt.legend(title='Covers Your Spread')
    plt.tight_layout()
    plt.savefig('pattern_plot.png', dpi=150)
    plt.close()
    print("Plot saved: pattern_plot.png")

if __name__ == '__main__':
    run_analysis()
