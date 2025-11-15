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

# Helper: normalize schedule DataFrame columns coming from nfl_data_py
def _normalize_schedule_columns(sched_df, source='schedules'):
    """Normalize schedule DataFrame columns so downstream code can safely use
    'home_team' and 'away_team'. Returns a copy with normalized column names.
    Raises a clear KeyError if neither column can be found.
    """
    df = sched_df.copy()
    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # common candidate names (extend if you see other names in your nfl_data_py version)
    home_candidates = ['home_team', 'home', 'home_team_name', 'home_name']
    away_candidates = ['away_team', 'away', 'away_team_name', 'away_name']

    col_map = {}
    for c in home_candidates:
        if c in df.columns:
            col_map[c] = 'home_team'
            break
    for c in away_candidates:
        if c in df.columns:
            col_map[c] = 'away_team'
            break

    if col_map:
        df = df.rename(columns=col_map)

    if 'home_team' not in df.columns or 'away_team' not in df.columns:
        raise KeyError(
            f"Expected 'home_team' and 'away_team' in schedule DataFrame from {source}. "
            f"Available columns: {list(df.columns)}"
        )

    return df

# === YOUR SPREADS ===
my_df = pd.read_csv('my_spreads.csv')
# normalize my_spreads columns (strip whitespace and lowercase)
my_df.columns = my_df.columns.str.strip().str.lower()
# validate required columns
required_my_cols = {'week', 'home_team', 'away_team', 'my_spread'}
missing = required_my_cols - set(my_df.columns)
if missing:
    raise KeyError(f"my_spreads.csv missing required columns: {missing}. Available: {list(my_df.columns)}")

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
            home = game.get('home_team')
            away = game.get('away_team')
            # be defensive if keys are missing
            if home is None or away is None:
                continue
            for book in game.get('bookmakers', []):
                if 'hardrock' in book.get('key', '').lower():
                    markets = book.get('markets', [])
                    if not markets:
                        continue
                    for outcome in markets[0].get('outcomes', []):
                        if outcome.get('name') == home:
                            spreads.append({
                                'week': week,
                                'home_team': home,
                                'away_team': away,
                                'hr_spread': outcome.get('point')
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
    # normalize schedule columns to avoid KeyError when nfl_data_py changes schema
    sched = _normalize_schedule_columns(sched, source=f"import_schedules({years})")
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
        # normalize schedule columns to avoid KeyError
        try:
            sched = _normalize_schedule_columns(sched, source="import_schedules([2025])")
        except KeyError as ke:
            print(f"Warning: schedule schema unexpected for week {week}: {ke}")
            sched = pd.DataFrame()

        if not sched.empty:
            sched = sched[sched['week'] == week]
            # if home_score/away_score missing this will raise later; keep defensive
            if {'home_score', 'away_score'}.issubset(sched.columns):
                sched['actual_margin'] = sched['home_score'] - sched['away_score']
            else:
                sched['actual_margin'] = None

        # Ensure hr_df has expected columns before merging
        if hr_df.empty or not {'home_team', 'away_team'}.issubset(hr_df.columns):
            print(f"Warning: no Hard Rock spreads for week {week} or unexpected columns: {list(hr_df.columns)}")
            # create empty DataFrame with expected cols so merge won't fail
            hr_df = pd.DataFrame(columns=['week', 'home_team', 'away_team', 'hr_spread'])

        merged = week_my.merge(hr_df, on=['home_team', 'away_team'], how='left')
        if not sched.empty and {'home_team', 'away_team', 'actual_margin'}.issubset(sched.columns):
            merged = merged.merge(sched[['home_team', 'away_team', 'actual_margin']], on=['home_team', 'away_team'], how='left')
        else:
            merged['actual_margin'] = None

        merged['ml_projection'] = merged.apply(lambda r: predict_spread(r['home_team'], r['away_team'], week), axis=1)
        merged['spread_diff'] = abs(merged['my_spread'] - merged['hr_spread'].fillna(0))
        merged['home_covers_my'] = merged['actual_margin'] > merged['my_spread']
        merged['home_covers_hr'] = merged['actual_margin'] > merged['hr_spread']
        results.append(merged)

    if results:
        df = pd.concat(results, ignore_index=True)
    else:
        df = pd.DataFrame()

    df.to_csv('full_analysis.csv', index=False)
    print("\nAnalysis saved to full_analysis.csv")

    # === PATTERNS ===
    print("\n" + "="*50)
    print("PATTERN ANALYSIS")
    print("="*50)
    if not df.empty:
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
    else:
        print("No results to analyze or plot.")


if __name__ == '__main__':
    run_analysis()

