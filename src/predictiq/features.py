"""
features.py - Feature engineering utilities for PredictIQ.

This module contains reusable functions for:
- Match result labeling
- Synthetic team strength calculation
- Tactical difference features
- Rolling form (last 5 matches)
- Utility helpers
"""

import numpy as np
import pandas as pd


# 1. Match Result Label
def compute_match_result(df):
    """
    Add a column 'match_result' to a Match dataframe.
    Output  {'home_win', 'draw', 'away_win'}
    """
    def label(row):
        if row["home_team_goal"] > row["away_team_goal"]:
            return "home_win"
        elif row["home_team_goal"] < row["away_team_goal"]:
            return "away_win"
        else:
            return "draw"

    df["match_result"] = df.apply(label, axis=1)
    return df


# 2. Synthetic Team Strength
def add_team_strength(df):
    """
    Compute synthetic team strength using core tactical attributes.
    Creates:
    - home_team_strength
    - away_team_strength
    - team_strength_diff
    """

    home_cols = [
        "home_buildUpPlaySpeed",
        "home_chanceCreationPassing",
        "home_chanceCreationShooting",
        "home_defencePressure",
    ]
    away_cols = [
        "away_buildUpPlaySpeed",
        "away_chanceCreationPassing",
        "away_chanceCreationShooting",
        "away_defencePressure",
    ]

    df["home_team_strength"] = df[home_cols].mean(axis=1)
    df["away_team_strength"] = df[away_cols].mean(axis=1)
    df["team_strength_diff"] = df["home_team_strength"] - df["away_team_strength"]

    return df


# 3. Tactical Difference Features
TACTICAL_COLS = [
    "buildUpPlaySpeed",
    "buildUpPlayPassing",
    "chanceCreationPassing",
    "chanceCreationShooting",
    "defencePressure",
    "defenceAggression",
    "defenceTeamWidth",
]

def add_tactical_differences(df):
    """
    Create tactical difference features:
        {feature}_diff = home_feature - away_feature
    """

    for col in TACTICAL_COLS:
        h = f"home_{col}"
        a = f"away_{col}"
        diff = f"{col}_diff"

        if h in df.columns and a in df.columns:
            df[diff] = df[h] - df[a]

    return df


# 4. Rolling Form Features (Last 5 Matches)
def compute_team_match_history(df_matches):
    """
    Expand a match table into a team-centered history table.
    Each row becomes two rows:
    - one from home team perspective
    - one from away team perspective
    """

    home = df_matches[[
        "match_api_id", "date",
        "home_team_api_id",
        "home_team_goal", "away_team_goal"
    ]].rename(columns={
        "home_team_api_id": "team_id",
        "home_team_goal": "goals_for",
        "away_team_goal": "goals_against",
    })
    home["is_home"] = 1

    away = df_matches[[
        "match_api_id", "date",
        "away_team_api_id",
        "away_team_goal", "home_team_goal"
    ]].rename(columns={
        "away_team_api_id": "team_id",
        "away_team_goal": "goals_for",
        "home_team_goal": "goals_against",
    })
    away["is_home"] = 0

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches = team_matches.sort_values(["team_id", "date"]).reset_index(drop=True)

    # determine result
    def outcome(row):
        if row["goals_for"] > row["goals_against"]:
            return "win"
        elif row["goals_for"] < row["goals_against"]:
            return "loss"
        return "draw"

    team_matches["result"] = team_matches.apply(outcome, axis=1)
    team_matches["win_flag"] = (team_matches["result"] == "win").astype(int)

    return team_matches


def compute_rolling_form(team_matches, window=5):
    """
    Compute rolling (last K matches) form features.
    Ensures temporal correctness using shift(1):
        - avg_goals_for_last5
        - avg_goals_against_last5
        - win_rate_last5
        - goal_diff_avg_last5
        - points_per_game_last5
    """

    team_matches = team_matches.sort_values(["team_id", "date"]).reset_index(drop=True)
    grouped = team_matches.groupby("team_id", group_keys=False)

    def add_roll(g):
        g = g.sort_values("date").copy()

        # shift by 1 to avoids leakage
        g["gf_shift"] = g["goals_for"].shift(1)
        g["ga_shift"] = g["goals_against"].shift(1)
        g["win_shift"] = g["win_flag"].shift(1)

        g["avg_goals_for_last5"] = g["gf_shift"].rolling(window, min_periods=1).mean()
        g["avg_goals_against_last5"] = g["ga_shift"].rolling(window, min_periods=1).mean()
        g["win_rate_last5"] = g["win_shift"].rolling(window, min_periods=1).mean()
        g["goal_diff_avg_last5"] = (g["gf_shift"] - g["ga_shift"]).rolling(window, min_periods=1).mean()
        g["points_per_game_last5"] = (g["win_shift"] * 3).rolling(window, min_periods=1).mean()

        return g

    return grouped.apply(add_roll)

