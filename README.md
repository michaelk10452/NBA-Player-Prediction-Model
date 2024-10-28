# NBA-Player-Prediction-Model

A Python-based machine learning application that predicts whether an NBA player will score above or below a target point total in their next game. The prediction is based on historical performance data from the last three seasons, recent games, and head-to-head matchups.

## Features

- Player performance analysis using NBA API data
- Historical data analysis from the last 3 seasons
- Recent form analysis (last 5 games)
- Head-to-head matchup analysis (last 3 games against specific opponent)
- Machine learning prediction using Logistic Regression with cross-validation
- Confidence scoring for predictions
- Home/Away game consideration

## Prerequisites

```bash
pip install nba_api scikit-learn pandas numpy
```

## Usage

1. Run the script:
```bash
python main.py
```

2. Follow the prompts to enter:
   - Player name (full name, e.g., "LeBron James")
   - Target points (e.g., "25")
   - Opponent team abbreviation (3 letters, e.g., "GSW")
   - Game location ("home" or "away")

3. The program will output:
   - Average points from last 5 games
   - Detailed stats from last 5 games
   - Stats from last 3 matchups against the specified opponent
   - Prediction whether the player will score above/below target points
   - Model accuracy and confidence scores

## How It Works

1. **Data Collection**:
   - Fetches player data using the NBA API
   - Retrieves game logs from the last 3 seasons
   - Collects recent game data and head-to-head matchups

2. **Feature Engineering**:
   - Combines historical and recent performance data
   - Incorporates game location (home/away) as a feature
   - Processes player statistics (points, rebounds, assists, steals, blocks)

3. **Machine Learning**:
   - Uses Logistic Regression for binary classification
   - Implements 5-fold cross-validation for robust accuracy measurement
   - Provides probability scores for predictions

## Limitations

- Requires active internet connection for NBA API access
- Historical data limited to last 3 seasons
- Does not account for injuries or lineup changes
- Limited to basic box score statistics

## Potential Improvements

1. **Enhanced Data Collection**:
   - Include player injury history
   - Add team roster context
   - Consider opponent defensive ratings
   - Include player rest days between games

2. **Advanced Analytics**:
   - Implement advanced statistics (PER, True Shooting %, etc.)
   - Add pace-adjusted statistics
   - Consider home/away performance splits
   - Include player matchup data

3. **Model Improvements**:
   - Test different ML algorithms (Random Forest, XGBoost)
   - Implement feature importance analysis
   - Add seasonal trends analysis
   - Consider player career trajectories

4. **User Experience**:
   - Add GUI interface
   - Include data visualization
   - Add batch prediction capability
   - Implement result export functionality

5. **Technical Enhancements**:
   - Add data caching to reduce API calls
   - Implement error handling for API failures
   - Add unit tests
   - Include data validation

## Error Handling

The program currently handles:
- Invalid player names
- Invalid team abbreviations
- Insufficient data for predictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Note

This tool is for educational and entertainment purposes only. Sports betting decisions should not be made solely based on these predictions.