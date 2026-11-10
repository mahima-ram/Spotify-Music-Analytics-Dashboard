# Spotify Listener Insights 🎧

An interactive music analytics dashboard that transforms Spotify listening data into personalized insights about listening habits, music discovery, and changes in musical preferences over time.

Built with **Python, Streamlit, Spotipy, Pandas, and Plotly**, the application connects to the Spotify Web API to analyze listening behavior across multiple time horizons and generate interpretable listener profiles.

## Overview

Spotify provides users with information about their top artists and tracks, but raw listening data doesn't always reveal broader patterns in listening behavior.

Spotify Listener Insights takes that data a step further by calculating behavioral metrics, combining them into higher-level dimensions, and presenting the results through an interactive dashboard.

The application analyzes listening behavior across three time ranges:

- **Short-term:** Recent listening preferences
- **Medium-term:** Intermediate listening patterns
- **Long-term:** More established musical preferences

It also incorporates recently played tracks to provide additional context about recent listening activity.

## Features

- **Spotify Authentication:** Connects to a user's Spotify account using OAuth authentication.
- **Multi-Timeframe Analysis:** Compares listening preferences across short-, medium-, and long-term time ranges.
- **Behavioral Metrics:** Calculates 13+ metrics to quantify different aspects of listening behavior.
- **Weighted Scoring:** Combines individual metrics into five higher-level behavioral dimensions.
- **Listener Classification:** Uses a rule-based classification system to generate personalized listener profiles.
- **Interactive Visualizations:** Uses Plotly to display listening patterns, artist and track trends, and changes in musical preferences.
- **Recently Played Analysis:** Incorporates recently played tracks into the listening analysis.

## How It Works

The application follows a data-processing pipeline:

```text
User Authentication
        |
        v
Spotify Web API
        |
        v
Spotipy Data Retrieval
        |
        v
Raw Listening Data (JSON)
        |
        v
Pandas Data Processing
        |
        v
13+ Behavioral Metrics
        |
        v
Weighted Scoring System
        |
        v
5 Behavioral Dimensions
        |
        v
Rule-Based Classification
        |
        v
Personalized Listener Profile
        |
        v
Plotly Visualizations
        |
        v
Streamlit Dashboard
```

### 1. Data Collection

The application uses Spotipy, a Python library for interacting with the Spotify Web API, to retrieve user listening data.

Spotify OAuth handles user authorization, allowing the application to access permitted listening information without collecting the user's Spotify password.

### 2. Data Processing and Feature Engineering

Spotify API responses are structured as JSON data. Pandas is used to organize and process the retrieved information for analysis.

The application calculates 13+ behavioral metrics designed to capture different aspects of listening behavior, including:

- Artist diversity
- Artist loyalty and persistence
- Music discovery
- Listening concentration
- Recency
- Overlap and stability across time ranges

These metrics transform raw listening information into measurable behavioral features.

### 3. Weighted Scoring

Individual metrics are combined into five higher-level dimensions:

- **Exploration:** How much the user explores different artists and music.
- **Loyalty:** How consistently the user returns to familiar artists.
- **Breadth:** How widely listening preferences are distributed.
- **Consistency:** How stable listening preferences are across time periods.
- **Recency:** How much recent listening activity influences the analysis.

A weighted scoring approach allows different metrics to contribute differently to each dimension.

### 4. Rule-Based Classification

The five dimensions are used by a rule-based classification system to generate personalized listener profiles.

Rather than using a machine learning model, the application uses explicitly defined scoring weights and classification rules. This makes the results interpretable and allows the reasoning behind a profile to be explained.

### 5. Data Visualization

Plotly powers the interactive charts, while Streamlit provides the application interface.

The dashboard allows users to explore listening trends, compare artists and tracks, and examine how musical preferences change across different time horizons.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic and data analysis |
| Streamlit | Interactive web application and dashboard |
| Spotipy | Python interface to the Spotify Web API |
| Spotify Web API | Listening data retrieval |
| Pandas | Data processing and metric calculation |
| Plotly | Interactive data visualization |
| Spotify OAuth | User authentication and authorization |

## Getting Started

### Prerequisites

- Python 3.10 or later
- A Spotify account
- A Spotify Developer application with API credentials

### 1. Clone the Repository

```bash
git clone https://github.com/mahima-ram/spotify-listener-insights.git
cd spotify-listener-insights
```

### 2. Install Dependencies

```bash
pip install streamlit spotipy pandas plotly python-dotenv
```

### 3. Configure Spotify API Credentials

Create an application in the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).

Configure the application's redirect URI to match the redirect URI used in your code.

Create a `.env` file in the project root:

```env
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback
```

Replace the example redirect URI with the exact URI configured in your Spotify Developer application and used by your code.

**Important:** Never commit your Spotify client secret, access tokens, or `.env` file to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
.cache
__pycache__/
*.pyc
```

### 4. Run the Application

If your main Streamlit file is named `spotify.py`, run:

```bash
streamlit run spotify.py
```

If your entry-point file has a different name, substitute it in the command.

Follow the Spotify authorization flow when prompted.

## Design Decisions

### Why Streamlit?

Streamlit allowed me to build an interactive data application entirely in Python without needing a separate frontend framework. This made it easier to focus on data analysis and translating the results into a usable dashboard.

### Why a Weighted Scoring System?

Not all behavioral metrics contribute equally to a particular listening dimension. Weighted scoring provides a way to combine multiple signals while explicitly controlling their relative importance.

### Why Rule-Based Classification Instead of Machine Learning?

The application does not currently have a labeled dataset of users and their corresponding listener profiles. A rule-based approach provides an interpretable alternative, allowing each classification to be traced back to its underlying metrics and scoring rules.

A potential future improvement would be to collect user feedback and labeled profiles to evaluate whether a supervised machine learning model could improve classification accuracy.

## Future Improvements

- Validate listener profiles through user testing and feedback.
- Refine metric weights based on observed user preferences.
- Expand comparisons across listening time periods.
- Improve API error handling and rate-limit management.
- Add automated tests for metric calculations and classification logic.
- Explore supervised machine learning approaches if sufficient labeled data becomes available.

## Author

Developed as a personal project to explore API integration, data analytics, feature engineering, and interactive application development.

**Technologies:** Python · Streamlit · Spotipy · Spotify Web API · Pandas · Plotly