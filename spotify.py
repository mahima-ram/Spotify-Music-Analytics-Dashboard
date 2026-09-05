"""
Spotify Listening Analyzer — Interactive Dashboard
----------------------------------------------------
A Streamlit app version of the analyzer. Instead of generating static
PNGs, this launches a live, interactive dashboard in your browser
where you can filter time ranges and hover over charts.

SETUP (same Spotify app as the CLI version):
1. Register an app at https://developer.spotify.com/dashboard
   (requires Premium). Redirect URI must be exactly:
   http://127.0.0.1:8888/callback
2. Install dependencies:
   pip install spotipy pandas plotly streamlit --break-system-packages
3. Set environment variables:s
   export SPOTIPY_CLIENT_ID="your_client_id"
   export SPOTIPY_CLIENT_SECRET="your_client_secret"
   export SPOTIPY_REDIRECT_URI="http://127.0.0.1:8888/callback"
4. Run:
   streamlit run spotify.py
   This opens a browser tab with the dashboard. The first time, it'll
   also pop a Spotify login/auth window — that's expected.

To host this live (so a recruiter can click a link instead of running
code locally), Streamlit Community Cloud (share.streamlit.io) is free
and connects directly to a GitHub repo.
"""
import os
from collections import Counter
from math import log2

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import spotipy
import streamlit as st
from spotipy.oauth2 import SpotifyOAuth

# ---- Spotify brand colors ----
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#191414"
SPOTIFY_WHITE = "#FFFFFF"
SPOTIFY_GRAY = "#B3B3B3"

TIME_RANGES = {
    "short_term": "Last 4 Weeks",
    "medium_term": "Last 6 Months",
    "long_term": "All Time",
}

st.set_page_config(
    page_title="Spotify Listening Analyzer",
    page_icon="🎧",
    layout="wide",
)


# -------------------------------------------------------------------
# Spotify API
# -------------------------------------------------------------------

@st.cache_resource
def get_spotify_client():
    # user-top-read = top artists/tracks
    # user-read-recently-played = recent listening behavior
    scope = "user-top-read user-read-recently-played"
    auth_manager = SpotifyOAuth(client_id=st.secrets["SPOTIPY_CLIENT_ID"],
        client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
        redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
    scope=scope,
    open_browser=True,)
    return spotipy.Spotify(auth_manager=auth_manager)


@st.cache_data(ttl=600)
def fetch_all_data(_sp, limit=20):
    """
    Fetch the Spotify data used by the dashboard.

    Important:
    We intentionally do NOT request Spotify's old popularity field.
    The profile is based on observable listening behavior instead.
    """
    data = {}

    for time_range in TIME_RANGES:
        artists = _sp.current_user_top_artists(
            limit=limit,
            time_range=time_range,
        )["items"]

        tracks = _sp.current_user_top_tracks(
            limit=limit,
            time_range=time_range,
        )["items"]

        data[time_range] = {
            "artists": artists,
            "tracks": tracks,
        }

    # Recently played tracks provide an additional behavioral signal.
    try:
        recent = _sp.current_user_recently_played(limit=50)
        data["recently_played"] = recent.get("items", [])
    except Exception:
        data["recently_played"] = []

    return data


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def safe_release_year(track):
    release_date = track.get("album", {}).get("release_date", "")
    if not release_date:
        return None

    try:
        return int(release_date[:4])
    except (TypeError, ValueError):
        return None


def unique_ids(items, key="id"):
    return {
        item.get(key)
        for item in items
        if item.get(key)
    }


def overlap_score(set_a, set_b):
    """Symmetric overlap score from 0-100."""
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    return (intersection / union) * 100 if union else 0.0


def normalized_entropy(items):
    """
    Shannon entropy normalized to 0-100.

    Higher = listening is spread more evenly across categories.
    Lower = listening is concentrated in a small number of categories.
    """
    if not items:
        return 0.0

    counts = Counter(items)
    total = sum(counts.values())

    if len(counts) <= 1:
        return 0.0

    probabilities = [
        count / total
        for count in counts.values()
    ]

    entropy = -sum(
        p * log2(p)
        for p in probabilities
        if p > 0
    )

    max_entropy = log2(len(counts))

    return (entropy / max_entropy) * 100 if max_entropy else 0.0


def rank_weighted_score(items):
    """
    Convert a ranked Spotify list into a 0-100 concentration score.

    The first item contributes the most weight.
    """
    if not items:
        return 0.0

    weights = list(range(len(items), 0, -1))
    total_weight = sum(weights)

    return sum(weights) / total_weight * 100


# -------------------------------------------------------------------
# Listening profile algorithm
# -------------------------------------------------------------------

def calculate_listening_profile(data):
    """
    Build a listening-behavior profile from Spotify data.

    The model intentionally avoids Spotify's removed popularity score.
    It instead derives behavioral signals from:
      - top artists
      - top tracks
      - album diversity
      - release years
      - cross-period overlap
      - recent listening
      - rank concentration

    Returns a dictionary of raw metrics, dimension scores,
    an overall profile score, and a profile classification.
    """

    short_artists = data["short_term"]["artists"]
    medium_artists = data["medium_term"]["artists"]
    long_artists = data["long_term"]["artists"]

    short_tracks = data["short_term"]["tracks"]
    medium_tracks = data["medium_term"]["tracks"]
    long_tracks = data["long_term"]["tracks"]

    recent_items = data.get("recently_played", [])

    # -------------------------
    # IDs
    # -------------------------

    short_artist_ids = unique_ids(short_artists)
    medium_artist_ids = unique_ids(medium_artists)
    long_artist_ids = unique_ids(long_artists)

    short_track_ids = unique_ids(short_tracks)
    medium_track_ids = unique_ids(medium_tracks)
    long_track_ids = unique_ids(long_tracks)

    short_album_ids = unique_ids(
        [t.get("album", {}) for t in short_tracks]
    )
    medium_album_ids = unique_ids(
        [t.get("album", {}) for t in medium_tracks]
    )
    long_album_ids = unique_ids(
        [t.get("album", {}) for t in long_tracks]
    )

    # -------------------------
    # 1. Artist diversity
    # -------------------------

    artist_diversity = normalized_entropy(
        [a.get("id") for a in short_artists if a.get("id")]
    )

    # -------------------------
    # 2. Track diversity
    # -------------------------

    track_diversity = normalized_entropy(
        [t.get("id") for t in short_tracks if t.get("id")]
    )

    # -------------------------
    # 3. Album diversity
    # -------------------------

    album_diversity = normalized_entropy(
        [
            t.get("album", {}).get("id")
            for t in short_tracks
            if t.get("album", {}).get("id")
        ]
    )

    # -------------------------
    # 4. Artist persistence
    # -------------------------

    persistent_artists = (
        short_artist_ids
        & medium_artist_ids
        & long_artist_ids
    )

    artist_persistence = (
        len(persistent_artists)
        / max(len(short_artist_ids), 1)
    ) * 100

    # -------------------------
    # 5. Artist turnover
    # -------------------------

    artist_turnover = 100 - overlap_score(
        short_artist_ids,
        long_artist_ids,
    )

    # -------------------------
    # 6. Recent artist discovery
    # -------------------------

    recent_artist_ids = unique_ids(
        [
            item.get("track", {}).get("artists", [{}])[0]
            for item in recent_items
            if item.get("track")
            and item.get("track", {}).get("artists")
        ]
    )

    new_recent_artists = recent_artist_ids - long_artist_ids

    recent_discovery = (
        len(new_recent_artists)
        / max(len(recent_artist_ids), 1)
    ) * 100

    # If recently-played data isn't available, use top-period turnover
    # as a fallback rather than returning a misleading zero.
    if not recent_items:
        recent_discovery = artist_turnover

    # -------------------------
    # 7. Taste stability
    # -------------------------

    short_medium = overlap_score(
        short_artist_ids,
        medium_artist_ids,
    )

    medium_long = overlap_score(
        medium_artist_ids,
        long_artist_ids,
    )

    taste_stability = (
        short_medium + medium_long
    ) / 2

    # -------------------------
    # 8. Track loyalty
    # -------------------------

    track_loyalty = overlap_score(
        short_track_ids,
        long_track_ids,
    )

    # -------------------------
    # 9. Album loyalty
    # -------------------------

    album_loyalty = overlap_score(
        short_album_ids,
        long_album_ids,
    )

    # -------------------------
    # 10. Release-era diversity
    # -------------------------

    release_years = [
        safe_release_year(track)
        for track in short_tracks
    ]
    release_years = [
        year for year in release_years
        if year is not None
    ]

    if release_years:
        year_span = max(release_years) - min(release_years)
        era_diversity = min(100, (year_span / 60) * 100)
    else:
        era_diversity = 0.0

    # -------------------------
    # 11. Recency preference
    # -------------------------

    if release_years:
        current_year = pd.Timestamp.now().year
        average_age = sum(
            max(0, current_year - year)
            for year in release_years
        ) / len(release_years)

        # 0 years old -> 100, 30+ years old -> 0.
        recency_preference = max(
            0,
            min(100, 100 - (average_age / 30) * 100)
        )
    else:
        recency_preference = 0.0

    # -------------------------
    # 12. Recent listening variety
    # -------------------------

    recent_track_ids = [
        item.get("track", {}).get("id")
        for item in recent_items
        if item.get("track", {}).get("id")
    ]

    recent_listening_variety = normalized_entropy(
        recent_track_ids
    )

    # -------------------------
    # 13. Artist rank concentration
    # -------------------------

    artist_rank_concentration = rank_weighted_score(
        short_artists
    )

    # -------------------------
    # Higher-level dimensions
    # -------------------------

    exploration = (
        artist_diversity * 0.25
        + track_diversity * 0.15
        + album_diversity * 0.10
        + artist_turnover * 0.20
        + recent_discovery * 0.20
        + era_diversity * 0.10
    )

    loyalty = (
        artist_persistence * 0.35
        + track_loyalty * 0.25
        + album_loyalty * 0.15
        + taste_stability * 0.25
    )

    breadth = (
        artist_diversity * 0.35
        + track_diversity * 0.25
        + album_diversity * 0.20
        + era_diversity * 0.20
    )

    consistency = (
        taste_stability * 0.50
        + artist_persistence * 0.30
        + track_loyalty * 0.20
    )

    recency = (
        recency_preference * 0.60
        + recent_discovery * 0.20
        + era_diversity * 0.20
    )

    # Overall "profile strength" is not a quality judgment.
    # It simply represents how much behavioral information is available.
    available_signals = [
        bool(short_artists),
        bool(medium_artists),
        bool(long_artists),
        bool(short_tracks),
        bool(medium_tracks),
        bool(long_tracks),
        bool(recent_items),
        bool(release_years),
    ]

    data_coverage = (
        sum(available_signals)
        / len(available_signals)
    ) * 100

    return {
        "Artist Diversity": round(artist_diversity),
        "Track Diversity": round(track_diversity),
        "Album Diversity": round(album_diversity),
        "Artist Persistence": round(artist_persistence),
        "Artist Turnover": round(artist_turnover),
        "Recent Discovery": round(recent_discovery),
        "Taste Stability": round(taste_stability),
        "Track Loyalty": round(track_loyalty),
        "Album Loyalty": round(album_loyalty),
        "Era Diversity": round(era_diversity),
        "Recency Preference": round(recency_preference),
        "Recent Listening Variety": round(recent_listening_variety),
        "Artist Rank Concentration": round(artist_rank_concentration),
        "Exploration": round(exploration),
        "Loyalty": round(loyalty),
        "Breadth": round(breadth),
        "Consistency": round(consistency),
        "Recency": round(recency),
        "Data Coverage": round(data_coverage),
    }


def classify_listener(profile):
    """
    Turn the numeric behavioral dimensions into an interpretable profile.
    """

    exploration = profile["Exploration"]
    loyalty = profile["Loyalty"]
    breadth = profile["Breadth"]
    consistency = profile["Consistency"]
    recency = profile["Recency"]

    # Strong exploration + breadth
    if exploration >= 72 and breadth >= 70:
        return (
            "The Explorer",
            "Your listening shows a strong appetite for variety, "
            "artist turnover, and musical discovery."
        )

    # Strong loyalty + consistency
    if loyalty >= 72 and consistency >= 70:
        return (
            "The Devotee",
            "Your listening is anchored by a stable core of artists "
            "and tracks that remain important across time."
        )

    # High breadth but lower loyalty
    if breadth >= 72 and loyalty < 55:
        return (
            "The Eclectic",
            "Your listening spans a broad range of artists, tracks, "
            "albums, and eras rather than settling into one core."
        )

    # High loyalty + moderate exploration
    if loyalty >= 62 and exploration >= 55:
        return (
            "The Curator",
            "You maintain a recognizable musical core while still "
            "making room for new discoveries."
        )

    # High recency
    if recency >= 72:
        return (
            "The Trend Surfer",
            "Your recent listening leans strongly toward newer music "
            "and changing discoveries."
        )

    # High consistency but not extreme loyalty
    if consistency >= 68:
        return (
            "The Comfort Listener",
            "Your listening patterns are relatively stable, with "
            "recurring artists and familiar musical choices."
        )

    return (
        "The Adaptive Listener",
        "Your listening balances familiar favorites with changing "
        "interests, producing a flexible musical profile."
    )


# -------------------------------------------------------------------
# Profile visualization
# -------------------------------------------------------------------

def render_listening_profile(data):
    profile = calculate_listening_profile(data)
    profile_name, description = classify_listener(profile)

    st.subheader("🎧 Your Listening Profile")

    st.markdown(
        f"<h2 style='color:{SPOTIFY_GREEN}; margin-bottom:0;'>"
        f"{profile_name}</h2>",
        unsafe_allow_html=True,
    )

    st.write(description)

    st.caption(
        f"Behavioral profile generated from 13 underlying metrics "
        f"across artists, tracks, albums, release eras, listening "
        f"history, and cross-period behavior. "
        f"Data coverage: {profile['Data Coverage']:.0f}%."
    )

    # High-level dimensions
    score_rows = [
        {"Metric": "Exploration", "Score": profile["Exploration"]},
        {"Metric": "Loyalty", "Score": profile["Loyalty"]},
        {"Metric": "Breadth", "Score": profile["Breadth"]},
        {"Metric": "Consistency", "Score": profile["Consistency"]},
        {"Metric": "Recency", "Score": profile["Recency"]},
    ]

    df = pd.DataFrame(score_rows)

    fig = px.bar(
        df,
        x="Score",
        y="Metric",
        orientation="h",
        range_x=[0, 100],
        text="Score",
        color_discrete_sequence=[SPOTIFY_GREEN],
        title="Listening Behavior Profile",
    )

    fig.update_traces(
        texttemplate="%{text}/100",
        textposition="outside",
    )

    fig.update_layout(
        plot_bgcolor=SPOTIFY_BLACK,
        paper_bgcolor=SPOTIFY_BLACK,
        font_color=SPOTIFY_WHITE,
        height=300,
        margin=dict(l=10, r=50, t=60, b=20),
        xaxis=dict(
            range=[0, 105],
            title="Behavioral score",
        ),
        yaxis=dict(
            title=None,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Detail cards
    cols = st.columns(5)

    detail_metrics = [
        ("Artist Diversity", profile["Artist Diversity"]),
        ("Recent Discovery", profile["Recent Discovery"]),
        ("Artist Persistence", profile["Artist Persistence"]),
        ("Taste Stability", profile["Taste Stability"]),
        ("Era Diversity", profile["Era Diversity"]),
    ]

    for col, (name, score) in zip(cols, detail_metrics):
        with col:
            st.metric(
                name,
                f"{score}/100",
            )

    with st.expander("View profile methodology"):
        st.markdown(
            """
            **The profile is a behavioral model, not a Spotify rating.**

            The algorithm combines 13 measurable signals:

            - Artist diversity
            - Track diversity
            - Album diversity
            - Artist persistence
            - Artist turnover
            - Recent artist discovery
            - Taste stability across time ranges
            - Track loyalty
            - Album loyalty
            - Release-era diversity
            - Recency preference
            - Recent listening variety
            - Artist rank concentration

            These signals are combined into five higher-level dimensions:
            **Exploration, Loyalty, Breadth, Consistency, and Recency**.

            The final listener type is assigned using transparent
            rule-based thresholds rather than a black-box model.
            """
        )

    return profile


# -------------------------------------------------------------------
# Existing visualizations
# -------------------------------------------------------------------

def render_release_year_histogram(tracks, label):
    """Distribution of release years for top tracks."""
    if not tracks:
        st.info(
            f"No top track data available for {label} yet."
        )
        return

    years = []

    for track in tracks:
        year = safe_release_year(track)
        if year is not None:
            years.append(year)

    if not years:
        st.info(
            f"No release date data available for {label}."
        )
        return

    df = pd.DataFrame({"Year": years})

    fig = px.histogram(
        df,
        x="Year",
        nbins=min(20, len(set(years))),
        color_discrete_sequence=[SPOTIFY_GREEN],
        title=f"Release Years of Your Top Tracks — {label}",
    )

    fig.update_layout(
        plot_bgcolor=SPOTIFY_BLACK,
        paper_bgcolor=SPOTIFY_BLACK,
        font_color=SPOTIFY_WHITE,
        height=400,
        bargap=0.1,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def render_artist_bar(artists, label, top_n=10):
    """
    Show personal artist rankings.

    This deliberately does NOT use Spotify popularity.
    """
    if not artists:
        st.info(
            f"No top artist data available for {label} yet."
        )
        return

    top = artists[:top_n]

    names = [
        artist["name"]
        for artist in top
    ][::-1]

    ranks = list(
        range(len(top), 0, -1)
    )

    fig = go.Figure(
        go.Bar(
            x=ranks,
            y=names,
            orientation="h",
            marker_color=SPOTIFY_GREEN,
            text=[
                f"#{i}"
                for i in range(len(top), 0, -1)
            ],
            textposition="outside",
            textfont=dict(
                color=SPOTIFY_WHITE,
                size=13,
            ),
        )
    )

    fig.update_layout(
        title=f"Your Top Artists — {label}",
        plot_bgcolor=SPOTIFY_BLACK,
        paper_bgcolor=SPOTIFY_BLACK,
        font_color=SPOTIFY_WHITE,
        height=400,
        xaxis=dict(
            range=[0, len(top) + 1],
            title="Spotify ranking",
            dtick=1,
        ),
        margin=dict(
            l=10,
            r=40,
            t=60,
            b=40,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


def render_artist_treemap(tracks, label):
    """
    Treemap of top tracks grouped by artist.

    Size is based on personal rank, not Spotify popularity.
    """
    if not tracks:
        st.info(
            f"No top track data available for {label} yet."
        )
        return

    rows = []

    for rank, track in enumerate(
        tracks,
        start=1,
    ):
        artist_name = (
            track["artists"][0]["name"]
            if track.get("artists")
            else "Unknown"
        )

        rows.append(
            {
                "Artist": artist_name,
                "Track": track["name"],
                "Weight": len(tracks) - rank + 1,
            }
        )

    df = pd.DataFrame(rows)

    fig = px.treemap(
        df,
        path=["Artist", "Track"],
        values="Weight",
        color="Weight",
        color_continuous_scale=[
            SPOTIFY_BLACK,
            SPOTIFY_GREEN,
        ],
        title=f"Top Tracks by Artist — {label}",
    )

    fig.update_layout(
        paper_bgcolor=SPOTIFY_BLACK,
        font_color=SPOTIFY_WHITE,
        height=450,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Main application
# -------------------------------------------------------------------

def main():
    st.markdown(
        f"<h1 style='color:{SPOTIFY_GREEN};'>"
        f"🎧 Spotify Listening Analyzer"
        f"</h1>",
        unsafe_allow_html=True,
    )

    st.caption(
        "An interactive analysis of your listening behavior, "
        "top artists, tracks, and musical taste over time."
    )

    sp = get_spotify_client()

    with st.spinner("Fetching your Spotify data..."):
        data = fetch_all_data(sp)

    # Sidebar filter
    st.sidebar.header("Filters")

    selected_range = st.sidebar.selectbox(
        "Time range",
        options=list(TIME_RANGES.keys()),
        format_func=lambda k: TIME_RANGES[k],
    )

    label = TIME_RANGES[selected_range]

    tab1, tab2, tab3 = st.tabs(
        [
            "Overview",
            "Artist Treemap",
            "Top Tracks",
        ]
    )

    # ---------------------------------------------------------------
    # Overview
    # ---------------------------------------------------------------

    with tab1:

        render_listening_profile(data)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            render_release_year_histogram(
                data[selected_range]["tracks"],
                label,
            )

        with col2:
            render_artist_bar(
                data[selected_range]["artists"],
                label,
            )

    # ---------------------------------------------------------------
    # Artist / Track Treemap
    # ---------------------------------------------------------------

    with tab2:
        render_artist_treemap(
            data[selected_range]["tracks"],
            label,
        )

    # ---------------------------------------------------------------
    # Top Tracks
    # ---------------------------------------------------------------

    with tab3:
        rows = [
            {
                "Rank": i + 1,
                "Track": track["name"],
                "Artist": ", ".join(
                    artist["name"]
                    for artist in track["artists"]
                ),
                "Album": track["album"]["name"],
                "Release Year": safe_release_year(track),
                "Duration (min)": round(
                    track.get("duration_ms", 0) / 60000,
                    2,
                ),
                "Explicit": track.get(
                    "explicit",
                    False,
                ),
            }
            for i, track in enumerate(
                data[selected_range]["tracks"]
            )
        ]

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()


# from collections import Counter

# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import spotipy
# import streamlit as st
# from spotipy.oauth2 import SpotifyOAuth

# # ---- Spotify brand colors, for a dashboard that feels "on brand" ----
# SPOTIFY_GREEN = "#1DB954"
# SPOTIFY_BLACK = "#191414"
# SPOTIFY_WHITE = "#FFFFFF"
# SPOTIFY_GRAY = "#B3B3B3"

# TIME_RANGES = {
#     "short_term": "Last 4 Weeks",
#     "medium_term": "Last 6 Months",
#     "long_term": "All Time",
# }

# st.set_page_config(page_title="Spotify Listening Analyzer", page_icon="🎧", layout="wide")


# # ---- Caching: avoid re-hitting the Spotify API on every interaction ----
# # @st.cache_resource keeps the same authenticated client across reruns
# @st.cache_resource
# def get_spotify_client():
#     scope = "user-top-read"
#     auth_manager = SpotifyOAuth(scope=scope, open_browser=True)
#     return spotipy.Spotify(auth_manager=auth_manager)


# # @st.cache_data caches the actual API results for 10 minutes, so
# # switching between tabs/filters doesn't re-fetch every time
# @st.cache_data(ttl=600)
# def fetch_all_data(_sp, limit=20):
#     data = {}

#     for time_range, label in TIME_RANGES.items():
#         artists = _sp.current_user_top_artists(
#             limit=limit,
#             time_range=time_range
#         )["items"]

#         tracks = _sp.current_user_top_tracks(
#             limit=limit,
#             time_range=time_range
#         )["items"]

#         # Explicitly fetch artist profiles so we get popularity
#         enriched_artists = []

#         for artist in artists:
#             try:
#                 artist_details = _sp.artist(artist["id"])

#                 # Copy the original artist data
#                 artist_copy = artist.copy()

#                 # Add popularity from the artist endpoint
#                 artist_copy["popularity"] = artist_details.get("popularity", 0)

#                 # Also make sure genres are available
#                 artist_copy["genres"] = artist_details.get("genres", [])

#                 enriched_artists.append(artist_copy)

#             except Exception:
#                 # Keep the artist even if the popularity request fails
#                 artist_copy = artist.copy()
#                 artist_copy["popularity"] = 0
#                 enriched_artists.append(artist_copy)

#         data[time_range] = {
#             "artists": enriched_artists,
#             "tracks": tracks
#         }

#     return data


# def genre_counts_from_artists(artists):
#     counter = Counter()
#     for artist in artists:
#         for genre in artist.get("genres", []):
#             counter[genre] += 1
#     return counter


# def render_release_year_histogram(tracks, label):
#     """Distribution of release years for your top tracks — old vs new music preference."""
#     if not tracks:
#         st.info(f"No top track data available for {label} yet.")
#         return
#     years = []
#     for track in tracks:
#         release_date = track["album"].get("release_date", "")
#         if release_date:
#             years.append(int(release_date[:4]))  # release_date starts with YYYY
#     if not years:
#         st.info(f"No release date data available for {label}.")
#         return
#     df = pd.DataFrame({"Year": years})
#     fig = px.histogram(
#         df,
#         x="Year",
#         nbins=min(20, len(set(years))),
#         color_discrete_sequence=[SPOTIFY_GREEN],
#         title=f"Release Years of Your Top Tracks — {label}",
#     )
#     fig.update_layout(
#         plot_bgcolor=SPOTIFY_BLACK,
#         paper_bgcolor=SPOTIFY_BLACK,
#         font_color=SPOTIFY_WHITE,
#         height=400,
#         bargap=0.1,
#     )
#     st.plotly_chart(fig, use_container_width=True)


# def render_mainstream_meter(data):
#     """Speedometer-style gauges showing average artist popularity per time range."""
#     from plotly.subplots import make_subplots

#     fig = make_subplots(
#         rows=1,
#         cols=3,
#         specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
#     )

#     for i, (time_range, label) in enumerate(TIME_RANGES.items(), start=1):
#         artists = data[time_range]["artists"]
#         avg_popularity = (
#             sum(a.get("popularity", 0) for a in artists) / len(artists) if artists else 0
#         )
#         fig.add_trace(
#             go.Indicator(
#                 mode="gauge+number",
#                 value=avg_popularity,
#                 title={"text": label, "font": {"color": SPOTIFY_WHITE, "size": 16}},
#                 number={"suffix": "", "font": {"color": SPOTIFY_GREEN, "size": 36}},
#                 gauge={
#                     "axis": {"range": [0, 100], "tickcolor": SPOTIFY_WHITE},
#                     "bar": {"color": SPOTIFY_GREEN},
#                     "bgcolor": SPOTIFY_BLACK,
#                     "borderwidth": 1,
#                     "bordercolor": SPOTIFY_GRAY,
#                     "steps": [
#                         {"range": [0, 40], "color": "#2a2a2a"},
#                         {"range": [40, 70], "color": "#3d3d3d"},
#                         {"range": [70, 100], "color": "#4f4f4f"},
#                     ],
#                 },
#             ),
#             row=1,
#             col=i,
#         )

#     fig.update_layout(
#         title={
#             "text": "Your Mainstream-o-Meter (0 = niche/underground, 100 = mainstream)",
#             "font": {"color": SPOTIFY_WHITE},
#         },
#         paper_bgcolor=SPOTIFY_BLACK,
#         font_color=SPOTIFY_WHITE,
#         height=320,
#         margin=dict(t=80, b=20),
#     )
#     st.plotly_chart(fig, use_container_width=True)

# def render_artist_bar(artists, label, top_n=10):
#     if not artists:
#         st.info(f"No top artist data available for {label} yet — Spotify needs more listening history to calculate this.")
#         return
#     top = artists[:top_n]
#     names = [a["name"] for a in top][::-1]
#     popularity = [a.get("popularity", 0) for a in top][::-1]
#     fig = go.Figure(
#         go.Bar(
#             x=popularity,
#             y=names,
#             orientation="h",
#             marker_color=SPOTIFY_GREEN,
#             text=popularity,
#             texttemplate="%{text}",
#             textposition="outside",
#             textfont=dict(color=SPOTIFY_WHITE, size=13),
#         )
#     )
#     fig.update_layout(
#         title=f"Top Artists — {label} (Spotify popularity score, 0-100)",
#         plot_bgcolor=SPOTIFY_BLACK,
#         paper_bgcolor=SPOTIFY_BLACK,
#         font_color=SPOTIFY_WHITE,
#         height=400,
#         xaxis=dict(range=[0, 105], title="Popularity"),
#         margin=dict(l=10, r=40, t=60, b=40),
#     )
#     st.plotly_chart(fig, use_container_width=True)


# def render_genre_shift(genre_counters_by_range):
#     """Animated-feeling comparison: grouped bar chart across all time ranges."""
#     overall = Counter()
#     for counter in genre_counters_by_range.values():
#         overall.update(counter)
#     top_genres = [g for g, _ in overall.most_common(8)]

#     if not top_genres:
#         st.info(
#             "No genre data was returned for your top artists in any time range. "
#             "This is a known Spotify API gap — many artists simply don't have "
#             "genre tags attached. Try the Top Tracks tab instead."
#         )
#         return

#     rows = []
#     for time_range, label in TIME_RANGES.items():
#         counter = genre_counters_by_range[time_range]
#         for genre in top_genres:
#             rows.append({"Genre": genre, "Time Range": label, "Count": counter.get(genre, 0)})

#     df = pd.DataFrame(rows)
#     fig = px.bar(
#         df,
#         x="Genre",
#         y="Count",
#         color="Time Range",
#         barmode="group",
#         color_discrete_sequence=[SPOTIFY_GREEN, "#1ED760", SPOTIFY_GRAY],
#         title="How Your Top Genres Compare Across Time Ranges",
#     )
#     fig.update_layout(
#         plot_bgcolor=SPOTIFY_BLACK,
#         paper_bgcolor=SPOTIFY_BLACK,
#         font_color=SPOTIFY_WHITE,
#         xaxis_tickangle=-30,
#         height=450,
#     )
#     st.plotly_chart(fig, use_container_width=True)


# def render_treemap(tracks, label):
#     """Treemap of top tracks grouped by artist — doesn't depend on genre data."""
#     if not tracks:
#         st.info(f"No top track data available for {label} yet.")
#         return
#     rows = []
#     for rank, track in enumerate(tracks, start=1):
#         artist_name = track["artists"][0]["name"] if track["artists"] else "Unknown"
#         rows.append(
#             {
#                 "Artist": artist_name,
#                 "Track": track["name"],
#                 # Higher score for higher-ranked (earlier) tracks
#                 "Weight": len(tracks) - rank + 1,
#             }
#         )
#     df = pd.DataFrame(rows)
#     fig = px.treemap(
#         df,
#         path=["Artist", "Track"],
#         values="Weight",
#         color="Weight",
#         color_continuous_scale=["#191414", "#1DB954"],
#         title=f"Top Tracks by Artist — {label}",
#     )
#     fig.update_layout(
#         paper_bgcolor=SPOTIFY_BLACK,
#         font_color=SPOTIFY_WHITE,
#         height=450,
#     )
#     st.plotly_chart(fig, use_container_width=True)


# def main():
#     st.markdown(
#         f"<h1 style='color:{SPOTIFY_GREEN};'>🎧 Spotify Listening Analyzer</h1>",
#         unsafe_allow_html=True,
#     )
#     st.caption("An interactive look at your top genres, artists, and how your taste shifts over time.")

#     sp = get_spotify_client()

#     with st.spinner("Fetching your Spotify data..."):
#         data = fetch_all_data(sp)

#     genre_counters_by_range = {
#         tr: genre_counts_from_artists(data[tr]["artists"]) for tr in TIME_RANGES
#     }

#     # Sidebar filter: lets a viewer pick which time range to explore
#     st.sidebar.header("Filters")
#     selected_range = st.sidebar.selectbox(
#         "Time range", options=list(TIME_RANGES.keys()), format_func=lambda k: TIME_RANGES[k]
#     )
#     label = TIME_RANGES[selected_range]

#     tab1, tab2, tab3 = st.tabs(["Overview", "Genre Treemap", "Top Tracks"])

#     with tab1:
#         col1, col2 = st.columns(2)
#         with col1:
#             render_release_year_histogram(data[selected_range]["tracks"], label)
#         with col2:
#             render_artist_bar(data[selected_range]["artists"], label)

#         st.divider()
#         render_mainstream_meter(data)

#     with tab2:
#         render_treemap(genre_counters_by_range[selected_range], label)

#     with tab3:
#         rows = [
#             {
#                 "Rank": i + 1,
#                 "Track": t["name"],
#                 "Artist": ", ".join(a["name"] for a in t["artists"]),
#                 "Album": t["album"]["name"],
#             }
#             for i, t in enumerate(data[selected_range]["tracks"])
#         ]
#         st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# if __name__ == "__main__":
#     main()
