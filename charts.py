"""
charts.py — Visualization functions for the dashboard.
Handles Dataset Statistics, Model Performance, and Top Words.
"""

import numpy as np                                              # numerical operations (sorting, arrays)
import streamlit as st                                          # web app framework
import plotly.graph_objects as go                                # low-level Plotly API for custom charts
import plotly.express as px                                      # high-level Plotly API (used for heatmap)
from sklearn.model_selection import train_test_split             # splits data into train/test sets
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score  # model evaluation metrics

# Shared color mapping so every chart uses the same green/gray/red scheme
COLOR_MAP = {1: "#63b022", 0: "#94a3b8", -1: "#d03b3b"}          # class code -> hex color (green/gray/red)
LABEL_MAP = {1: "Positive", 0: "Neutral", -1: "Negative"}        # class code -> human-readable label
ORDER = [1, 0, -1]                                                # display order: Positive, Neutral, Negative


def render_dataset_stats(df):
    """Metric cards + donut chart + stacked proportion bar."""
    st.markdown("## 📊 Dataset statistics")                      # section header

    counts = df["category"].value_counts().reindex(ORDER).fillna(0).astype(int)
    # ^ count rows per class, force them into Positive/Neutral/Negative order,
    #   fill any missing class with 0, and cast to whole numbers

    c1, c2, c3, c4 = st.columns(4)                                # 4 equal-width side-by-side columns
    c1.metric("Total comments", f"{len(df):,}")                   # total row count, comma-formatted (e.g. 37,249)
    c2.metric("Positive", f"{counts[1]:,}")                       # count where category == 1
    c3.metric("Neutral", f"{counts[0]:,}")                        # count where category == 0
    c4.metric("Negative", f"{counts[-1]:,}")                      # count where category == -1

    col1, col2 = st.columns(2)                                    # two columns: donut chart | stacked bar

    with col1:
        fig = go.Figure(data=[go.Pie(                             # build a pie chart figure
            labels=[LABEL_MAP[k] for k in ORDER],                 # slice labels in Positive/Neutral/Negative order
            values=[counts[k] for k in ORDER],                    # slice sizes matching that same order
            hole=0.6,                                             # hole size 0.6 turns the pie into a donut
            marker=dict(colors=[COLOR_MAP[k] for k in ORDER]),    # slice colors matching our color scheme
            sort=False,                                           # keep our explicit order, don't auto-sort by size
            textinfo="none",                                      # hide default labels on the slices themselves
        )])
        fig.update_layout(template="plotly_dark",                 # use Plotly's built-in dark theme
                           title="Sentiment distribution",         # chart title
                           paper_bgcolor="rgba(0,0,0,0)",           # transparent outer background
                           plot_bgcolor="rgba(0,0,0,0)",            # transparent plotting area background
                           margin=dict(t=50, b=10, l=10, r=10),     # tight margins around the chart
                           height=300)                              # fixed chart height in pixels
        st.plotly_chart(fig, use_container_width=True)              # render it, stretching to the column's width

    with col2:
        pct = (counts / counts.sum() * 100).round(1)                # convert raw counts to percentages, 1 decimal
        fig2 = go.Figure()                                          # start an empty figure
        for k in ORDER:                                             # add one bar segment per class
            fig2.add_trace(go.Bar(
                x=[pct[k]], y=["All comments"], orientation="h",    # horizontal bar, single row labeled "All comments"
                name=f"{LABEL_MAP[k]} {pct[k]}%",                   # legend text shows label + percentage
                marker_color=COLOR_MAP[k],                          # bar segment color
            ))
        fig2.update_layout(barmode="stack",                         # stack the 3 segments into one bar
                            template="plotly_dark",
                            title="Proportion breakdown",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(visible=False),               # hide the x-axis (percentages shown in legend)
                            yaxis=dict(visible=False),                # hide the y-axis (only one row anyway)
                            legend=dict(orientation="h", y=-0.3),      # horizontal legend below the chart
                            margin=dict(t=50, b=10, l=10, r=10),
                            height=300)
        st.plotly_chart(fig2, use_container_width=True)


def render_model_performance(model, vectorizer, df):
    """Accuracy/F1/algorithm metric cards + confusion matrix heatmap."""
    st.markdown("## 🎯 Model performance")                         # section header

    X = vectorizer.transform(df["clean_comment"])                  # convert all cleaned comments into TF-IDF vectors
    y = df["category"]                                             # true labels for every comment

    _, X_test, _, y_test = train_test_split(                       # recreate a held-out test split for evaluation
        X, y, test_size=0.2, random_state=42, stratify=y           # 20% test, fixed seed, keep class balance
    )                                                               # (the underscores discard the training portions,
                                                                     # since we only need the test set here)
    y_pred = model.predict(X_test)                                 # run the model on the test set
    acc = accuracy_score(y_test, y_pred)                            # compute overall accuracy
    f1 = f1_score(y_test, y_pred, average="weighted")               # compute F1 score, weighted by class size

    c1, c2, c3 = st.columns(3)                                      # 3 columns for the performance metrics
    c1.metric("Accuracy", f"{acc * 100:.2f}%")                      # accuracy as a percentage, 2 decimals
    c2.metric("F1 score", f"{f1:.2f}")                              # F1 score, 2 decimals
    c3.metric("Algorithm", "Linear SVM")                            # static label, not computed

    cm = confusion_matrix(y_test, y_pred, labels=ORDER)             # build the confusion matrix in our class order
    fig = px.imshow(                                                # render the matrix as a colored heatmap
        cm, text_auto=True,                                         # show the raw numbers inside each cell
        x=[f"Pred. {LABEL_MAP[k].lower()[:3]}" for k in ORDER],     # x-axis labels: "Pred. pos", "Pred. neu", etc.
        y=[f"Actual {LABEL_MAP[k].lower()[:3]}" for k in ORDER],    # y-axis labels: "Actual pos", "Actual neu", etc.
        color_continuous_scale="Blues",                             # blue color gradient for cell intensity
        labels=dict(color="Count"),                                 # legend title for the color scale
    )
    fig.update_layout(template="plotly_dark",
                       title="Confusion matrix",
                       paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       coloraxis_showscale=False,                    # hide the color scale bar (keeps it compact)
                       margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)


def render_top_words(model, vectorizer, top_n=8):
    """Three side-by-side bar charts: top TF-IDF words per sentiment class."""
    st.markdown("## 🔑 Top words per sentiment")                    # section header

    feature_names = np.array(vectorizer.get_feature_names_out())    # array of every word in the TF-IDF vocabulary
    classes = list(model.classes_)                                  # e.g. [-1, 0, 1], in the order sklearn stores them
    cols = st.columns(len(ORDER))                                   # one column per class (3 total)

    for i, cls in enumerate(ORDER):                                 # loop through Positive, Neutral, Negative
        idx_in_model = classes.index(cls)                           # find where this class sits in model.coef_
        coefs = model.coef_[idx_in_model]                           # get that class's word weight (coefficient) array
        top_idx = np.argsort(coefs)[-top_n:][::-1]                  # indices of the top_n highest weights, descending
        top_words = feature_names[top_idx]                          # look up the actual words at those indices
        top_scores = coefs[top_idx]                                 # and their corresponding weights

        fig = go.Figure(go.Bar(
            x=top_scores[::-1], y=top_words[::-1], orientation="h", # reverse order so the strongest word ends up on top
            marker_color=COLOR_MAP[cls],                            # bar color matches this class's scheme color
        ))
        fig.update_layout(template="plotly_dark", title=LABEL_MAP[cls],
                           paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)",
                           height=320, margin=dict(t=40, b=10, l=10, r=10))
        with cols[i]:                                                # place this chart in its own column
            st.plotly_chart(fig, use_container_width=True)