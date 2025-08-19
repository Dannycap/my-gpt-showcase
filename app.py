"""
Interactive Efficient Frontier Web App
=====================================

This Streamlit application recreates the functionality of your
`plot_3_efficient_frontier` notebook without exposing the code cells to
visitors. Users can adjust model parameters, run the efficient frontier
optimization, visualize the results, and use a *chat-style* interface to
request different kinds of explanations (e.g., "focus on drawdowns",
"explain like I'm new to portfolio theory") powered by OpenAI.
No personalized financial advice is provided.

Dependencies
------------

The app relies on the following Python packages:

* streamlit
* pandas
* numpy
* skfolio
* plotly
* openai

Install them with pip:

    pip install streamlit pandas numpy skfolio plotly openai

Running the App
---------------

Once dependencies are installed, run the app from a terminal with:

    streamlit run app.py

Streamlit will start a local development server and open the app in
your default browser.

Note on OpenAI API Keys
-----------------------

If you would like the application to generate chat-based explanations,
enter a valid OpenAI API key in the sidebar. If left blank, the chat
feature will show an informative message instead of calling the API.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go  # noqa: F401 (kept for compatibility)
from typing import Tuple, Optional

# ----------------------------------------------------------------------
# Optional dependencies (skfolio)
# ----------------------------------------------------------------------
try:
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio import PerfMeasure, RatioMeasure, RiskMeasure
except ImportError:
    load_sp500_dataset = None
    MeanRisk = None
    PerfMeasure = None
    RatioMeasure = None
    RiskMeasure = None

# ----------------------------------------------------------------------
# Optional dependency (openai) - support v1.x client and v0.x fallback
# ----------------------------------------------------------------------
try:
    import openai  # type: ignore
    OpenAIClient = getattr(openai, "OpenAI", None)
except ImportError:
    openai = None  # type: ignore
    OpenAIClient = None


# =============================================================================
# Data / Modeling helpers
# =============================================================================
def load_data() -> pd.DataFrame:
    """Load and return the S&P 500 sample price dataset via skfolio."""
    if load_sp500_dataset is None:
        raise RuntimeError(
            "skfolio is not installed. Please install skfolio to load the dataset."
        )
    return load_sp500_dataset()


def compute_efficient_frontier(
    prices: pd.DataFrame,
    test_size: float = 0.33,
    efficient_frontier_size: int = 30,
    risk_measure=None,
    min_return: Optional[np.ndarray] = None,
) -> Tuple[object, object, object]:
    """
    Fit a mean–risk model and compute train/test populations plus a combined population.

    Returns
    -------
    population_train, population_test, population
    """
    from skfolio.preprocessing import prices_to_returns
    from sklearn.model_selection import train_test_split

    if risk_measure is None:
        risk_measure = RiskMeasure.VARIANCE

    # Convert prices to returns
    returns = prices_to_returns(prices)

    # Train/test split without shuffling: first part for training
    X_train, X_test = train_test_split(returns, test_size=test_size, shuffle=False)

    # Configure the optimization model
    model = MeanRisk(
        risk_measure=risk_measure,
        efficient_frontier_size=efficient_frontier_size,
        portfolio_params=dict(name=risk_measure.name.capitalize()),
        min_return=min_return,
    )

    # Fit the model and predict on train and test sets
    model.fit(X_train)
    population_train = model.predict(X_train)
    population_test = model.predict(X_test)

    # Tag the portfolios for color coding later
    population_train.set_portfolio_params(tag="Train")
    population_test.set_portfolio_params(tag="Test")

    # Concatenate populations
    population = population_train + population_test
    return population_train, population_test, population


def plot_population(
    population,
    x=None,
    y=None,
    color_scale=None,
    hover_measures=None,
):
    """Generate a Plotly scatter chart for the population of portfolios."""
    # Set defaults here to avoid referencing skfolio enums at import time if missing
    x = x or RiskMeasure.ANNUALIZED_STANDARD_DEVIATION
    y = y or PerfMeasure.ANNUALIZED_MEAN
    color_scale = color_scale or RatioMeasure.ANNUALIZED_SHARPE_RATIO
    hover_measures = hover_measures or [
        RiskMeasure.MAX_DRAWDOWN,
        RatioMeasure.ANNUALIZED_SORTINO_RATIO,
    ]
    fig = population.plot_measures(
        x=x,
        y=y,
        color_scale=color_scale,
        hover_measures=hover_measures,
    )
    return fig


def summarize_population(population) -> pd.DataFrame:
    """Return a summary DataFrame for a population of portfolios."""
    return population.summary()


# =============================================================================
# Utility helpers
# =============================================================================
def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that a DataFrame has unique column names for Streamlit display.
    """
    df = df.copy()
    seen = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    df.columns = new_cols
    return df


# =============================================================================
# Chat helper (non-advisory, user-steerable)
# =============================================================================
def chat_with_model(
    api_key: str,
    messages: list,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Send a chat turn to OpenAI using either openai v1.x client or v0.x fallback.
    `messages` must be a list of dicts: [{"role": "...", "content": "..."}]
    """
    if not api_key:
        return "No API key provided. Enter one in the sidebar to chat."
    if openai is None:
        return "openai package is not installed. Please install it to use the chat."

    try:
        if OpenAIClient is not None:
            client = OpenAIClient(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.5,
                max_tokens=800,
            )
            return resp.choices[0].message.content
        else:
            openai.api_key = api_key  # type: ignore
            resp = openai.ChatCompletion.create(  # type: ignore
                model=model,
                messages=messages,
                temperature=0.5,
                max_tokens=800,
            )
            return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling OpenAI: {e}"


# =============================================================================
# App
# =============================================================================
def main() -> None:
    # Configure page (must be first Streamlit call)
    icon_path = "icon.png"
    if os.path.exists(icon_path):
        st.set_page_config(
            page_title="Efficient Frontier Explorer",
            page_icon=icon_path,
            layout="wide",
        )
        st.image(icon_path, width=80)
    else:
        st.set_page_config(page_title="Efficient Frontier Explorer", layout="wide")

    st.title("Efficient Frontier Explorer")
    st.write("Explore the efficient frontier using your own data or sample data.")

    # -----------------------
    # Sidebar: parameters
    # -----------------------
    st.sidebar.header("Model Parameters")
    if MeanRisk is None or RiskMeasure is None:
        st.sidebar.warning(
            "skfolio is not available. Install it to enable optimization and charts."
        )

    test_size = st.sidebar.slider(
        "Test set size", min_value=0.1, max_value=0.9, value=0.33, step=0.05
    )
    frontier_size = st.sidebar.slider(
        "Number of portfolios", min_value=5, max_value=100, value=30, step=5
    )

    # Protect against skfolio missing
    risk_measure = None
    if RiskMeasure is not None:
        risk_measure_option = st.sidebar.selectbox(
            "Risk measure",
            options=[
                (RiskMeasure.VARIANCE, "Variance"),
                (RiskMeasure.SEMI_VARIANCE, "Semi-Variance"),
                (RiskMeasure.CVAR, "Conditional Value at Risk"),
            ],
            format_func=lambda x: x[1],
        )
        risk_measure = risk_measure_option[0]

    min_return_enabled = st.sidebar.checkbox(
        "Specify minimum annualized return?", value=False
    )
    if min_return_enabled:
        min_ret_input = st.sidebar.text_input(
            "Minimum annualized returns (comma-separated, e.g. 0.05,0.10)",
            "0.05,0.10,0.15",
        )
        try:
            # Convert annualized levels to daily (approx 252 trading days)
            min_return_values = [
                float(x.strip()) / 252 for x in min_ret_input.split(",") if x.strip()
            ]
            min_return = np.array(min_return_values)
        except ValueError:
            st.sidebar.error("Could not parse minimum return values.")
            min_return = None
    else:
        min_return = None

    api_key = st.sidebar.text_input(
        "OpenAI API key (optional)", type="password", placeholder="sk-..."
    )
    run_button = st.sidebar.button("Run optimization")

    uploaded_file = st.sidebar.file_uploader(
        "Upload your own price CSV (optional)",
        type=["csv"],
        help="CSV with a date column and columns per asset; if provided, it replaces the default S&P 500 sample dataset.",
    )

    # Placeholders
    plot_placeholder = st.empty()
    summary_placeholder = st.empty()
    chat_placeholder = st.empty()

    # -----------------------
    # Run
    # -----------------------
    if run_button:
        # Load data (uploaded or sample)
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                # Infer date column
                date_col = None
                if "Date" in df.columns:
                    date_col = "Date"
                elif df.columns[0].lower().startswith("date"):
                    date_col = df.columns[0]

                if date_col is not None:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.set_index(date_col)
                else:
                    df.index = pd.to_datetime(df.iloc[:, 0])
                    df = df.drop(df.columns[0], axis=1)

                prices = df.sort_index()
            else:
                prices = load_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

        if MeanRisk is None or RiskMeasure is None:
            st.error("Optimization unavailable because skfolio is not installed.")
            return

        with st.spinner("Computing efficient frontier..."):
            try:
                population_train, population_test, population = compute_efficient_frontier(
                    prices=prices,
                    test_size=test_size,
                    efficient_frontier_size=frontier_size,
                    risk_measure=risk_measure,
                    min_return=min_return,
                )
            except Exception as e:
                st.error(f"Error during optimization: {e}")
                return

            # -----------------------------------------------------------------
            # Additional charts/info
            # -----------------------------------------------------------------
            # 1) Daily returns line chart
            st.subheader("Daily Returns for Each Asset")
            try:
                import plotly.express as px

                returns = prices.pct_change().dropna()
                fig_returns = px.line(
                    returns,
                    x=returns.index,
                    y=returns.columns,
                    labels={"value": "Daily Return", "variable": "Ticker", "x": "Date"},
                )
                plot_placeholder.plotly_chart(
                    fig_returns, use_container_width=True, key="returns_chart"
                )
            except Exception as e:
                st.write(f"Could not plot daily returns: {e}")

            # 2) Efficient frontier scatter (train + test)
            st.subheader("Efficient Frontier (Train + Test)")
            try:
                fig_frontier = plot_population(population)
                st.plotly_chart(fig_frontier, use_container_width=True, key="frontier_chart")
            except Exception as e:
                st.write(f"Could not plot efficient frontier: {e}")

            # 3) Weights shape
            try:
                sample_portfolio = population_train[0]
                weights_shape = sample_portfolio.weights.shape
            except Exception:
                weights_shape = ("unknown",)
            st.write(f"Weights array shape: {weights_shape}")

            # 4) Portfolio composition
            st.subheader("Portfolio Composition")
            col3, col4 = st.columns(2)
            with col3:
                st.write("Train portfolios composition")
                try:
                    fig_train = population_train.plot_composition()
                    st.plotly_chart(fig_train, use_container_width=True, key="train_composition")
                except Exception as e:
                    st.write(f"Could not plot train composition: {e}")
            with col4:
                st.write("Test portfolios composition")
                try:
                    fig_test = population_test.plot_composition()
                    st.plotly_chart(fig_test, use_container_width=True, key="test_composition")
                except Exception as e:
                    st.write(f"Could not plot test composition: {e}")

            # 5) Measures (e.g., Sharpe)
            try:
                measures_df = population_test.measures(
                    measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO
                )
                measures_df_unique = make_unique_columns(measures_df)
                st.subheader("Test Portfolio Measures (Annualized Sharpe Ratio)")
                st.dataframe(measures_df_unique)
            except Exception as e:
                st.write(f"Could not compute portfolio measures: {e}")

            # 6) Summary statistics
            summary_stats = summarize_population(population)
            summary_stats_unique = make_unique_columns(summary_stats)
            st.subheader("Summary Statistics")
            st.dataframe(summary_stats_unique)

            # 7) Optional: constrained frontier (train only)
            if min_return is not None and len(min_return) > 0:
                try:
                    from skfolio.optimization import MeanRisk as MR

                    model2 = MR(
                        risk_measure=risk_measure,
                        min_return=min_return,
                        portfolio_params=dict(name=risk_measure.name.capitalize()),
                    )
                    # Train slice only
                    n_train = int(len(prices.pct_change().dropna()) * (1 - test_size))
                    population_min = model2.fit_predict(
                        prices.pct_change().dropna().iloc[:n_train]
                    )

                    st.subheader("Efficient Frontier with Minimum Return Constraint (Train)")
                    fig_min = population_min.plot_measures(
                        x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
                        y=PerfMeasure.ANNUALIZED_MEAN,
                        color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
                        hover_measures=[
                            RiskMeasure.MAX_DRAWDOWN,
                            RatioMeasure.ANNUALIZED_SORTINO_RATIO,
                        ],
                    )
                    st.plotly_chart(fig_min, use_container_width=True, key="min_frontier_chart")
                except Exception as e:
                    st.write(f"Could not compute constrained efficient frontier: {e}")

            # 8) Interactive, non-advisory analysis chat
            st.subheader("Interactive Analysis Chat")

            with st.expander("Chat settings", expanded=True):
                preset = st.selectbox(
                    "Starting instruction (choose one or write your own below)",
                    [
                        "Summarize key risk/return patterns and trade-offs. Avoid giving advice.",
                        "Explain how to read the efficient frontier for a beginner. No recommendations.",
                        "Compare train vs test portfolios and highlight robustness concerns.",
                        "Identify outliers in Sharpe/Sortino and discuss plausible reasons (data-driven only).",
                        "Translate the stats into plain English for a non-technical audience.",
                    ],
                    index=0,
                    help="This is a starting steer. You can override or add more instructions in the chat.",
                )
                custom_instruction = st.text_area(
                    "Optional: add your own instruction",
                    placeholder="e.g., Focus on downside risk and maximum drawdown. Avoid prescriptive portfolio advice.",
                )
                include_tables = st.checkbox(
                    "Attach summary table context to each message",
                    value=True,
                    help="Includes a compact copy of the current summary stats so the model can reference them.",
                )

            # Initialize chat state
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            system_preamble = (
                "You are an analytical assistant. You explain observations from provided data "
                "in a neutral, informational way. Do NOT give personalized financial advice, "
                'allocations, or recommendations. Avoid prescriptive language like "you should". '
                "Be clear and concise."
            )

            base_messages = [{"role": "system", "content": system_preamble}]

            # Optional: provide compact table context
            if include_tables:
                compact_table = summary_stats_unique.to_csv(index=True)
                # Keep under a cap to avoid token bloat
                if len(compact_table) > 15000:
                    compact_table = compact_table[:15000]
                base_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Context: CSV of summary statistics for efficient-frontier portfolios.\n"
                            "Use this only as reference.\n\n" + compact_table
                        ),
                    }
                )

            # Show chat history
            for m in st.session_state.chat_messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])

            # Starter suggestion
            starter = custom_instruction.strip() if custom_instruction.strip() else preset

            if len(st.session_state.chat_messages) == 0:
                with st.chat_message("assistant"):
                    st.markdown(
                        "Tell me **how** you'd like these results interpreted (e.g., "
                        "_‘focus on downside risk and avoid advice’_, _‘explain like I’m new to portfolio theory’_, "
                        "_‘compare train vs test robustness’_)."
                    )
                    if starter:
                        st.caption(f"Suggested start: _{starter}_")

            # Chat input
            user_turn = st.chat_input("Type your instruction or question about the results…")
            if user_turn:
                st.session_state.chat_messages.append({"role": "user", "content": user_turn})
                with st.chat_message("user"):
                    st.markdown(user_turn)

                composed = base_messages + [
                    {"role": "user", "content": f"Interpretation steer: {starter}"},
                ] + st.session_state.chat_messages

                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        reply = chat_with_model(api_key=api_key, messages=composed)
                        st.markdown(reply)
                st.session_state.chat_messages.append({"role": "assistant", "content": reply})

            st.caption(
                "Note: This chat explains patterns in the uploaded/sample data. "
                "It avoids prescriptive or personalized financial advice."
            )


if __name__ == "__main__":
    main()
