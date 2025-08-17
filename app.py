"""
Interactive Efficient Frontier Web App
====================================

This Streamlit application recreates the functionality of your
`plot_3_efficient_frontier` notebook without exposing the code cells to
visitors.  Users can adjust model parameters, run the efficient
frontier optimization, visualize the results and optionally get a
natural‑language summary powered by OpenAI.  To run the app locally
you'll need to install a few dependencies (see instructions below).

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

```bash
pip install streamlit pandas numpy skfolio plotly openai
```

Running the App
---------------

Once dependencies are installed, run the app from a terminal with:

```bash
streamlit run app.py
```

Streamlit will start a local development server and open the app in
your default browser.

Note on OpenAI API Keys
-----------------------

If you would like the application to generate a natural‑language summary
of your results, enter a valid OpenAI API key in the text field.  If
left blank, the summary step will be skipped.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
# Brand tokens (match investgptai.io)
COLOR_PRIMARY = "#3B82F6"   # blue
COLOR_BG      = "#F9FAFB"   # page bg
COLOR_CARD    = "#FFFFFF"   # card bg
COLOR_TEXT    = "#1F2937"   # dark text

st.set_page_config(
    page_title="Efficient Frontier Explorer",
    page_icon="icon.png",   # keep your logo next to app.py
    layout="wide"
)

# Plotly theme to match site
pio.templates["investgptai"] = pio.templates["plotly_white"]
pio.templates["investgptai"].layout.update(
    paper_bgcolor=COLOR_CARD,
    plot_bgcolor=COLOR_CARD,
    font=dict(color=COLOR_TEXT, family="Inter, system-ui, Segoe UI, Arial, sans-serif"),
    colorway=["#3B82F6","#0EA5E9","#10B981","#F59E0B","#EF4444","#8B5CF6","#14B8A6","#F472B6"],
    xaxis=dict(gridcolor="#E5E7EB"),
    yaxis=dict(gridcolor="#E5E7EB")
)
pio.templates.default = "investgptai"

# Light CSS polish
st.markdown(f"""
<style>
  .stApp {{ background:{COLOR_BG}; }}
  .block-container {{ padding-top:1.5rem; padding-bottom:3rem; }}
  h1,h2,h3,h4,h5,h6 {{ color:{COLOR_TEXT}; letter-spacing:.1px; }}
  .stButton > button {{
    background:{COLOR_PRIMARY}; color:#fff; border:0; border-radius:12px;
    padding:.6rem 1rem; box-shadow:0 4px 12px rgba(59,130,246,.15);
  }}
  .stButton > button:hover {{ filter:brightness(1.04); }}
  a, a:visited {{ color:{COLOR_PRIMARY}; }}
</style>
""", unsafe_allow_html=True)
# --- /Theme -------------------------------------------------------------------
try:
    # Import optional dependencies when available.  These are only
    # required for the optimization and plotting functions provided by
    # skfolio.
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio import PerfMeasure, RatioMeasure, RiskMeasure
except ImportError:
    load_sp500_dataset = None
    MeanRisk = None
    PerfMeasure = None
    RatioMeasure = None
    RiskMeasure = None

try:
    # Import the openai package.  Different versions expose different APIs.
    import openai  # type: ignore
    # The modern v1.x library exposes an `OpenAI` class for client
    # instantiation; older versions (<1.0) do not.  We'll detect and
    # accommodate both.
    OpenAIClient = getattr(openai, "OpenAI", None)
except ImportError:
    openai = None  # type: ignore
    OpenAIClient = None


def load_data() -> pd.DataFrame:
    """Load and return the S&P 500 price dataset.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame of daily price data for a collection of
        S&P 500 constituents.  Raises an informative exception if
        `skfolio` is not installed.
    """
    if load_sp500_dataset is None:
        raise RuntimeError(
            "skfolio is not installed. Please install skfolio to load the dataset."
        )
    return load_sp500_dataset()


def compute_efficient_frontier(
    prices: pd.DataFrame,
    test_size: float = 0.33,
    efficient_frontier_size: int = 30,
    risk_measure: RiskMeasure = None,
    min_return: np.ndarray | None = None,
) -> tuple:
    """Fit a mean‑risk model and compute the population of efficient portfolios.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data with datetime index and asset tickers as columns.
    test_size : float, optional
        Fraction of the data to hold out for testing.  The first portion of
        the dataset is used for training and the remainder for testing.
    efficient_frontier_size : int, optional
        Number of portfolios along the efficient frontier.
    risk_measure : skfolio.RiskMeasure, optional
        Choice of risk measure.  If `None`, variance is used.
    min_return : np.ndarray or None, optional
        Target return levels for the optimization.  Leave `None` to fit
        without a return constraint.

    Returns
    -------
    tuple
        (population_train, population_test, population) where each
        population is a list of portfolios.  The combined `population` is
        the concatenation of train and test portfolios.
    """
    from skfolio.preprocessing import prices_to_returns
    from sklearn.model_selection import train_test_split

    if risk_measure is None:
        risk_measure = RiskMeasure.VARIANCE

    # Convert prices to returns
    returns = prices_to_returns(prices)

    # Train/test split without shuffling: first part for training
    X_train, X_test = train_test_split(
        returns, test_size=test_size, shuffle=False
    )

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
    x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    hover_measures=None,
):
    """Generate a Plotly scatter chart for the population of portfolios.

    This wraps the built‑in `skfolio.population.plot_measures` method so that
    the resulting figure can be embedded into the Streamlit app.
    """
    if hover_measures is None:
        hover_measures = [RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO]

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


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that a DataFrame has unique column names.

    Streamlit (via PyArrow) cannot display DataFrames with duplicate column
    names.  This function appends a suffix ("_1", "_2", ...) to any
    duplicated columns to make them unique.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that may contain duplicate column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with unique column names.
    """
    df = df.copy()
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    df.columns = new_cols
    return df


def generate_chatgpt_summary(summary_stats: pd.DataFrame, api_key: str) -> str:
    """Generate a narrative summary using OpenAI's ChatGPT.

    This helper supports both the v1.x `openai.OpenAI` client class and
    earlier versions of the library where `openai` exposes top‑level
    `ChatCompletion` methods.

    Parameters
    ----------
    summary_stats : pd.DataFrame
        The summary statistics DataFrame returned by `population.summary()`.
    api_key : str
        A valid OpenAI API key.  If empty or invalid, an informative
        message is returned instead of calling the API.

    Returns
    -------
    str
        The assistant's response or an error message.
    """
    if not api_key:
        return "No API key provided. Skipping ChatGPT summary."
    if openai is None:
        return "openai package is not installed. Please install it to use ChatGPT."

    prompt = (
        "Here are summary statistics for a set of efficient frontier portfolios:\n"
        f"{summary_stats.to_string()}\n\n"
        "Please summarize the results for portfolios (ptf0 to ptf29), highlight any patterns or notable features, "
        "and explain their potential implications."
    )
    try:
        # v1.x of openai-python: use OpenAI client class
        if OpenAIClient is not None:
            client = OpenAIClient(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            return response.choices[0].message.content
        # v0.x of openai-python: set API key and call ChatCompletion directly
        else:
            openai.api_key = api_key  # type: ignore
            response = openai.ChatCompletion.create(  # type: ignore
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful financial analyst."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling OpenAI: {e}"


def main() -> None:
    """Run the Streamlit app."""
    # Configure the page.  If an "icon.png" file is present in the current
    # directory, use it as the page icon; otherwise fall back to default.
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
    st.write(
        "Explore the efficient frontier of portfolios using your own data or sample data."
    )

    # Sidebar for user inputs
    st.sidebar.header("Model Parameters")
    test_size = st.sidebar.slider(
        "Test set size", min_value=0.1, max_value=0.9, value=0.33, step=0.05
    )
    frontier_size = st.sidebar.slider(
        "Number of portfolios", min_value=5, max_value=100, value=30, step=5
    )
    risk_measure_option = st.sidebar.selectbox(
        "Risk measure",
        options=[
            (RiskMeasure.VARIANCE, "Variance"),
            (RiskMeasure.SEMI_VARIANCE, "Semi‑Variance"),
            (RiskMeasure.CVAR, "Conditional Value at Risk"),
        ],
        format_func=lambda x: x[1],
    )
    risk_measure = risk_measure_option[0]
    # Optional minimum return constraint
    min_return_enabled = st.sidebar.checkbox(
        "Specify minimum annualized return?", value=False
    )
    if min_return_enabled:
        min_ret_input = st.sidebar.text_input(
            "Minimum annualized returns (comma‑separated, e.g. 0.05,0.10)",
            "0.05,0.10,0.15",
        )
        try:
            min_return_values = [float(x.strip()) / 252 for x in min_ret_input.split(",") if x.strip()]
            min_return = np.array(min_return_values)
        except ValueError:
            st.sidebar.error("Could not parse minimum return values.")
            min_return = None
    else:
        min_return = None

    api_key = st.sidebar.text_input(
        "OpenAI API key (optional)",
        type="password",
        placeholder="sk-..."
    )

    run_button = st.sidebar.button("Run optimization")

    # Allow the user to upload their own CSV of price data.  The CSV should
    # contain a date column (either unnamed index or named "Date") and
    # columns for each asset's prices.  Dates will be parsed automatically.
    uploaded_file = st.sidebar.file_uploader(
        "Upload your own price CSV (optional)",
        type=["csv"],
        help="CSV with a date column and columns per asset; if provided, it will replace the default SP500 dataset.",
    )

    # Placeholder containers for results
    plot_placeholder = st.empty()
    summary_placeholder = st.empty()
    chat_placeholder = st.empty()

    if run_button:
        # Load data: either the user's uploaded CSV or the built‑in S&P 500 dataset.
        try:
            if uploaded_file is not None:
                # Read the uploaded CSV.  Use the first column or a "Date" column as the index.
                df = pd.read_csv(uploaded_file)
                # Determine date column
                date_col = None
                # If first column looks like a date or is named "Date", use it
                if "Date" in df.columns:
                    date_col = "Date"
                elif df.columns[0].lower().startswith("date"):
                    date_col = df.columns[0]
                if date_col is not None:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.set_index(date_col)
                else:
                    # Try converting the index from the unnamed first column
                    df.index = pd.to_datetime(df.iloc[:, 0])
                    df = df.drop(df.columns[0], axis=1)
                prices = df.sort_index()
            else:
                prices = load_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

        with st.spinner("Computing efficient frontier..."):
            try:
                population_train, population_test, population = compute_efficient_frontier(
                    prices,
                    test_size=test_size,
                    efficient_frontier_size=frontier_size,
                    risk_measure=risk_measure,
                    min_return=min_return,
                )
            except Exception as e:
                st.error(f"Error during optimization: {e}")
                return

            # -----------------------------------------------------------------
            # Compute and display additional notebook charts and information
            # -----------------------------------------------------------------
            # 1. Compute daily returns and display line chart
            returns = prices.pct_change().dropna()
            st.subheader("Daily Returns for Each Asset")
            # Use Plotly for an interactive multi‑line chart
            import plotly.express as px
            fig_returns = px.line(
                returns,
                x=returns.index,
                y=returns.columns,
                labels={"value": "Daily Return", "variable": "Ticker", "x": "Date"},
            )
            plot_placeholder.plotly_chart(fig_returns, use_container_width=True, key="returns_chart")

            # 2. Plot efficient frontier scatter for train+test
            st.subheader("Efficient Frontier (Train + Test)")
            fig = plot_population(population)
            st.plotly_chart(fig, use_container_width=True, key="frontier_chart")

            # 3. Show shape of weights array (number of portfolios x number of assets)
            # Use the fitted model from the training set to get weights_.  All
            # portfolios in population share the same weight dimension.
            try:
                sample_portfolio = population_train[0]
                weights_shape = sample_portfolio.weights.shape
            except Exception:
                weights_shape = ("unknown",)
            st.write(f"Weights array shape: {weights_shape}")

            # 4. Plot composition of train and test portfolios
            st.subheader("Portfolio Composition")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Train portfolios composition")
                fig_train = population_train.plot_composition()
                st.plotly_chart(fig_train, use_container_width=True, key="train_composition")
            with col2:
                st.write("Test portfolios composition")
                fig_test = population_test.plot_composition()
                st.plotly_chart(fig_test, use_container_width=True, key="test_composition")

            # 5. Show performance measures of the test portfolios (e.g., Sharpe ratio)
            try:
                measures_df = population_test.measures(
                    measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO
                )
                # Ensure unique column names for display
                measures_df_unique = make_unique_columns(measures_df)
                st.subheader("Test Portfolio Measures (Annualized Sharpe Ratio)")
                st.dataframe(measures_df_unique)
            except Exception as e:
                st.write(f"Could not compute portfolio measures: {e}")

            # 6. Show summary statistics
            summary_stats = summarize_population(population)
            summary_stats_unique = make_unique_columns(summary_stats)
            st.subheader("Summary Statistics")
            st.dataframe(summary_stats_unique)

            # 7. If a minimum return array is provided, run a second optimization on the
            #    training set only and display its efficient frontier
            if min_return is not None and len(min_return) > 0:
                try:
                    from skfolio.optimization import MeanRisk as MR
                    model2 = MR(
                        risk_measure=risk_measure,
                        min_return=min_return,
                        portfolio_params=dict(name=risk_measure.name.capitalize()),
                    )
                    population_min = model2.fit_predict(returns.iloc[: int(len(returns) * (1 - test_size))])
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

            # 8. ChatGPT summary
            st.subheader("ChatGPT Summary")
            chat_result = generate_chatgpt_summary(summary_stats_unique, api_key)
            st.write(chat_result)


if __name__ == "__main__":
    main()
