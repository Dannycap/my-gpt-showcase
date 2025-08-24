import numpy as np
import pandas as pd
from typing import Optional, Tuple

try:
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio import PerfMeasure, RatioMeasure, RiskMeasure
except ImportError as e:  # pragma: no cover - handled at runtime
    raise RuntimeError("skfolio must be installed to use finance logic") from e

from skfolio.preprocessing import prices_to_returns
from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    """Return sample S&P 500 price data."""
    return load_sp500_dataset()


def compute_efficient_frontier(
    prices: pd.DataFrame,
    test_size: float = 0.33,
    efficient_frontier_size: int = 30,
    risk_measure: Optional[RiskMeasure] = None,
    min_return: Optional[np.ndarray] = None,
) -> Tuple[object, object, object]:
    """Fit a mean–risk model and compute train/test populations."""
    if risk_measure is None:
        risk_measure = RiskMeasure.VARIANCE

    returns = prices_to_returns(prices)
    X_train, X_test = train_test_split(returns, test_size=test_size, shuffle=False)

    model = MeanRisk(
        risk_measure=risk_measure,
        efficient_frontier_size=efficient_frontier_size,
        portfolio_params=dict(name=risk_measure.name.capitalize()),
        min_return=min_return,
    )
    model.fit(X_train)
    population_train = model.predict(X_train)
    population_test = model.predict(X_test)

    population_train.set_portfolio_params(tag="Train")
    population_test.set_portfolio_params(tag="Test")
    population = population_train + population_test
    return population_train, population_test, population


def summarize_population(population) -> pd.DataFrame:
    """Return summary stats for a population of portfolios."""
    return population.summary()
