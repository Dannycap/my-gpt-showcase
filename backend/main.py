from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from .logic import (
    load_data,
    compute_efficient_frontier,
    summarize_population,
)
from skfolio import RiskMeasure, PerfMeasure, RatioMeasure

app = FastAPI(title="Finance API")


class FrontierRequest(BaseModel):
    test_size: float = 0.33
    efficient_frontier_size: int = 30
    risk_measure: str = "VARIANCE"
    min_return: Optional[List[float]] = None


@app.get("/")
def read_root():
    return {"message": "Finance API is running"}


@app.post("/efficient-frontier")
def efficient_frontier(req: FrontierRequest):
    prices = load_data()
    risk_measure = getattr(RiskMeasure, req.risk_measure)
    min_return = np.array(req.min_return) if req.min_return else None

    _, _, population = compute_efficient_frontier(
        prices,
        test_size=req.test_size,
        efficient_frontier_size=req.efficient_frontier_size,
        risk_measure=risk_measure,
        min_return=min_return,
    )

    frame = population.to_frame(
        measures=[
            RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
            PerfMeasure.ANNUALIZED_MEAN,
            RatioMeasure.ANNUALIZED_SHARPE_RATIO,
        ]
    )
    summary = summarize_population(population)

    return {
        "plot": {
            "risk": frame["ANNUALIZED_STANDARD_DEVIATION"].tolist(),
            "return": frame["ANNUALIZED_MEAN"].tolist(),
            "sharpe": frame["ANNUALIZED_SHARPE_RATIO"].tolist(),
        },
        "summary": summary.to_dict(orient="records"),
    }
