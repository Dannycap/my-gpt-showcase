import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import zipfile
from functools import reduce

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
  [data-testid="stToolbar"] { display: none !important; }
  footer, [data-testid="stStatusWidget"] { display: none !important; height: 0 !important; }
  .block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; }
  button[title="View fullscreen"],
  button[aria-label="View fullscreen"],
  [data-testid="StyledFullScreenButton"],
  [data-testid="stElementToolbar"] { display: none !important; }
</style>
<script src="https://cdn.jsdelivr.net/npm/iframe-resizer/js/iframeResizer.contentWindow.min.js"></script>
""", unsafe_allow_html=True)


# (Optional safety: ensure the sidebar header shows)
with st.sidebar:
    st.header("SEC Settings")

# ==============================================
# SEC EDGAR helpers
# ==============================================
SEC_FILES = "https://www.sec.gov/files"
SEC_DATA = "https://data.sec.gov"


def _normalize_ticker_records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    return []


@st.cache_data(show_spinner=False)
def cik_lookup(ticker: str, ua: str) -> str:
    url = f"{SEC_FILES}/company_tickers.json"
    headers = {"User-Agent": ua}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    t = ticker.upper().strip()
    for rec in _normalize_ticker_records(data):
        try:
            if rec["ticker"].upper() == t:
                return str(rec["cik_str"]).zfill(10)
        except Exception:
            continue
    raise ValueError("Ticker not found on SEC mapping.")


@st.cache_data(show_spinner=False)
def get_company_facts(cik: str, ua: str) -> dict:
    url = f"{SEC_DATA}/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {"User-Agent": ua}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


# ==============================================
# XBRL extraction helpers
# ==============================================
ANNUAL_FORMS = {"10-K", "20-F", "40-F"}
QUARTER_FORMS = {"10-Q"}


def _empty_annual_frame() -> pd.DataFrame:
    cols = ["fy", "fp", "form", "filed", "end", "val"]
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return _ensure_fy_int64(df)


def _ensure_fy_int64(df: pd.DataFrame) -> pd.DataFrame:
    if "fy" in df.columns:
        df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
    return df


def _extract_annual_series(facts: dict, tag: str, unit_preference=("USD", "shares"), limit: int = 20, aggregate_quarters: bool = True) -> pd.DataFrame:
    # Tag may live under us-gaap or dei
    tag_data = None
    for ns in ("us-gaap", "dei"):
        try:
            tag_data = facts["facts"][ns][tag]
            break
        except KeyError:
            continue
    if tag_data is None:
        return _empty_annual_frame()

    series = None
    for unit in unit_preference:
        if unit in tag_data.get("units", {}):
            series = tag_data["units"][unit]
            break
    if not series:
        return _empty_annual_frame()

    rows = []
    for item in series:
        if item.get("fy") is None:
            continue
        if item.get("fp") != "FY":
            continue
        if item.get("form") not in ANNUAL_FORMS:
            continue
        rows.append({
            "fy": item.get("fy"),
            "fp": item.get("fp"),
            "form": item.get("form"),
            "filed": item.get("filed"),
            "end": item.get("end"),
            "val": item.get("val"),
        })
    df = pd.DataFrame(rows)
    if df.empty and aggregate_quarters:
        # Fallback: sum quarters per FY
        q_rows = []
        for item in series:
            if item.get("fy") is None:
                continue
            if item.get("fp") in {"Q1","Q2","Q3","Q4"} and item.get("form") in QUARTER_FORMS:
                q_rows.append({
                    "fy": item.get("fy"),
                    "fp": item.get("fp"),
                    "form": item.get("form"),
                    "filed": item.get("filed"),
                    "end": item.get("end"),
                    "val": item.get("val"),
                })
        if q_rows:
            qdf = pd.DataFrame(q_rows)
            agg = qdf.groupby("fy", as_index=False)["val"].sum()
            agg["fp"] = "FY_from_Q"
            agg["form"] = "10-Q (agg)"
            agg["filed"] = pd.NaT
            agg["end"] = pd.NaT
            df = agg[["fy","fp","form","filed","end","val"]]
    if df.empty:
        return _empty_annual_frame()
    df["filed_dt"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.sort_values(["fy", "filed_dt"]).drop_duplicates("fy", keep="last")
    df = df.sort_values("fy", ascending=False).head(limit)
    df = df.drop(columns=["filed_dt"]).reset_index(drop=True)
    return _ensure_fy_int64(df)


def _extract_coalesced(facts: dict, tags, unit_preference=("USD",), limit: int = 20, label: str | None = None) -> pd.DataFrame:
    frames = []
    for tg in tags:
        df = _extract_annual_series(facts, tg, unit_preference=unit_preference, limit=limit)
        if not df.empty and "val" in df.columns:
            frames.append(df[["fy", "val"]].rename(columns={"val": tg}))
    if not frames:
        return pd.DataFrame({"fy": pd.Series(dtype="Int64"), (label or (tags[0] if tags else "value")): pd.Series(dtype="float64")})
    merged = reduce(lambda l, r: l.merge(r, on="fy", how="outer"), frames).sort_values("fy").reset_index(drop=True)
    merged = _ensure_fy_int64(merged)
    tag_cols = [c for c in merged.columns if c != "fy"]
    merged[label or tags[0]] = merged[tag_cols].bfill(axis=1).iloc[:, 0]
    out = merged[["fy", (label or tags[0])]]
    return _ensure_fy_int64(out)


def _value_column(df: pd.DataFrame, year_col: str = "fy") -> str | None:
    for c in df.columns:
        if c != year_col:
            return c
    return None


# ==============================================
# Yahoo Finance integration (statements + growth + shares)
# ==============================================
@st.cache_data(show_spinner=False)
def yahoo_5y_growth_estimate(ticker: str) -> float | None:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        et = getattr(t, "earnings_trend", None)
        if isinstance(et, pd.DataFrame) and not et.empty:
            cols = {c.lower(): c for c in et.columns}
            if "period" in cols and "growth" in cols:
                mask = et[cols["period"]].astype(str).str.contains("5y|long", case=False, na=False)
                subset = et[mask]
                if not subset.empty:
                    v = subset[cols["growth"]].iloc[0]
                    if isinstance(v, str) and "%" in v:
                        return float(v.replace("%", "").strip()) / 100.0
                    elif pd.api.types.is_number(v):
                        est = float(v)
                        if est > 1:
                            est /= 100.0
                        return est
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def yahoo_statements(ticker: str):
    """Return dict of income/balance/cashflow frames with ['fy', <metrics...>] and sharesOutstanding."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        def tidy(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame({"fy": pd.Series(dtype="Int64")})
            df = df.copy()
            # Columns are dates; convert to year
            def col_to_year(c):
                try:
                    return int(pd.to_datetime(str(c)).year)
                except Exception:
                    try:
                        return int(str(c)[:4])
                    except Exception:
                        return None
            years = [col_to_year(c) for c in df.columns]
            good = [i for i, y in enumerate(years) if y is not None]
            df = df.iloc[:, good]
            years = [years[i] for i in good]
            df.columns = years
            df = df.T.reset_index().rename(columns={"index": "fy"})
            return _ensure_fy_int64(df)

        inc_raw = getattr(t, "financials", pd.DataFrame())
        bs_raw = getattr(t, "balance_sheet", pd.DataFrame())
        cf_raw = getattr(t, "cashflow", pd.DataFrame())

        inc = tidy(inc_raw)
        bs = tidy(bs_raw)
        cf = tidy(cf_raw)

        # Map Yahoo row names -> our labels
        def pick(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame({"fy": pd.Series(dtype="Int64")})
            df_l = df.copy()
            if isinstance(inc_raw, pd.DataFrame) and not inc_raw.empty:
                index_names = {str(i).lower(): str(i) for i in inc_raw.index}
            else:
                index_names = {}
            # Build by looking up source index rows from the original (pre-tidy) df
            out = pd.DataFrame({"fy": df_l["fy"]})
            for label, yahoo_keys in mapping.items():
                val = None
                for yk in yahoo_keys:
                    # Try exact row from raw df
                    try:
                        row = inc_raw.loc[yk] if yk in inc_raw.index else None
                    except Exception:
                        row = None
                    if row is None:
                        # Try case-insensitive
                        candidates = [k for k in inc_raw.index.astype(str) if str(k).lower() == str(yk).lower()] if isinstance(inc_raw, pd.DataFrame) else []
                        row = inc_raw.loc[candidates[0]] if candidates else None
                    if row is not None:
                        # Align to years we kept
                        ser = row.dropna()
                        try:
                            ser.index = [int(pd.to_datetime(str(x)).year) for x in ser.index]
                        except Exception:
                            try:
                                ser.index = [int(str(x)[:4]) for x in ser.index]
                            except Exception:
                                continue
                        val = df_l["fy"].map(ser.to_dict())
                        break
                if val is None:
                    out[label] = np.nan
                else:
                    out[label] = val.values
            return _ensure_fy_int64(out)

        inc_map = {
            "Revenue": ["Total Revenue"],
            "Cost of Revenue": ["Cost Of Revenue", "Cost of Revenue"],
            "Gross Profit": ["Gross Profit"],
            "Operating Income": ["Operating Income"],
            "Pretax Income": ["Income Before Tax", "Ebt"],
            "Net Income": ["Net Income"],
            # EPS diluted isn't always available in financials; skip for now
        }
        bs_map = {
            "Cash & Equivalents": ["Cash And Cash Equivalents", "Cash And Cash Equivalents And Short Term Investments"],
            "Total Assets": ["Total Assets"],
            "Total Liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
            "Long-Term Debt": ["Long Term Debt"],
            "Debt Current": ["Short Long Term Debt", "Short Term Debt"],
            "Current Assets": ["Total Current Assets"],
            "Current Liabilities": ["Total Current Liabilities"],
            "Shareholders' Equity": ["Total Stockholder Equity", "Total Equity Gross Minority Interest"],
        }
        cf_map = {
            "CFO": ["Total Cash From Operating Activities"],
            "CapEx": ["Capital Expenditures"],
            "CF from Investing": ["Total Cashflows From Investing Activities"],
            "CF from Financing": ["Total Cash From Financing Activities"],
        }

        inc_y = pick(inc, inc_map)
        bs_y = pick(bs, bs_map)
        cf_y = pick(cf, cf_map)

        # Shares outstanding from summary info
        shares = None
        try:
            info = getattr(t, "info", {}) or {}
            if isinstance(info, dict):
                so = info.get("sharesOutstanding")
                if so:
                    shares = float(so)
        except Exception:
            shares = None

        return {"income": inc_y, "balance": bs_y, "cashflow": cf_y, "shares": shares}
    except Exception:
        return {"income": pd.DataFrame({"fy": pd.Series(dtype="Int64")}), "balance": pd.DataFrame({"fy": pd.Series(dtype="Int64")}), "cashflow": pd.DataFrame({"fy": pd.Series(dtype="Int64")}), "shares": None}


# ==============================================
# Builders: merge Yahoo (primary) + SEC (fallback)
# ==============================================
def _coalesce_on_fy(primary: pd.DataFrame, fallback: pd.DataFrame, label: str) -> pd.DataFrame:
    """Merge two ['fy', value] frames; prefer primary non-null."""
    p = primary.rename(columns={_value_column(primary, "fy") or "val": label}) if (isinstance(primary, pd.DataFrame) and not primary.empty) else pd.DataFrame({"fy": pd.Series(dtype="Int64"), label: pd.Series(dtype="float64")})
    f = fallback.rename(columns={_value_column(fallback, "fy") or "val": label}) if (isinstance(fallback, pd.DataFrame) and not fallback.empty) else pd.DataFrame({"fy": pd.Series(dtype="Int64"), label: pd.Series(dtype="float64")})
    p = _ensure_fy_int64(p); f = _ensure_fy_int64(f)
    merged = p.merge(f, on="fy", how="outer", suffixes=("_p","_f"))
    if merged.empty:
        return pd.DataFrame({"fy": pd.Series(dtype="Int64"), label: pd.Series(dtype="float64")})
    merged[label] = merged.get(f"{label}_p")
    merged[label] = merged[label].where(pd.notna(merged[label]), merged.get(f"{label}_f"))
    return merged[["fy", label]].sort_values("fy")


@st.cache_data(show_spinner=False)
def build_financials_hybrid(facts: dict, ticker: str, start_year: int = 2015) -> pd.DataFrame:
    y = yahoo_statements(ticker)
    # SEC
    cfo_sec = _extract_coalesced(facts, ["NetCashProvidedByUsedInOperatingActivities",
                                         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
                                         "NetCashProvidedByUsedInOperatingActivitiesIndirectMethod",
                                         "CashFlowsFromUsedInOperatingActivities"], label="CFO")
    capex_sec = _extract_coalesced(facts, ["PaymentsToAcquirePropertyPlantAndEquipment","CapitalExpenditures","PurchaseOfPropertyAndEquipment"], label="CapEx")
    revenue_sec = _extract_coalesced(facts, ["RevenueFromContractWithCustomerExcludingAssessedTax","SalesRevenueNet","Revenues"], label="Revenue")
    shares_sec = _extract_annual_series(facts, "CommonStockSharesOutstanding", unit_preference=("shares",))

    # Yahoo slices (already ['fy', labels...])
    cfo_y = y["cashflow"][["fy","CFO"]] if "CFO" in y["cashflow"].columns else pd.DataFrame({"fy": pd.Series(dtype="Int64"), "CFO": pd.Series(dtype="float64")})
    capex_y = y["cashflow"][["fy","CapEx"]] if "CapEx" in y["cashflow"].columns else pd.DataFrame({"fy": pd.Series(dtype="Int64"), "CapEx": pd.Series(dtype="float64")})
    rev_y = y["income"][["fy","Revenue"]] if "Revenue" in y["income"].columns else pd.DataFrame({"fy": pd.Series(dtype="Int64"), "Revenue": pd.Series(dtype="float64")})

    cfo = _coalesce_on_fy(cfo_y, cfo_sec, "CFO")
    capex = _coalesce_on_fy(capex_y, capex_sec, "CapEx")
    revenue = _coalesce_on_fy(rev_y, revenue_sec, "Revenue")

    # Cash & Debt from SEC (more reliable historically)
    cash = _extract_coalesced(facts, ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"], label="Cash")
    debt_lt = _extract_coalesced(facts, ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"], label="Long-Term Debt")
    debt_st = _extract_coalesced(facts, ["DebtCurrent", "ShortTermBorrowings", "ShortTermDebt", "CommercialPaper", "LongTermDebtCurrent", "CurrentPortionOfLongTermDebt"], label="Current Debt")
    if not debt_lt.empty or not debt_st.empty:
        d1 = debt_lt.rename(columns={_value_column(debt_lt,"fy") or "val": "lt"}) if not debt_lt.empty else pd.DataFrame({"fy": pd.Series(dtype="Int64"), "lt": pd.Series(dtype="float64")})
        d2 = debt_st.rename(columns={_value_column(debt_st,"fy") or "val": "st"}) if not debt_st.empty else pd.DataFrame({"fy": pd.Series(dtype="Int64"), "st": pd.Series(dtype="float64")})
        debt = d1.merge(d2, on="fy", how="outer")
        debt["Debt"] = debt[["lt","st"]].sum(axis=1, skipna=True)
        debt = debt[["fy","Debt"]]
    else:
        debt = pd.DataFrame({"fy": pd.Series(dtype="Int64"), "Debt": pd.Series(dtype="float64")})

    # Shares: prefer Yahoo summary; fallback to SEC annual shares
    shares = None
    if y.get("shares"):
        shares = y["shares"]
    if shares is None and not shares_sec.empty and "val" in shares_sec.columns:
        shares = float(shares_sec["val"].dropna().iloc[-1]) if not shares_sec["val"].dropna().empty else None

    dfs = {"CFO": cfo, "CapEx": capex, "Revenue": revenue, "Cash": cash, "Debt": debt}
    out = None
    for name, df in dfs.items():
        if df is None or df.empty:
            col = pd.DataFrame({"fy": pd.Series(dtype="Int64"), name: pd.Series(dtype="float64")})
        else:
            vcol = "val" if "val" in df.columns else (_value_column(df, "fy") or "val")
            if vcol not in df.columns:
                col = pd.DataFrame({"fy": pd.Series(dtype="Int64"), name: pd.Series(dtype="float64")})
            else:
                col = df[["fy", vcol]].rename(columns={vcol: name})
        col = _ensure_fy_int64(col)
        out = col if out is None else out.merge(col, on="fy", how="outer")

    out = out[out["fy"] >= start_year] if "fy" in out.columns else out
    out = out.sort_values("fy").reset_index(drop=True)
    out["FCF"] = out["CFO"] - out["CapEx"]
    # stash shares scalar
    out.attrs["shares_outstanding_scalar"] = shares
    return out


@st.cache_data(show_spinner=False)
def build_statements_hybrid(facts: dict, ticker: str, start_year: int = 2015):
    y = yahoo_statements(ticker)

    def extract_coalesced(label, tag_list, unit=("USD",)):
        df = _extract_coalesced(facts, tag_list, unit_preference=unit, label=label)
        if df is None or df.empty:
            return pd.DataFrame({"fy": pd.Series(dtype="Int64"), label: pd.Series(dtype="float64")})
        if "val" in df.columns and label not in df.columns:
            df = df.rename(columns={"val": label})
        return _ensure_fy_int64(df)

    def merge_map(tag_map, yahoo_df):
        frames = []
        # Yahoo primary
        if isinstance(yahoo_df, pd.DataFrame) and not yahoo_df.empty:
            ycols = [c for c in yahoo_df.columns if c != "fy"]
            for c in ycols:
                frames.append(yahoo_df[["fy", c]].rename(columns={c: c}))
        # SEC fallback by map
        for label, tags, unit in tag_map:
            if tags is None:
                continue
            unit_pref = (unit,) if isinstance(unit, str) else unit
            frames.append(extract_coalesced(label, tags, unit_pref))
        if not frames:
            return pd.DataFrame({"fy": pd.Series(dtype="Int64")})
        merged = reduce(lambda l, r: pd.merge(l, r, on="fy", how="outer"), frames).sort_values("fy")
        # coalesce duplicates (prefer Yahoo col if duplicates exist)
        for label, _, _ in tag_map:
            if f"{label}_x" in merged.columns and f"{label}_y" in merged.columns:
                merged[label] = merged[f"{label}_x"].where(pd.notna(merged[f"{label}_x"]), merged[f"{label}_y"])
                merged.drop(columns=[f"{label}_x", f"{label}_y"], inplace=True)
        return _ensure_fy_int64(merged)

    income_map = [
        ("Revenue", ["RevenueFromContractWithCustomerExcludingAssessedTax","RevenueFromContractWithCustomerIncludingAssessedTax","SalesRevenueNet","Revenues","SalesRevenueGoodsNet","SalesRevenueServicesNet"], "USD"),
        ("Cost of Revenue", ["CostOfRevenue","CostOfGoodsAndServicesSold","CostOfGoodsSold","CostOfSales"], "USD"),
        ("Gross Profit", ["GrossProfit"], "USD"),
        ("Operating Income", ["OperatingIncomeLoss","IncomeFromOperations"], "USD"),
        ("Pretax Income", ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest","IncomeBeforeIncomeTaxes","IncomeLossFromContinuingOperationsBeforeIncomeTaxes","IncomeBeforeIncomeTaxesAndMinorityInterest"], "USD"),
        ("Net Income", ["NetIncomeLoss"], "USD"),
    ]
    balance_map = [
        ("Cash & Equivalents", ["CashAndCashEquivalentsAtCarryingValue","CashCashEquivalentsAndShortTermInvestments","CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"], "USD"),
        ("Total Assets", ["Assets"], "USD"),
        ("Total Liabilities", ["Liabilities"], "USD"),
        ("Liabilities Current", ["LiabilitiesCurrent"], "USD"),
        ("Liabilities Noncurrent", ["LiabilitiesNoncurrent"], "USD"),
        ("Long-Term Debt", ["LongTermDebt","LongTermDebtNoncurrent","LongTermDebtAndCapitalLeaseObligations"], "USD"),
        ("Debt Current", ["DebtCurrent","ShortTermBorrowings","ShortTermDebt","CommercialPaper","LongTermDebtCurrent","CurrentPortionOfLongTermDebt"], "USD"),
        ("Current Assets", ["AssetsCurrent"], "USD"),
        ("Current Liabilities", ["LiabilitiesCurrent"], "USD"),
        ("Shareholders' Equity", ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest","StockholdersEquity"], "USD"),
    ]
    cashflow_map = [
        ("CFO", ["NetCashProvidedByUsedInOperatingActivities","NetCashProvidedByUsedInOperatingActivitiesContinuingOperations","NetCashProvidedByUsedInOperatingActivitiesIndirectMethod","CashFlowsFromUsedInOperatingActivities"], "USD"),
        ("CapEx", ["PaymentsToAcquirePropertyPlantAndEquipment","CapitalExpenditures","PaymentsToAcquireProductiveAssets","PaymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets","PurchaseOfPropertyAndEquipment"], "USD"),
        ("CF from Investing", ["NetCashProvidedByUsedInInvestingActivities"], "USD"),
        ("CF from Financing", ["NetCashProvidedByUsedInFinancingActivities"], "USD"),
        ("Free Cash Flow", None, "USD"),
    ]

    inc = merge_map(income_map, y["income"])
    bal = merge_map(balance_map, y["balance"])
    cf  = merge_map(cashflow_map, y["cashflow"])

    # Deriveds
    if "Gross Profit" not in inc.columns and all(c in inc.columns for c in ["Revenue", "Cost of Revenue"]):
        inc["Gross Profit"] = inc["Revenue"] - inc["Cost of Revenue"]
    if "Cost of Revenue" not in inc.columns and all(c in inc.columns for c in ["Revenue", "Gross Profit"]):
        inc["Cost of Revenue"] = inc["Revenue"] - inc["Gross Profit"]

    if "Total Liabilities" not in bal.columns and all(c in bal.columns for c in ["Liabilities Current", "Liabilities Noncurrent"]):
        bal["Total Liabilities"] = bal[["Liabilities Current","Liabilities Noncurrent"]].sum(axis=1, skipna=True)
    if "Shareholders' Equity" not in bal.columns and all(c in bal.columns for c in ["Total Assets", "Total Liabilities"]):
        bal["Shareholders' Equity"] = bal["Total Assets"] - bal["Total Liabilities"]
    if "Total Liabilities" not in bal.columns and all(c in bal.columns for c in ["Total Assets", "Shareholders' Equity"]):
        bal["Total Liabilities"] = bal["Total Assets"] - bal["Shareholders' Equity"]
    if "Debt Current" in bal.columns and "Current Debt" not in bal.columns:
        bal["Current Debt"] = bal["Debt Current"]

    # Filter by start year
    for df in (inc, bal, cf):
        if "fy" in df.columns:
            df.dropna(subset=["fy"], inplace=True)
            df = _ensure_fy_int64(df)
            df = df[df["fy"] >= start_year]
        # reassign back
    inc = inc[inc["fy"] >= start_year] if "fy" in inc.columns else inc
    bal = bal[bal["fy"] >= start_year] if "fy" in bal.columns else bal
    cf  = cf[cf["fy"] >= start_year] if "fy" in cf.columns else cf

    inc = inc.sort_values("fy").reset_index(drop=True)
    bal = bal.sort_values("fy").reset_index(drop=True)
    cf  = cf.sort_values("fy").reset_index(drop=True)

    if "CFO" in cf.columns and "CapEx" in cf.columns:
        cf["Free Cash Flow"] = cf["CFO"] - cf["CapEx"]

    return {"income": inc, "balance": bal, "cashflow": cf, "shares": y.get("shares")}


# ==============================================
# Display helpers
# ==============================================
def years_as_columns(df: pd.DataFrame, year_col: str = "fy") -> pd.DataFrame:
    if df is None or df.empty or year_col not in df.columns:
        return df
    tmp = df.copy()
    metric_cols = [c for c in tmp.columns if c != year_col and pd.api.types.is_numeric_dtype(tmp[c])]
    if not metric_cols:
        metric_cols = [c for c in tmp.columns if c != year_col]
    cols = [year_col] + metric_cols
    tmp = tmp[cols].sort_values(year_col)
    wide = tmp.set_index(year_col).T.reset_index().rename(columns={"index": "Metric"})
    return wide


def _series_for_chart(df: pd.DataFrame, years_as_cols: bool, metric: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Year", "Value"])
    try:
        if years_as_cols:
            if "Metric" not in df.columns:
                return pd.DataFrame(columns=["Year", "Value"])
            tmp = df.set_index("Metric")
            if metric not in tmp.index:
                return pd.DataFrame(columns=["Year", "Value"])
            row = tmp.loc[metric]
            s = row.dropna()
            out = pd.DataFrame({"Year": s.index.astype(str), "Value": pd.to_numeric(s.values, errors="coerce")}).dropna()
            return out
        else:
            if "fy" not in df.columns or metric not in df.columns:
                return pd.DataFrame(columns=["Year", "Value"])
            sub = df[["fy", metric]].dropna()
            out = pd.DataFrame({"Year": sub["fy"].astype(str), "Value": pd.to_numeric(sub[metric].values, errors="coerce")}).dropna()
            return out
    except Exception:
        return pd.DataFrame(columns=["Year", "Value"])


# ==============================================
# DCF engine
# ==============================================
def dcf_two_stage_fade(last_fcf: float, years: int, g1: float, wacc: float, terminal_growth: float, fade_years: int = 0):
    if years <= 0:
        raise ValueError("years must be positive")
    if wacc <= terminal_growth:
        raise ValueError("WACC must be greater than terminal growth.")

    fade_years = max(0, min(int(fade_years), int(years)))
    growths = []
    for t in range(1, years + 1):
        if fade_years == 0 or t <= years - fade_years:
            g_t = g1
        else:
            j = t - (years - fade_years)
            frac = j / fade_years
            g_t = g1 + (terminal_growth - g1) * frac
        growths.append(g_t)

    fcfs = []
    cur = last_fcf
    for g in growths:
        cur = cur * (1 + g)
        fcfs.append(cur)

    disc = np.array([1 / ((1 + wacc) ** t) for t in range(1, years + 1)])
    pv_fcfs = np.array(fcfs) * disc
    tv = fcfs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_tv = tv / ((1 + wacc) ** years)
    ev = pv_fcfs.sum() + pv_tv
    return ev, pv_fcfs, pv_tv, np.array(fcfs), np.array(growths)


# ==============================================
# App UI
# ==============================================
st.set_page_config(page_title="DCF", layout="wide")
st.title("DCF")
st.caption("Hybrid fundamentals: Yahoo primary with SEC fallbacks (2015+), charts, and DCF.")

with st.sidebar:
    st.header("SEC Settings")
    ua_email = st.text_input("Your email (required by SEC)", placeholder="you@example.com")
    ticker = st.text_input("Ticker", value="AAPL").upper()

    st.markdown("---")
    st.header("History Window")
    start_year = st.slider("Start year", 2000, 2020, 2015, help="Only include fundamentals from this fiscal year onward.")

    st.markdown("---")
    st.header("DCF Assumptions")
    years = st.slider("Projection years", 3, 10, 5)
    growth = st.slider("FCF growth (Stage 1)", -0.2, 0.3, 0.08, step=0.005)
    fade_years = st.slider("Fade to terminal over last N years", 0, 5, 2)
    wacc = st.slider("Discount rate (WACC)", 0.02, 0.20, 0.10, step=0.005)
    tgr = st.slider("Terminal growth", 0.0, 0.05, 0.025, step=0.0025)

    st.markdown("---")
    st.header("Analyst Estimates (Yahoo)")
    use_yahoo = st.toggle("Use Yahoo analyst 5y growth if available", value=True)

    st.markdown("---")
    st.header("Balance Sheet Adjustments")
    override_cash = st.number_input("Override Cash (USD, optional)", min_value=0.0, value=0.0, step=1e6, format="%0.2f")
    override_debt = st.number_input("Override Debt (USD, optional)", min_value=0.0, value=0.0, step=1e6, format="%0.2f")

    st.markdown("---")
    st.header("Display Options")
    years_as_cols = st.toggle("Show years as columns", value=True)

    st.markdown("---")
    load = st.button("Load / Refresh data", type="primary")

# Initialize store
if "store" not in st.session_state:
    st.session_state.store = None

must_load = False
if load or st.session_state.store is None or st.session_state.store.get("ticker") != ticker:
    must_load = True

if must_load:
    try:
        ua = f"DCF-App/1.0 ({ua_email})" if ua_email else "DCF-App/1.0 (contact@example.com)"
        cik = cik_lookup(ticker, ua)
        facts = get_company_facts(cik, ua)
        fins = build_financials_hybrid(facts, ticker, start_year=start_year)
        stmts = build_statements_hybrid(facts, ticker, start_year=start_year)

        # DCF readiness
        fcf_ok = not fins.empty and ("FCF" in fins.columns) and not fins["FCF"].dropna().empty

        st.session_state.store = {
            "ticker": ticker,
            "ua": ua,
            "cik": cik,
            "facts": facts,
            "fins": fins,
            "stmts": stmts,
            "fcf_ok": fcf_ok,
        }
        st.success(f"Loaded fundamentals for {ticker} (CIK {cik}).")
    except Exception as e:
        st.exception(e)

store = st.session_state.store
if not store or store.get("ticker") != ticker:
    st.stop()

fins = store["fins"]
stmts = store["stmts"]
fcf_ok = store["fcf_ok"]

# Yahoo growth
footnotes = []
yahoo_growth = yahoo_5y_growth_estimate(ticker) if use_yahoo else None
g1 = growth
if use_yahoo and yahoo_growth is not None:
    g1 = float(yahoo_growth)
    footnotes.append(f"Stage 1 growth uses Yahoo Finance 'Next 5 Years (per annum)' estimate: {g1:.2%}.")
elif use_yahoo and yahoo_growth is None:
    footnotes.append("Yahoo Finance 5y growth estimate not available; using the slider value for Stage 1 growth.")

# Shares from hybrid (scalar in attrs) or fallback compute
shares_scalar = fins.attrs.get("shares_outstanding_scalar", None)
if shares_scalar is None:
    # Try balance sheet Yahoo 'Shareholders' Equity' divided by book value per share? (skip)
    pass

# Compute DCF
if fcf_ok:
    last_row = fins.dropna(subset=["FCF"]).iloc[-1]
    last_fcf = float(last_row["FCF"])
    cash_val = override_cash if override_cash > 0 else float(fins["Cash"].dropna().iloc[-1]) if "Cash" in fins.columns and not fins["Cash"].dropna().empty else 0.0
    debt_val = override_debt if override_debt > 0 else float(fins["Debt"].dropna().iloc[-1]) if "Debt" in fins.columns and not fins["Debt"].dropna().empty else 0.0
    shares_val = shares_scalar if shares_scalar is not None else np.nan

    ev, pv_fcfs, pv_tv, fcfs, growths = dcf_two_stage_fade(last_fcf, years, g1, wacc, tgr, fade_years=fade_years)
    equity_value = ev + cash_val - debt_val
    price_per_share = equity_value / shares_val if pd.notna(shares_val) and shares_val > 0 else np.nan
else:
    last_fcf = np.nan
    cash_val = float(fins["Cash"].dropna().iloc[-1]) if "Cash" in fins.columns and not fins["Cash"].dropna().empty else 0.0
    debt_val = float(fins["Debt"].dropna().iloc[-1]) if "Debt" in fins.columns and not fins["Debt"].dropna().empty else 0.0
    shares_val = shares_scalar if shares_scalar is not None else np.nan
    ev = pv_fcfs = pv_tv = equity_value = price_per_share = np.nan
    fcfs = growths = np.array([])

kpis = {
    "Ticker": store["ticker"],
    "CIK": store["cik"],
    "Last FCF (USD)": last_fcf,
    "Cash (USD)": cash_val,
    "Debt (USD)": debt_val,
    "Enterprise Value (EV)": ev,
    "Equity Value": equity_value,
    "Shares (outstanding)": shares_val,
    "Implied Price / Share": price_per_share,
    "Projection years": years,
    "Stage 1 growth (g1)": g1,
    "Fade years": fade_years,
    "WACC": wacc,
    "Terminal growth": tgr,
}
kpi_df = pd.DataFrame({"Metric": list(kpis.keys()), "Value": list(kpis.values())})

if fcf_ok:
    proj_years = list(range(1, years + 1))
    proj = pd.DataFrame({
        "Year": [f"Year {i}" for i in proj_years],
        "Assumed Growth": [float(x) for x in growths] if len(growths) else [np.nan]*years,
        "FCF": [float(x) for x in fcfs] if len(fcfs) else [np.nan]*years,
        "PV of FCF": [float(x) for x in pv_fcfs] if isinstance(pv_fcfs, np.ndarray) and pv_fcfs.size else [np.nan]*years,
    })
else:
    proj = pd.DataFrame({
        "Year": [f"Year {i}" for i in range(1, years + 1)],
        "Assumed Growth": [np.nan]*years,
        "FCF": [np.nan]*years,
        "PV of FCF": [np.nan]*years,
    })

# Sensitivity
sens_df = None
if fcf_ok and pd.notna(shares_val) and shares_val > 0:
    wacc_grid = np.round(np.linspace(max(0.05, wacc-0.03), min(0.20, wacc+0.03), 7), 4)
    tgr_grid = np.round(np.linspace(max(0.0, tgr-0.01), min(0.05, tgr+0.01), 5), 4)
    sens = []
    for w in wacc_grid:
        row_vals = []
        for g in tgr_grid:
            try:
                ev_, _, _, _, _ = dcf_two_stage_fade(last_fcf, years, g1, w, g, fade_years=fade_years)
                eq = ev_ + cash_val - debt_val
                price = eq / shares_val
            except Exception:
                price = np.nan
            row_vals.append(price)
        sens.append(row_vals)
    sens_df = pd.DataFrame(sens, index=[f"WACC {x:.2%}" for x in wacc_grid], columns=[f"g {x:.2%}" for x in tgr_grid])

# Tabs
tab_dcf, tab_is, tab_bs, tab_cf, tab_export = st.tabs(["DCF", "Income", "Balance", "Cash Flow", "Export"])

# Footnotes
if pd.isna(shares_val):
    footnotes.append("Shares outstanding not returned by Yahoo summary; price per share left blank.")

with tab_dcf:
    st.subheader("DCF Results")
    st.dataframe(kpi_df, use_container_width=True)

    st.subheader("Projected FCF & PV")
    st.dataframe(proj, use_container_width=True)
    st.caption(f"Terminal value PV: {pv_tv:,.0f}" if isinstance(pv_tv, (int,float,np.floating)) and np.isfinite(pv_tv) else "Terminal value PV: n/a")
    if footnotes:
        with st.expander("Footnotes / assumptions"):
            for i, note in enumerate(footnotes, 1):
                st.markdown(f"{i}. {note}")

    st.subheader("FCF Build (annual)")
    st.dataframe(years_as_columns(fins) if years_as_cols else fins, use_container_width=True)

# Statement tabs + charts
def statement_block(title: str, df: pd.DataFrame, key_prefix: str):
    st.subheader(f"{title} (annual)")
    st.dataframe(years_as_columns(df) if years_as_cols else df, use_container_width=True)
    st.markdown("### Chart this statement")
    _df_display = years_as_columns(df) if years_as_cols else df
    if _df_display is None or _df_display.empty:
        st.info("No data to chart.")
        return
    if years_as_cols:
        metric_options = [m for m in _df_display.get("Metric", []).tolist()] if "Metric" in _df_display.columns else []
    else:
        metric_options = [c for c in _df_display.columns if c != "fy" and pd.api.types.is_numeric_dtype(_df_display[c])]
    if not metric_options:
        st.info("No numeric metrics found to chart.")
        return
    c1, c2 = st.columns([2,1])
    with c1:
        sel_metric = st.selectbox("Metric", metric_options, key=f"{key_prefix}_metric")
    with c2:
        chart_type = st.radio("Chart type", ["Line", "Bar"], horizontal=True, key=f"{key_prefix}_chart")
    chart_df = _series_for_chart(_df_display, years_as_cols, sel_metric)
    if chart_df.empty:
        st.info("Nothing to plot for the selected metric.")
    else:
        if chart_type == "Line":
            st.line_chart(chart_df, x="Year", y="Value", use_container_width=True)
        else:
            st.bar_chart(chart_df, x="Year", y="Value", use_container_width=True)

with tab_is:
    statement_block("Income Statement", stmts["income"], "is")

with tab_bs:
    statement_block("Balance Sheet", stmts["balance"], "bs")

with tab_cf:
    statement_block("Cash Flow Statement", stmts["cashflow"], "cf")

def _excel_bytes_or_zip_csvs(ticker: str, stmts: dict, fins: pd.DataFrame, kpi_df: pd.DataFrame, proj: pd.DataFrame, sens_df: pd.DataFrame | None):
    engine = None
    try:
        import xlsxwriter  # noqa
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa
            engine = "openpyxl"
        except Exception:
            engine = None
    if engine:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine=engine) as writer:
            stmts["income"].to_excel(writer, index=False, sheet_name="Income_Statement")
            stmts["balance"].to_excel(writer, index=False, sheet_name="Balance_Sheet")
            stmts["cashflow"].to_excel(writer, index=False, sheet_name="Cash_Flow")
            fins.to_excel(writer, index=False, sheet_name="FCF_Build")
            kpi_df.to_excel(writer, index=False, sheet_name="DCF_KPIs")
            proj.to_excel(writer, index=False, sheet_name="Projection")
            if sens_df is not None:
                sens_df.to_excel(writer, sheet_name="Sensitivity")
        return ("excel", buf.getvalue(), engine)
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{ticker}_income.csv", stmts["income"].to_csv(index=False))
        z.writestr(f"{ticker}_balance.csv", stmts["balance"].to_csv(index=False))
        z.writestr(f"{ticker}_cashflow.csv", stmts["cashflow"].to_csv(index=False))
        z.writestr(f"{ticker}_fcf_build.csv", fins.to_csv(index=False))
        z.writestr(f"{ticker}_dcf_kpis.csv", kpi_df.to_csv(index=False))
        z.writestr(f"{ticker}_projection.csv", proj.to_csv(index=False))
        if sens_df is not None:
            z.writestr(f"{ticker}_sensitivity.csv", sens_df.to_csv())
    return ("zip", zbuf.getvalue(), None)

with tab_export:
    st.subheader("Export Results")
    inc_out = years_as_columns(stmts["income"]) if years_as_cols else stmts["income"]
    bal_out = years_as_columns(stmts["balance"]) if years_as_cols else stmts["balance"]
    cf_out  = years_as_columns(stmts["cashflow"]) if years_as_cols else stmts["cashflow"]
    fins_out = years_as_columns(fins) if years_as_cols else fins

    colA, colB = st.columns(2)
    with colA:
        st.download_button("Income Statement (CSV)", data=inc_out.to_csv(index=False).encode("utf-8"), file_name=f"{store['ticker']}_income.csv", mime="text/csv")
        st.download_button("Balance Sheet (CSV)", data=bal_out.to_csv(index=False).encode("utf-8"), file_name=f"{store['ticker']}_balance.csv", mime="text/csv")
        st.download_button("Cash Flow Statement (CSV)", data=cf_out.to_csv(index=False).encode("utf-8"), file_name=f"{store['ticker']}_cashflow.csv", mime="text/csv")
    with colB:
        st.download_button("DCF KPIs (CSV)", data=kpi_df.to_csv(index=False).encode("utf-8"), file_name=f"{store['ticker']}_dcf_kpis.csv", mime="text/csv")
        st.download_button("Projection (CSV)", data=proj.to_csv(index=False).encode("utf-8"), file_name=f"{store['ticker']}_projection.csv", mime="text/csv")
        if sens_df is not None:
            st.download_button("Sensitivity (CSV)", data=sens_df.to_csv().encode("utf-8"), file_name=f"{store['ticker']}_sensitivity.csv", mime="text/csv")

    export_stmts = {"income": inc_out, "balance": bal_out, "cashflow": cf_out}
    kind, payload, engine = _excel_bytes_or_zip_csvs(store["ticker"], export_stmts, fins_out, kpi_df, proj, sens_df)
    if kind == "excel":
        st.download_button(
            "All Sheets (Excel)",
            data=payload,
            file_name=f"{store['ticker']}_dcf_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help=f"Generated with pandas ExcelWriter engine={engine}"
        )
    else:
        st.info("Excel engine not available in this environment; offering a ZIP of CSVs instead.")
        st.download_button(
            "All Sheets (ZIP of CSV)",
            data=payload,
            file_name=f"{store['ticker']}_dcf_export.zip",
            mime="application/zip"
        )
