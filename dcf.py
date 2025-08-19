import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import zipfile

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


def _extract_annual_series(facts: dict, tag: str, unit_preference=("USD", "shares"), limit: int = 10) -> pd.DataFrame:
    try:
        tag_data = facts["facts"]["us-gaap"][tag]
    except KeyError:
        return pd.DataFrame(columns=["fy", "fp", "form", "filed", "end", "val"]).astype({"fy": "Int64"})

    series = None
    for unit in unit_preference:
        if unit in tag_data.get("units", {}):
            series = tag_data["units"][unit]
            break
    if not series:
        return pd.DataFrame(columns=["fy", "fp", "form", "filed", "end", "val"]).astype({"fy": "Int64"})

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
    if df.empty:
        return df.astype({"fy": "Int64"})
    df["filed_dt"] = pd.to_datetime(df["filed"], errors="coerce")
    df = df.sort_values(["fy", "filed_dt"]).drop_duplicates("fy", keep="last")
    df = df.sort_values("fy", ascending=False).head(limit)
    return df.drop(columns=["filed_dt"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def build_financials(facts: dict) -> pd.DataFrame:
    cfo = _extract_annual_series(facts, "NetCashProvidedByUsedInOperatingActivities")
    capex = _extract_annual_series(facts, "PaymentsToAcquirePropertyPlantAndEquipment")
    revenue = _extract_annual_series(facts, "Revenues")
    if revenue.empty:
        revenue = _extract_annual_series(facts, "SalesRevenueNet")
    net_income = _extract_annual_series(facts, "NetIncomeLoss")

    shares = _extract_annual_series(facts, "CommonStockSharesOutstanding", unit_preference=("shares",))
    if shares.empty:
        shares = _extract_annual_series(facts, "EntityCommonStockSharesOutstanding", unit_preference=("shares",))

    cash = _extract_annual_series(facts, "CashAndCashEquivalentsAtCarryingValue")
    if cash.empty:
        cash = _extract_annual_series(facts, "CashCashEquivalentsAndShortTermInvestments")
    debt = _extract_annual_series(facts, "LongTermDebtNoncurrent")
    if debt.empty:
        debt = _extract_annual_series(facts, "LongTermDebt")

    dfs = {"CFO": cfo, "CapEx": capex, "Revenue": revenue, "Net Income": net_income, "Shares": shares, "Cash": cash, "Debt": debt}
    out = None
    for name, df in dfs.items():
        col = df[["fy", "val"]].rename(columns={"val": name}) if not df.empty else pd.DataFrame({"fy": [], name: []})
        out = col if out is None else out.merge(col, on="fy", how="outer")
    out = out.sort_values("fy").reset_index(drop=True)
    out["FCF"] = out["CFO"] - out["CapEx"]
    return out


@st.cache_data(show_spinner=False)
def build_statements(facts: dict, limit: int = 10):
    def extract_first(tag_list, unit=("USD",)):
        for tg in tag_list:
            df = _extract_annual_series(facts, tg, unit_preference=unit, limit=limit)
            if not df.empty:
                return df[["fy", "val"]].rename(columns={"val": tg})
        return pd.DataFrame({"fy": [], tag_list[0]: []})

    def merge_map(tag_map):
        out = None
        for label, tags, unit in tag_map:
            if tags is None:
                col = pd.DataFrame({"fy": []})
            else:
                unit_pref = (unit,) if isinstance(unit, str) else unit
                base = extract_first(tags, unit_pref)
                if not base.empty:
                    base = base.rename(columns={base.columns[-1]: label})
                else:
                    base = pd.DataFrame({"fy": [], label: []})
                col = base
            out = col if out is None else out.merge(col, on="fy", how="outer")
        if out is None:
            out = pd.DataFrame({"fy": []})
        return out

    income_map = [
        ("Revenue", ["Revenues", "SalesRevenueNet"], "USD"),
        ("Cost of Revenue", ["CostOfRevenue"], "USD"),
        ("Gross Profit", ["GrossProfit"], "USD"),
        ("Operating Income", ["OperatingIncomeLoss"], "USD"),
        ("Pretax Income", ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "IncomeBeforeIncomeTaxes"], "USD"),
        ("Net Income", ["NetIncomeLoss"], "USD"),
        ("EPS (Diluted)", ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"], "USD/shares"),
    ]
    balance_map = [
        ("Cash & Equivalents", ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"], "USD"),
        ("Total Assets", ["Assets"], "USD"),
        ("Total Liabilities", ["Liabilities"], "USD"),
        ("Long-Term Debt", ["LongTermDebt", "LongTermDebtNoncurrent"], "USD"),
        ("Current Assets", ["AssetsCurrent"], "USD"),
        ("Current Liabilities", ["LiabilitiesCurrent"], "USD"),
        ("Shareholders' Equity", ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"], "USD"),
    ]
    cashflow_map = [
        ("CFO", ["NetCashProvidedByUsedInOperatingActivities"], "USD"),
        ("CapEx", ["PaymentsToAcquirePropertyPlantAndEquipment"], "USD"),
        ("CF from Investing", ["NetCashProvidedByUsedInInvestingActivities"], "USD"),
        ("CF from Financing", ["NetCashProvidedByUsedInFinancingActivities"], "USD"),
        ("Free Cash Flow", None, "USD"),
    ]

    inc = merge_map(income_map)
    bal = merge_map(balance_map)
    cf = merge_map(cashflow_map)
    if "CFO" in cf.columns and "CapEx" in cf.columns:
        cf["Free Cash Flow"] = cf["CFO"] - cf["CapEx"]

    inc = inc.sort_values("fy").reset_index(drop=True)
    bal = bal.sort_values("fy").reset_index(drop=True)
    cf = cf.sort_values("fy").reset_index(drop=True)

    return {"income": inc, "balance": bal, "cashflow": cf}


# ==============================================
# DCF engine
# ==============================================
def dcf_two_stage(last_fcf: float, g_years: int, growth: float, wacc: float, terminal_growth: float):
    fcfs = [last_fcf * ((1 + growth) ** t) for t in range(1, g_years + 1)]
    disc = np.array([1 / ((1 + wacc) ** t) for t in range(1, g_years + 1)])
    pv_fcfs = np.array(fcfs) * disc

    fcf_terminal_next = fcfs[-1] * (1 + terminal_growth)
    if wacc <= terminal_growth:
        raise ValueError("WACC must be greater than terminal growth.")
    tv = fcf_terminal_next / (wacc - terminal_growth)
    pv_tv = tv / ((1 + wacc) ** g_years)

    ev = pv_fcfs.sum() + pv_tv
    return ev, pv_fcfs, pv_tv


# ==============================================
# App UI (no emojis) + persistent data
# ==============================================
st.set_page_config(page_title="DCF", layout="wide")
with open("styles.css") as css:
    st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
st.title("DCF")
st.caption("Load a stock once, then navigate between DCF, Income, Balance, and Cash Flow without re-running fetches.")

with st.sidebar:
    st.header("SEC Settings")
    ua_email = st.text_input(
        "Your email (required by SEC)",
        placeholder="you@example.com",
        help="Used when requesting SEC filings",
    )
    ticker = st.text_input(
        "Ticker",
        value="AAPL",
        help="Public company ticker symbol",
    ).upper()

    st.markdown("---")
    st.header("DCF Assumptions")
    years = st.slider("Projection years", 3, 10, 5, help="Years to project free cash flow")
    growth = st.slider(
        "FCF growth (Years 1..N)",
        -0.2,
        0.3,
        0.08,
        step=0.01,
        help="Annual growth rate for projected FCF",
    )
    wacc = st.slider(
        "Discount rate (WACC)",
        0.02,
        0.20,
        0.10,
        step=0.005,
        help="Weighted average cost of capital",
    )
    tgr = st.slider(
        "Terminal growth",
        0.0,
        0.05,
        0.025,
        step=0.0025,
        help="Perpetual growth after projection period",
    )

    st.markdown("---")
    st.header("Balance Sheet Adjustments")
    override_cash = st.number_input("Override Cash (USD, optional)", min_value=0.0, value=0.0, step=1e6, format="%0.2f")
    override_debt = st.number_input("Override Debt (USD, optional)", min_value=0.0, value=0.0, step=1e6, format="%0.2f")

    st.markdown("---")
    st.header("Q&A (optional)")
    enable_ai = st.toggle("Enable Q&A", value=False)
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    temperature = st.slider(
        "Creativity (temperature)",
        0.0,
        1.0,
        0.2,
        0.05,
        help="Higher values yield more varied answers",
    )

    st.markdown("---")
    load = st.button("Load / Refresh data", type="primary")

# Initialize store
if "store" not in st.session_state:
    st.session_state.store = None
if "derived" not in st.session_state:
    st.session_state.derived = None

# Load or refresh data only when button clicked OR when ticker changed with no store
must_load = False
if load:
    must_load = True
elif st.session_state.store is None:
    must_load = True
elif st.session_state.store and st.session_state.store.get("ticker") != ticker:
    st.info("Ticker changed. Click Load / Refresh data to fetch new company data.")
else:
    pass

if must_load:
    try:
        ua = f"DCF-App/1.0 ({ua_email})" if ua_email else "DCF-App/1.0 (contact@example.com)"
        cik = cik_lookup(ticker, ua)
        facts = get_company_facts(cik, ua)
        fins = build_financials(facts)
        stmts = build_statements(facts)
        if fins.empty or fins["FCF"].dropna().empty:
            st.error("Couldn't derive FCF from SEC facts (CFO or CapEx missing). Try a different ticker.")
        else:
            st.session_state.store = {
                "ticker": ticker,
                "ua": ua,
                "cik": cik,
                "facts": facts,
                "fins": fins,
                "stmts": stmts,
            }
            st.success(f"Loaded fundamentals for {ticker} (CIK {cik}).")
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        if code in (403, 429):
            st.error("SEC blocked the request (HTTP %s). Enter a real email in the sidebar and click Load again." % code)
        else:
            st.exception(e)
    except Exception as e:
        st.exception(e)

store = st.session_state.store
if not store or store.get("ticker") != ticker:
    st.stop()

# Recompute DCF every run using current sliders, reusing fetched financials
fins = store["fins"]
stmts = store["stmts"]

last_row = fins.dropna(subset=["FCF"]).iloc[-1]
last_fcf = float(last_row["FCF"])
cash = override_cash if override_cash > 0 else float(fins["Cash"].dropna().iloc[-1]) if not fins["Cash"].dropna().empty else 0.0
debt = override_debt if override_debt > 0 else float(fins["Debt"].dropna().iloc[-1]) if not fins["Debt"].dropna().empty else 0.0
shares = float(fins["Shares"].dropna().iloc[-1]) if not fins["Shares"].dropna().empty else np.nan

ev, pv_fcfs, pv_tv = dcf_two_stage(last_fcf, years, growth, wacc, tgr)
equity_value = ev + cash - debt
price_per_share = equity_value / shares if pd.notna(shares) and shares > 0 else np.nan

kpis = {
    "Ticker": store["ticker"],
    "CIK": store["cik"],
    "Last FCF (USD)": last_fcf,
    "Cash (USD)": cash,
    "Debt (USD)": debt,
    "Enterprise Value (EV)": ev,
    "Equity Value": equity_value,
    "Shares (outstanding)": shares,
    "Implied Price / Share": price_per_share,
    "Years": years,
    "FCF growth": growth,
    "WACC": wacc,
    "Terminal growth": tgr,
}
kpi_df = pd.DataFrame({"Metric": list(kpis.keys()), "Value": list(kpis.values())})
proj = pd.DataFrame({
    "Year": [f"Year {i}" for i in range(1, years + 1)],
    "FCF": [last_fcf * ((1 + growth) ** i) for i in range(1, years + 1)],
    "PV of FCF": pv_fcfs,
})

# Sensitivity
sens_df = None
if pd.notna(shares) and shares > 0:
    wacc_grid = np.round(np.linspace(max(0.05, wacc-0.03), min(0.20, wacc+0.03), 7), 4)
    tgr_grid = np.round(np.linspace(max(0.0, tgr-0.01), min(0.05, tgr+0.01), 5), 4)
    sens = []
    for w in wacc_grid:
        row_vals = []
        for g in tgr_grid:
            try:
                ev_, _, _ = dcf_two_stage(last_fcf, years, growth, w, g)
                eq = ev_ + cash - debt
                price = eq / shares
            except Exception:
                price = np.nan
            row_vals.append(price)
        sens.append(row_vals)
    sens_df = pd.DataFrame(sens, index=[f"WACC {x:.2%}" for x in wacc_grid], columns=[f"g {x:.2%}" for x in tgr_grid])

# Persist derived for Q&A/export
st.session_state.derived = {
    "kpis": kpis, "kpi_df": kpi_df, "proj": proj, "sens_df": sens_df,
    "pv_fcfs": pv_fcfs, "pv_tv": pv_tv, "last_fcf": last_fcf,
    "shares": shares, "cash": cash, "debt": debt,
}

# =============== Tabs ===============
tab_dcf, tab_is, tab_bs, tab_cf, tab_export = st.tabs(
    ["DCF", "Income", "Balance", "Cash Flow", "Export"]
)

def _build_context_blob(ctx_store: dict, derived: dict, max_rows: int = 8) -> str:
    fins_rows = ctx_store["fins"].head(max_rows).to_dict(orient="records")
    proj_rows = derived["proj"].head(max_rows).to_dict(orient="records")
    kpis_local = derived["kpis"]
    parts = [
        f"Ticker: {ctx_store.get('ticker')}  |  CIK: {ctx_store.get('cik')}",
        "\n[KPIs]\n" + pd.Series(kpis_local).to_string(),
        "\n[Financials sample]\n" + pd.DataFrame(fins_rows).to_csv(index=False),
        "\n[Projection sample]\n" + pd.DataFrame(proj_rows).to_csv(index=False),
    ]
    return "\n".join([str(p) for p in parts])


def _ask_openai(api_key: str, model: str, temperature: float, question: str, context_blob: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system = (
        "You are a precise financial analyst. Answer ONLY using the provided context. "
        "Cite exact figures from the tables when relevant. If something is unknown, say you don't know."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context_blob}\n\nQuestion: {question}"},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": 800,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text[:500]}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def render_qa_panel(suffix: str):
    st.markdown("---")
    st.subheader("Q&A")
    if enable_ai:
        with st.form(f"qa_form_{suffix}"):
            q = st.text_area("Your question", placeholder="For example: What drives the implied price most? Any red flags in recent FCF trends?", height=100, key=f"q_{suffix}")
            show_ctx = st.checkbox("Show context sent to the model", value=False, key=f"showctx_{suffix}")
            asked = st.form_submit_button("Ask")
        if asked:
            if not openai_key:
                st.error("Please enter your OpenAI API key in the sidebar.")
            else:
                try:
                    ctx_blob = _build_context_blob(st.session_state.store, st.session_state.derived)
                    if show_ctx:
                        with st.expander("Context preview"):
                            st.code(ctx_blob[:5000])
                    answer = _ask_openai(openai_key, model, temperature, q, ctx_blob)
                    st.success("Answer")
                    st.write(answer)
                except Exception as e:
                    st.exception(e)
    else:
        st.info("Enable Q&A in the sidebar to ask questions.")


with tab_dcf:
    st.subheader("DCF Results")
    st.dataframe(kpi_df, use_container_width=True)
    st.subheader("Projected FCF & PV")
    st.dataframe(proj, use_container_width=True)
    st.caption(f"Terminal value PV: {pv_tv:,.0f}")
    if sens_df is not None:
        st.subheader("Sensitivity (Implied Price) – WACC vs Terminal Growth")
        st.dataframe(sens_df.style.format("{:.2f}"), use_container_width=True)
    else:
        st.info("Shares outstanding not found; sensitivity not shown as price per share.")
    render_qa_panel("dcf")

with tab_is:
    st.subheader("Income Statement (annual)")
    st.dataframe(stmts["income"], use_container_width=True)
    render_qa_panel("is")

with tab_bs:
    st.subheader("Balance Sheet (annual)")
    st.dataframe(stmts["balance"], use_container_width=True)
    render_qa_panel("bs")

with tab_cf:
    st.subheader("Cash Flow Statement (annual)")
    st.dataframe(stmts["cashflow"], use_container_width=True)
    render_qa_panel("cf")

def _excel_bytes_or_zip_csvs(ticker: str, stmts: dict, fins: pd.DataFrame, kpi_df: pd.DataFrame, proj: pd.DataFrame, sens_df: pd.DataFrame | None):
    # Try Excel with xlsxwriter, then openpyxl; else return a ZIP of CSVs.
    engine = None
    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401
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

    # Fallback: ZIP of CSVs
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
    colA, colB = st.columns(2)
    with colA:
        st.download_button("Income Statement (CSV)", data=stmts["income"].to_csv(index=False).encode("utf-8"), file_name=f"{store['ticker']}_income.csv", mime="text/csv")
        st.download_button("Balance Sheet (CSV)", data=stmts["balance"].to_csv(index=False).encode("utf-8"), file_name=f"{store['ticker']}_balance.csv", mime="text/csv")
        st.download_button("Cash Flow Statement (CSV)", data=stmts["cashflow"].to_csv(index=False).encode("utf-8"), file_name=f"{store['ticker']}_cashflow.csv", mime="text/csv")
    with colB:
        st.download_button("DCF KPIs (CSV)", data=kpi_df.to_csv(index=False).encode("utf-8"), file_name=f"{store['ticker']}_dcf_kpis.csv", mime="text/csv")
        st.download_button("Projection (CSV)", data=proj.to_csv(index=False).encode("utf-8"), file_name=f"{store['ticker']}_projection.csv", mime="text/csv")
        if sens_df is not None:
            st.download_button("Sensitivity (CSV)", data=sens_df.to_csv().encode("utf-8"), file_name=f"{store['ticker']}_sensitivity.csv", mime="text/csv")

    kind, payload, engine = _excel_bytes_or_zip_csvs(store["ticker"], stmts, fins, kpi_df, proj, sens_df)
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
    render_qa_panel("export")

st.caption("Tip: After you load a ticker, you can switch tabs freely. Change DCF sliders any time; the DCF recomputes from stored fundamentals without reloading from the SEC.")
