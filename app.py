import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Retail Net Revenue Drivers", layout="wide")

st.title("Retail Dashboard: What Drives Net Revenue Across Countries")
st.caption("Pricing • Inventory • Customer Loyalty • Product Ratings • Sales Rep Performance")

# -----------------------
# Data Loader
# -----------------------
uploaded = st.file_uploader("Upload your retail dataset (CSV or Excel)", type=["csv", "xlsx"])

@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

if uploaded is None:
    st.info("Upload your dataset to generate the dashboard. If you want, tell me your column names and I’ll map them for you.")
    st.stop()

df = load_data(uploaded)
df.columns = [c.strip() for c in df.columns]

st.subheader("1) Column mapping (adjust if needed)")
cols = df.columns.tolist()

# --- Best-effort defaults (you can change in UI) ---
def pick(defaults):
    for d in defaults:
        if d in cols: return d
    return None

country_col  = st.selectbox("Country column", cols, index=cols.index(pick(["Country","country"])) if pick(["Country","country"]) in cols else 0)
netrev_col   = st.selectbox("Net Revenue column", cols, index=cols.index(pick(["Net Revenue","NetRevenue","net_revenue","Revenue","Sales"])) if pick(["Net Revenue","NetRevenue","net_revenue","Revenue","Sales"]) in cols else 0)
price_col    = st.selectbox("Unit Price column (optional)", ["(none)"] + cols, index=0)
discount_col = st.selectbox("Discount % / Discount column (optional)", ["(none)"] + cols, index=0)
qty_col      = st.selectbox("Quantity column (optional)", ["(none)"] + cols, index=0)
inv_col      = st.selectbox("Inventory Level column (optional)", ["(none)"] + cols, index=0)
rating_col   = st.selectbox("Product Rating column (optional)", ["(none)"] + cols, index=0)
loyalty_col  = st.selectbox("Customer Loyalty column (optional)", ["(none)"] + cols, index=0)
rep_col      = st.selectbox("Sales Rep column (optional)", ["(none)"] + cols, index=0)
category_col = st.selectbox("Category/Product column (optional)", ["(none)"] + cols, index=0)

# -----------------------
# Clean numeric columns
# -----------------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

df[netrev_col] = to_num(df[netrev_col])

if discount_col != "(none)":
    df[discount_col] = to_num(df[discount_col])
    # If discount looks like 0-100, convert to 0-1
    if df[discount_col].dropna().quantile(0.9) > 1.5:
        df[discount_col] = df[discount_col] / 100.0

if inv_col != "(none)":
    df[inv_col] = to_num(df[inv_col])

if rating_col != "(none)":
    df[rating_col] = to_num(df[rating_col])

if loyalty_col != "(none)":
    df[loyalty_col] = to_num(df[loyalty_col])

# -----------------------
# Filters
# -----------------------
st.sidebar.header("Filters")
countries = sorted(df[country_col].dropna().unique().tolist())
sel_countries = st.sidebar.multiselect("Country", countries, default=countries)
df_f = df[df[country_col].isin(sel_countries)].copy()

if category_col != "(none)":
    cats = sorted(df_f[category_col].dropna().unique().tolist())
    sel_cats = st.sidebar.multiselect("Category/Product", cats, default=cats[:min(len(cats), 10)] if len(cats)>0 else [])
    if len(sel_cats) > 0:
        df_f = df_f[df_f[category_col].isin(sel_cats)]

# -----------------------
# KPI Computations
# -----------------------
total_netrev = df_f[netrev_col].sum(skipna=True)

# Pricing leakage: estimate discount leakage if we have discount + (price & qty OR gross revenue)
pricing_leak = None
if discount_col != "(none)":
    # If gross revenue exists, use it; else compute from price * qty
    gross = None
    if price_col != "(none)" and qty_col != "(none)":
        df_f["_gross"] = to_num(df_f[price_col]) * to_num(df_f[qty_col])
        gross = df_f["_gross"]
    elif "Gross Revenue" in df_f.columns:
        gross = to_num(df_f["Gross Revenue"])

    if gross is not None:
        # Revenue "given away" as discounts (approx)
        pricing_leak = (gross * df_f[discount_col]).sum(skipna=True)

# Excess inventory flag
excess_inv_value = None
excess_inv_rows = pd.DataFrame()
if inv_col != "(none)":
    # rule-of-thumb: top 20% inventory levels are "excess"
    thr = df_f[inv_col].quantile(0.8)
    excess_inv_rows = df_f[df_f[inv_col] >= thr].copy()
    excess_inv_value = excess_inv_rows[inv_col].sum(skipna=True)

# Low rating impact
low_rating_rows = pd.DataFrame()
if rating_col != "(none)":
    low_rating_rows = df_f[df_f[rating_col] <= 2].copy()
    low_rating_netrev = low_rating_rows[netrev_col].sum(skipna=True)
else:
    low_rating_netrev = None

# Rep performance
rep_perf = None
if rep_col != "(none)":
    agg = {netrev_col: "sum"}
    if discount_col != "(none)":
        agg[discount_col] = "mean"
    rep_perf = df_f.groupby(rep_col, dropna=False).agg(agg).reset_index().sort_values(netrev_col, ascending=False)

# -----------------------
# KPI cards
# -----------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Net Revenue", f"{total_netrev:,.0f}")

if pricing_leak is not None:
    c2.metric("Estimated Discount Leakage", f"{pricing_leak:,.0f}")
else:
    c2.metric("Estimated Discount Leakage", "N/A", help="Needs Discount and Gross Revenue (or Unit Price + Quantity)")

if excess_inv_value is not None:
    c3.metric("Excess Inventory (units)", f"{excess_inv_value:,.0f}")
else:
    c3.metric("Excess Inventory", "N/A")

if rating_col != "(none)":
    c4.metric("Net Revenue from Low-Rated (<=2) Items", f"{low_rating_netrev:,.0f}")
else:
    c4.metric("Low-Rating Revenue", "N/A")

st.divider()

# -----------------------
# Visuals (dashboard grid)
# -----------------------
left, right = st.columns(2)

# Net revenue by country
country_rev = df_f.groupby(country_col)[netrev_col].sum().reset_index().sort_values(netrev_col, ascending=False)
fig_country = px.bar(country_rev, x=country_col, y=netrev_col, title="Net Revenue by Country")
left.plotly_chart(fig_country, use_container_width=True)

# Pricing impact
if discount_col != "(none)":
    # if we don't have margin, show Net Revenue vs Discount
    fig_pricing = px.scatter(
        df_f, x=discount_col, y=netrev_col, color=country_col,
        title="Pricing Efficiency: Discount vs Net Revenue",
        hover_data=[country_col] + ([rep_col] if rep_col != "(none)" else [])
    )
    right.plotly_chart(fig_pricing, use_container_width=True)
else:
    right.info("Add a Discount column mapping to see Pricing Efficiency visuals.")

left2, right2 = st.columns(2)

# Inventory vs rating
if inv_col != "(none)" and rating_col != "(none)":
    inv_rating = df_f.groupby(rating_col)[inv_col].mean().reset_index().sort_values(rating_col)
    fig_inv = px.bar(inv_rating, x=rating_col, y=inv_col, title="Inventory Efficiency: Avg Inventory by Product Rating")
    left2.plotly_chart(fig_inv, use_container_width=True)
elif inv_col != "(none)":
    inv_country = df_f.groupby(country_col)[inv_col].mean().reset_index()
    fig_inv = px.bar(inv_country, x=country_col, y=inv_col, title="Avg Inventory by Country")
    left2.plotly_chart(fig_inv, use_container_width=True)
else:
    left2.info("Map Inventory Level column to see inventory inefficiency.")

# Sales rep performance
if rep_perf is not None:
    if discount_col != "(none)":
        fig_rep = px.scatter(rep_perf, x=netrev_col, y=discount_col, text=rep_col,
                             title="Sales Rep Performance: Net Revenue vs Avg Discount")
    else:
        fig_rep = px.bar(rep_perf, x=rep_col, y=netrev_col, title="Sales Rep Performance: Net Revenue by Rep")
    fig_rep.update_traces(textposition="top center")
    right2.plotly_chart(fig_rep, use_container_width=True)
else:
    right2.info("Map Sales Rep column to see rep performance.")

st.divider()

# Loyalty analysis
if loyalty_col != "(none)":
    # bucket loyalty into Low/Med/High
    bins = df_f[loyalty_col].quantile([0, 0.33, 0.66, 1]).values
    bins = np.unique(bins)
    if len(bins) >= 4:
        df_f["_loyal_bucket"] = pd.cut(df_f[loyalty_col], bins=bins, include_lowest=True, labels=["Low", "Medium", "High"])
        loyalty_rev = df_f.groupby("_loyal_bucket")[netrev_col].sum().reset_index()
        fig_loy = px.pie(loyalty_rev, names="_loyal_bucket", values=netrev_col, hole=0.4,
                         title="Net Revenue Share by Loyalty Segment")
        st.plotly_chart(fig_loy, use_container_width=True)
    else:
        st.info("Not enough variation in loyalty values to segment.")
else:
    st.info("Map Loyalty Score column to see loyalty impact.")

# -----------------------
# Recommendations (rule-based)
# -----------------------
st.subheader("Revenue Loss Hotspots & Recommendations")

recs = []

if pricing_leak is not None and total_netrev > 0:
    leak_pct = pricing_leak / (pricing_leak + total_netrev)
    recs.append((
        "Inefficient pricing / discount leakage",
        f"Discounts account for an estimated {pricing_leak:,.0f} in revenue given away. "
        f"Recommendation: set discount guardrails (e.g., approvals > 15%), and target discounts only to price-sensitive segments."
    ))

if inv_col != "(none)":
    recs.append((
        "Excess inventory risk",
        "Recommendation: identify top-20% inventory SKUs and apply markdown/clearance plans; reduce replenishment until sell-through improves."
    ))

if rating_col != "(none)":
    recs.append((
        "Low product ratings driving lost demand",
        "Recommendation: prioritize quality fixes for <=2-star products; pause promotions on poorly rated SKUs and redirect to 4–5 star items."
    ))

if rep_col != "(none)":
    recs.append((
        "Underperforming sales efforts / inconsistent discounting",
        "Recommendation: compare reps on net revenue AND average discount; coach reps who discount heavily without delivering higher revenue."
    ))

if not recs:
    st.warning("I need at least some of: Discount, Inventory, Rating, Loyalty, Sales Rep columns to generate full recommendations.")
else:
    rec_df = pd.DataFrame(recs, columns=["Issue area", "Recommendation"])
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

st.caption("Tip: Use the filters on the left to create country-specific screenshots for your PPT.")