"""
NZ Rental Market Analysis — Streamlit Web App
158.755 Data Science Project 4 · Massey University 2026
Data: MBIE Tenancy Services Rental Bond Data (CC-BY-3.0-NZ)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
 
st.set_page_config(page_title="NZ Rental Market Forecasting", page_icon="🏠", layout="wide")
st.title("🏠 NZ Rental Market Analysis & Forecasting")
st.markdown("**158.755 Data Science Project 4 — Massey University 2026**")
st.markdown("*Data: MBIE Tenancy Services Rental Bond Data, Feb 1993 – Mar 2026 · CC-BY-3.0-NZ*")
 
@st.cache_data
def load_data():
    for enc in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            df = pd.read_csv(
                'detailed-monthly-tla-tenancy.csv',
                encoding=enc,
                quotechar='"',
                skipinitialspace=True,
            )
            numeric_cols = ['Lodged Bonds', 'Active Bonds', 'Closed Bonds', 'Median Rent',
                            'Geometric Mean Rent', 'Upper Quartile Rent', 'Lower Quartile Rent',
                            'Log Std Dev Weekly Rent']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(',', '').str.strip().str.strip('"'),
                        errors='coerce'
                    )
            if 'Time Frame' in df.columns:
                df['Time Frame'] = pd.to_datetime(
                    df['Time Frame'].astype(str).str.strip().str.strip('"'), errors='coerce'
                )
            if 'Location' in df.columns:
                df['Location'] = df['Location'].astype(str).str.strip().str.strip('"')
            df = df.dropna(subset=['Time Frame'])
            return df.sort_values(['Location', 'Time Frame']).reset_index(drop=True)
        except Exception:
            continue
    st.error("Could not load data file.")
    return pd.DataFrame()
 
@st.cache_data
def load_model_results():
    try:
        if os.path.exists('models/model_results.csv'):
            return pd.read_csv('models/model_results.csv')
    except Exception:
        pass
    return None
 
@st.cache_data
def load_cv_results():
    try:
        if os.path.exists('models/cv_results.csv'):
            return pd.read_csv('models/cv_results.csv')
    except Exception:
        pass
    return None
 
df = load_data()
if df.empty:
    st.stop()
 
all_locations = sorted([x for x in df['Location'].unique().tolist() if isinstance(x, str) and x != 'nan'])
CITIES = ['ALL', 'Auckland', 'Wellington City', 'Christchurch City', 'Hamilton City',
          'Tauranga City', 'Dunedin City', 'Queenstown-Lakes District', 'Palmerston North City']
CITIES = [c for c in CITIES if c in all_locations]
 
# Sidebar
st.sidebar.header("⚙️ Configuration")
default_idx = all_locations.index('ALL') if 'ALL' in all_locations else 0
selected_loc = st.sidebar.selectbox("Location", all_locations, index=default_idx)
compare_mode = st.sidebar.checkbox("Compare cities", False)
compare_locs = []
if compare_mode:
    compare_locs = st.sidebar.multiselect("Select cities to compare", CITIES, default=CITIES[:5])
 
# Top metrics
loc_data = df[df['Location'] == selected_loc].sort_values('Time Frame')
if len(loc_data) == 0:
    st.warning(f"No data found for {selected_loc}")
    st.stop()
 
latest = loc_data.iloc[-1]
year_ago = loc_data.iloc[-13] if len(loc_data) >= 13 else loc_data.iloc[0]
 
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📍 Location", selected_loc if len(selected_loc) <= 15 else selected_loc[:15] + "…")
col2.metric("💰 Median Rent", f"${latest['Median Rent']:.0f}/wk",
            f"${latest['Median Rent'] - year_ago['Median Rent']:+.0f} vs 1yr ago")
col3.metric("📊 IQR", f"${latest['Lower Quartile Rent']:.0f}–${latest['Upper Quartile Rent']:.0f}/wk")
col4.metric("🏘️ Active Bonds", f"{latest['Active Bonds']:,.0f}")
col5.metric("📅 Latest data", latest['Time Frame'].strftime('%b %Y'))
st.divider()
 
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trend", "🗺️ All TLAs", "📊 YoY Growth", "🤖 Forecast", "📋 Model Results"])
 
with tab1:
    st.subheader(f"Median Weekly Rent — {selected_loc}")
    if compare_mode and compare_locs:
        fig = go.Figure()
        for loc in compare_locs:
            sub = df[df['Location'] == loc].sort_values('Time Frame')
            fig.add_trace(go.Scatter(x=sub['Time Frame'], y=sub['Median Rent'],
                                     name=loc.replace(' City', '').replace(' District', ''), mode='lines'))
        fig.update_layout(title="Median Weekly Rent Comparison", yaxis_title="NZD/week", height=430)
    else:
        sub = loc_data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub['Time Frame'], y=sub['Upper Quartile Rent'],
                                 fill=None, mode='lines', line_color='rgba(52,152,219,0.3)', name='Upper Q'))
        fig.add_trace(go.Scatter(x=sub['Time Frame'], y=sub['Lower Quartile Rent'],
                                 fill='tonexty', mode='lines', line_color='rgba(52,152,219,0.3)',
                                 fillcolor='rgba(52,152,219,0.15)', name='Lower Q'))
        fig.add_trace(go.Scatter(x=sub['Time Frame'], y=sub['Median Rent'],
                                 mode='lines', line=dict(color='#e74c3c', width=2.5), name='Median'))
        # Use shapes + annotations instead of add_vline to avoid plotly version issues
        events = [('GFC', '2008-09-01', 'gray'), ('Chch EQ', '2011-02-01', 'orange'),
                  ('COVID-19', '2020-03-01', 'purple'), ('Rate hikes', '2022-03-01', 'red')]
        shapes = []
        annotations = []
        for event, date, color in events:
            shapes.append(dict(type='line', x0=date, x1=date, y0=0, y1=1,
                               xref='x', yref='paper',
                               line=dict(color=color, width=1.5, dash='dash'), opacity=0.6))
            annotations.append(dict(x=date, y=1, xref='x', yref='paper',
                                    text=event, showarrow=False,
                                    font=dict(size=10, color=color),
                                    xanchor='left', yanchor='top'))
        fig.update_layout(title=f"{selected_loc} — Median Rent with IQR Band",
                          yaxis_title="NZD/week", height=430,
                          shapes=shapes, annotations=annotations)
    st.plotly_chart(fig, use_container_width=True)
 
    col_a, col_b = st.columns(2)
    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=loc_data['Time Frame'], y=loc_data['Lodged Bonds'],
                              name='Bonds lodged', marker_color='steelblue', opacity=0.7))
        fig2.update_layout(title='Monthly Bond Lodgements', yaxis_title='Bonds/month', height=300)
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=loc_data['Time Frame'], y=loc_data['Active Bonds'],
                                  fill='tozeroy', line=dict(color='seagreen', width=2)))
        fig3.update_layout(title='Active Bonds (rental stock)', yaxis_title='Active bonds', height=300)
        st.plotly_chart(fig3, use_container_width=True)
 
with tab2:
    st.subheader("All Territorial Authorities — Latest Median Rent Snapshot")
    snap = df[df['Location'] != 'ALL']
    latest_snap = snap.sort_values('Time Frame').groupby('Location').last().reset_index()
    latest_snap = latest_snap[['Location', 'Median Rent', 'Upper Quartile Rent', 'Lower Quartile Rent',
                                'Active Bonds']].sort_values('Median Rent', ascending=False)
    latest_snap.columns = ['Location', 'Median ($/wk)', 'Upper Q', 'Lower Q', 'Active Bonds']
    latest_snap = latest_snap.dropna(subset=['Median ($/wk)'])
 
    fig_bar = px.bar(latest_snap.head(30), x='Median ($/wk)', y='Location',
                     orientation='h', color='Median ($/wk)', color_continuous_scale='Reds',
                     title='Top 30 Most Expensive TLAs (latest month)')
    fig_bar.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)
    st.dataframe(latest_snap.style.background_gradient(subset=['Median ($/wk)'], cmap='Reds')
                 .format({'Median ($/wk)': '${:.0f}', 'Upper Q': '${:.0f}',
                          'Lower Q': '${:.0f}', 'Active Bonds': '{:,.0f}'}),
                 use_container_width=True, height=400)
 
with tab3:
    st.subheader("Year-on-Year Rent Growth by City")
    yoy_data = []
    for loc in CITIES:
        sub = df[df['Location'] == loc].set_index('Time Frame').sort_index()
        if len(sub) > 12:
            sub = sub.copy()
            sub['yoy'] = sub['Median Rent'].pct_change(12) * 100
            sub['location'] = loc
            yoy_data.append(sub[['yoy', 'location']].dropna())
    if yoy_data:
        yoy_df = pd.concat(yoy_data).reset_index()
        yoy_df.columns = ['date', 'yoy', 'location']
        selected_yoy = st.multiselect("Cities", CITIES, default=CITIES[:3])
        if selected_yoy:
            fig_yoy = px.line(yoy_df[yoy_df['location'].isin(selected_yoy)],
                              x='date', y='yoy', color='location',
                              title='Year-on-Year Rent Growth (%)')
            fig_yoy.add_hline(y=0, line_dash='dash', line_color='black', opacity=0.5)
            fig_yoy.update_layout(yaxis_title='YoY change (%)', height=420)
            st.plotly_chart(fig_yoy, use_container_width=True)
 
with tab4:
    st.subheader("📈 National Rent Forecast")
    st.info("Forecast uses scenario-based projections. See notebook Section 9 for full SARIMAX methodology.")
    nat_data = df[df['Location'] == 'ALL'].set_index('Time Frame').sort_index()
    fc_months = st.slider("Forecast horizon (months)", 6, 36, 12)
 
    last_val = nat_data['Median Rent'].iloc[-1]
    fc_dates = pd.date_range(nat_data.index[-1] + pd.DateOffset(months=1), periods=fc_months, freq='MS')
    scenarios = {"Optimistic (+2%/yr)": 0.02 / 12, "Base (+4%/yr)": 0.04 / 12, "Pessimistic (+6%/yr)": 0.06 / 12}
    colors_sc = {"Optimistic (+2%/yr)": "#27ae60", "Base (+4%/yr)": "#e67e22", "Pessimistic (+6%/yr)": "#e74c3c"}
 
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=nat_data.index[-36:], y=nat_data['Median Rent'].values[-36:],
                                name='Historical', line=dict(color='#2c3e50', width=2.5)))
    for label, g in scenarios.items():
        preds = [last_val * (1 + g) ** (i + 1) for i in range(fc_months)]
        fig_fc.add_trace(go.Scatter(x=fc_dates, y=preds, name=label,
                                    line=dict(color=colors_sc[label], width=1.8, dash='dot')))
 
    mr = load_model_results()
    if mr is not None and 'MAPE' in mr.columns:
        best = mr.loc[mr['MAPE'].idxmin()]
        st.success(f"✅ Best model: **{best['model']}** | MAPE = {best['MAPE']:.2f}% | MAE = ${best['MAE']:.1f}/wk")
 
    fig_fc.update_layout(title=f"NZ National Median Rent — {fc_months}-Month Scenarios",
                         yaxis_title="NZD/week", height=430)
    st.plotly_chart(fig_fc, use_container_width=True)
 
with tab5:
    st.subheader("Model Performance (from notebook)")
    res = load_model_results()
    if res is not None:
        try:
            st.dataframe(res.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE'], color='#c6efce')
                         .highlight_max(subset=['R2'], color='#c6efce')
                         .format({'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'R2': '{:.4f}', 'MAPE': '{:.2f}%'}),
                         use_container_width=True)
            fig_r = px.bar(res.sort_values('MAPE'), x='model', y='MAPE',
                           color='MAPE', color_continuous_scale='RdYlGn_r',
                           title='MAPE by Model — lower is better')
            fig_r.update_layout(xaxis_tickangle=-20)
            st.plotly_chart(fig_r, use_container_width=True)
        except Exception:
            st.dataframe(res, use_container_width=True)
    else:
        st.warning("Model results not available.")
 
    cv = load_cv_results()
    if cv is not None:
        st.subheader("Walk-Forward Cross-Validation Results")
        st.dataframe(cv, use_container_width=True)
 
    st.markdown("""
**Key results (test window 2022–2023):**
- SARIMAX + Lodged Bonds: **MAPE 1.78%**, MAE $9.95/wk, R² 0.582
- SARIMA(1,1,2)(0,1,1,12): **MAPE 1.79%**, MAE $10.02/wk, R² 0.578
- Seasonal Naive baseline: MAPE 6.14% (both models beat baseline by >3× on MAPE)
- Walk-forward CV mean MAPE: 1.68% ± 0.78% — stable across all folds
    """)
 
st.divider()
st.caption("MBIE Tenancy Services Rental Bond Data (CC-BY-3.0-NZ) | 158.755 Massey University 2026")
