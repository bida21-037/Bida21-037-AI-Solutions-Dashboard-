import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
import numpy as np
import hashlib
import numpy
import joblib
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import os
import gzip
# --------------------- Page Config ---------------------
st.set_page_config(page_title="Executive Dashboard", layout="wide")

st.markdown("""
    <style>
        /* Reduce top and bottom padding of the main container */
        .block-container {
            padding-top: 0.3rem;
            padding-bottom: 0.3rem;
        }

        /* Reduce vertical spacing between elements */
        .stMetric, .stPlotlyChart, .stSubheader, .stMarkdown, .stSelectbox {
            margin-bottom: 0.3rem !important;
        }

        /* Remove extra spacing between rows */
        .element-container {
            padding-bottom: 0.2rem !important;
        }

        .kpi-box {
            margin-bottom: 0.3rem;    
        }
        /* Optional: tighten up column spacing */
        .stColumns {
            gap: 0.5rem !important;
        }
    </style>
""", unsafe_allow_html=True)

# User credentials (replace with your own, or store securely)
users = {
    "Lorato": hashlib.sha256("rati123".encode()).hexdigest(),
    "rep1": hashlib.sha256("sales2025".encode()).hexdigest(),
}

# Function to check credentials
def login(username, password):
    if username in users:
        return users[username] == hashlib.sha256(password.encode()).hexdigest()
    return False

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title(" Login Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.success("Login successful!")
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()  # Prevent app from loading until logged in

# Load both datasets
df_2024 = pd.read_csv("Clean_WebServerLogs_Enriched (2024)_compressed.csv.gz", compression='gzip', parse_dates=["timestamp"])
df_2025 = pd.read_csv("Clean_WebServerLogs_Enriched (2025)_compressed.csv.gz", compression='gzip', parse_dates=["timestamp"])

# Explicitly convert timestamp column
df_2024["timestamp"] = pd.to_datetime(df_2024["timestamp"], errors='coerce')
df_2025["timestamp"] = pd.to_datetime(df_2025["timestamp"], errors='coerce')

# Combine datasets for easier comparison
df_2024["year"] = 2024
df_2025["year"] = 2025
df_2024["month"] = df_2024["timestamp"].dt.month
df_2025["month"] = df_2025["timestamp"].dt.month
df_combined = pd.concat([df_2024, df_2025], ignore_index=True)

# Sidebar with user profile and styled menu
with st.sidebar:
    # Centered image using columns
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image("user.png", width=140)

    # Centered name and email using HTML
    st.markdown(
        """
        <div style="text-align: center;">
            <h4 style="margin-bottom: 5px;">Lorato Mokgatlhe </h4>
            <p style="margin-top: 0;">Lorato.m@example.com</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Custom CSS for sidebar nav
    st.markdown("""
        <style>
            /* Add space between nav links */
            .nav-pills .nav-link {
                margin-bottom: 1px;
                background-color: #D4A3C4;
                border-radius: 8px;
                color: #333;
                transition: all 0.3s ease;
            }
            .nav-pills .nav-link:hover {
                background-color: #d9e7ff;
                color: #000;
            }
            .nav-pills .nav-link.active {
                background-color: #A8D1E7 !important;
                color: white !important;
            }
                
        </style>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title="Dashboard",
        options=["Sales Team View", "Sales Representative","Marketing View", "Manager View", "AI Model"],
        icons=["people-fill", "person-lines-fill", "bullseye", "person-badge-fill", "cpu-fill"],
        menu_icon="columns-gap",
        default_index=0
    )

    custom_palette = ["#A8D1E7", "#FCFAF2", "#FFBFC5", "#EB8DB5", "#D4A3C4"]

    # Add logout button
    if st.session_state.get("logged_in", False):
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

#------------------------------
# Executive Overview Tab
#------------------------------

if selected == "Sales Team View":
     
    # Constants
    team_target = 300_000_000

    def calculate_kpis(df):
        total_sales = df['sales_amount'].sum()
        total_visits = df['session_id'].nunique()
        total_conversions = df[df['conversion_made'] == True]['session_id'].nunique()
        conversion_rate = (total_conversions / total_visits) * 100 if total_visits > 0 else 0
        demo_requests = df[df['is_demo_request'] == True].shape[0]
        ai_interactions = df[df['is_virtual_assistant'] == True].shape[0]
        return total_sales, conversion_rate, demo_requests, ai_interactions
        # --- KPI Calculations ---
    def delta_percent(current, previous):
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        return ((current - previous) / previous) * 100

    sales_2025, cr_2025, demos_2025, ai_2025 = calculate_kpis(df_2025)
    sales_2024, cr_2024, demos_2024, ai_2024 = calculate_kpis(df_2024)

    delta_sales = delta_percent(sales_2025, sales_2024)
    delta_cr = delta_percent(cr_2025, cr_2024)
    delta_demos = delta_percent(demos_2025, demos_2024)
    delta_ai = delta_percent(ai_2025, ai_2024)

    # --- Page Title ---
    st.title("📊 Sales Team Overview")
    
    # --- CSS Styling ---
    st.markdown("""
    <style>
        .kpi-card {
            border: 1px solid #d4d4d4;
            border-radius: 6px;
            padding: 6px 8px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            height: 90px;
        }
        .kpi-title {
            font-size: 12px;
            color: #555;
            margin-bottom: 2px;
        }
        .kpi-value {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 2px;
        }
        .kpi-delta {
            font-size: 16px;
            margin-top: 2px;
        }
        .positive {
            color: #2ecc71;
        }
        .negative {
            color: #e74c3c;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- KPI Cards ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Sales ($)</div>
            <div class="kpi-value">${sales_2025:,.2f}</div>
            <div class="kpi-delta {'positive' if delta_sales >= 0 else 'negative'}">
                {'↑' if delta_sales >= 0 else '↓'} {abs(delta_sales):.1f}% vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Conversion Rate</div>
            <div class="kpi-value">{cr_2025:.1f}%</div>
            <div class="kpi-delta {'positive' if delta_cr >= 0 else 'negative'}">
                {'↑' if delta_cr >= 0 else '↓'} {abs(delta_cr):.1f}% vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Demo Requests</div>
            <div class="kpi-value">{demos_2025}</div>
            <div class="kpi-delta {'positive' if delta_demos >= 0 else 'negative'}">
                {'↑' if delta_demos >= 0 else '↓'} {abs(delta_demos):.1f}% vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">AI Assistant Interactions</div>
            <div class="kpi-value">{ai_2025}</div>
            <div class="kpi-delta {'positive' if delta_ai >= 0 else 'negative'}">
                {'↑' if delta_ai >= 0 else '↓'} {abs(delta_ai):.1f}% vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 0px;'></div>", unsafe_allow_html=True)

    # --- Container Wrapper Function (no titles) ---
    def container_wrap(fig):
        st.markdown("""
            <div style="padding: 0px 0px; background-color: #f9f9f9; border-radius: 12px; box-shadow: 1px 2px 6px rgba(0,0,0,0.1); margin-bottom: 20px;">
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    # --- First Row ---
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    # --- 1: Gauge Chart with Legend ---
    with row1_col1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sales_2025,
            delta={'reference': team_target, 'valueformat': '.0f'},
            gauge={
                'axis': {'range': [None, team_target]},
                'bar': {'color': "#00CC96"},
                'steps': [
                    {'range': [0, team_target * 0.5], 'color': "#FF4B4B"},
                    {'range': [team_target * 0.5, team_target * 0.8], 'color': "#FFA500"},
                    {'range': [team_target * 0.8, team_target], 'color': "#4CAF50"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': team_target
                }
            }
        ))
        gauge.update_layout(height=200, width= 200, margin=dict(t=20, b=10), title="2025 Sales Target", showlegend=False)
        container_wrap(gauge)

        # Custom legend below
        st.markdown("""
            <div style='font-size:13px; margin-top:-10px'>
                <b>Legend:</b><br>
                <span style='color:#FF4B4B'>■ Below 50%</span> &nbsp;
                <span style='color:#FFA500'>■ 50-80%</span> &nbsp;
                <span style='color:#4CAF50'>■ 80-100%</span>
            </div>
        """, unsafe_allow_html=True)
    # --- 3: Conversion Rate Trend ---
    with row1_col2:
        conv_month = df_combined.groupby(['year', 'month']).apply(
            lambda df: df[df['conversion_made'] == True]['session_id'].nunique() /
                    df['session_id'].nunique() * 100
        ).reset_index(name='conversion_rate')
        fig_conv = px.line(conv_month, x='month', y='conversion_rate', title= 'Conversion Rate Trend',color='year',
                        markers=True)
        fig_conv.update_layout(height=250, width=200, margin=dict(t=20, b=10), xaxis_title="Month", yaxis_title="Conversion Rate (%)")
        container_wrap(fig_conv)

    # --- 2: Monthly Sales Chart (Grouped) ---
    with row1_col3:
        
        # Filter and group data
        monthly_sales = df_combined[df_combined['conversion_made'] == True] \
            .groupby(['year', 'month'])['sales_amount'].sum().reset_index()
        
        # Convert month numbers to month abbreviations (1 → 'Jan', 2 → 'Feb', etc.)
        month_abbr_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        monthly_sales['month_abbr'] = monthly_sales['month'].map(month_abbr_map)
        
        # Define performance label function (return labels, not colors)
        def performance_label(sales):
            if sales < 12_500_000:
                return 'Low Performance'
            elif 12_500_000 <= sales <= 13_000_000:
                return 'Moderate Performance'
            else:
                return 'High Performance'


        # Apply performance label
        monthly_sales['performance'] = monthly_sales['sales_amount'].apply(performance_label)

        # Define color map using the desired hex values
        performance_colors = {
            'Low Performance': '#e74c3c',
            'Moderate Performance': '#f39c12',
            'High Performance': '#2ecc71'
        }


        # Create tabs for each year
        tab1, tab2 = st.tabs(["2024", "2025"])
        
        with tab1:
            # Filter for 2024
            sales_2024 = monthly_sales[monthly_sales['year'] == 2024].sort_values('month')
            
            if not sales_2024.empty:
                # Create bar chart for 2024
                fig_2024 = px.bar(
                    sales_2024,
                    x='month_abbr',
                    y='sales_amount',
                    color='performance',
                    title='2024 Monthly Sales',
                    color_discrete_map=performance_colors,
                    text='sales_amount'
                )

                # Update layout
                fig_2024.update_layout(
                    height=250,
                    margin=dict(t=30, b=10),
                    xaxis_title="Month",
                    yaxis_title="Sales Amount ($)",
                    showlegend=True,
                    xaxis={'categoryorder': 'array', 'categoryarray': list(month_abbr_map.values())}
                )
                
                # Format hover and text labels
                fig_2024.update_traces(
                    selector=dict(type='bar'),
                    texttemplate='$%{y:,.0f}',
                    textposition='outside',  # 'top center' is not valid for bar traces — use 'outside', 'inside', etc.
                    hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.0f}',
                    textfont_size=12
                )

                
                st.plotly_chart(fig_2024, use_container_width=True)
            else:
                st.warning("No 2024 data available")
        
        with tab2:
            # Filter for 2025
            sales_2025 = monthly_sales[monthly_sales['year'] == 2025].sort_values('month')
            
            if not sales_2025.empty:
                # Create bar chart for 2025
                fig_2025 = px.bar(
                    sales_2025,
                    x='month_abbr',
                    y='sales_amount',
                    color='performance',
                    title='2025 Monthly Sales',
                    color_discrete_map=performance_colors,
                    text='sales_amount'
                )

                # Update layout
                fig_2025.update_layout(
                    height=300,
                    margin=dict(t=40, b=20),
                    xaxis_title="Month",
                    yaxis_title="Sales Amount ($)",
                    showlegend=False,
                    xaxis={'categoryorder': 'array', 'categoryarray': list(month_abbr_map.values())}
                )
                
                # Format hover and text labels
                fig_2025.update_traces(
                    selector=dict(type='bar'),
                    texttemplate='$%{y:,.0f}',
                    textposition='outside',  # 'top center' is not valid for bar traces — use 'outside', 'inside', etc.
                    hovertemplate='<b>%{x}</b><br>Sales: $%{y:,.0f}',
                    textfont_size=12
                )

                
                st.plotly_chart(fig_2025, use_container_width=True)
            else:
                st.warning("No 2025 data available")

    # --- Second Row ---
    row2_col1, row2_col2 = st.columns(2)

# --- 4: Product Demand Shift (Side-by-side) ---
    with row2_col1:
        # Prepare data
        demand_2025 = df_2025[df_2025['conversion_made'] == True]['product_name'].value_counts().rename("2025")
        demand_2024 = df_2024[df_2024['conversion_made'] == True]['product_name'].value_counts().rename("2024")
        
        demand_df = pd.concat([demand_2024, demand_2025], axis=1).fillna(0)
        demand_df.index.name = "Product"
        demand_df = demand_df.reset_index()
        
        # Create tabs
        tab1, tab2 = st.tabs(["2024", "2025"])
        
        with tab1:
            if not demand_2024.empty:
                # Sort 2024 data by sales count
                demand_2024_sorted = demand_df[['Product', '2024']].sort_values('2024', ascending=False)
                
                # Create performance tiers (top 20% green, middle 60% orange, bottom 20% red)
                total_products = len(demand_2024_sorted)
                demand_2024_sorted['performance'] = pd.qcut(
                    demand_2024_sorted['2024'],
                    q=[0, 0.2, 0.8, 1],
                    labels=['red', 'orange', 'green']
                )
                
                fig_2024 = px.bar(
                    demand_2024_sorted,
                    x="Product",
                    y="2024",
                    title='2024 Product Demand Shift',
                    color='performance',
                    color_discrete_map={'red': 'red', 'orange': 'orange', 'green': 'green'},
                    text='2024'
                )
                
                fig_2024.update_layout(
                    height=300,
                    margin=dict(t=20, b=10),
                    xaxis_title="Product",
                    yaxis_title="Sales Count",
                    showlegend=False,
                    xaxis={'categoryorder': 'total descending'}
                )
                
                fig_2024.update_traces(
                    texttemplate='%{y}',
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Sales: %{y}'
                )

                st.plotly_chart(fig_2024, use_container_width=True)
            else:
                st.warning("No 2024 data available")
        
        with tab2:
            if not demand_2025.empty:
                # Sort 2025 data by sales count
                demand_2025_sorted = demand_df[['Product', '2025']].sort_values('2025', ascending=False)
                
                # Create performance tiers
                total_products = len(demand_2025_sorted)
                demand_2025_sorted['performance'] = pd.qcut(
                    demand_2025_sorted['2025'],
                    q=[0, 0.2, 0.8, 1],
                    labels=['red', 'orange', 'green']
                )
                
                fig_2025 = px.bar(
                    demand_2025_sorted,
                    x="Product",
                    y="2025",
                    title='2025 Product Demand ',
                    color='performance',
                    color_discrete_map={'red': 'red', 'orange': 'orange', 'green': 'green'},
                    text='2025'
                )
                
                fig_2025.update_layout(
                    height=300,
                    margin=dict(t=20, b=10),
                    xaxis_title="Product",
                    yaxis_title="Sales Count",
                    showlegend=False,
                    xaxis={'categoryorder': 'total descending'}
                )
                
                fig_2025.update_traces(
                    texttemplate='%{y}',
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Sales: %{y}'
                )

                st.plotly_chart(fig_2025, use_container_width=True)
            else:
                st.warning("No 2025 data available")

    # --- 5: Category Trends by Month (Line Chart with Color by Year) ---
    with row2_col2:
        # Aggregate sales by year, month, and category
        cat_trend = df_combined[df_combined['conversion_made'] == True] \
            .groupby(['year', 'month', 'page_category'])['sales_amount'].sum().reset_index()

        # Convert month to abbreviations
        month_abbr_map = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        cat_trend['month_abbr'] = cat_trend['month'].map(month_abbr_map)

        # Plot: line chart with color by page category, dash by year
        fig_cat = px.line(
            cat_trend,
            x='month_abbr',
            y='sales_amount',
            color='page_category',
            line_dash='year',
            markers=True,
            title='Sales Trend by Page Category and Year'
        )

        # Sort X-axis by actual month order
        fig_cat.update_xaxes(categoryorder='array', categoryarray=list(month_abbr_map.values()))

        # Layout adjustments
        fig_cat.update_layout(
            height=330,
            margin=dict(t=80, b=20),
            xaxis_title='Month',
            yaxis_title='Sales Amount ($)',
            legend_title='Category / Year'
        )

        container_wrap(fig_cat)

#-------------------------
#      SALES Rep
#-------------------------
elif selected == "Sales Representative":
    st.markdown("<div style='margin-top:10px; margin-bottom:10px'></div>", unsafe_allow_html=True)
    st.title("Sales Representative View")

    rep_targets = {
        "Alice": 90_000_000,
        "Bob": 70_000_000,
        "Cleo": 70_000_000,
        "Dan": 50_000_000,
        "Emma": 50_000_000
    }
    
    # --- Filters ---
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        selected_year = st.selectbox("Year", sorted(df_combined["year"].unique(), reverse=True))
    with col2:
        # Add "All" option for sales rep
        sales_rep_options = ["All"] + sorted(df_combined["sales_rep"].unique().tolist())
        selected_rep = st.selectbox("Sales Representative", sales_rep_options)
    with col3:
        # Add "All" option for region
        region_options = ["All"] + sorted(df_combined["region"].unique().tolist())
        selected_region = st.selectbox("Region", region_options)

    # --- Filtered Data ---
    # Apply filters based on selections (including "All" option)
    df_filtered = df_combined.copy()
    if selected_rep != "All":
        df_filtered = df_filtered[df_filtered["sales_rep"] == selected_rep]
    if selected_region != "All":
        df_filtered = df_filtered[df_filtered["region"] == selected_region]
    
    df_2025_rep = df_filtered[df_filtered["year"] == 2025]
    df_2024_rep = df_filtered[df_filtered["year"] == 2024]
    
    # For rep target progress (needs separate filtering)
    if selected_rep == "All":
        df_2025_filtered = df_2025.copy()
    else:
        df_2025_filtered = df_2025[df_2025['sales_rep'] == selected_rep]
        
    if selected_rep == "All":
        df_2024_filtered = df_2024.copy()
    else:
        df_2024_filtered = df_2024[df_2024['sales_rep'] == selected_rep]

    # --- KPIs ---
    total_2025 = df_2025_rep["sales_amount"].sum()
    total_2024 = df_2024_rep["sales_amount"].sum()
    delta_sales = total_2025 - total_2024

    conv_2025 = df_2025_rep[df_2025_rep["conversion_made"] == True]["session_id"].nunique() / df_2025_rep["session_id"].nunique() * 100
    conv_2024 = df_2024_rep[df_2024_rep["conversion_made"] == True]["session_id"].nunique() / df_2024_rep["session_id"].nunique() * 100
    delta_conv = conv_2025 - conv_2024

    avg_deal_2025 = df_2025_rep[df_2025_rep["conversion_made"] == True]["sales_amount"].mean()
    avg_deal_2024 = df_2024_rep[df_2024_rep["conversion_made"] == True]["sales_amount"].mean()
    delta_avg = avg_deal_2025 - avg_deal_2024
    
    # --- KPI Calculations ---
    delta_sales_pct = (delta_sales / total_2024 * 100) if total_2024 > 0 else 0
    delta_conv_pct = (delta_conv / conv_2024 * 100) if conv_2024 > 0 else 0
    delta_avg_pct = (delta_avg / avg_deal_2024 * 100) if avg_deal_2024 > 0 else 0

    # --- CSS Styling ---
    st.markdown("""
    <style>
        .kpi-card {
            border: 1px solid #d4d4d4;
            border-radius: 6px;
            padding: 6px 8px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            height: 90px;
        }
        .kpi-title {
            font-size: 12px;
            color: #555;
            margin-bottom: 2px;
        }
        .kpi-value {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 2px;
        }
        .kpi-delta {
            font-size: 16px;
            margin-top: 2px;
        }
        .positive {
            color: #2ecc71;
        }
        .negative {
            color: #e74c3c;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- KPI Display ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Sales</div>
            <div class="kpi-value">${total_2025:,.0f}</div>
            <div class="kpi-delta {'positive' if delta_sales >= 0 else 'negative'}">
                {f"{'↑' if delta_sales >= 0 else '↓'} ${abs(delta_sales):,.0f} ({abs(delta_sales_pct):.1f}%)" if total_2024 > 0 else "N/A"} vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Conversion Rate</div>
            <div class="kpi-value">{conv_2025:.2f}%</div>
            <div class="kpi-delta {'positive' if delta_conv >= 0 else 'negative'}">
                {f"{'↑' if delta_conv >= 0 else '↓'} {abs(delta_conv):.2f}% ({abs(delta_conv_pct):.1f}%)" if conv_2024 > 0 else "N/A"} vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Deal Size</div>
            <div class="kpi-value">${avg_deal_2025:,.0f}</div>
            <div class="kpi-delta {'positive' if delta_avg >= 0 else 'negative'}">
                {f"{'↑' if delta_avg >= 0 else '↓'} ${abs(delta_avg):,.0f} ({abs(delta_avg_pct):.1f}%)" if avg_deal_2024 > 0 else "N/A"} vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Visuals in 2 Rows with 2 Charts Each ---
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # 1. Monthly Sales Trend
    monthly_sales = df_filtered[df_filtered["conversion_made"] == True].groupby(["year", "month"])["sales_amount"].sum().reset_index()
    fig_month = px.line(monthly_sales, x="month", y="sales_amount", color="year", 
                       title="Monthly Sales Trend",
                       labels={"month": "Month", "sales_amount": "Sales Amount ($)", "year": "Year"})
    fig_month.update_layout(height=200, width=200, margin=dict(t=40, b=20))
    row1_col1.plotly_chart(fig_month, use_container_width=True)

    # 2. Year-over-Year Sales (Gauge) - Improved with currency and delta label
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=total_2025,
        number={'prefix': "$", 'valueformat': ',.0f'},
        delta={'reference': total_2024, 'valueformat': '$,.0f', 'prefix': "Δ: ", 'suffix': " vs 2024"},
        gauge={
            'axis': {'range': [None, max(total_2025, total_2024) * 1.1]},
            'bar': {'color': "#5CCC00"},
            'steps': [
                {'range': [0, total_2024 * 0.5], 'color': "#FF4B4B"},
                {'range': [total_2024 * 0.5, total_2024 * 0.9], 'color': "#FFA500"},
                {'range': [total_2024 * 0.9, total_2024], 'color': "#4CAF50"},
                {'range': [total_2024, max(total_2025, total_2024) * 1.1], 'color': "#4CAF50"}  # Extended green
            ],
            'threshold': {
                'line': {'color': "black", 'width': 2},
                'thickness': 1,
                'value': total_2024
            }
        },
        title = {'text': "2025 Sales vs 2024 Sales"}
    ))
    gauge.update_layout(height=200, width=200, margin=dict(t=40, b=20))
    row1_col2.plotly_chart(gauge, use_container_width=True)

    # --- Product Mix Chart (Filtered by Rep, Region, Year) ---
    product_mix_df = df_filtered[df_filtered["year"] == selected_year]
    product_mix = product_mix_df.groupby("product_name")["sales_amount"].sum().reset_index()
    product_mix = product_mix[product_mix["product_name"] != ""]  # Exclude empty names
    product_mix = product_mix.sort_values(by="sales_amount", ascending=False)  # Sort descending

    # Create performance categories
    max_sales = product_mix["sales_amount"].max()
    product_mix["performance"] = pd.cut(
        product_mix["sales_amount"],
        bins=[0, max_sales*0.33, max_sales*0.66, max_sales],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    # Create color mapping
    color_map = {
        "High": "#4CAF50",  # Green
        "Medium": "#FFA500",  # Orange
        "Low": "#FF4B4B"  # Red
    }

    # Set chart dimensions based on selection
    if selected_rep == "All":
        chart_height = 350  # Taller for "All" view
        legend_position = -0.3
        font_size = 9
    else:
        chart_height = 200  # Standard height
        legend_position = -0.5
        font_size = 10

    # Create the figure
    fig_mix = px.bar(
        product_mix,
        x="sales_amount",
        y="product_name",
        orientation="h",
        title=f"Product Sales Performance - {selected_year}",
        color="performance",
        color_discrete_map=color_map,
        labels={
            "sales_amount": "Total Sales ($)", 
            "product_name": "Product", 
            "performance": "Performance"
        },
        category_orders={"performance": ["High", "Medium", "Low"]}
    )

    # Update layout
    fig_mix.update_layout(
        height=chart_height,
        margin=dict(t=40, b=40, l=100),
        legend_title_text="Performance",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_position,
            xanchor="center",
            x=0.5
        ),
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )

    # Adjust y-axis for "All" view
    if selected_rep == "All":
        fig_mix.update_yaxes(
            tickfont=dict(size=font_size),
            automargin=True
        )

    row2_col1.plotly_chart(fig_mix, use_container_width=True)
    # --- Rep Target Progress ---
    with row2_col2:
        # Remove spacing below the heading
        st.markdown("""
            <style>
                .target-progress-heading h4 {
                    margin-bottom: 0.2rem !important;
                }
                .compact-progress:first-child,
                .rep-progress:first-child {
                    margin-top: 0 !important;
                }
            </style>
        """, unsafe_allow_html=True)

        # Heading without extra margin
        st.markdown('<div class="target-progress-heading"><h4>🎯 Rep Target Progress</h4></div>', unsafe_allow_html=True)
        

        # More compact CSS for "All" view
        if selected_rep == "All":
            st.markdown("""
                <style>
                    .compact-progress {
                        margin-top: 2px;
                        margin-bottom: 1px;
                        padding: 2px 2px;
                        background-color: #f8f8f8;
                        border-radius: 4px;
                        font-size: 0.5em;
                    }
                    .compact-progress h5 {
                        margin: 0 0 2px 0;
                        font-size: 16px;
                        display: flex;
                        justify-content: space-between;
                    }
                    .compact-progress .bar-container {
                        height: 4px;
                        background-color: #e0e0e0;
                        border-radius: 4px;
                        overflow: hidden;
                        margin-bottom: 4px;
                    }
                    .compact-progress .bar-fill {
                        height: 80%;
                    }
                    .compact-progress span {
                        font-size: 10px;
                        display: block;
                        line-height: 1.2;
                    }
                    .progress-row {
                        display: flex;
                        justify-content: space-between;
                        width: 80%;
                    }
                    .progress-name {
                        flex: 1;
                        white-space: nowrap;
                        overflow: hidden;
                        text-overflow: ellipsis;
                    }
                    .progress-numbers {
                        flex-shrink: 0;
                        margin-left: 8px;
                        text-align: right;
                    }
                </style>
            """, unsafe_allow_html=True)
        else:
            # Original styling for single rep view
            st.markdown("""
                <style>
                    .rep-progress {
                        margin-top: 4px;
                        margin-bottom: 4px;
                        padding: 6px 10px;
                        background-color: #f0f0f0;
                        border-radius: 6px;
                    }
                    .rep-progress h5 {
                        margin: 0 0 4px 0;
                        font-size: 14px;
                    }
                    .rep-progress .bar-container {
                        height: 10px;
                        background-color: #ddd;
                        border-radius: 5px;
                        overflow: hidden;
                        margin-bottom: 10px;
                    }
                    .rep-progress .bar-fill {
                        height: 100%;
                    }
                    .rep-progress span {
                        font-size: 12px;
                    }
                </style>
            """, unsafe_allow_html=True)

        def get_status(progress):
            if progress >= 1.0:
                return "🏆 Target achieved! Excellent work!", "#4CAF50"
            elif progress >= 0.8:
                return "🎯 Almost there! Keep going!", "#4CAF50"
            elif progress >= 0.5:
                return "⚠️ Halfway to target. Push a bit more!", "#FFA500"
            else:
                return "🚧 Off to a slow start. You got this!", "#FF4B4B"

        if selected_rep == "All":
            for rep, target in rep_targets.items():
                achieved = df_2025[df_2025["sales_rep"] == rep]["sales_amount"].sum()
                progress = achieved / target
                status_text, status_color = get_status(progress)
                bar_width = min(progress * 100, 100)
                
                st.markdown(f"""
                    <div class="compact-progress">
                        <h5 class="progress-row">
                            <span class="progress-name">{rep}</span>
                            <span class="progress-numbers">${achieved:,.0f} / ${target:,.0f}</span>
                        </h5>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: {bar_width}%; background-color: {status_color};"></div>
                        </div>
                        <span style='color:{status_color}'>{status_text}</span>
                    </div>
                """, unsafe_allow_html=True)

        elif selected_rep in rep_targets:
            target = rep_targets[selected_rep]
            achieved = df_2025_filtered["sales_amount"].sum()
            progress = achieved / target
            status_text, status_color = get_status(progress)
            bar_width = min(progress * 100, 100)

            st.markdown(f"""
                <div class="rep-progress">
                    <h5>{selected_rep}: ${achieved:,.0f} / ${target:,.0f}</h5>
                    <div class="bar-container">
                        <div class="bar-fill" style="width: {bar_width}%; background-color: {status_color};"></div>
                    </div>
                    <span style='color:{status_color}'>{status_text}</span>
                </div>
            """, unsafe_allow_html=True)
#-------------------------------------------
#  Sales Manager Tab
# ------------------------------------------
elif selected == "Manager View":
    st.title("👔 Manager Dashboard")
    st.markdown('<style>div.block-container{padding-top:0rem !important;padding-bottom: 1rem;} header, footer {visibility: hidden;} </style>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:10px; margin-bottom:10px'></div>", unsafe_allow_html=True)
    
    
    # Remove whitespace
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.5rem;
                padding-bottom: 0rem;
            }
            .stPlotlyChart {
                margin-top: 0rem;
                margin-bottom: 0rem;
            }
            [data-testid="stVerticalBlock"] {
                gap: 0.2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Filters Row ---
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        # Year Filter
        with col1:
            year_filter = st.selectbox("Year", ["All", "2024", "2025"], index=1)
        
        # Quarter Filter
        with col2:
            quarter_filter = st.selectbox("Quarter", ["All", "Q1", "Q2", "Q3", "Q4"])
        
        # Month Filter
        with col3:
            month_filter = st.selectbox("Month", ["All"] + list(range(1, 13)))
        
        # Region Filter
        with col4:
            region_filter = st.selectbox("Region", ["All"] + sorted(df_combined['region'].dropna().unique().tolist()))
    
    # --- Data Filtering ---
    def filter_data(df, year=None):
        filtered = df.copy()
        
        # Apply year filter if specified
        if year:
            filtered = filtered[filtered['year'] == year]
        
        # Apply quarter filter
        if quarter_filter != "All":
            q_num = int(quarter_filter[1])
            filtered = filtered[filtered['timestamp'].dt.quarter == q_num]
        
        # Apply month filter
        if month_filter != "All":
            filtered = filtered[filtered['timestamp'].dt.month == month_filter]
        
        # Apply region filter
        if region_filter != "All":
            filtered = filtered[filtered['region'] == region_filter]
        
        return filtered
    
    # Get filtered data for each year
    df_2024_filtered = filter_data(df_2024, 2024)
    df_2025_filtered = filter_data(df_2025, 2025)
    
    # Get combined filtered data for visualizations
    if year_filter == "All":
        filtered_df = pd.concat([df_2024_filtered, df_2025_filtered])
    elif year_filter == "2024":
        filtered_df = df_2024_filtered
    else:
        filtered_df = df_2025_filtered
    
    # --- KPI Calculations ---
    def calculate_kpis(df, year_label):
        kpis = {
            'year': year_label,
            'total_sales': 0,
            'conversion_rate': 0,
            'best_product': "N/A",
            'demo_requests': 0,
            'total_visits': 0
        }
        
        if not df.empty:
            # Total Sales
            kpis['total_sales'] = df['sales_amount'].sum()
            
            # Conversion Rate
            kpis['total_visits'] = df['session_id'].nunique()
            conversions = df[df['conversion_made'] == True]['session_id'].nunique()
            kpis['conversion_rate'] = (conversions / kpis['total_visits']) * 100 if kpis['total_visits'] > 0 else 0
            
            # Best Product
            converted = df[df['conversion_made'] == True]
            if not converted.empty and 'product_name' in converted.columns:
                best = converted['product_name'].mode()
                if not best.empty:
                    kpis['best_product'] = best[0]
            
            # Demo Requests
            if 'is_demo_request' in df.columns:
                kpis['demo_requests'] = df['is_demo_request'].sum()
        
        return kpis
    
    # Calculate KPIs for both years
    kpis_2024 = calculate_kpis(df_2024_filtered, "2024")
    kpis_2025 = calculate_kpis(df_2025_filtered, "2025")
    
    # --- Display KPIs ---
    st.markdown("""
    <style>
        .kpi-card {
            border: 1px solid #d4d4d4;
            border-radius: 6px;
            padding: 6px 8px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            height: 90px;
        }
        .kpi-title {
            font-size: 12px;
            color: #555;
            margin-bottom: 2px;
        }
        .kpi-value {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 2px;
        }
        .kpi-delta {
            font-size: 16px;
            margin-top: 2px;
        }
        .positive {
            color: #2ecc71;
        }
        .negative {
            color: #e74c3c;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Sales YTD</div>
            <div class="kpi-value">${kpis_2025['total_sales']:,.0f}</div>
            <div class="kpi-delta {'positive' if kpis_2025['total_sales'] >= kpis_2024['total_sales'] else 'negative'}">
                {f"↑ {(kpis_2025['total_sales'] - kpis_2024['total_sales'])/kpis_2024['total_sales']*100:.1f}%" if kpis_2024['total_sales'] > 0 else "N/A"} vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Conversion Rate</div>
            <div class="kpi-value">{kpis_2025['conversion_rate']:.1f}%</div>
            <div class="kpi-delta {'positive' if kpis_2025['conversion_rate'] >= kpis_2024['conversion_rate'] else 'negative'}">
                {f"↑ {(kpis_2025['conversion_rate'] - kpis_2024['conversion_rate'])/kpis_2024['conversion_rate']*100:.1f}%" if kpis_2024['conversion_rate'] > 0 else "N/A"} vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        best_product = kpis_2025['best_product'][:15] + '...' if len(kpis_2025['best_product']) > 15 else kpis_2025['best_product']
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Best-Selling Product</div>
            <div class="kpi-value">{best_product}</div>
            <div class="kpi-delta">
                {'Same as 2024' if kpis_2025['best_product'] == kpis_2024['best_product'] else 'Changed from 2024'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Demo Requests</div>
            <div class="kpi-value">{kpis_2025['demo_requests']}</div>
            <div class="kpi-delta {'positive' if kpis_2025['demo_requests'] >= kpis_2024['demo_requests'] else 'negative'}">
                {f"↑ {(kpis_2025['demo_requests'] - kpis_2024['demo_requests'])/kpis_2024['demo_requests']*100:.1f}%" if kpis_2024['demo_requests'] > 0 else "N/A"} vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)
    
     # --- Main Content Area ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Row for first two charts
        row1_col1, row1_col2 = st.columns(2)
        
        with row1_col1:
            # Monthly Sales Trend (Line Chart)
            monthly_sales = filtered_df[filtered_df['conversion_made'] == True].groupby(['year', 'month'])['sales_amount'].sum().reset_index()
            
            fig_monthly = px.line(monthly_sales, x='month', y='sales_amount', color='year',
                                title='Monthly Sales Trend',
                                labels={'sales_amount': 'Sales Amount', 'month': 'Month'},
                                color_discrete_sequence=["#636EFA", "#EF553B"])
            fig_monthly.update_layout(height=200, margin=dict(l=0 , r=0, t=25, b=8))
            
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        with row1_col2:
            # Product Mix (Horizontal Bar Chart with performance colors)
            product_sales = filtered_df[filtered_df['conversion_made'] == True].groupby('product_name')['sales_amount'].sum().reset_index()
            product_sales = product_sales.sort_values('sales_amount', ascending=False)

            # Define performance levels
            def get_performance_color(sales):
                if sales < 80_000_000:
                    return "#e74c3c"   # Red (Low)
                elif sales < 100_000_000:
                    return "#f39c12"   # Orange (Medium)
                else:
                    return "#2ecc71"   # Green (High)

            product_sales['color'] = product_sales['sales_amount'].apply(get_performance_color)

            # Calculate total and each product's percentage
            total_sales = product_sales['sales_amount'].sum()
            product_sales['percentage'] = (product_sales['sales_amount'] / total_sales * 100).round(1)

            # Create a label like "AI Solutions: 40%"
            product_sales['label'] = product_sales.apply(
                lambda row: f"{row['percentage']:.1f}% of total sales", axis=1
            )


            # Create bar chart with manual colors
            fig_products = px.bar(
                product_sales,
                y='product_name',
                x='sales_amount',
                color='color',
                title='Product Sales Mix',
                labels={'sales_amount': 'Total Sales', 'product_name': 'Product'},
                color_discrete_map='identity',
                text='label'  # This adds labels to bars
            )

            fig_products.update_layout(
                height=250,
                margin=dict(l=0.8, r=0.8, t=25, b=8),
                showlegend=False,  # We'll create a manual legend
                yaxis={'categoryorder': 'total ascending'}
            )

            fig_products.update_traces(
                textposition='inside',
                textfont_size=10,
                hovertemplate='<b>%{y}</b><br>Sales: $%{x:,.0f}<br>Contribution: %{text}'
            )

            st.plotly_chart(fig_products, use_container_width=True)

            # Custom Legend
            st.markdown("""
            <div style="display: flex; gap: 10px; font-size: 12px; margin-top: -10px;">
                <div style="display: flex; align-items: center;"><div style="width: 12px; height: 12px; background-color: #2ecc71; margin-right: 4px;"></div> High Sales (&gt; $15K)</div>
                <div style="display: flex; align-items: center;"><div style="width: 12px; height: 12px; background-color: #f39c12; margin-right: 4px;"></div> Medium ($5K – $15K)</div>
                <div style="display: flex; align-items: center;"><div style="width: 12px; height: 12px; background-color: #e74c3c; margin-right: 4px;"></div> Low (&lt; $5K)</div>
            </div>
            """, unsafe_allow_html=True)

        # Row for third chart (full width)
        # Top 5 Best-Selling Products (Lollipop Chart)
        # Compute top products
        top_products = filtered_df[filtered_df['conversion_made'] == True]['product_name'].value_counts().head(5).reset_index()
        top_products.columns = ['product_name', 'count']

        # Assign performance color based on sales count
        def get_performance_color(count):
            if count < 60_000:
                return "#e74c3c"   # Red (Low)
            elif count < 62_000:
                return "#f39c12"   # Orange (Medium)
            else:
                return "#2ecc71"   # Green (High)

        top_products['color'] = top_products['count'].apply(get_performance_color)

        # Lollipop chart (scatter + lines)
        fig_lollipop = px.scatter(
            top_products,
            x='count',
            y='product_name',
            title='Best-Selling Products',
            labels={'count': 'Number of Sales', 'product_name': 'Product'},
            color='color',
            color_discrete_map='identity'
        )

        # Add stems (lines)
        for i, row in top_products.iterrows():
            fig_lollipop.add_shape(
                type='line',
                x0=0, y0=row['product_name'],
                x1=row['count'], y1=row['product_name'],
                line=dict(color='blue', width=2)
            )

        fig_lollipop.update_traces(marker=dict(size=12))
        fig_lollipop.update_layout(
            height=250,
            margin=dict(l=0.6, r=0, t=25, b=0.8),
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )

        st.plotly_chart(fig_lollipop, use_container_width=True)

        # Custom legend for performance tiers
        st.markdown("""
        <div style="display: flex; gap: 10px; font-size: 12px; margin-top: -10px;">
            <div style="display: flex; align-items: center;"><div style="width: 12px; height: 12px; background-color: #2ecc71; margin-right: 4px;"></div> High Sales (&gt; 30)</div>
            <div style="display: flex; align-items: center;"><div style="width: 12px; height: 12px; background-color: #f39c12; margin-right: 4px;"></div> Medium (10 – 30)</div>
            <div style="display: flex; align-items: center;"><div style="width: 12px; height: 12px; background-color: #e74c3c; margin-right: 4px;"></div> Low (&lt; 10)</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Region-wise Target Achievement (Progress Bars)
        st.markdown("### Region Performance")
        
        # Calculate region targets (example: using mean + 20% as target)
        region_sales = filtered_df[filtered_df['conversion_made'] == True].groupby('region')['sales_amount'].sum().reset_index()
        region_sales['target'] = region_sales['sales_amount'].mean() * 1.2
        region_sales['pct_achieved'] = (region_sales['sales_amount'] / region_sales['target']) * 100
        
        # Sort by achievement percentage
        region_sales = region_sales.sort_values('pct_achieved', ascending=False)
        
        # Create custom progress bars
        for _, row in region_sales.iterrows():
            st.markdown(f"**{row['region']}**")
            
            # Determine color based on achievement
            if row['pct_achieved'] >= 100:
                color = "#2ecc71"  # Green
            elif row['pct_achieved'] >= 75:
                color = "#f39c12"  # Orange
            else:
                color = "#e74c3c"  # Red
            
            st.markdown(f"""
            <div style="margin-bottom: 0.5px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.3px;">
                    <span>${row['sales_amount']:,.0f}</span>
                    <span>{row['pct_achieved']:.0f}%</span>
                </div>
                <div style="height: 6px; background-color: #ecf0f1; border-radius: 5px;">
                    <div style="width: {min(row['pct_achieved'], 100)}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
                </div>
                <div style="font-size: 12px; color: #7f8c8d; text-align: right; margin-top: 1px;">
                    Target: ${row['target']:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

#---------------------------------
#  MARKETING ANALYTICS
#---------------------------------
elif selected == "Marketing View":
    st.title("📢 Marketing Dashboard")
    st.markdown('<style>div.block-container{padding-top:0rem !important;padding-bottom: 1rem;} header, footer {visibility: hidden;} </style>', unsafe_allow_html=True)

    
    # Remove whitespace
    st.markdown("""
        <style>
            .block-container {
                padding-top: 0.5rem;
                padding-bottom: 0rem;
            }
            .stPlotlyChart {
                margin-top: 0rem;
                margin-bottom: 0rem;
            }
            [data-testid="stVerticalBlock"] {
                gap: 0.2rem;
            }
            .kpi-card {
                border: 1px solid #d4d4d4;
                border-radius: 6px;
                padding: 6px 8px;
                background-color: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
                height: 90px;
            }
            .kpi-title {
                font-size: 12px;
                color: #555;
                margin-bottom: 2px;
            }
            .kpi-value {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 2px;
            }
            .kpi-delta {
                font-size: 16px;
                margin-top: 2px;
            }
            .positive {
                color: #2ecc71;
            }
            .negative {
                color: #e74c3c;
            }
            [data-testid="stHorizontalBlock"] {
                gap: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- Filters Row ---
    with st.container():
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Year Filter
        with col1:
            year_filter = st.selectbox("Year", ["All", "2024", "2025"], index=1)
        
        # Quarter Filter
        with col2:
            quarter_filter = st.selectbox("Quarter", ["All", "Q1", "Q2", "Q3", "Q4"])
        
        # Month Filter
        with col3:
            month_names = ["All"] + ["January", "February", "March", "April", "May", "June", 
                                    "July", "August", "September", "October", "November", "December"]
            month_filter = st.selectbox("Month", month_names)
        
        # Campaign Filter
        with col4:
            campaign_filter = st.selectbox("Campaign ID", ["All"] + sorted(df_combined['campaign_id'].dropna().unique().tolist()))
        
        # Region Filter
        with col5:
            region_filter = st.selectbox("Region", ["All"] + sorted(df_combined['region'].dropna().unique().tolist()))
    
    # --- Data Filtering ---
    def filter_data(df, year=None):
        filtered = df.copy()
        
        # Apply year filter if specified
        if year:
            filtered = filtered[filtered['year'] == year]
        
        # Apply quarter filter
        if quarter_filter != "All":
            q_num = int(quarter_filter[1])
            filtered = filtered[filtered['timestamp'].dt.quarter == q_num]
        
        # Apply month filter
        if month_filter != "All":
            month_num = month_names.index(month_filter)  # Get month number from name
            filtered = filtered[filtered['timestamp'].dt.month == month_num]
        
        # Apply campaign filter
        if campaign_filter != "All":
            filtered = filtered[filtered['campaign_id'] == campaign_filter]
        
        # Apply region filter
        if region_filter != "All":
            filtered = filtered[filtered['region'] == region_filter]
        
        return filtered
    
    # Get filtered data for each year
    df_2024_filtered = filter_data(df_2024, 2024)
    df_2025_filtered = filter_data(df_2025, 2025)
    
    # Get combined filtered data for visualizations
    if year_filter == "All":
        filtered_df = pd.concat([df_2024_filtered, df_2025_filtered])
    elif year_filter == "2024":
        filtered_df = df_2024_filtered
    else:
        filtered_df = df_2025_filtered
    
    # --- KPI Calculations ---
    def calculate_kpis(df, year_label):
        kpis = {
            'year': year_label,
            'total_sales': 0,
            'conversion_rate': 0,
            'click_through_rate': 0,
            'roi': 0,
            'total_impressions': 0,
            'total_sessions': 0,
            'total_demo_requests': 0,
            'total_conversions': 0
        }
        
        if not df.empty:
            # Total Sales
            kpis['total_sales'] = df['sales_amount'].sum()
            
            # Conversion Rate
            kpis['total_sessions'] = df['session_id'].nunique()
            kpis['total_conversions'] = df[df['conversion_made'] == True]['session_id'].nunique()
            kpis['conversion_rate'] = (kpis['total_conversions'] / kpis['total_sessions']) * 100 if kpis['total_sessions'] > 0 else 0
            
            # Click-through Rate (CTR)
            # Assuming 'page_views_per_session' can represent impressions
            kpis['total_impressions'] = df['page_views_per_session'].sum()
            kpis['click_through_rate'] = (kpis['total_sessions'] / kpis['total_impressions']) * 100 if kpis['total_impressions'] > 0 else 0
            
            # ROI (Return on Investment)
            # Assuming we have a campaign cost estimate (using $1000 as placeholder)
            campaign_cost = 100_000  # This should be replaced with actual campaign cost data
            kpis['roi'] = ((kpis['total_sales'] - campaign_cost) / campaign_cost) * 100 if campaign_cost > 0 else 0
            
            # Demo Requests
            if 'is_demo_request' in df.columns:
                kpis['total_demo_requests'] = df['is_demo_request'].sum()
        
        return kpis
    
    # Calculate KPIs for both years
    kpis_2024 = calculate_kpis(df_2024_filtered, "2024")
    kpis_2025 = calculate_kpis(df_2025_filtered, "2025")
    
    # --- Display KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Click-Through Rate</div>
            <div class="kpi-value">{kpis_2025['click_through_rate']:.1f}%</div>
            <div class="kpi-delta {'positive' if kpis_2025['click_through_rate'] >= kpis_2024['click_through_rate'] else 'negative'}">
                {f"↑ {(kpis_2025['click_through_rate'] - kpis_2024['click_through_rate'])/kpis_2024['click_through_rate']*100:.1f}%" if kpis_2024['click_through_rate'] > 0 else "N/A"} vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Conversion Rate</div>
            <div class="kpi-value">{kpis_2025['conversion_rate']:.1f}%</div>
            <div class="kpi-delta {'positive' if kpis_2025['conversion_rate'] >= kpis_2024['conversion_rate'] else 'negative'}">
                {f"↑ {(kpis_2025['conversion_rate'] - kpis_2024['conversion_rate'])/kpis_2024['conversion_rate']*100:.1f}%" if kpis_2024['conversion_rate'] > 0 else "N/A"} vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Sales</div>
            <div class="kpi-value">${kpis_2025['total_sales']:,.0f}</div>
            <div class="kpi-delta {'positive' if kpis_2025['total_sales'] >= kpis_2024['total_sales'] else 'negative'}">
                {f"↑ {(kpis_2025['total_sales'] - kpis_2024['total_sales'])/kpis_2024['total_sales']*100:.1f}%" if kpis_2024['total_sales'] > 0 else "N/A"} vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">ROI</div>
            <div class="kpi-value">{kpis_2025['roi']:.1f}%</div>
            <div class="kpi-delta {'positive' if kpis_2025['roi'] >= kpis_2024['roi'] else 'negative'}">
                {f"↑ {(kpis_2025['roi'] - kpis_2024['roi'])/kpis_2024['roi']*100:.1f}%" if kpis_2024['roi'] > 0 else "N/A"} vs 2024
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Main Content Area ---
    col1, col2 = st.columns([2, 3])
    

    with col1:
        campaign_stats = filtered_df.groupby(['campaign_id', 'year']).agg({
                'session_id': 'nunique',
                'is_product_page': 'sum',
                'is_demo_request': 'sum',
                'conversion_made': 'sum',
                'sales_amount': 'sum'
            }).reset_index()
            
        # Calculate metrics
        campaign_stats['ctr'] = (campaign_stats['session_id'] / (campaign_stats['session_id'] * 3)) * 100
        campaign_stats['conversion_rate'] = (campaign_stats['conversion_made'] / campaign_stats['session_id']) * 100
        campaign_stats['conversion_rate'] = campaign_stats['conversion_rate'].replace([np.inf, -np.inf], 0)
        
            # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["By Total Sales", "By Conversion Rate", "By Engagement"])
        
        with tab1:
            # Sort by sales and show all years if "All" is selected
            if year_filter == "All":
                top_campaigns = campaign_stats.sort_values('sales_amount', ascending=False).head(10)
                fig_sales = px.bar(top_campaigns,
                                    x='campaign_id', y='sales_amount',
                                    title='Top Campaigns by Sales',
                                    labels={'sales_amount': 'Total Sales', 'campaign_id': 'Campaign ID'},
                                    color='year',
                                    barmode='group',
                                    color_discrete_sequence=["#636EFA", "#EF553B"])
            else:
                top_campaigns = campaign_stats.sort_values('sales_amount', ascending=False).head(10)
                fig_sales = px.bar(top_campaigns,
                                    x='campaign_id', y='sales_amount',
                                    title='Top Campaigns by Sales',
                                    labels={'sales_amount': 'Total Sales', 'campaign_id': 'Campaign ID'},
                                    color='sales_amount',
                                    color_continuous_scale='Blues')
            fig_sales.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=True)
            st.plotly_chart(fig_sales, use_container_width=True)
        
        with tab2:
            # Sort by conversion rate
            top_campaigns = campaign_stats.sort_values('conversion_rate', ascending=False).head(10)
            if year_filter == "All":
                fig_conversion = px.bar(top_campaigns,
                                        x='campaign_id', y='conversion_rate',
                                        title='Top Campaigns by Conversion Rate',
                                        labels={'conversion_rate': 'Conversion Rate (%)', 'campaign_id': 'Campaign ID'},
                                        color='year',
                                        barmode='group',
                                        color_discrete_sequence=["#636EFA", "#EF553B"])
            else:
                fig_conversion = px.bar(top_campaigns,
                                        x='campaign_id', y='conversion_rate',
                                        title='Top Campaigns by Conversion Rate',
                                        labels={'conversion_rate': 'Conversion Rate (%)', 'campaign_id': 'Campaign ID'},
                                        color='conversion_rate',
                                        color_continuous_scale='Blues')
            fig_conversion.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=True)
            st.plotly_chart(fig_conversion, use_container_width=True)
        
        with tab3:
            # Sort by engagement (demo requests)
            top_campaigns = campaign_stats.sort_values('is_demo_request', ascending=False).head(10)
            if year_filter == "All":
                fig_engagement = px.bar(top_campaigns,
                                        x='campaign_id', y='is_demo_request',
                                        title='Top Campaigns by Demo Requests',
                                        labels={'is_demo_request': 'Demo Requests', 'campaign_id': 'Campaign ID'},
                                        color='year',
                                        barmode='group',
                                        color_discrete_sequence=["#636EFA", "#EF553B"])
            else:
                fig_engagement = px.bar(top_campaigns,
                                        x='campaign_id', y='is_demo_request',
                                        title='Top Campaigns by Demo Requests',
                                        labels={'is_demo_request': 'Demo Requests', 'campaign_id': 'Campaign ID'},
                                        color='is_demo_request',
                                        color_continuous_scale='Blues')
            fig_engagement.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), showlegend=True)
            st.plotly_chart(fig_engagement, use_container_width=True)
    

        if not filtered_df.empty and 'campaign_id' in filtered_df.columns:
            # Funnel: Campaign Engagement → Conversion
            funnel_data = {
                'Stage': ['Sessions', 'Product Page Views', 'Demo Requests', 'Conversions'],
                'Count': [
                    filtered_df['session_id'].nunique(),
                    filtered_df[filtered_df['is_product_page'] == True]['session_id'].nunique(),
                    filtered_df[filtered_df['is_demo_request'] == True]['session_id'].nunique(),
                    filtered_df[filtered_df['conversion_made'] == True]['session_id'].nunique()
                ]
            }

            fig_funnel = px.funnel(funnel_data, x='Count', y='Stage', title='Marketing Funnel')
            fig_funnel.update_layout(height=150, margin=dict(l=0, r=0, t=25, b=0))
            st.plotly_chart(fig_funnel, use_container_width=True)

        
    
    with col2:
        # Engagement Time Distribution
        if 'session_duration' in filtered_df.columns:
                # Create bins for session duration
            filtered_df['duration_bin'] = pd.cut(filtered_df['session_duration'], 
                                            bins=[0, 30, 60, 120, 300, 600, float('inf')],
                                            labels=['<30s', '30-60s', '1-2m', '2-5m', '5-10m', '>10m'])
            
            duration_dist = filtered_df.groupby(['year', 'duration_bin'])['session_id'].count().reset_index()
            duration_dist.columns = ['Year', 'Duration', 'Sessions']
            
            fig_duration = px.line(duration_dist, x='Duration', y='Sessions', color='Year',
                                title='Session Duration Distribution',
                                labels={'Sessions': 'Number of Sessions', 'Duration': 'Session Duration'},
                                color_discrete_sequence=["#14DB1E", "#E4861B"])
            fig_duration.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_duration, use_container_width=True)

        st.markdown("### Campaign Goals")
        
        # Define some example targets (these should be replaced with actual targets)
        campaign_budget = 10_000_000
        budget_spent = 650_000  # Example value
        lead_target = 500_000
        leads_reached = filtered_df['session_id'].nunique()
        conversion_target = 100_000
        conversions_reached = filtered_df[filtered_df['conversion_made'] == True]['session_id'].nunique()
        
        # Create 3 columns for the gauges
        gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
        
        with gauge_col1:
            # Budget Spent
            pct_spent = (budget_spent / campaign_budget) * 100
            gauge_color = "#2ecc71" if pct_spent >= 80 else "#f39c12" if pct_spent >= 50 else "#0be75f"
            
            fig_gauge1 = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pct_spent,
                number={'suffix': '%'},
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Budget Spent<br>${budget_spent:,.0f} / ${campaign_budget:,.0f}"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 50], 'color': "#e74c3c"},
                        {'range': [50, 80], 'color': "#f39c12"},
                        {'range': [80, 100], 'color': "#2ecc71"}
                    ],
                }
            ))

            fig_gauge1.update_layout(height=150, margin=dict(l=0, r=0, t=80, b=4))
            st.plotly_chart(fig_gauge1, use_container_width=True)
        
        with gauge_col2:
            # Lead Target
            pct_leads = (leads_reached / lead_target) * 100
            gauge_color = "#2ecc71" if pct_leads >= 80 else "#f39c12" if pct_leads >= 50 else "#e74c3c"
            
            fig_gauge2 = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pct_leads,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Lead Target<br>{leads_reached}/{lead_target}"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 50], 'color': "#e74c3c"},
                        {'range': [50, 80], 'color': "#f39c12"},
                        {'range': [80, 100], 'color': "#2ecc71"}
                    ]
                }
            ))
            fig_gauge2.update_layout(height=150, margin=dict(l=0, r=0, t=80, b=4))
            st.plotly_chart(fig_gauge2, use_container_width=True)
        
        with gauge_col3:
            # Conversion Target
            pct_conversions = (conversions_reached / conversion_target) * 100
            gauge_color = "#2ecc71" if pct_conversions >= 80 else "#f39c12" if pct_conversions >= 50 else "#e74c3c"
            
            fig_gauge3 = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pct_conversions,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Conversion Target<br>{conversions_reached}/{conversion_target}"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 50], 'color': "#e74c3c"},
                        {'range': [50, 80], 'color': "#f39c12"},
                        {'range': [80, 100], 'color': "#2ecc71"}
                    ]
                }
            ))
            fig_gauge3.update_layout(height=150, margin=dict(l=0, r=0, t=80, b=4))
            st.plotly_chart(fig_gauge3, use_container_width=True)

    # Top Referring Channels (keep this below the campaign performance section)
        if 'referrer' in filtered_df.columns:
            referrer_stats = filtered_df.groupby('referrer').agg({
                'session_id': 'nunique',
                'conversion_made': 'sum'
            }).reset_index().sort_values('session_id', ascending=False)
            
            referrer_thresholds = {
                'excellent': 200_000,
                'high': 150_000,
                'medium': 80_000,
                'low': 40_000
            }

            def referrer_performance(value, thresholds):
                if value >= thresholds['excellent']:
                    return 'Excellent'
                elif value >= thresholds['high']:
                    return 'High'
                elif value >= thresholds['medium']:
                    return 'Medium'
                else:
                    return 'Low'

            referrer_stats['performance'] = referrer_stats['session_id'].apply(
                lambda x: referrer_performance(x, referrer_thresholds)
            )

            referrer_color_map = {
                'Low': '#FF4B4B',       # Red
                'Medium': '#FFA500',    # Orange
                'High': '#4CAF50',      # Green
                'Excellent': '#2E8B57'  # Dark Green
            }

            top_referrers = referrer_stats.sort_values('session_id', ascending=False).head(10)

            fig_referrers = px.bar(
                top_referrers,
                x='session_id',
                y='referrer',
                color='performance',
                color_discrete_map=referrer_color_map,
                title='Top Referring Channels (Performance Levels)',
                labels={'session_id': 'Total Sessions', 'referrer': 'Referrer'}
            )

            fig_referrers.update_layout(height=170, margin=dict(l=0, r=0, t=30, b=0), showlegend=True)
            st.plotly_chart(fig_referrers, use_container_width=True)

#---------------------------------
#  AI MODEL
#---------------------------------

elif selected == "AI Model":
    st.title("🔮 AI Conversion Prediction")
    st.markdown('<style>div.block-container{padding-top:0rem !important;padding-bottom: 1rem;} header, footer {visibility: hidden;} </style>', unsafe_allow_html=True)

    st.markdown("""
        <style>
            /* Reduce padding on main container */
            .appview-container .main .block-container {
                padding-top: 0.4rem;
                padding-bottom: 0.5rem;
                padding-left: 0.3rem;
                padding-right: 0.3rem;
            }

            /* Reduce vertical spacing between widgets */
            div[data-testid="stVerticalBlock"] > div {
                margin-bottom: 0.1rem;
            }

            /* Compact metrics and dataframe spacing */
            div[data-testid="stMetric"] {
                padding: 0.25rem 0.3rem;
            }
            .stDataFrame {
                padding: 0.3rem 0;
            }
        </style>
    """, unsafe_allow_html=True)


    st.write("Use the trained logistic regression model to predict whether a user interaction will convert into a sale.")
    prediction_mode = st.radio("Choose Prediction Mode:", ["Single Prediction", "Batch Prediction (Upload CSV)"], horizontal=True)
    st.markdown("---")
    model = joblib.load("logistic_model_pipeline.pkl")

    if prediction_mode == "Single Prediction":
        st.markdown("## Input Features (Single)")

        # Group inputs into 3 columns per row
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            region = st.selectbox("Region", ["Europe", "North America", "Asia", "South America", "Africa", "Oceania"])
        with col1b:
            sales_channel = st.selectbox("Sales Channel", ["Web", "Mobile", "In-person"])
        with col1c:
            product_price = st.number_input("Product Price", min_value=0.0, value=50.0)

        col2a, col2b, col2c = st.columns(3)
        with col2a:
            session_duration = st.slider("Session Duration (sec)", min_value=0, max_value=600, value=120)
        with col2b:
            page_views = st.slider("Page Views", min_value=1, max_value=50, value=5)
        with col2c:
            ai_assistant_used = st.selectbox("AI Assistant Used", ["Yes", "No"])

        col3a, col3b, col3c = st.columns(3)
        with col3a:
            hour = st.slider("Hour of Day", min_value=0, max_value=23, value=14)
        with col3b:
            day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        with col3c:
            st.empty()

        predict_col1, predict_col2 = st.columns([3, 1])
        with predict_col1:
            if st.button("🔍 Predict Conversion"):
                # Load a sample row from the original dataset
                sample_input = pd.read_csv("Clean_WebServerLogs_Enriched (2).csv").iloc[[0]].copy()

                # Overwrite relevant fields with user inputs
                sample_input["region"] = region
                sample_input["sales_channel"] = sales_channel
                sample_input["product_price"] = product_price
                sample_input["session_duration"] = session_duration
                sample_input["page_views"] = page_views
                sample_input["is_virtual_assistant"] = 1 if ai_assistant_used == "Yes" else 0
                sample_input["hour"] = hour
                sample_input["hour_of_day"] = hour  # if needed
                sample_input["day_of_week"] = day_of_week

                # Drop target if present
                sample_input = sample_input.drop(columns=["conversion_made"], errors="ignore")

                # Make prediction
                prediction = model.predict(sample_input)[0]
                probability = model.predict_proba(sample_input)[0][1]

                # Display result
                st.success(f"**Predicted Conversion:** {'Yes' if prediction == 1 else 'No'}")
                st.metric(label="Conversion Probability", value=f"{probability:.2%}")

                with predict_col2:
                    st.subheader("📈 Result")
                    st.markdown(f"### {'✅ Likely to Convert' if prediction == 1 else '❌ Unlikely to Convert'}")
                    st.metric("Conversion Probability", f"{probability:.2%}")
                    st.progress(probability)

        st.markdown("---")
        st.subheader("📊 Model Performance")

        # Metrics display
        try:
            metrics = joblib.load("logistic_model_metrics.pkl")
            st.success("Model metrics loaded successfully.")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            col2.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            col3.metric("Recall", f"{metrics.get('recall', 0):.2%}")
            col4.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
        except Exception as e:
            st.error(f"❌ Failed to load metrics: {e}")

        try:
            fig = joblib.load("confusion_matrix_fig.pkl")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Confusion matrix not available.")

    elif prediction_mode == "Batch Prediction (Upload CSV)":
        st.markdown("## 📤 Upload CSV for Batch Prediction")
        st.info("The CSV should include the following columns: `region`, `sales_channel`, `product_price`, `session_duration`, `page_views_per_session`, `is_virtual_assistant`, `hour`, `day_of_week`")

        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("CSV uploaded successfully. Preview below:")
                st.dataframe(df.head())

                if st.button("⚙️ Run Batch Predictions"):
                    # Ensure binary conversion of 'ai_assistant_used'
                    df['is_virtual_assistant'] = df['is_virtual_assistant'].map({'Yes': 1, 'No': 0}).fillna(df['is_virtual_assistant'])

                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)[:, 1]

                    df['predicted_conversion'] = predictions
                    df['conversion_probability'] = probabilities

                    st.success("Predictions completed.")
                    st.dataframe(df.head())

                    # Download button
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='conversion_predictions.csv',
                        mime='text/csv'
                    )

            except Exception as e:
                st.error(f"Error processing file: {e}")
    
