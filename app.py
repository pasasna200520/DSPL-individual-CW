import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Sri Lanka Bus Routes Analysis",
    page_icon="🚌",
    layout="wide"
)

# Define function to load and clean data
@st.cache_data
def load_data():
    # Read the data
    df = pd.read_csv('cleaned_dataset.csv')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Convert Travel_Time to minutes for easier analysis
    def convert_to_minutes(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            # Extract hours and minutes from the string format
            parts = time_str.split()
            hours = int(parts[2].split(':')[0])
            minutes = int(parts[2].split(':')[1])
            return hours * 60 + minutes
        except:
            return np.nan
    
    df['Travel_Time_Minutes'] = df['Travel_Time'].apply(convert_to_minutes)
    
    # Remove rows with missing data
    df = df.dropna(subset=['No_of_Buses', 'Distance_KM', 'Travel_Time_Minutes'])
    
    # Create a derived column for bus frequency (buses per 100 km)
    df['Buses_Per_100KM'] = (df['No_of_Buses'] / df['Distance_KM']) * 100
    
    # Calculate average speed (km/h)
    df['Avg_Speed_KMH'] = df['Distance_KM'] / (df['Travel_Time_Minutes'] / 60)
    
    return df

# Load data
df = load_data()

# Header section
st.title("Sri Lanka Inter-Provincial Bus Routes Analysis")
st.markdown("### Analysis of Bus Frequency, Distance and Travel Time")

# Overview metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Routes", df.shape[0])
with col2:
    st.metric("Total Buses", int(df['No_of_Buses'].sum()))
with col3:
    st.metric("Avg. Distance", f"{df['Distance_KM'].mean():.1f} KM")
with col4:
    avg_time_min = df['Travel_Time_Minutes'].mean()
    hours = int(avg_time_min // 60)
    minutes = int(avg_time_min % 60)
    st.metric("Avg. Travel Time", f"{hours}h {minutes}m")

st.markdown("---")

# Sidebar filters
st.sidebar.header("Filters")

# Distance range filter
distance_range = st.sidebar.slider(
    "Distance Range (KM)",
    min_value=int(df['Distance_KM'].min()),
    max_value=int(df['Distance_KM'].max()),
    value=(int(df['Distance_KM'].min()), int(df['Distance_KM'].max()))
)

# Travel time range filter (in hours)
max_time = int(df['Travel_Time_Minutes'].max() // 60) + 1
time_range = st.sidebar.slider(
    "Travel Time Range (Hours)",
    min_value=0,
    max_value=max_time,
    value=(0, max_time)
)

# Number of buses filter
max_buses = int(df['No_of_Buses'].max())
buses_range = st.sidebar.slider(
    "Number of Buses",
    min_value=1,
    max_value=max_buses,
    value=(1, max_buses)
)

# Apply filters
filtered_df = df[
    (df['Distance_KM'] >= distance_range[0]) & 
    (df['Distance_KM'] <= distance_range[1]) &
    (df['Travel_Time_Minutes'] >= time_range[0] * 60) & 
    (df['Travel_Time_Minutes'] <= time_range[1] * 60) &
    (df['No_of_Buses'] >= buses_range[0]) & 
    (df['No_of_Buses'] <= buses_range[1])
]

# Main analysis section
st.header("Correlation Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Buses vs Distance")
    fig = px.scatter(
        filtered_df, 
        x='Distance_KM', 
        y='No_of_Buses',
        color='Avg_Speed_KMH',
        hover_data=['Route_No', 'Origin', 'Destination', 'Travel_Time_Minutes'],
        trendline="ols",
        title="Correlation between Number of Buses and Distance"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate correlation coefficient
    corr_buses_distance = filtered_df['No_of_Buses'].corr(filtered_df['Distance_KM'])
    st.write(f"Correlation coefficient: {corr_buses_distance:.3f}")
    
    if corr_buses_distance > 0.5:
        st.write("Strong positive correlation: As distance increases, more buses are allocated.")
    elif corr_buses_distance < -0.5:
        st.write("Strong negative correlation: Fewer buses are allocated to longer routes.")
    else:
        st.write("Weak correlation: Distance has limited influence on bus allocation.")

with col2:
    st.subheader("Buses vs Travel Time")
    fig = px.scatter(
        filtered_df, 
        x='Travel_Time_Minutes', 
        y='No_of_Buses',
        color='Distance_KM',
        hover_data=['Route_No', 'Origin', 'Destination'],
        trendline="ols",
        title="Correlation between Number of Buses and Travel Time"
    )
    fig.update_layout(
        height=500,
        xaxis_title="Travel Time (Minutes)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate correlation coefficient
    corr_buses_time = filtered_df['No_of_Buses'].corr(filtered_df['Travel_Time_Minutes'])
    st.write(f"Correlation coefficient: {corr_buses_time:.3f}")
    
    if corr_buses_time > 0.5:
        st.write("Strong positive correlation: Routes with longer travel times have more buses.")
    elif corr_buses_time < -0.5:
        st.write("Strong negative correlation: Routes with longer travel times have fewer buses.")
    else:
        st.write("Weak correlation: Travel time has limited influence on bus allocation.")

# Additional analysis
st.header("Additional Insights")

tab1, tab2, tab3 = st.tabs(["Route Distribution", "Speed Analysis", "Geographic Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 routes by number of buses
        top_routes = filtered_df.sort_values('No_of_Buses', ascending=False).head(10)
        fig = px.bar(
            top_routes,
            y='Route_No',
            x='No_of_Buses',
            color='Distance_KM',
            orientation='h',
            hover_data=['Origin', 'Destination'],
            title="Top 10 Routes by Number of Buses"
        )
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution of buses per route
        fig = px.histogram(
            filtered_df,
            x='No_of_Buses',
            nbins=20,
            title="Distribution of Buses per Route"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Average speed vs distance
        fig = px.scatter(
            filtered_df,
            x='Distance_KM',
            y='Avg_Speed_KMH',
            color='No_of_Buses',
            hover_data=['Route_No', 'Origin', 'Destination'],
            trendline="ols",
            title="Average Speed vs Distance"
        )
        fig.update_layout(height=500, yaxis_title="Average Speed (KM/H)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Speed distribution
        fig = px.histogram(
            filtered_df,
            x='Avg_Speed_KMH',
            nbins=20,
            title="Distribution of Average Speed"
        )
        fig.update_layout(height=500, xaxis_title="Average Speed (KM/H)")
        st.plotly_chart(fig, use_container_width=True)
        
        avg_speed = filtered_df['Avg_Speed_KMH'].mean()
        st.write(f"Average speed across all routes: {avg_speed:.1f} KM/H")

with tab3:
    # Create a simple frequency count for origins and destinations
    origin_counts = filtered_df['Origin'].value_counts().reset_index()
    origin_counts.columns = ['Location', 'Count']
    origin_counts['Type'] = 'Origin'
    
    dest_counts = filtered_df['Destination'].value_counts().reset_index()
    dest_counts.columns = ['Location', 'Count']
    dest_counts['Type'] = 'Destination'
    
    location_counts = pd.concat([origin_counts, dest_counts])
    
    # Group by location and sum counts
    location_totals = location_counts.groupby('Location')['Count'].sum().reset_index()
    location_totals = location_totals.sort_values('Count', ascending=False)
    
    # Display top locations
    fig = px.bar(
        location_totals.head(15),
        x='Location',
        y='Count',
        title="Top 15 Locations by Route Frequency"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show route frequency for top origin-destination pairs
    route_counts = filtered_df.groupby(['Origin', 'Destination']).size().reset_index(name='Frequency')
    route_counts = route_counts.sort_values('Frequency', ascending=False)
    
    st.subheader("Top Origin-Destination Pairs")
    st.dataframe(route_counts.head(10))

# Table view of the data
st.header("Data Explorer")
cols_to_show = ['Route_No', 'Origin', 'Destination', 'No_of_Buses', 'Distance_KM', 
                'Travel_Time', 'Avg_Speed_KMH', 'Buses_Per_100KM']
st.dataframe(filtered_df[cols_to_show])

# Add download button for the filtered data
@st.cache_data
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(filtered_df[cols_to_show])
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name='sri_lanka_bus_routes_filtered.csv',
    mime='text/csv',
)

# Correlation matrix
st.header("Correlation Matrix")
corr_cols = ['No_of_Buses', 'Distance_KM', 'Travel_Time_Minutes', 'Trips_Per_Day', 'KM_Per_Day', 'Avg_Speed_KMH']
corr_matrix = filtered_df[corr_cols].corr()

fig = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    title="Correlation Matrix"
)
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# Summary and findings
st.header("Key Findings")

# Calculate key correlations
corr_buses_dist = filtered_df['No_of_Buses'].corr(filtered_df['Distance_KM'])
corr_buses_time = filtered_df['No_of_Buses'].corr(filtered_df['Travel_Time_Minutes'])
corr_dist_time = filtered_df['Distance_KM'].corr(filtered_df['Travel_Time_Minutes'])

st.write(f"""
### Summary of Analysis:

1. **Buses vs Distance Correlation: {corr_buses_dist:.3f}**
   - {'Positive correlation indicates more buses tend to be allocated to longer routes' if corr_buses_dist > 0 else 'Negative correlation indicates fewer buses tend to be allocated to longer routes'}

2. **Buses vs Travel Time Correlation: {corr_buses_time:.3f}**
   - {'Positive correlation indicates routes with longer travel times have more buses' if corr_buses_time > 0 else 'Negative correlation indicates routes with longer travel times have fewer buses'}

3. **Distance vs Travel Time Correlation: {corr_dist_time:.3f}**
   - This strong correlation is expected as longer distances generally require more travel time

4. **Average Speed: {filtered_df['Avg_Speed_KMH'].mean():.1f} KM/H**
   - This represents the average speed across all routes in the filtered dataset
""")

st.info("""
**Tips for using this dashboard:**
- Use the filters on the sidebar to narrow down routes by distance, travel time, and number of buses
- Interact with the charts by hovering over data points for more details
- Download the filtered data for further analysis
""")

# Footer
st.markdown("---")
st.caption("Data Source: Sri Lanka Inter-Provincial Bus Routes Dataset")