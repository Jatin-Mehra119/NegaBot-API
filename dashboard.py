"""
NegaBot Streamlit Dashboard
Admin Analytics UI for tweet sentiment classification
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
import time
from database import get_all_predictions
import re

# Configure Streamlit page
st.set_page_config(
    page_title="NegaBot Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    .stAlert {
        border-radius: 10px;
    }
    .main-header {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load prediction data from database"""
    try:
        predictions = get_all_predictions()
        if predictions:
            df = pd.DataFrame(predictions)
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['date'] = df['created_at'].dt.date
            df['hour'] = df['created_at'].dt.hour
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def create_word_cloud(texts, title):
    """Create word cloud from list of texts"""
    if not texts:
        return None
    
    # Combine all texts
    combined_text = ' '.join(texts)
    
    # Clean text (remove URLs, mentions, special characters)
    combined_text = re.sub(r'http\S+', '', combined_text)
    combined_text = re.sub(r'@\w+', '', combined_text)
    combined_text = re.sub(r'[^\w\s]', '', combined_text)
    
    if len(combined_text.strip()) == 0:
        return None
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(combined_text)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    return fig

def filter_dataframe(df, sentiment_filter, date_range, search_term):
    """Apply filters to dataframe"""
    filtered_df = df.copy()
    
    # Sentiment filter
    if sentiment_filter != "All":
        filtered_df = filtered_df[filtered_df['sentiment'] == sentiment_filter]
    
    # Date range filter
    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'] >= start_date) & 
            (filtered_df['date'] <= end_date)
        ]
    
    # Search term filter
    if search_term:
        filtered_df = filtered_df[
            filtered_df['text'].str.contains(search_term, case=False, na=False)
        ]
    
    return filtered_df

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown("<h1 class='main-header'>🤖 NegaBot Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df.empty:
        st.warning("📭 No prediction data found. Make some predictions using the API first!")
        st.markdown("""
        ### Quick Start:
        1. Start the API: `uvicorn api:app --reload`
        2. Make predictions via POST to `/predict`
        3. Come back to see the analytics!
        """)
        return
    
    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")
    
    # Sentiment filter
    sentiment_options = ["All"] + list(df['sentiment'].unique())
    sentiment_filter = st.sidebar.selectbox("Sentiment", sentiment_options)
    
    # Date range filter
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Search filter
    search_term = st.sidebar.text_input("Search in tweets", "")
    
    # Apply filters
    filtered_df = filter_dataframe(df, sentiment_filter, date_range, search_term)
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = len(filtered_df)
        st.metric("📊 Total Predictions", total_predictions)
    
    with col2:
        positive_count = len(filtered_df[filtered_df['sentiment'] == 'Positive'])
        positive_pct = (positive_count / total_predictions * 100) if total_predictions > 0 else 0
        st.metric("😊 Positive", f"{positive_count} ({positive_pct:.1f}%)")
    
    with col3:
        negative_count = len(filtered_df[filtered_df['sentiment'] == 'Negative'])
        negative_pct = (negative_count / total_predictions * 100) if total_predictions > 0 else 0
        st.metric("😞 Negative", f"{negative_count} ({negative_pct:.1f}%)")
    
    with col4:
        avg_confidence = filtered_df['confidence'].mean() if not filtered_df.empty else 0
        st.metric("🎯 Avg Confidence", f"{avg_confidence:.2%}")
    
    st.markdown("---")
    
    # Charts section
    if not filtered_df.empty:
        
        # Sentiment distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Sentiment Distribution")
            sentiment_counts = filtered_df['sentiment'].value_counts()
            
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution",
                color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("📊 Confidence Distribution")
            fig_hist = px.histogram(
                filtered_df,
                x='confidence',
                nbins=20,
                title="Confidence Score Distribution",
                color='sentiment',
                color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C'}
            )
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Time series analysis
        st.subheader("📅 Predictions Over Time")
        
        if len(filtered_df) > 1:
            # Daily predictions
            daily_counts = filtered_df.groupby(['date', 'sentiment']).size().reset_index(name='count')
            
            fig_time = px.line(
                daily_counts,
                x='date',
                y='count',
                color='sentiment',
                title="Daily Sentiment Trends",
                color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C'}
            )
            fig_time.update_layout(xaxis_title="Date", yaxis_title="Number of Predictions")
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Hourly distribution
            col1, col2 = st.columns(2)
            
            with col1:
                hourly_counts = filtered_df.groupby('hour').size()
                fig_hourly = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title="Predictions by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Count'}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with col2:
                # Confidence by sentiment
                fig_box = px.box(
                    filtered_df,
                    x='sentiment',
                    y='confidence',
                    title="Confidence Score by Sentiment",
                    color='sentiment',
                    color_discrete_map={'Positive': '#2E8B57', 'Negative': '#DC143C'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Word clouds
        st.subheader("☁️ Word Clouds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            positive_texts = filtered_df[filtered_df['sentiment'] == 'Positive']['text'].tolist()
            if positive_texts:
                fig_wc_pos = create_word_cloud(positive_texts, "Positive Sentiment Word Cloud")
                if fig_wc_pos:
                    st.pyplot(fig_wc_pos)
                else:
                    st.info("Not enough positive text data for word cloud")
            else:
                st.info("No positive predictions found")
        
        with col2:
            negative_texts = filtered_df[filtered_df['sentiment'] == 'Negative']['text'].tolist()
            if negative_texts:
                fig_wc_neg = create_word_cloud(negative_texts, "Negative Sentiment Word Cloud")
                if fig_wc_neg:
                    st.pyplot(fig_wc_neg)
                else:
                    st.info("Not enough negative text data for word cloud")
            else:
                st.info("No negative predictions found")
        
        # Recent predictions table
        st.subheader("📝 Recent Predictions")
        
        # Show recent predictions with formatting
        recent_df = filtered_df.head(20).copy()
        recent_df['text'] = recent_df['text'].str[:100] + '...'  # Truncate long texts
        recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x:.2%}")
        recent_df['created_at'] = recent_df['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        display_df = recent_df[['created_at', 'text', 'sentiment', 'confidence']].rename(columns={
            'created_at': 'Timestamp',
            'text': 'Tweet Text',
            'sentiment': 'Sentiment',
            'confidence': 'Confidence'
        })
        
        st.dataframe(display_df, use_container_width=True)
        
        # Export data option
        st.subheader("💾 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"negabot_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = filtered_df.to_json(orient='records', date_format='iso')
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=f"negabot_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🤖 NegaBot Analytics Dashboard | Last updated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
