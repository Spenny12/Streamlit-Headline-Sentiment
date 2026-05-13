# app.py

import streamlit as st
import feedparser
import google.generativeai as genai
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta

# --- Core Functions ---

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_and_parse_feed(url):
    """Fetches and parses an RSS feed, caching the results for 1 hour."""
    try:
        d = feedparser.parse(url)
        entries = []
        for entry in d.entries:
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                entries.append({
                    'title': entry.title,
                    'link': entry.link,
                    'published_parsed': entry.published_parsed
                })
        return entries
    except Exception as e:
        st.error(f"Could not parse feed {url}. Error: {e}")
        return []

def analyze_headlines_batch(headlines_chunk, keywords, model):
    """
    Uses Gemini to determine relevance and sentiment for a batch of headlines in one API call.
    Expects a list of dictionaries with 'id' and 'headline'.
    """
    keyword_str = ", ".join(keywords)

    prompt = f"""
    You are an analytical JSON API. Analyze the following list of headlines against these topics: {keyword_str}.

    For each headline, determine:
    1. 'matched_keyword': The topic it is most related to (strictly choose from the list). If none, output null.
    2. 'sentiment': If a topic matched, is the headline 'Positive', 'Negative', or 'Neutral' about that topic? If no topic matched, output null.

    Headlines to analyze:
    """

    for item in headlines_chunk:
        prompt += f"\nID: {item['id']} | Headline: \"{item['headline']}\""

    prompt += """

    Return ONLY a valid JSON array of objects. Do not include markdown formatting blocks like ```json.
    Format exactly like this:
    [
      {"id": 0, "matched_keyword": "Topic 1", "sentiment": "Positive"},
      {"id": 1, "matched_keyword": null, "sentiment": null}
    ]
    """

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Clean up potential markdown formatting from the LLM response
        if text.startswith('```json'):
            text = text[7:-3].strip()
        elif text.startswith('```'):
            text = text[3:-3].strip()

        return json.loads(text)
    except json.JSONDecodeError:
        st.error("Failed to parse Gemini output as JSON. Retrying or skipping batch might be needed.")
        return []
    except Exception as e:
        st.error(f"Gemini API Error during batch processing: {e}")
        return []

# --- Streamlit UI and Main Application Flow ---

st.set_page_config(layout="wide", page_title="Headline Sentiment Analyser")
st.title("Headline Sentiment Analyser")

# 1. User Inputs in the Sidebar
with st.sidebar:
    st.header("Configuration")

    try:
        default_key = st.secrets["gemini"]["api_key"]
    except (FileNotFoundError, KeyError):
        default_key = ""

    gemini_api_key = st.text_input("Enter your Gemini API Key", type="password", value=default_key)

    feeds_input = st.text_area("Enter RSS Feed URLs (one per line)", height=150)
    keywords_input = st.text_area("Enter Keywords (one per line)", height=150)

    st.header("Date Range Settings")
    time_options = ["Last 1 Week", "Last 1 Month", "Last 3 Months", "Last 6 Months", "Last 12 Months", "Custom Date Range"]
    selected_time = st.selectbox("Select Time Range", time_options)

    # Date range calculations
    end_datetime = datetime.now(timezone.utc)

    if selected_time == "Custom Date Range":
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", end_datetime.date() - timedelta(days=14))
        end_date = col2.date_input("End Date", end_datetime.date())

        start_datetime = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_datetime = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
    else:
        if selected_time == "Last 1 Week":
            start_datetime = end_datetime - timedelta(weeks=1)
        elif selected_time == "Last 1 Month":
            start_datetime = end_datetime - relativedelta(months=1)
        elif selected_time == "Last 3 Months":
            start_datetime = end_datetime - relativedelta(months=3)
        elif selected_time == "Last 6 Months":
            start_datetime = end_datetime - relativedelta(months=6)
        elif selected_time == "Last 12 Months":
            start_datetime = end_datetime - relativedelta(months=12)

if not gemini_api_key:
    st.info("Enter Gemini key to the left. Ask Tom if unsure or need the key.")
    st.stop()

if st.button("Analyse Feeds"):
    try:
        genai.configure(api_key=gemini_api_key)
        # Using flash model as it is faster and cheaper for batch classification
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Failed to configure Gemini API. Please check your key. Error: {e}")
        st.stop()

    feeds = [feed.strip() for feed in feeds_input.split('\n') if feed.strip()]
    initial_keywords = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]

    if not feeds or not initial_keywords:
        st.warning("Please provide at least one RSS feed and one keyword.")
        st.stop()

    st.success(f"Fetching articles from {start_datetime.strftime('%Y-%m-%d')} to {end_datetime.strftime('%Y-%m-%d')}...")

    # Step 1: Collect and filter all headlines from cached feeds
    headlines_to_process = []
    current_id = 0

    with st.spinner("Downloading and filtering RSS feeds..."):
        for feed_url in feeds:
            entries = fetch_and_parse_feed(feed_url)
            for entry in entries:
                pub_date = datetime(*entry['published_parsed'][:6], tzinfo=timezone.utc)
                if start_datetime <= pub_date <= end_datetime:
                    headlines_to_process.append({
                        "id": current_id,
                        "headline": entry['title'],
                        "link": entry['link'],
                        "date": pub_date.strftime('%Y-%m-%d')
                    })
                    current_id += 1

    st.info(f"Found {len(headlines_to_process)} articles in the date range. Starting AI Analysis in batches...")

    # Step 2: Process in batches of 20
    BATCH_SIZE = 20
    final_results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, len(headlines_to_process), BATCH_SIZE):
        batch = headlines_to_process[i:i + BATCH_SIZE]
        status_text.write(f"Analyzing batch {i//BATCH_SIZE + 1} of {(len(headlines_to_process)-1)//BATCH_SIZE + 1}...")

        batch_results = analyze_headlines_batch(batch, initial_keywords, model)

        # Map AI results back to the original headline data
        for res in batch_results:
            if res.get("matched_keyword"): # If not null
                original_item = next((item for item in batch if item["id"] == res["id"]), None)
                if original_item:
                    final_results.append({
                        "Headline": original_item["headline"],
                        "Link": original_item["link"],
                        "Matched Keyword": res["matched_keyword"],
                        "Sentiment": res["sentiment"],
                        "Date": original_item["date"]
                    })

        # Update progress bar
        progress = min(1.0, (i + BATCH_SIZE) / len(headlines_to_process))
        progress_bar.progress(progress)

    status_text.empty() # Clear the status text

    # Step 3: Display Results, Visualizations, and Export
    st.subheader("Analysis Complete")

    if not final_results:
        st.info("No matching articles found based on your keywords in the selected date range.")
    else:
        st.write(f"Found **{len(final_results)}** highly relevant articles.")

        df = pd.DataFrame(final_results)

        # Create columns for the table and the chart
        col_table, col_chart = st.columns([2, 1])

        with col_table:
            st.dataframe(
                df,
                column_config={
                    "Link": st.column_config.LinkColumn("Link", display_text="🔗 Read Article")
                },
                use_container_width=True
            )

            # CSV Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f'sentiment_analysis_{datetime.now().strftime("%Y%m%d")}.csv',
                mime='text/csv',
            )

        with col_chart:
            st.write("**Sentiment Breakdown**")
            # Count sentiments and plot
            sentiment_counts = df['Sentiment'].value_counts()
            st.bar_chart(sentiment_counts, color="#1E90FF")
