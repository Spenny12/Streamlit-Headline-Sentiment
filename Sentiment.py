# app.py

import streamlit as st
import feedparser
import google.generativeai as genai
from datetime import datetime, timedelta, timezone

# --- Core Functions ---

def expand_keywords_with_gemini(keywords, model):
    """Uses Gemini to expand a list of keywords into a list of semantic variations."""
    expanded_keywords = set()
    st.write("Expanding keywords with Gemini for semantic search...") # Replaces print()
    my_bar = st.progress(0)
    for i, keyword in enumerate(keywords):
        try:
            prompt = f"""
            Given the primary keyword, generate a short, comma-separated list of highly related terms, names, or common acronyms a news headline might use. Include the original keyword.
            Example for "Donald Trump": Donald Trump, Trump
            Example for "Artificial Intelligence": Artificial Intelligence, AI, machine learning
            Primary Keyword: "{keyword}"
            """
            response = model.generate_content(prompt)
            variations = [term.strip() for term in response.text.split(',')]
            for var in variations:
                if var:
                    expanded_keywords.add(var)
            st.text(f"  '{keyword}' -> {', '.join(variations)}") # Replaces print()
        except Exception as e:
            st.warning(f"Could not expand keyword '{keyword}' due to API error: {e}") # Replaces print()
            expanded_keywords.add(keyword)
        my_bar.progress((i + 1) / len(keywords))

    return list(expanded_keywords)


def get_gemini_sentiment(headline, term, model):
    """Performs sentiment analysis using the Gemini API."""
    try:
        prompt = f"""
        Analyze the sentiment of the following headline strictly in relation to the term '{term}'. Is the headline positive, negative, or neutral about '{term}'?
        Answer with only one word: Positive, Negative, or Neutral. Headline: "{headline}"
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini API Error: {e}") # Replaces print()
        return "API Error"

# --- Streamlit UI and Main Application Flow ---

st.set_page_config(layout="wide", page_title="Headline Sentiment Analyzer")
st.title("📰 Headline Sentiment Analyzer")

# 1. User Inputs in the Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Use st.secrets for the API key if available, otherwise use text_input
    try:
        default_key = st.secrets["gemini"]["api_key"]
    except (FileNotFoundError, KeyError):
        default_key = ""
        
    gemini_api_key = st.text_input("Enter your Gemini API Key", type="password", value=default_key)

    feeds_input = st.text_area("Enter RSS Feed URLs (one per line)", "http://feeds.bbci.co.uk/news/rss.xml\nhttps://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml", height=150)
    keywords_input = st.text_area("Enter Keywords (one per line)", "NVIDIA\nArtificial Intelligence\nUK election", height=150)

# Main container for the app logic
if not gemini_api_key:
    st.info("Please enter your Gemini API key in the sidebar to begin.")
    st.stop()

# Configure the Gemini API
genai.configure(api_key=gemini_api_key)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Start analysis when the button is clicked
if st.button("🚀 Analyze Feeds"):
    # Convert text area inputs to lists
    feeds = [feed.strip() for feed in feeds_input.split('\n') if feed.strip()]
    initial_keywords = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]

    if not feeds or not initial_keywords:
        st.warning("Please provide at least one RSS feed and one keyword.")
        st.stop()

    # Run the analysis logic
    with st.spinner("Expanding keywords..."):
        keywords = expand_keywords_with_gemini(initial_keywords, model)
    st.success(f"Searching with {len(keywords)} total terms...")

    results = []
    one_week_ago = datetime.now(timezone.utc) - timedelta(days=7)

    st.subheader("Processing Feeds...")
    status_area = st.container() # Create a container to show progress

    for feed_url in feeds:
        status_area.write(f"Parsing feed: {feed_url}")
        try:
            d = feedparser.parse(feed_url)
            for entry in d.entries:
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    if pub_date >= one_week_ago:
                        headline_lower = entry.title.lower()
                        for keyword in keywords:
                            if keyword.lower() in headline_lower:
                                sentiment = get_gemini_sentiment(entry.title, keyword, model)
                                results.append({
                                    "Headline": entry.title,
                                    "Link": entry.link,
                                    "Matched Keyword": keyword,
                                    "Sentiment": sentiment,
                                    "Date": pub_date.strftime('%Y-%m-%d')
                                })
                                break # Move to next article after first match
        except Exception as e:
            st.error(f"Could not parse feed {feed_url}. Error: {e}")


    # 4. Display results in the app
    st.subheader("Analysis Complete")
    if not results:
        st.info("No new matching articles found in the last week.")
    else:
        st.write(f"Found {len(results)} matching articles.")
        # Use st.dataframe to show results in a nice table
        st.dataframe(
            results,
            column_config={
                "Link": st.column_config.LinkColumn("Link", display_text="🔗 Read Article")
            },
            use_container_width=True
        )

