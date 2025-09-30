# app.pyF

import streamlit as st
import feedparser
import google.generativeai as genai
from datetime import datetime, timedelta, timezone

# --- Core Functions ---

def check_article_relevance(headline, keywords, model):
    """
    Uses Gemini to determine if a headline is relevant to a list of user-defined keywords.
    """
    try:
        # Create a comma-separated string of keywords for the prompt
        keyword_str = ", ".join(keywords)
        
        prompt = f"""
        Analyse the following headline and determine if it is directly about or related to any of these topics: {keyword_str}.

        If it is related, respond with ONLY the topic from the list that it is most related to.
        If it is not related to any of the topics, respond with ONLY the word "None".

        Headline: "{headline}"
        """
        response = model.generate_content(prompt)
        result = response.text.strip()

        # If the result from Gemini is one of our keywords, it's a match.
        if result in keywords:
            return result # Return the matched keyword
        else:
            return None # Return None if Gemini says "None" or something unexpected
            
    except Exception as e:
        st.error(f"Gemini API Error during relevance check: {e}")
        return None


def get_gemini_sentiment(headline, term, model):
    """Performs sentiment analysis using the Gemini API."""
    try:
        prompt = f"""
        Analyse the sentiment of the following headline strictly in relation to the term '{term}'. Is the headline positive, negative, or neutral about '{term}'?
        Answer with only one word: Positive, Negative, or Neutral. Headline: "{headline}"
        """
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini API Error: {e}") # Replaces print()
        return "API Error"

# --- Streamlit UI and Main Application Flow ---

st.set_page_config(layout="wide", page_title="Headline Sentiment Analyser")
st.title("📰 Headline Sentiment Analyser")

# 1. User Inputs in the Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Use st.secrets for the API key if available, otherwise use text_input
    try:
        default_key = st.secrets["gemini"]["api_key"]
    except (FileNotFoundError, KeyError):
        default_key = ""
        
    gemini_api_key = st.text_input("Enter your Gemini API Key", type="password", value=default_key)

    feeds_input = st.text_area("Enter RSS Feed URLs (one per line)", height=150)
    keywords_input = st.text_area("Enter Keywords (one per line)", height=150)

# Main container for the app logic
if not gemini_api_key:
    st.info("Enter Gemini key to the left. Ask Tom if unsure or need the key.")
    st.stop()

# Start analysis when the button is clicked
if st.button("🚀 Analyze Feeds"):

    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Failed to configure Gemini API. Please check your key. Error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to configure Gemini API. Please check your key. Error: {e}")
        st.stop()
    # Convert text area inputs to lists
    feeds = [feed.strip() for feed in feeds_input.split('\n') if feed.strip()]
    initial_keywords = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]

    if not feeds or not initial_keywords:
        st.warning("Please provide at least one RSS feed and one keyword.")
        st.stop()

    # --- THIS SECTION IS REPLACED ---
    # The call to expand_keywords_with_gemini is gone. We use initial_keywords directly.
    st.success(f"Checking articles for relevance against your {len(initial_keywords)} topics...")

    results = []
    two_weeks_ago = datetime.now(timezone.utc) - timedelta(days=14)

    st.subheader("Processing Feeds...")
    status_area = st.container() 

    for feed_url in feeds:
        status_area.write(f"Parsing feed: {feed_url}")
        try:
            d = feedparser.parse(feed_url)
            for entry in d.entries:
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                    if pub_date >= two_weeks_ago:
                        # --- NEW LOGIC STARTS HERE ---
                        # Instead of looping through keywords, we make one call to Gemini.
                        matched_keyword = check_article_relevance(entry.title, initial_keywords, model)
                        
                        if matched_keyword:
                            # If a relevant keyword was returned, proceed with sentiment analysis
                            st.write(f"  Found Relevant Article: '{entry.title}' (Topic: {matched_keyword})")
                            sentiment = get_gemini_sentiment(entry.title, matched_keyword, model)
                            st.write(f"    Sentiment: {sentiment}")

                            results.append({
                                "Headline": entry.title,
                                "Link": entry.link,
                                "Matched Keyword": matched_keyword,
                                "Sentiment": sentiment,
                                "Date": pub_date.strftime('%Y-%m-%d')
                            })
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




















