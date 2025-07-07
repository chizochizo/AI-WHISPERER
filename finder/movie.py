import streamlit as st
import pandas as pd
import openai
from gtts import gTTS
from io import BytesIO

# ========== API Keys ==========
openai.api_key = st.secrets["GROQ_API_KEY"]
openai.api_base = "https://api.groq.com/openai/v1"

# ========== Load Data ==========
df = pd.read_csv("movies.csv")

# ========== Page Setup ==========
st.set_page_config(page_title="🎬 Movie Finder", layout="wide")
st.title("🎬 AI Movie Finder")

# ========== Default Session Values ==========
defaults = {
    "search": "",
    "year": "All",
    "rating": "All",
    "genre": "All",
    "chatbot_reply": "",
    "clear_triggered": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ========== Clear Button Behavior ==========
if st.session_state.clear_triggered:
    st.session_state["search"] = ""
    st.session_state["year"] = "All"
    st.session_state["rating"] = "All"
    st.session_state["genre"] = "All"
    st.session_state["chatbot_reply"] = ""
    st.session_state.clear_triggered = False
    st.rerun()

# ========== Voice Playback ==========
def play_audio(text):
    tts = gTTS(text)
    mp3_fp = BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    st.audio(mp3_fp.read(), format='audio/mp3')

# ========== Layout ==========
left, right = st.columns([3, 1])

# ========== LEFT PANEL ==========
with left:
    with st.form("search_form"):
        st.subheader("🔎 Search and Filter")

        search_input = st.text_input("🎬 Enter movie title", value=st.session_state.search)

        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
        with col1:
            selected_year = st.selectbox("📅 Year", ["All"] + [str(y) for y in range(1990, 2026)], key="form_year")
        with col2:
            selected_rating = st.selectbox("⭐ Rating", ["All"] + [str(r) for r in range(0, 11)], key="form_rating")
        with col3:
            selected_genre = st.selectbox("🎭 Genre", ["All"] + sorted(df["genre"].unique()), key="form_genre")
        with col4:
            search = st.form_submit_button("🔍 Search")
        with col5:
            clear = st.form_submit_button("🧹 Clear")
            if clear:
                st.session_state.clear_triggered = True

    # Sync only on Search button
    if search:
        st.session_state["search"] = search_input
        st.session_state["year"] = selected_year
        st.session_state["rating"] = selected_rating
        st.session_state["genre"] = selected_genre

    # Use session_state values for filtering
    search_query = st.session_state.search.lower()
    filtered = df.copy()

    if st.session_state.year != "All":
        filtered = filtered[filtered["year"] == int(st.session_state.year)]
    if st.session_state.rating != "All":
        filtered = filtered[filtered["rating"] >= int(st.session_state.rating)]
    if st.session_state.genre != "All":
        filtered = filtered[filtered["genre"] == st.session_state.genre]

    if search_query:
        exact = filtered[filtered["title"].str.lower() == search_query]
        partial = filtered[filtered["title"].str.lower().str.contains(search_query)]
        filtered = pd.concat([exact, partial]).drop_duplicates()

    # Always show trending genres
    st.subheader("🔥 Trending Genres This Week")
    top_genres = df["genre"].value_counts().head(3).index.tolist()
    st.markdown(" | ".join([f"**🎬 {g}**" for g in top_genres]))

    # Show results
    if search:
        if filtered.empty:
            st.warning("❌ Movie not found — not released or missing from dataset.")
        else:
            st.success(f"🎉 Found {len(filtered)} movie(s)")
            for _, row in filtered.iterrows():
                st.markdown(f"""
                #### 🎞️ {row['title']} ({row['year']})
                - **Genre**: {row['genre']}
                - **Rating**: ⭐ {row['rating']}
                - **Description**: {row['description']}
                ---
                """)

# ========== RIGHT PANEL ==========
with right:
    st.subheader("🤖 Ask MovieBot")

    user_prompt = st.text_area(
        "Ask anything about movies",
        placeholder="e.g. Suggest a movie after 2019"
    )

    if st.button("Ask"):
        if user_prompt.strip():
            try:
                with st.spinner("💬 Thinking..."):
                    response_stream = openai.ChatCompletion.create(
                        model="llama3-70b-8192",
                        messages=[
                            {"role": "system", "content": "You are a helpful AI movie assistant."},
                            {"role": "user", "content": user_prompt}
                        ],
                        stream=True,
                        temperature=0.7
                    )

                    reply = ""
                    placeholder = st.empty()
                    for chunk in response_stream:
                        if "choices" in chunk:
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta:
                                reply += delta["content"]
                                placeholder.markdown(reply)

                    st.session_state.chatbot_reply = reply
            except Exception as e:
                st.error(f"❌ {e}")
        else:
            st.warning("❗ Please enter a question.")

    # 🔊 Always visible speak button
    if st.button("🔊 Speak Response"):
        text = st.session_state.get("chatbot_reply", "No response yet. Ask me something first.")
        play_audio(text)
