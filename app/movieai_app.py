import streamlit as st
from app.movai_core import create_agent   

st.set_page_config(page_title="🎬 MovAI - MovieBot")
st.title("🎬 MovAI - MovieBot")
st.markdown("Ask anything about movies!")

# ── API key input ──────────────────────────────────────────────────────────────
api_key = st.text_input("🔑 Enter your Google Gemini API key", type="password")

# ── Chat history for UI only ( tuples of role, text ) ─────────────────────────
if "chat_ui" not in st.session_state:
    st.session_state.chat_ui = []

# ── Rebuild the agent whenever the API key CHANGES ────────────────────────────
if api_key and api_key != st.session_state.get("saved_api_key"):
    st.session_state.saved_api_key = api_key
    st.session_state.agent = create_agent(api_key)

# If no key yet, stop here
if "agent" not in st.session_state:
    st.warning("🚨 Please enter your API key to start chatting.")
    st.stop()

# ── Reset chat button – clears UI + agent memory –────────────────────────────
def reset_chat():
    st.session_state.chat_ui = []
    st.session_state.agent.memory.clear()     # LangChain memory reset

st.sidebar.button("🔄  Reset chat", on_click=reset_chat)

# ── Display past messages ─────────────────────────────────────────────────────
for role, msg in st.session_state.chat_ui:
    with st.chat_message(role):
        st.markdown(msg)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your message here...")

if user_input:
    # show user bubble
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_ui.append(("user", user_input))

    # call the agent
    with st.chat_message("assistant"):
        with st.spinner("🎬 MovieBot is thinking..."):
            try:
                reply = st.session_state.agent.run(user_input)
            except Exception as e:
                st.error(f"❌ Error running agent: {e}")
                st.stop()
            st.markdown(reply)

    st.session_state.chat_ui.append(("assistant", reply))
