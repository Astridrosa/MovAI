import streamlit as st
from movai.movai_core import create_agent   

st.set_page_config(page_title="🎬 MovAI - Your Movie Chatbot")
# ... sisanya tetap ...

st.title("🎬 MovAI - MovieBot")
st.markdown("Ask anything about movies")

# === Input Gemini API Key ===
google_api_key = st.text_input("🔑 Enter your Google Gemini API Key", type="password")

# === Initialize state: chat history and agent ===
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "agent" not in st.session_state:
    st.session_state.agent = create_agent(google_api_key)

# === Setup agent only after API key provided ===
if not google_api_key:
    st.warning("🚨 Please enter your Google Gemini API key to start chatting.")
else:
    if st.session_state.agent is None:
        try:
            st.session_state.agent = create_agent(google_api_key)
            st.success("✅ API Key entered successfully. You can now start chatting with MovieBot!")
            st.markdown("👋 Hi there! I'm MovAI. Feel free to ask me anything about movies ")
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {e}")
            st.stop()

# === Display previous chat messages ===
for role, msg in st.session_state.chat_messages:
    with st.chat_message(role):
        st.markdown(msg)

# === Chat input box ===
user_input = st.chat_input("Type your message here...")

# === On message input, run agent and display response ===
if user_input and st.session_state.agent:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_messages.append(("user", user_input))

    friendly_input = f"{user_input}. Please answer simply and include a short follow-up."

    with st.chat_message("assistant"):
        with st.spinner("🎮 MovieBot is thinking..."):
            try:
                response = st.session_state.agent.run(friendly_input)
                st.markdown(response)
                st.session_state.chat_messages.append(("assistant", response))
            except Exception as e:
                st.error(f"❌ Error running agent: {e}")
