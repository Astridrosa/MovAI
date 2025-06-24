# === Imports =================================================================
import ast
import pandas as pd
from functools import lru_cache

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# === Data loading & cleaning ==================================================
def _clean_genre(x: str):
    if pd.isna(x) or not isinstance(x, str):
        return []
    try:
        genres = ast.literal_eval(x)
        if isinstance(genres, list):
            return [g.strip().lower() for g in genres if isinstance(g, str)]
    except Exception:
        return [g.strip().lower() for g in x.split(",")]
    return []

def load_data() -> pd.DataFrame:
    url = (
        "https://raw.githubusercontent.com/Astridrosa/MovAI/"
        "refs/heads/master/data/IMDB_Movie.csv"
    )
    df = pd.read_csv(url)
    df["genre_list"]       = df["genre"].apply(_clean_genre)
    df["movie_name_clean"] = df["movie_name"].fillna("").str.lower()
    df["director_clean"]   = df["director"].fillna("").str.lower()
    df["cast_clean"]       = df["cast"].fillna("").str.lower()
    return df

DF = load_data()

# === Vector store (cached per-API-key) =======================================
@lru_cache(maxsize=3)
def _vectorstore(api_key: str):
    docs = [
        Document(
            page_content=(
                f"Title: {row.movie_name}\n"
                f"Genre: {row.genre}\n"
                f"Director: {row.director}\n"
                f"Cast: {row.cast}\n"
                f"Year: {row.year}"
            )
        )
        for _, row in DF.iterrows()
    ]
    chunks = CharacterTextSplitter(
        chunk_size=300, chunk_overlap=30
    ).split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )
    return FAISS.from_documents(chunks, embeddings)

def _rag_answer(api_key: str, question: str) -> str:
    retriever = _vectorstore(api_key).as_retriever()
    docs      = retriever.get_relevant_documents(question)
    if not docs:
        return "Sorry, I couldn't find relevant information."

    context = "\n\n".join(d.page_content for d in docs[:5])
    prompt  = (
        "You are a helpful movie expert.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=api_key, temperature=0.5
    )
    return llm.predict(prompt)

# === Simple Data-frame helpers ===============================================
def search_movie(title: str):
    m = DF[DF.movie_name_clean.str.contains(title.strip().lower(), na=False)]
    return (
        m[["movie_name", "genre", "director"]].head(3).to_string(index=False)
        if not m.empty else f"No movie titled '{title}'."
    )

def recommend_genre(genre: str):
    m = DF[DF.genre_list.apply(lambda gs: genre.lower() in gs)]
    return (
        m[["movie_name", "genre"]].head(5).to_string(index=False)
        if not m.empty else f"No movies found for genre '{genre}'."
    )

def movies_by_year(year: str):
    try:
        y = int(year)
    except ValueError:
        return "Invalid year."
    m = DF[DF.year == y]
    return (
        m[["movie_name", "year"]].head(5).to_string(index=False)
        if not m.empty else f"No movies released in {year}."
    )

def director_movies(name: str):
    m = DF[DF.director_clean.str.contains(name.strip().lower(), na=False)]
    return (
        m[["movie_name", "director"]].head(5).to_string(index=False)
        if not m.empty else f"No movies by director '{name}'."
    )

def actor_movies(name: str):
    m = DF[DF.cast_clean.str.contains(name.strip().lower(), na=False)]
    return (
        m[["movie_name", "cast"]].head(5).to_string(index=False)
        if not m.empty else f"No movies with actor '{name}'."
    )

def recommend_mood(mood: str):
    mood_map = {
        "happy": "comedy", "sad": "drama", "excited": "action",
        "romantic": "romance", "scary": "horror", "thrilling": "thriller",
    }
    g = mood_map.get(mood.lower())
    return recommend_genre(g) if g else "Mood not recognised."

# === Agent factory ============================================================
def create_agent(api_key: str):
    # 1. Memory – chat_history disimpan otomatis oleh LangChain
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    # 2. LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7
    )

    # 3. Tools
    tools = [
        Tool(
            name="AskDB",
            func=lambda q: _rag_answer(api_key, q),
            description="Free-form movie questions via database search.",
        ),
        Tool("Search",  search_movie,      "Find a movie by title."),
        Tool("Genre",   recommend_genre,   "Recommend movies by genre."),
        Tool("Year",    movies_by_year,    "Find movies from a specific year."),
        Tool("Director",director_movies,   "Find movies by director."),
        Tool("Actor",   actor_movies,      "Find movies with a given actor."),
        Tool("Mood",    recommend_mood,    "Recommend movies by mood."),
    ]

    # 4. Initialize conversational agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
    )

    # (opsional) cek isi memori kosong
    print("✅ MEMORY CHECK:", memory.load_memory_variables({}))
    return agent


# === Cara pakai ===============================================================
if __name__ == "__main__":
    GOOGLE_API_KEY = "YOUR_API_KEY_HERE"          # ganti dengan milikmu
    moviebot = create_agent(GOOGLE_API_KEY)

    # panggil agent – CUKUP kirim field 'input'
    reply = moviebot.invoke({
        "input": "Rekomendasikan film action era 2010-an dong!"
    })
    print("🎬 BOT:", reply["output"])
