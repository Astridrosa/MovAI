# === Import Libraries ===
import pandas as pd
import ast

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# === Load and preprocess the dataset ===

def clean_genre(x):
    if pd.isna(x) or not isinstance(x, str):
        return []
    try:
        genres = ast.literal_eval(x)
        if isinstance(genres, list):
            return [g.strip().lower() for g in genres if isinstance(g, str)]
    except:
        return [g.strip().lower() for g in x.split(',')]
    return []

def load_data():
    url = "https://raw.githubusercontent.com/Astridrosa/MovAI/refs/heads/master/data/IMDB_Movie.csv"
    df = pd.read_csv(url)
    df['genre_list'] = df['genre'].apply(clean_genre)
    df['movie_name_clean'] = df['movie_name'].fillna('').str.lower()
    df['director_clean'] = df['director'].fillna('').str.lower()
    df['cast_clean'] = df['cast'].fillna('').str.strip().str.lower()
    return df

df = load_data()

# === RAG helpers === ----------------------------------------------------------
def _build_vectorstore(api_key: str):
    docs = [
        Document(
            page_content=(
                f"Title: {row.movie_name}\nGenre: {row.genre}\n"
                f"Director: {row.director}\nCast: {row.cast}\nYear: {row.year}"
            )
        )
        for _, row in df.iterrows()
    ]

    texts = CharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )
    return FAISS.from_documents(texts, embeddings)

def rag_search(api_key: str, query: str) -> str:
    retriever = _build_vectorstore(api_key).as_retriever()
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "No relevant information found."

    context = "\n\n".join(doc.page_content for doc in docs[:5])
    prompt = (
        "You are a helpful movie expert AI.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=api_key, temperature=0.5
    )
    return llm.predict(prompt)

# === Simple search/recommend functions (non-LLM) =============================
def search_movie(title: str):
    m = df[df.movie_name_clean.str.contains(title.strip().lower(), na=False)]
    return (
        m[["movie_name", "genre", "director"]].head(3).to_string(index=False)
        if not m.empty else f"Movie '{title}' not found."
    )

def recommend_by_genre(genre: str):
    m = df[df.genre_list.apply(lambda gs: genre.lower() in gs)]
    return (
        m[["movie_name", "genre"]].head().to_string(index=False)
        if not m.empty else "No genre match found."
    )

def movies_by_year(year: str):
    try:
        y = int(year)
    except ValueError:
        return "Invalid year format."
    m = df[df.year == y]
    return (
        m[["movie_name", "year"]].head(5).to_string(index=False)
        if not m.empty else f"No movies from {year}."
    )

def director_movies(name: str):
    m = df[df.director_clean.str.contains(name.strip().lower(), na=False)]
    return (
        m[["movie_name", "director"]].head(5).to_string(index=False)
        if not m.empty else f"No movies by director '{name}'."
    )

def actor_movies(actor: str):
    m = df[df.cast_clean.str.contains(actor.strip().lower(), na=False)]
    return (
        m[["movie_name", "cast"]].head(5).to_string(index=False)
        if not m.empty else f"No movies with actor '{actor}'."
    )

def recommend_by_mood(mood: str):
    mood_map = {
        "happy": "comedy", "sad": "drama", "excited": "action",
        "romantic": "romance", "scary": "horror", "thrilling": "thriller",
    }
    genre = mood_map.get(mood.lower())
    return (
        recommend_by_genre(genre)
        if genre else f"Unrecognised mood '{mood}'."
    )

# === Agent factory ============================================================
def create_agent(api_key: str):
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True  # <-- kunci agar list BaseMessage
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7
    )

    tools = [
        Tool("RAGSearch",      lambda q: rag_search(api_key, q),
             "Answer any movie-related question using the database."),
        Tool("SearchMovie",      search_movie,      "Find a movie by title."),
        Tool("RecommendGenre",   recommend_by_genre, "Recommend movies by genre."),
        Tool("MoviesByYear",     movies_by_year,     "Find movies from a given year."),
        Tool("DirectorMovies",   director_movies,    "Find movies by a director."),
        Tool("ActorMovies",      actor_movies,       "Find movies featuring an actor."),
        Tool("RecommendMood",    recommend_by_mood,  "Recommend movies by mood."),
    ]

    return initialize_agent(
        tools, llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
    )
