# === Import Libraries ===
import pandas as pd
import ast
from langchain.agents import Tool, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.agents.agent_types import AgentType

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

# === RAG ===
def get_vectorstore(api_key):
    docs = []
    for _, row in df.iterrows():
        content = f"""Title: {row['movie_name']}
Genre: {row['genre']}
Director: {row['director']}
Cast: {row['cast']}
Year: {row['year']}"""
        docs.append(Document(page_content=content))

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

def rag_search_movies(api_key, query):
    vectorstore = get_vectorstore(api_key)
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    
    if not docs:
        return "No relevant information found."

    context = "\n\n".join([doc.page_content for doc in docs[:5]])

    prompt = f"""You are a helpful movie expert AI.

Use the following context to answer the user's question:
{context}

Question: {query}
Answer:"""

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.5
    )

    return llm.predict(prompt)

# --- Tool Functions ---
def search_movie(title):
    title = title.strip().lower()
    result = df[df['movie_name_clean'].str.contains(title, na=False)]
    return result[['movie_name', 'genre', 'director']].head(3).to_string(index=False) if not result.empty else f"Movie '{title}' not found."

def recommend_movies_by_genre(genre):
    genre = genre.lower()
    result = df[df['genre_list'].apply(lambda genres: genre in genres)]
    return result[['movie_name', 'genre']].head().to_string(index=False) if not result.empty else "No genre match found."

def get_movies_by_year(year):
    try:
        year = int(year)
    except:
        return "Invalid year format."
    result = df[df['year'] == year]
    return result[['movie_name', 'year']].head(5).to_string(index=False) if not result.empty else "No movies found for that year."

def get_director_movies(name):
    name = name.strip().lower()
    result = df[df['director_clean'].str.contains(name)]
    return result[['movie_name', 'director']].head(5).to_string(index=False) if not result.empty else f"No movies found by director '{name}'."

def get_movies_by_actor(actor_name):
    actor_name = actor_name.strip().lower()
    result = df[df['cast_clean'].str.contains(actor_name, na=False)]
    return result[['movie_name', 'cast']].head(5).to_string(index=False) if not result.empty else f"No movies found with actor '{actor_name}'."

def recommend_movies_by_mood(mood):
    mood_map = {
        "happy": "comedy",
        "sad": "drama",
        "excited": "action",
        "romantic": "romance",
        "scary": "horror",
        "thrilling": "thriller"
    }
    genre = mood_map.get(mood.lower())
    if genre:
        return recommend_movies_by_genre(genre)
    else:
        return f"Sorry, I don't recognize the mood '{mood}'. Try: happy, sad, excited, romantic, scary, thrilling."

# === Agent Creation ===
from langchain.schema import BaseMessage  # untuk validasi memory jika perlu

def create_agent(api_key):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )

    def rag_tool_func(query):
        return rag_search_movies(api_key, query)

    tools = [
        Tool(name="RAGSearch", func=rag_tool_func, description="Jawab pertanyaan film menggunakan database."),
        Tool(name="SearchMovie", func=search_movie, description="Cari film berdasarkan judul."),
        Tool(name="RecommendByGenre", func=recommend_movies_by_genre, description="Rekomendasi film berdasarkan genre."),
        Tool(name="MoviesByYear", func=get_movies_by_year, description="Cari film dari tahun tertentu."),
        Tool(name="DirectorMovies", func=get_director_movies, description="Cari film dari sutradara tertentu."),
        Tool(name="ActorMovies", func=get_movies_by_actor, description="Cari film berdasarkan aktor."),
        Tool(name="RecommendByMood", func=recommend_movies_by_mood, description="Rekomendasi film berdasarkan suasana hati."),
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent
