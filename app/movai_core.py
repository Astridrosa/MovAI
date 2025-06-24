# === Import Libraries ===
import pandas as pd
import ast
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
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
    url = "https://raw.githubusercontent.com/Astridrosa/MovieAI/main/IMDB_Movie.csv"
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
def create_agent(api_key):
    global memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7
    )

    # Function RAG to acces memory
    def rag_tool_func(query):
        memory_vars = memory.load_memory_variables({})
        history = memory_vars.get("chat_history", "")

        formatted = "Conversation History:\n"
        for turn in history.split("\n"):
            if "Human:" in turn:
                formatted += f"User: {turn.split('Human:')[-1].strip()}\n"
            elif "AI:" in turn:
                formatted += f"Assistant: {turn.split('AI:')[-1].strip()}\n"

        full_query = f"""{formatted}

Current question: {query}
Use the conversation above to answer accurately."""

        return rag_search_movies(api_key, full_query)

    tools = [
        Tool(name="RAGSearch", func=rag_tool_func, description="Answer any movie-related question (title, actor, director, genre, year, etc.) using database information"),
        Tool(name="SearchMovie", func=search_movie, description="Search movie by title"),
        Tool(name="RecommendByGenre", func=recommend_movies_by_genre, description="Recommend movies based on genre"),
        Tool(name="MoviesByYear", func=get_movies_by_year, description="Find movies from a specific year"),
        Tool(name="DirectorMovies", func=get_director_movies, description="Find all movies by a specific director"),
        Tool(name="ActorMovies", func=get_movies_by_actor, description="Find movies with a specific actor"),
        Tool(name="RecommendByMood", func=recommend_movies_by_mood, description="Recommend movies based on mood (happy, sad, excited, etc.)"),
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
