import streamlit as st
import requests
import json
import logging
import re
import os
import time
import urllib3
from pymongo import MongoClient
from duckduckgo_search import DDGS
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, Tuple

# SSL 
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='chatbot_app.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# --- Database and API Configuration ---
JSON_SOURCE_file = 'fixed_data.json'
apiKey = "86070930-6a32-466d-abf3-0727ab21773d" 
basicUrl = "https://genai.hkbu.edu.hk/api/v0/rest"
modelName = "gpt-4.1-mini"
apiVersion = "2024-12-01-preview"

# --- MongoDB Configuration ---
# MongoDB Atlas 
MONGO_URI = "mongodb+srv://jacky173173_db_user:173173173@cluster0.7rbkruk.mongodb.net/?retryWrites=true&w=majority"
DB_NAME = "hkbu_admissions"

# --- Helper Functions ---

def get_faculty_name(code: str, data: Dict) -> str:
    for faculty in data.get("faculties", []):
        for programme in faculty.get("programmes", []):
            if programme.get("code") == code:
                return faculty.get("name", "")
    return ""

def find_program_url(code: str, data: Dict) -> str:
    """Helper to find the URL for a given programme code."""
    for faculty in data.get("faculties", []):
        for programme in faculty.get("programmes", []):
            if programme.get("code") == code:
                return programme.get("information_website", "")
    return ""

def perform_web_search(query: str, programme_code: str, programme_data: Dict) -> str:
    """
    Uses DuckDuckGo API directly (DDGS) to find information.
    Does NOT use BeautifulSoup/Scraping.
    Handles DNS/Connection errors gracefully.
    """
    clean_query = query.replace(programme_code, "").strip()
    faculty_name = get_faculty_name(programme_code, programme_data)
    
    search_queries = []
    
    
    if faculty_name:
        search_queries.append(f"HKBU {faculty_name} {clean_query}")
    
    
    search_queries.append(f"HKBU {programme_code} {clean_query}")

    
    search_queries.append(f"HKBU {programme_code} course information")

    combined_results = ""
    
    # DDGS Object
    ddgs = DDGS()

    for q in search_queries:
        logger.info(f"Asking Search API to find: {q}")
        try:
            # max_results=5: 
            # backend="api": 
            results = ddgs.text(q, max_results=5, backend="api")
            
            if results:
                combined_results += f"\n\n--- Search Results for '{q}' ---\n"
                for res in results:
                    title = res.get('title', 'No Title')
                    link = res.get('href', 'No Link')
                    body = res.get('body', '')
                    
                    combined_results += f"Title: {title}\nLink: {link}\nSnippet: {body}\n\n"
                
                if len(combined_results) > 1500:
                    break
                    
        except Exception as e:
            logger.warning(f"Search API error for '{q}': {e}")
            time.sleep(1) 
            continue

    return combined_results

def extract_code_and_query(query: str, data: Dict) -> Tuple[str | None, str]:
    query_upper = query.upper()
    code_match = re.search(r'(JS\d{4}|[A-Z]{2,4}-[A-Z]{2,4})', query_upper)
    if code_match:
        code = code_match.group(0)
        if any(prog.get("code", "").upper() == code for fac in data.get("faculties", []) for prog in fac.get("programmes", [])):
            logger.info(f"Found and verified code '{code}' in query.")
            return code, query
    return None, query

# --- MongoDB Backend Logic ---

def get_mongo_client():
    """ MongoDB connect"""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # test
        client.server_info()
        return client
    except Exception as e:
        logger.critical(f"Cannot connect to MongoDB: {e}")
        return None

def setup_database(json_file: str):
    """
     JSON to MongoDB。
    """
    logger.info(f"Setting up MongoDB from '{json_file}'...")
    
    client = get_mongo_client()
    if not client:
        raise ConnectionError("MongoDB connection failed.")
        
    db = client[DB_NAME]
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. Programmes Collection
        col_programmes = db["programmes"]
        
        col_programmes.delete_many({})
        
        programmes_to_insert = []
        for faculty in data.get("faculties", []):
            for programme in faculty.get("programmes", []):
                
                programme['faculty_name'] = faculty.get("name")
                programmes_to_insert.append(programme)
        
        if programmes_to_insert:
            col_programmes.insert_many(programmes_to_insert)
            logger.info(f"Inserted {len(programmes_to_insert)} programmes into MongoDB.")

        # 2. General Info Collection
        col_general = db["general_info"]
        col_general.delete_many({})
        if "general_notes" in data:
            col_general.insert_one({"key": "general_notes", "value": data['general_notes']})

    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        raise e
    finally:
        client.close()

def load_data_from_db():
    logger.info(f"Loading data from MongoDB: {DB_NAME}")
    
    client = get_mongo_client()
    if not client:
        raise ConnectionError("MongoDB connection failed.")
    
    db = client[DB_NAME]
    
    try:
        
        cursor = db["programmes"].find({}, {"_id": 0})
        
        faculties = {}
        for prog in cursor:
            faculty_name = prog.pop("faculty_name", "Unknown Faculty")
            
            if faculty_name not in faculties:
                faculties[faculty_name] = {"name": faculty_name, "programmes": []}
            
            faculties[faculty_name]["programmes"].append(prog)

        # General Notes
        general_note_doc = db["general_info"].find_one({"key": "general_notes"}, {"_id": 0})
        general_notes = general_note_doc.get("value", {}) if general_note_doc else {}

        reconstructed_data = {
            "faculties": list(faculties.values()),
            "general_notes": general_notes
        }
        
        logger.info("Successfully loaded data from MongoDB.")
        return reconstructed_data

    finally:
        client.close()

def create_documents(data):
    docs = []
    logger.info("Starting to create documents from programme data...")

    for faculty in data.get("faculties", []):
        for programme in faculty.get("programmes", []):
            meta = {"programme_code": programme.get("code"), "title": programme.get("title"), "faculty": faculty.get("name")}
            
            if programme.get("programme_description"):
                docs.append(Document(page_content=f"Programme Description for {programme.get('title')} ({programme.get('code')}): {programme.get('programme_description')}", metadata=meta))
            
            if programme.get("career_opportunities"):
                careers_str = json.dumps(programme.get('career_opportunities'), indent=2, ensure_ascii=False)
                docs.append(Document(page_content=f"Career and Study Pathways for {programme.get('title')} ({programme.get('code')}): {careers_str}", metadata=meta))

            req_parts = []
            if programme.get("core_subjects_requirement"): 
                req_parts.append(f"Core Subjects: {json.dumps(programme.get('core_subjects_requirement'))}")
            if programme.get("elective_subjects_requirement"): 
                req_parts.append(f"Elective Subjects: {json.dumps(programme.get('elective_subjects_requirement'))}")
            if programme.get("programme_specific_admission_requirements"): 
                req_parts.append(f"Specific Requirements: {programme.get('programme_specific_admission_requirements')}")
            
            if req_parts:
                docs.append(Document(page_content=f"Admission Requirements for {programme.get('title')} ({programme.get('code')}): {'; '.join(req_parts)}", metadata=meta))
            
            other_info_parts = []
            if programme.get("notes"):
                other_info_parts.append(f"Notes: {json.dumps(programme.get('notes'), ensure_ascii=False)}")
            if programme.get("information_website"):
                other_info_parts.append(f"Official Website: {programme.get('information_website')}")
            
            if other_info_parts:
                 docs.append(Document(page_content=f"Other Information for {programme.get('title')} ({programme.get('code')}): {'; '.join(other_info_parts)}", metadata=meta))

    if data.get("general_notes"):
        docs.append(Document(page_content=f"General Admission Notes: {json.dumps(data.get('general_notes'))}", metadata={"type": "general_notes", "title": "General Notes"}))
    
    return docs

@st.cache_resource
def initialize_chatbot():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        
        
        client = get_mongo_client()
        if client:
            db = client[DB_NAME]
            
            if db["programmes"].count_documents({}) == 0:
                if os.path.exists(JSON_SOURCE_file):
                    setup_database(JSON_SOURCE_file)
                else:
                    st.error(f"Error: '{JSON_SOURCE_file}' not found for initial setup.")
                    return None
            client.close()
        else:
            st.error("Could not connect to MongoDB. Please check your internet connection or firewall.")
            return None

        programme_data = load_data_from_db()
        all_documents = create_documents(programme_data)
        vectorstore = FAISS.from_documents(all_documents, embeddings)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert assistant for Hong Kong Baptist University (HKBU) admissions for the 2025 entry.\n"
                "You have access to database records and internet search summaries.\n"
                "1. Base your answer on the provided Context.\n"
                "2. When answering, first state the full programme title and code.\n"
                "3. If the search results contain lists of names (professors/staff), summarize them clearly.\n"
                "4. **CRITICAL**: If the context contains a section starting with '[Metadata] Official Website:', you MUST use this URL.\n"
                "5. **FALLBACK RULE**: If you cannot find the specific answer, respond with: **'Based on the available data, I cannot find the specific details. Please refer to the official programme website: [Insert URL from Metadata]'**."
            )),
            ("user", "Context:\n{context}\n\nQuestion:\n{question}")
        ])

        logger.info("Chatbot initialized successfully.")
        return {
            "programme_data": programme_data,
            "all_documents": all_documents,
            "vectorstore": vectorstore,
            "prompt_template": prompt_template
        }
    except Exception as e:
        logger.critical(f"Failed to initialize chatbot: {e}", exc_info=True)
        st.error(f"Chatbot initialization error: {e}")
        return None

def get_response(user_query: str, chat_history: list, chatbot_data: dict) -> str:
    programme_data = chatbot_data['programme_data']
    all_documents = chatbot_data['all_documents']
    vectorstore = chatbot_data['vectorstore']
    prompt_template = chatbot_data['prompt_template']

    programme_code, search_query = extract_code_and_query(user_query, programme_data)
    
    relevant_docs = []
    web_content = ""
    forced_url = "" 

    if programme_code:
        logger.info(f"Code '{programme_code}' found. Using direct manual filtering.")
        relevant_docs = [doc for doc in all_documents if doc.metadata.get("programme_code") == programme_code]
        
        forced_url = find_program_url(programme_code, programme_data)
        if forced_url:
            logger.info(f"Identified official URL: {forced_url}")

        logger.info("Performing Web Search via Tool...")
        web_content = perform_web_search(user_query, programme_code, programme_data)
        
    else:
        logger.info("No code found. Using general semantic search.")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(search_query)

    context = ""
    if relevant_docs:
        context += "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    
    if web_content:
        context += web_content
        
    if forced_url:
        context += f"\n\n[Metadata] Official Website: {forced_url}"
    
    if not context:
        context = "No relevant information found."

    final_messages_from_template = prompt_template.format_messages(context=context, question=user_query)
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    api_messages = [{"role": role_map.get(msg.type, "user"), "content": msg.content} for msg in final_messages_from_template]
    
    conversation = api_messages 
    payload = {'messages': conversation, 'max_tokens': 6000}
    
    try:
        response = requests.post(
            f"{basicUrl}/deployments/{modelName}/chat/completions?api-version={apiVersion}",
            json=payload, headers={'Content-Type': 'application/json', 'api-key': apiKey}
        )
        response.raise_for_status()
        data = response.json()
        return data['choices'][0]['message']['content']
    except requests.RequestException as e:
        error_details = e.response.text if e.response else str(e)
        logger.error(f"API request failed: {e}. Details: {error_details}")
        return f"Sorry, API request failed: {str(e)}"

#  Streamlit UI

def main():
    st.set_page_config(page_title="HKBU Admissions Chatbot", page_icon="🎓")
    st.title("🎓 HKBU Admissions Chatbot")
    st.caption("A HKBU course chatbot")

    if "text_size" not in st.session_state:
        st.session_state.text_size = "Medium"
    if "text_align" not in st.session_state:
        st.session_state.text_align = "Left"

    with st.expander("⚙️ Display Settings"):
        st.radio("Text Size", ["Small", "Medium", "Large"], key="text_size", horizontal=True)
        st.radio("Text Alignment", ["Left", "Center", "Right"], key="text_align", horizontal=True)

    size_map = {"Small": "0.9rem", "Medium": "1rem", "Large": "1.2rem"}
    text_align_map = {"Left": "left", "Center": "center", "Right": "right"}
    current_font_size = size_map[st.session_state.text_size]
    current_text_align = text_align_map[st.session_state.text_align]

    st.markdown(f"""
    <style>
        .stChatMessage p, .stChatMessage li {{
            font-size: {current_font_size} !important;
            text-align: {current_text_align} !important;
        }}
    </style>
    """, unsafe_allow_html=True)

    chatbot_data = initialize_chatbot()

    if not chatbot_data:
        st.error("Chatbot Initialization failed. Please check if MongoDB is running.")
        st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Please enter your HKBU course question ..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching database and web..."):
                response = get_response(prompt, [], chatbot_data)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":

    main()


