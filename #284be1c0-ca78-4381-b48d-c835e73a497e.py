import streamlit as st
import requests
import json
import logging
import re
import sqlite3
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, Tuple

# Configuration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='chatbot_app.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# --- Database and API Configuration ---
DB_FILE = 'hkbu_admissions.db'
JSON_SOURCE_file = 'fixed_data.json'
apiKey = "7cad0daa-7d09-4e4a-9f4c-d70a159c32ea" #API Key
basicUrl = "https://genai.hkbu.edu.hk/api/v0/rest"
modelName = "gpt-4.1-mini"
apiVersion = "2024-12-01-preview"

# Backend Logic

@st.cache_resource
def initialize_chatbot():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        if not os.path.exists(DB_FILE):
            if not os.path.exists(JSON_SOURCE_file):
                st.error(f"Error:can not find out the sourse file '{JSON_SOURCE_file}'")
                return None
            setup_database(JSON_SOURCE_file)
        
        programme_data = load_data_from_db()
        all_documents = create_documents(programme_data)
        vectorstore = FAISS.from_documents(all_documents, embeddings)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert assistant for Hong Kong Baptist University (HKBU) admissions for the 2025 entry. Your task is to answer user questions based *only* on the provided context.\n"
                "Follow these rules strictly:\n"
                "1. Base your entire answer on the information given in the 'Context' section. Do not use any external knowledge.\n"
                "2. When answering, first state the full programme title and code (e.g., 'Bachelor of Science (Hons) (JS2510)').\n"
                "3. Structure your answer clearly. Use bullet points for lists like requirements or career options.\n"
                "4. If the context does not contain the answer to the question, you MUST respond with only this exact phrase: 'Based on the information provided, I can't find any relevant information.' Do not apologize or add any other text."
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
        st.error(f"Chatbot A fatal error occurred during initialization: {e}")
        return None

def setup_database(json_file: str):
    logger.info(f"Database file '{DB_FILE}' not found. Creating and populating from '{json_file}'...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('DROP TABLE IF EXISTS programmes')
    cursor.execute('DROP TABLE IF EXISTS general_info')
    cursor.execute('''
    CREATE TABLE programmes (
        code TEXT PRIMARY KEY, title TEXT, faculty_name TEXT, admissions_type TEXT,
        fund_type TEXT, first_year_intake INTEGER, credits_required_for_graduation INTEGER,
        programme_description TEXT, core_subjects_requirement TEXT, elective_subjects_requirement TEXT,
        programme_specific_admission_requirements TEXT, m1_m2 TEXT, category_b_c TEXT,
        subject_weights TEXT, jupas_admission_score REAL, career_opportunities TEXT,
        notes TEXT, information_website TEXT
    )''')
    cursor.execute('CREATE TABLE general_info (key TEXT PRIMARY KEY, value TEXT)')
    for faculty in data.get("faculties", []):
        for programme in faculty.get("programmes", []):
            cursor.execute('INSERT OR REPLACE INTO programmes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                           (programme.get("code"), programme.get("title"), faculty.get("name"), programme.get("admissions_type"), programme.get("fund_type"), programme.get("first_year_intake"), programme.get("credits_required_for_graduation"), programme.get("programme_description"), json.dumps(programme.get("core_subjects_requirement")), json.dumps(programme.get("elective_subjects_requirement")), programme.get("programme_specific_admission_requirements"), programme.get("m1_m2"), programme.get("category_b_c"), json.dumps(programme.get("subject_weights")), programme.get("jupas_admission_score"), json.dumps(programme.get("career_opportunities")), json.dumps(programme.get("notes")), programme.get("information_website")))
    if "general_notes" in data:
        cursor.execute('INSERT OR REPLACE INTO general_info (key, value) VALUES (?, ?)', ('general_notes', json.dumps(data['general_notes'])))
    conn.commit()
    conn.close()
    logger.info("Database setup complete.")

def load_data_from_db():
    logger.info(f"Loading programme data from database: '{DB_FILE}'")
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM programmes")
    rows = cursor.fetchall()
    faculties = {}
    for row in rows:
        programme_dict = dict(row)
        faculty_name = programme_dict.pop("faculty_name")
        for key in ['core_subjects_requirement', 'elective_subjects_requirement', 'subject_weights', 'career_opportunities', 'notes']:
            if programme_dict.get(key) and isinstance(programme_dict[key], str):
                try:
                    programme_dict[key] = json.loads(programme_dict[key])
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode JSON for key '{key}' in programme '{programme_dict['code']}'")
        if faculty_name not in faculties:
            faculties[faculty_name] = {"name": faculty_name, "programmes": []}
        faculties[faculty_name]["programmes"].append(programme_dict)
    cursor.execute("SELECT value FROM general_info WHERE key = 'general_notes'")
    general_notes_row = cursor.fetchone()
    general_notes = json.loads(general_notes_row['value']) if general_notes_row else {}
    conn.close()
    reconstructed_data = {
        "faculties": list(faculties.values()),
        "general_notes": general_notes
    }
    logger.info("Successfully loaded and reconstructed all columns from database.")
    return reconstructed_data

# --- THIS IS THE CORRECTED FUNCTION ---
def create_documents(data):
    """
    Create document blocks for RAG.
    Adopt a "one document per course, one topic" strategy to improve search accuracy.
    """
    docs = []
    logger.info("Starting to create documents from programme data...")
    for faculty in data.get("faculties", []):
        for programme in faculty.get("programmes", []):
            meta = {"programme_code": programme.get("code"), "title": programme.get("title"), "faculty": faculty.get("name")}
            
            # doc 1: programme_description
            if programme.get("programme_description"):
                docs.append(Document(page_content=f"Programme Description for {programme.get('title')} ({programme.get('code')}): {programme.get('programme_description')}", metadata=meta))
            
            # doc 2: career_opportunities
            if programme.get("career_opportunities"):
                careers_str = json.dumps(programme.get('career_opportunities'), indent=2, ensure_ascii=False)
                docs.append(Document(page_content=f"Career and Study Pathways for {programme.get('title')} ({programme.get('code')}): {careers_str}", metadata=meta))

            # doc 3: core_subjects_requirement
            req_parts = []
            if programme.get("core_subjects_requirement"): 
                req_parts.append(f"Core Subjects: {json.dumps(programme.get('core_subjects_requirement'))}")
            if programme.get("elective_subjects_requirement"): 
                req_parts.append(f"Elective Subjects: {json.dumps(programme.get('elective_subjects_requirement'))}")
            if programme.get("programme_specific_admission_requirements"): 
                req_parts.append(f"Specific Requirements: {programme.get('programme_specific_admission_requirements')}")
            
            if req_parts:
                docs.append(Document(page_content=f"Admission Requirements for {programme.get('title')} ({programme.get('code')}): {'; '.join(req_parts)}", metadata=meta))
            
            # doc 4: notes and inforamtion_website (Merge these clutter information)
            other_info_parts = []
            if programme.get("notes"):
                other_info_parts.append(f"Notes: {json.dumps(programme.get('notes'), ensure_ascii=False)}")
            if programme.get("information_website"):
                other_info_parts.append(f"Official Website: {programme.get('information_website')}")
            
            if other_info_parts:
                 docs.append(Document(page_content=f"Other Information for {programme.get('title')} ({programme.get('code')}): {'; '.join(other_info_parts)}", metadata=meta))

    # doc 5: general_notes
    if data.get("general_notes"):
        docs.append(Document(page_content=f"General Admission Notes: {json.dumps(data.get('general_notes'))}", metadata={"type": "general_notes", "title": "General Notes"}))
    
    logger.info(f"Successfully created {len(docs)} focused documents for RAG.")
    return docs
# --- END OF CORRECTED FUNCTION ---

def extract_code_and_query(query: str, data: Dict) -> Tuple[str | None, str]:
    query_upper = query.upper()
    code_match = re.search(r'(JS\d{4}|[A-Z]{2,4}-[A-Z]{2,4})', query_upper)
    if code_match:
        code = code_match.group(0)
        if any(prog.get("code", "").upper() == code for fac in data.get("faculties", []) for prog in fac.get("programmes", [])):
            logger.info(f"Found and verified code '{code}' in query.")
            return code, query
    return None, query

def get_response(user_query: str, chat_history: list, chatbot_data: dict) -> str:
    programme_data = chatbot_data['programme_data']
    all_documents = chatbot_data['all_documents']
    vectorstore = chatbot_data['vectorstore']
    prompt_template = chatbot_data['prompt_template']

    programme_code, search_query = extract_code_and_query(user_query, programme_data)
    
    relevant_docs = []
    if programme_code:
        logger.info(f"Code '{programme_code}' found. Using direct manual filtering.")
        relevant_docs = [doc for doc in all_documents if doc.metadata.get("programme_code") == programme_code]
    else:
        logger.info("No code found. Using general semantic search.")
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(search_query)

    if not relevant_docs:
        logger.warning(f"No documents retrieved for query: '{search_query}'")
        context = "No relevant information found."
    else:
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
    
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
        return f"Sorry,API request fail,please try again：{str(e)}"

#  Streamlit UI

def main():
    st.set_page_config(page_title="HKBU Admissions Chatbot", page_icon="🎓")
    st.title("🎓 HKBU Admissions Chatbot")
    st.caption("A HKBU course chatbot")

    chatbot_data = initialize_chatbot()

    if not chatbot_data:
        st.error("Chatbot Initialization failed")
        st.stop()
    
    # Sidebar code has been removed
    
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
            with st.spinner("Search..."):
                response = get_response(prompt, [], chatbot_data)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()