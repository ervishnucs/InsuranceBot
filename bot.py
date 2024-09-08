import streamlit as st
import os
import nltk
from thirdai import licensing, neural_db as ndb
from openai import OpenAI
from dotenv import load_dotenv
from streamlit_chat import message

# Download NLTK data
nltk.download("punkt")

# Load environment variables from .env file
load_dotenv()

# Access the keys
openai_api_key = os.getenv("OPENAI_API_KEY")
thirdai_key = os.getenv("THIRDAI_KEY")

# Licensing and setup
if thirdai_key:
    licensing.activate(thirdai_key)

openai_client = OpenAI(api_key=openai_api_key)

# Initialize the neural database
db = ndb.NeuralDB()
insertable_docs = []
doc_files = [
    "cash-back-plan-brochuree.pdf",
    "gold-brochure (2).pdf",
    "guaranteed-monthly-income-plan.pdf.coredownload.inline.pdf",
    "guaranteed-protection-plus-plan-brochure.pdf",
    "indiafirst-csc-shubhlabh-plan-brochure - Copy - Copy.pdf",
    "indiafirst-life-fortune-plus-plan-brochure.pdf.coredownload.inline.pdf",
]

for file in doc_files:
    doc = ndb.PDF(file)
    insertable_docs.append(doc)

db.insert(insertable_docs, train=False)

def generate_answers(query, references):
    context = "\n\n".join(references[:3])
    prompt = (
        "answer with little respect.\n"
        f"You are an insurance chatbot designed to provide information about insurance plans. "
        f"Respond to the following question by providing clear, accurate, and relevant information about insurance plans. "
        f"Make sure to address the specifics of the plan and any important details that might help the user make an informed decision.\n"
        f"Question: {query}\n"
        f"Context: {context}\n"
        "Provide a short answer that helps the user understand their options.\n"
        "Give a specific answer within 3-4 lines.\n"
        "Ask if there is any additional doubt on the provided answer.\n"
    )

    messages = [{"role": "user", "content": prompt}]
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, temperature=0
    )
    return response.choices[0].message.content

def generate_queries_chatgpt(original_query):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates multiple search queries based on a single input query."},
            {"role": "user", "content": f"Generate multiple search queries related to: {original_query}"},
            {"role": "user", "content": "OUTPUT (5 queries):"}
        ]
    )
    generated_queries = response.choices[0].message.content.strip().split("\n")
    return generated_queries

def get_references(query):
    search_results = db.search(query, top_k=50)
    references = [result.text for result in search_results]
    return references

def reciprocal_rank_fusion(reference_list, k=60):
    fused_scores = {}
    for i in reference_list:
        for rank, j in enumerate(i):
            if j not in fused_scores:
                fused_scores[j] = 0
            fused_scores[j] += 1 / ((rank+1) + k)
    reranked_results = {j: score for j, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    return reranked_results

def get_answer(query, r):
    return generate_answers(query=query, references=r)

# Initialize session state to store the conversation
if 'responses' not in st.session_state:
    st.session_state['responses'] = []
    
if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Callback function to process input
def process_input():
    user_input = st.session_state.input
    if user_input:
        # Append user query to requests
        st.session_state['requests'].append(user_input)
        
        # Generate multiple search queries using ChatGPT
        query_list = generate_queries_chatgpt(user_input)
        
        # Get references for each generated query
        reference_list = [get_references(q) for q in query_list]
        
        # Apply reciprocal rank fusion to merge results
        r = reciprocal_rank_fusion(reference_list)
        
        # Get ranked reference list
        ranked_reference_list = [i for i in r.keys()]
        
        # Get the final answer using the references
        ans = get_answer(user_input, ranked_reference_list)
        
        # Append the bot's answer to responses
        st.session_state['responses'].append(ans)
        
        # Clear the input field after submission
        st.session_state.input = ""
        st.experimental_rerun()

# Streamlit Title
st.title("Insurance Bot")

# Check if the initial prompt needs to be added
if len(st.session_state['requests']) == 0 and len(st.session_state['responses']) == 0:
    st.session_state['responses'].append("How can I assist you with your insurance queries?")

# Display the conversation history like a chat interface
for i in range(len(st.session_state['responses'])):
    message(st.session_state['responses'][i], key=str(i) + '_bot')
    
    # Display user message on the right, only if the user has submitted a query
    if i < len(st.session_state['requests']):
        message(st.session_state['requests'][i], is_user=True, key=str(i) + '_user')

# Input box for user queries, using the process_input function to handle changes
st.text_input("Do you have any questions?", key="input", on_change=process_input)
