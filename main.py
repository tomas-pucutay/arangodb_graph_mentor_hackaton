from src.agent import query_graph_rag
from src.agent import best_action_plan_per_competence, suggest_action_plans_for_competence
from src.agent import find_best_competences_for_action_plan, find_duration_action_plan

import streamlit as st
import time

def show_home():
    st.title("Welcome")
    st.image("./assets/development.webp", use_container_width=True)
    st.markdown("### Come and discover your best version.")

def show_chatbot():
    st.title("Competency Development Mentor")
    st.write("Ask questions about your professional growth and get structured answers.")
    
    categories = {
        "General questions": ["What competences exist?"],
        "Start Here: Find your competence": ["Write about you and let the system find the most suitable competence to develop"],
        "Now, generate an action plan": ["In my last meeting with my boss, he told me that I need to improve negotiation, what can I do?"],
        "Best way to achieve a competence": ["What is the best activity to improve my communication?"],
        "Time to spend on action plans": ["How much time will it take me to finish my activity of case study for a new development?"],
        "Relevance of my action plan": ["How relevant is learn better presentations to competencies?"]
    }
    
    selection = st.radio("Choose a category:", list(categories.keys()))
    for example in categories[selection]:
        st.markdown(f"- Example: {example}")
    
    user_input = st.text_input("Write your question:")
    if user_input:
        response = query_graph_rag(user_input)
        
        if selection == "Now, generate an action plan":
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Raw data:", suggest_action_plans_for_competence(user_input), height=300, key="raw_now")
            with col2:
                st.text_area("Answer:", response, height=300, key="answer_now")

        elif selection == "Best way to achieve a competence":
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Raw data:", best_action_plan_per_competence(user_input), height=300, key="raw_best")
            with col2:
                st.text_area("Answer:", response, height=300, key="answer_best")

        elif selection == "Time to spend on action plans":
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Raw data:", find_duration_action_plan(user_input), height=300, key="raw_time")
            with col2:
                st.text_area("Answer:", response, height=300, key="answer_time")

        elif selection == "Relevance of my action plan":
            col1, col2 = st.columns(2)
            with col1:
                st.text_area("Raw data:", find_best_competences_for_action_plan(user_input), height=300, key="raw_relevance")
            with col2:
                st.text_area("Answer:", response, height=300, key="answer_relevance")

        else:
            st.text_area("Answer:", response, height=300, key="default_answer")

def main():
    st.set_page_config(page_title="Competencies development", layout="wide")
    
    st.markdown("""
        <style>
        body { font-family: 'Arial', sans-serif; background-color: #f4f4f9; }
        .stButton>button { background-color: #0052cc; color: white; border-radius: 8px; padding: 8px 16px; }
        .stTextInput>div>div>input { border-radius: 8px; border: 1px solid #ccc; padding: 10px; }
        </style>
        """, unsafe_allow_html=True)
    
    opcion = st.sidebar.radio("Navigation", ["Home", "Chatbot"])
    if opcion == "Home":
        show_home()
    else:
        show_chatbot()

if __name__ == "__main__":
    main()