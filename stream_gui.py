import streamlit as st
from Model_Connection_Rag import question,query_openrouter,faiss_loading  # Make sure this function returns a string
from Vector_Data_Store_Embed import create_index_file

st.title("Medical Chatbot")

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
query = st.text_input("Ask a question about cancer:")

# Handle submission
if st.button("Ask"):
    if query:
        with st.spinner("Thinking..."):
            try:
                answer = question(query)

                if answer:
                    st.session_state.history.append((query, answer))
                else:
                    st.warning("⚠️ No answer returned from model.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Display history
if st.session_state.history:
    st.markdown("---")
    st.subheader("Chat History")
    for q, a in reversed(st.session_state.history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")

