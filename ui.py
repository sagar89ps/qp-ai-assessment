import streamlit as st
import requests

st.title("Document Upload and Query")

# File Upload Section
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

if uploaded_file is not None:
    # Use the filename and the file content
    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
    with st.spinner("Uploading..."):
        try:
            # Ensure correct port and full URL
            response = requests.post("http://localhost:8005/upload", files=files)
            if response.status_code == 200:
                st.success("File uploaded and processed successfully!")
                st.json(response.json())
                
                # Store the filename in session state for potential future use
                st.session_state.uploaded_filename = uploaded_file.name
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Connection Error: {e}")
            st.error("Possible reasons:\n"
                     "1. Ensure the FastAPI server is running\n"
                     "2. Check that port 8005 is correct\n"
                     "3. Verify no other service is using this port")

# Question Asking Section
question = st.text_input("Ask a question about the uploaded document")

if question:
    try:
        # Send the question in the correct JSON format
        response = requests.post("http://localhost:8005/query", json={"question": question})
        if response.status_code == 200:
            st.write(f"Answer: {response.json()['answer']}")
        else:
            st.error(f"Query Error: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")