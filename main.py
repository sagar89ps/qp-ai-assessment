from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from pydantic import BaseModel
from document_parser import parse_document
from vector_storage import vector_store
from llm_handler import answer_question
import multipart

app = FastAPI()

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryModel(BaseModel):
    question: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = f"temp/{file.filename}"
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Parse the document
        text = parse_document(file_path)
        
        # Add chunks to vector store
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        vector_store.add(chunks)
        
        return {
            "filename": file.filename, 
            "message": "Document uploaded and processed",
            "text_length": len(text)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_chatbot(query: QueryModel):
    context = vector_store.search(query.question)
    answer = answer_question(query.question, context)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)