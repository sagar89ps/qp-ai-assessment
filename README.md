# qp-ai-assessment

# Contextual Chatbot

## Project Overview
A contextual chatbot that allows users to upload documents and ask questions based on the document's content.

## Features
- Document Upload (PDF/DOCX)
- Semantic Search
- Question Answering
- Performance Evaluation

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/qp-ai-assessment.git
cd qp-ai-assessment
```

2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Running the Application
1. Start Backend
```bash
uvicorn backend.main:app --reload
```

2. Start Frontend
```bash
streamlit run frontend/ui.py
```

## Testing
```bash
python -m pytest tests/
```

## MLOps Pipeline
Refer to `mlops/pipeline.drawio` for the complete MLOps workflow.

## Performance Evaluation
Run performance tests using:
```bash
python mlops/performance_evaluation.py
```

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

