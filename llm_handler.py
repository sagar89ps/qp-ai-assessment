from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a more robust question-answering model
qa_pipeline = pipeline(
    "question-answering", 
    model="deepset/roberta-base-squad2",
    # Alternatively: "distilbert-base-uncased-distilled-squad"
)

def answer_question(question, context, max_answer_length=100):
    """
    Enhanced question answering function with more robust error handling
    
    Args:
        question (str): User's question
        context (list): List of context tuples (text, distance)
        max_answer_length (int, optional): Maximum length of the answer
    
    Returns:
        str: Generated answer
    """
    try:
        if not context:
            return "I couldn't find relevant information to answer your question."
        
        # Combine context chunks, prioritizing most similar
        context_text = " ".join([chunk for chunk, _ in sorted(context, key=lambda x: x[1])[:3]])
        
        # Perform question answering
        response = qa_pipeline(question=question, context=context_text)
        
        # Truncate answer if too long
        answer = response['answer'][:max_answer_length] + '...' if len(response['answer']) > max_answer_length else response['answer']
        
        # Log confidence and score for debugging
        logger.info(f"Answer confidence: {response['score']}, Length: {len(answer)}")
        
        return answer if response['score'] > 0.1 else "I'm not confident about the answer based on the available context."
    
    except Exception as e:
        logger.error(f"Error in question answering: {e}")
        return "I encountered an error while trying to answer your question."
