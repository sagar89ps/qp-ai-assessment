# import json
# from typing import List, Dict
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# class ChatbotEvaluator:
#     def __init__(self, ground_truth_file: str):
#         """
#         Initialize evaluator with ground truth data
        
#         Args:
#             ground_truth_file (str): Path to JSON file with test cases
#         """
#         with open(ground_truth_file, 'r') as f:
#             self.ground_truth = json.load(f)
    
#     def evaluate(self, predict_func):
#         """
#         Evaluate chatbot performance
        
#         Args:
#             predict_func (callable): Function to get chatbot predictions
        
#         Returns:
#             Dict with performance metrics
#         """
#         predictions = []
#         true_labels = []
        
#         for test_case in self.ground_truth:
#             context = test_case.get('context', [])
#             question = test_case['question']
#             expected_answer = test_case['answer']
            
#             # Get chatbot prediction
#             prediction = predict_func(question, context)
            
#             # Compare prediction with expected answer
#             is_correct = self._is_answer_correct(prediction, expected_answer)
            
#             predictions.append(is_correct)
#             true_labels.append(True)
        
#         # Calculate metrics
#         accuracy = accuracy_score(true_labels, predictions)
#         precision, recall, f1, _ = precision_recall_fscore_support(
#             true_labels, predictions, average='binary'
#         )
        
#         return {
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1_score": f1
#         }
    
#     def _is_answer_correct(self, prediction: str, expected: str, threshold: float = 0.7):
#         """
#         Check if prediction matches expected answer
        
#         Args:
#             prediction (str): Chatbot's prediction
#             expected (str): Expected answer
#             threshold (float): Similarity threshold
        
#         Returns:
#             bool: Whether prediction is considered correct
#         """
#         # Implement your similarity checking logic here
#         # This could use techniques like:
#         # - Levenshtein distance
#         # - Semantic similarity
#         # - Keyword matching
#         return prediction.lower() in expected.lower()

# # Example usage
# def predict_func(question, context):
#     # Placeholder for actual prediction function
#     return "Sample answer"

# # Load test cases and run evaluation
# evaluator = ChatbotEvaluator('test_cases.json')
# results = evaluator.evaluate(predict_func)
# print(json.dumps(results, indent=2))



import json
import random
import difflib
import re
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ChatbotEvaluator:
    def __init__(self, ground_truth_file: str):
        """
        Initialize evaluator with ground truth data
        
        Args:
            ground_truth_file (str): Path to JSON file with test cases
        """
        with open(ground_truth_file, 'r') as f:
            data = json.load(f)
            self.ground_truth = data['test_cases']
    
    def evaluate(self, predict_func):
        """
        Evaluate chatbot performance
        
        Args:
            predict_func (callable): Function to get chatbot predictions
        
        Returns:
            Dict with performance metrics
        """
        predictions = []
        true_labels = []
        detailed_results = []
        
        for test_case in self.ground_truth:
            context = test_case.get('context', [])
            question = test_case['question']
            expected_answer = test_case['answer']
            
            # Get chatbot prediction
            prediction = predict_func(question, context)
            
            # Compare prediction with expected answer
            is_correct = self._is_answer_correct(prediction, expected_answer)
            
            predictions.append(is_correct)
            true_labels.append(True)
            
            # Store detailed result for analysis
            detailed_results.append({
                'question': question,
                'expected': expected_answer,
                'predicted': prediction,
                'is_correct': is_correct
            })
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Handle potential precision calculation issues
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary', zero_division=0
            )
        except Exception:
            precision = recall = f1 = 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "detailed_results": detailed_results
        }
    
    def _is_answer_correct(self, prediction: str, expected: str, threshold: float = 0.6):
        """
        Advanced similarity checking method
        
        Args:
            prediction (str): Chatbot's predicted answer
            expected (str): Ground truth answer
            threshold (float): Similarity threshold
        
        Returns:
            bool: Whether prediction is considered correct
        """
        # Preprocess texts
        def preprocess(text: str) -> str:
            """Clean and normalize text for comparison"""
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation and extra whitespace
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        
        # Exact match check (case-insensitive)
        if preprocess(prediction) == preprocess(expected):
            return True
        
        # Levenshtein-based similarity
        def levenshtein_similarity(str1: str, str2: str) -> float:
            """Calculate Levenshtein similarity ratio"""
            matcher = difflib.SequenceMatcher(None, str1, str2)
            return matcher.ratio()
        
        # Enhanced keyword matching with semantic understanding
        def advanced_keyword_match(prediction: str, expected: str) -> float:
            """
            Calculate a more sophisticated keyword matching score
            
            Considers:
            - Presence of key terms
            - Partial matches
            - Order of important words
            """
            pred_words = preprocess(prediction).split()
            exp_words = preprocess(expected).split()
            
            # Check for core terms
            core_terms = [word for word in exp_words if len(word) > 3]
            
            # Track matching core terms
            matched_core_terms = [
                term for term in core_terms 
                if any(term in pred_word or pred_word in term for pred_word in pred_words)
            ]
            
            # Calculate match ratio
            core_match_ratio = len(matched_core_terms) / len(core_terms) if core_terms else 0
            
            return core_match_ratio
        
        # Semantic similarity using TF-IDF and cosine similarity
        def semantic_similarity(text1: str, text2: str) -> float:
            """Calculate semantic similarity using TF-IDF and cosine similarity"""
            vectorizer = TfidfVectorizer().fit_transform([text1, text2])
            cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
            return cosine_sim
        
        # Compute individual similarity scores
        levenshtein_score = levenshtein_similarity(
            preprocess(prediction), 
            preprocess(expected)
        )
        
        semantic_score = semantic_similarity(
            preprocess(prediction), 
            preprocess(expected)
        )
        
        keyword_score = advanced_keyword_match(prediction, expected)
        
        # Combine scores with more balanced weighting
        combined_score = (
            0.3 * levenshtein_score + 
            0.4 * semantic_score + 
            0.3 * keyword_score
        )
        
        # Return True if combined score exceeds threshold
        return combined_score >= threshold

# Example more realistic prediction function with some variations
def predict_func(question, context):
    # Simulating imperfect predictions with some variations
    predictions = {
        "What is the capital of France?": [
            "Paris is the capital of France.",
            "The capital of France happens to be Paris.",
            "Paris, located in France, serves as its capital city.",
            "France's main city is Paris."
        ],
        "How many planets are in our solar system?": [
            "There are 8 planets in our solar system.",
            "Our solar system contains 8 planets.",
            "Currently, 8 planets are recognized in the solar system.",
            "8 planets exist in the solar system."
        ],
        "Who wrote Romeo and Juliet?": [
            "William Shakespeare wrote Romeo and Juliet.",
            "The play Romeo and Juliet was written by Shakespeare.",
            "Shakespeare is the author of Romeo and Juliet.",
            "Romeo and Juliet was penned by William Shakespeare."
        ],
        "What is the largest mammal in the world?": [
            "The blue whale is the largest mammal in the world.",
            "Blue whales are considered the largest mammals.",
            "In terms of size, blue whales top the list of mammals.",
            "Largest mammal? That would be the blue whale."
        ],
        "When was the Eiffel Tower built?": [
            "The Eiffel Tower was built in 1889 for the World's Fair in Paris.",
            "In 1889, during the World's Fair, the Eiffel Tower was constructed.",
            "Paris saw the Eiffel Tower's construction in 1889 for the World's Fair.",
            "The famous tower was built in 1889 as part of the World's Fair."
        ]
    }
    
    # Introduce some randomness and potential incorrect responses
    if random.random() < 0.1:  # 10% chance of a completely wrong answer
        return "I'm not sure about the answer."
    
    # Randomly select from predefined variations or similar answers
    return random.choice(predictions.get(question, ["I don't know."]))

# Prepare test cases JSON
test_cases = {
    "test_cases": [
        {
            "question": "What is the capital of France?",
            "context": ["European countries", "Geography"],
            "answer": "Paris is the capital of France."
        },
        {
            "question": "How many planets are in our solar system?",
            "context": ["Astronomy", "Planetary science"],
            "answer": "There are 8 planets in our solar system."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "context": ["Literature", "Shakespeare"],
            "answer": "William Shakespeare wrote Romeo and Juliet."
        },
        {
            "question": "What is the largest mammal in the world?",
            "context": ["Marine biology", "Animals"],
            "answer": "The blue whale is the largest mammal in the world."
        },
        {
            "question": "When was the Eiffel Tower built?",
            "context": ["Paris history", "Architecture"],
            "answer": "The Eiffel Tower was built in 1889 for the World's Fair in Paris."
        }
    ]
}

# Save test cases
with open('test_cases.json', 'w') as f:
    json.dump(test_cases, f, indent=4)

# Run multiple evaluations to show performance variability
print("Running 5 evaluations to demonstrate performance variability:")
for i in range(5):
    print(f"\nEvaluation {i+1}:")
    evaluator = ChatbotEvaluator('test_cases.json')
    results = evaluator.evaluate(predict_func)
    
    # Print key metrics
    print(f"Accuracy: {results['accuracy']:.2f}")
    print(f"Precision: {results['precision']:.2f}")
    print(f"Recall: {results['recall']:.2f}")
    print(f"F1 Score: {results['f1_score']:.2f}")
    
    # Optionally print detailed results
    print("\nDetailed Results:")
    for result in results['detailed_results']:
        print(f"Question: {result['question']}")
        print(f"Expected: {result['expected']}")
        print(f"Predicted: {result['predicted']}")
        print(f"Correct: {result['is_correct']}\n")