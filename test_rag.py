import os
import pickle
import pytest
from google import genai
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel  # NEW: Import DeepEval's Gemini Wrapper
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

# Import your actual app functions and global state
from app import ask_video, STATE

# ==========================================
# TEST CONFIGURATION
# ==========================================
GEMINI_API_KEY = "Paste your active key here"

# Ensure this matches your actual cache file name inside the LensAI folder
CACHE_FILE = "Paste your pkl file name here.pkl" 

# The question we are testing
TEST_QUERY = "What is the main difference between filter and map functions?"

# DeepEval needs a "perfect" answer to calculate Precision and Recall
EXPECTED_OUTPUT = "The filter function returns the actual elements of the list that satisfy the condition (e.g., [2, 4, 6, 8, 0]), while the map function returns boolean values (True/False) representing whether each element satisfied the condition."

# ==========================================
# SETUP FIXTURE (Runs before the test)
# ==========================================
@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Loads your actual video database into memory without running the Flask server."""
    if os.path.exists(CACHE_FILE):
        print(f"\n📦 Loading test database from {CACHE_FILE}...")
        with open(CACHE_FILE, "rb") as f:
            cache_data = pickle.load(f)
            STATE['vector_db'] = cache_data["vector_db"]
    else:
        raise FileNotFoundError(f"Cache file {CACHE_FILE} not found. Please check the filename.")

# ==========================================
# THE ACTUAL TEST
# ==========================================
def test_rag_pipeline_full():
    client = genai.Client(api_key=GEMINI_API_KEY)

    print(f"\n🤖 Asking Gemini: '{TEST_QUERY}'")
    
    # 1. Run your actual hybrid search and generation pipeline
    answer, image_path, top_k = ask_video(client, TEST_QUERY)

    # 2. Extract the raw text chunks the Hybrid Search found
    retrieved_contexts = [item['context'] for item in top_k]
    
    print("\n✅ Gemini answered. Handing over to DeepEval to grade ALL metrics...")

    # 3. Create the DeepEval Test Case
    test_case = LLMTestCase(
        input=TEST_QUERY,
        actual_output=answer,
        expected_output=EXPECTED_OUTPUT, 
        retrieval_context=retrieved_contexts
    )

    # 4. FIX: Initialize Gemini 3.1 Flash Lite as the Metric Judge Model
    judge_model = GeminiModel(
        model="gemini-3.1-flash-lite",
        api_key=GEMINI_API_KEY,
        temperature=0
    )

    # 5. Pass the Gemini 3.1 Flash Lite judge directly into each metric
    faithfulness = FaithfulnessMetric(threshold=0.7, model=judge_model)
    answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=judge_model)
    contextual_relevancy = ContextualRelevancyMetric(threshold=0.7, model=judge_model)
    contextual_precision = ContextualPrecisionMetric(threshold=0.7, model=judge_model)
    contextual_recall = ContextualRecallMetric(threshold=0.7, model=judge_model)

    # 6. Run the evaluation!
    assert_test(test_case, [
        faithfulness, 
        answer_relevancy, 
        contextual_relevancy, 
        contextual_precision, 
        contextual_recall
    ])