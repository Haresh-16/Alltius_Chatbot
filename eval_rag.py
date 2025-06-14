#FUTURE WORK!
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from dotenv import load_dotenv

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

load_dotenv()
from rag_system import initialize_rag_system

def create_evaluation_dataset():
    """
    Creates a synthetic evaluation dataset for multi-turn conversations.

    NOTE: The 'ground_truth' answers here are synthetic. For a real evaluation,
    you should replace them with expert-verified, factual answers based on your
    actual knowledge base documents.
    """
    evaluation_data = [
        {
            "turn_id": 1,
            "question": "How do I add funds to my Angel One account?",
            "ground_truth": "You can add funds to your Angel One account using UPI, Net Banking, or by transferring funds via NEFT/RTGS from your registered bank account. This can be done through the Angel One mobile app or website.",
        },
        {
            "turn_id": 2, # Follow-up question
            "question": "Are there any charges for using UPI?",
            "ground_truth": "No, there are no charges for adding funds to your Angel One account using UPI.",
        },
        {
            "turn_id": 1,
            "question": "What are the eligibility requirements for getting a health insurance plan?",
            "ground_truth": "Eligibility for health insurance plans typically depends on age, medical history, and income. Specific requirements vary by plan, but generally, applicants must be within a certain age bracket (e.g., 18-65 years) and may need to undergo a medical check-up.",
        },
        {
            "turn_id": 2, # Follow-up question
            "question": "What if I have a pre-existing condition like diabetes?",
            "ground_truth": "If you have a pre-existing condition like diabetes, it may be covered after a specific waiting period, which is typically 2 to 4 years. Some plans may offer coverage from day one with an additional premium.",
        },
    ]
    return evaluation_data

def run_rag_and_generate_results(rag_system, eval_dataset):
    """
    Runs the RAG system over the evaluation dataset and collects results.
    """
    results = []
    for turn in sorted(eval_dataset, key=lambda x: x['turn_id']):
        print(f"Processing question (Turn {turn['turn_id']}): {turn['question']}")
        rag_output = rag_system.query(turn['question'])
        results.append({
            "question": turn["question"],
            "answer": rag_output["answer"],
            "contexts": rag_output["contexts"],
            "ground_truth": turn["ground_truth"],
        })
    return results

def main():
    """
    Main function to run the RAG evaluation.
    """
    print("🚀 Starting RAG evaluation script...")

    # 1. Initialize the RAG system
    print("Initializing RAG system...")
    rag_system = initialize_rag_system()
    if not rag_system:
        print("❌ Failed to initialize RAG system. Exiting.")
        return

    # 2. Create the evaluation questions and ground truths
    print("Creating evaluation dataset...")
    evaluation_dataset = create_evaluation_dataset()

    # 3. Run the RAG system and collect results
    print("Running RAG system over evaluation questions...")
    results_list = run_rag_and_generate_results(rag_system, evaluation_dataset)
    results_dataset = Dataset.from_list(results_list)

    # 4. Choose and wrap your custom models for RAGAs
    print("Initializing and wrapping evaluation LLM (Mistral)...")
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens=512,
        temperature=0.1,
    )
    # Wrap your LLM
    ragas_llm = LangchainLLMWrapper(llm)

    # Choose Embeddings model
    print("Initializing and wrapping evaluation embeddings (all-MiniLM-L6-v2)...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Wrap embeddings model
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings_model)


    # 5. Define the metrics for evaluation
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # 6. Run the evaluation
    print("\n🔬 Evaluating results with RAGAs...")
    result = evaluate(
        dataset=results_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # 7. Display the results
    print("\n📊 Evaluation Results:")
    df = result.to_pandas()
    print(df)

if __name__ == "__main__":
    main()