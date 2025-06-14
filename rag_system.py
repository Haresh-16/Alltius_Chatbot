# rag_system.py

import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime

# LangChain imports
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_huggingface import HuggingFaceEndpoint

# Pinecone and embeddings
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    HEALTH_PLAN = "health_plan"
    ANGELONE_SUPPORT = "angelone_support"
    IRRELEVANT = "irrelevant"

@dataclass
class RetrievalResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    namespace: str

@dataclass
class QueryAnalysis:
    query_type: QueryType
    namespace: str
    confidence: float
    reasoning: str

class MultiAgentRAGSystem:
    def __init__(self,
                 pinecone_api_key: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.pinecone_api_key = pinecone_api_key
        self.index_name = "rag-chatbot-index"
        self.embedding_model_name = embedding_model
        self.llm_model_id = llm_model
        
        # Add internal logging for Streamlit app
        self.internal_logs = []
        self.max_logs = 50  # Keep last 50 logs

        self._setup_pinecone()
        self._setup_embedding_model()
        self._setup_llm()
        self._setup_agents()

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )

    def _log_activity(self, agent: str, action: str):
        """Log agent activity for display in Streamlit app"""
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "agent": agent,
            "action": action
        }
        self.internal_logs.append(log_entry)
        
        # Keep only the last N logs to prevent memory issues
        if len(self.internal_logs) > self.max_logs:
            self.internal_logs = self.internal_logs[-self.max_logs:]
        
        # Also log to standard logger
        logger.info(f"LOG: {agent} - {action}")

    def _setup_pinecone(self):
        try:
            self._log_activity("System", "Connecting to Pinecone...")
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pc.Index(self.index_name)
            self._log_activity("System", f"✅ Connected to Pinecone index: {self.index_name}")
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            self._log_activity("System", f"❌ Pinecone connection failed: {str(e)[:50]}...")
            logger.error(f"Error connecting to Pinecone: {e}")
            raise

    def _setup_embedding_model(self):
        try:
            self._log_activity("System", "Loading embedding model...")
            self.encoder = SentenceTransformer(self.embedding_model_name)
            self._log_activity("System", f"✅ Loaded embedding model: {self.embedding_model_name}")
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            self._log_activity("System", f"❌ Embedding model failed: {str(e)[:50]}...")
            logger.error(f"Error loading embedding model: {e}")
            raise

    def _setup_llm(self):
        # MODIFIED: Check for the standard Hugging Face token name
        if not os.getenv("HUGGING_FACE_HUB_TOKEN"):
            self._log_activity("System", "❌ HUGGING_FACE_HUB_TOKEN not found")
            logger.error("Hugging Face token (HUGGING_FACE_HUB_TOKEN) not found in environment variables.")
            raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable not set")
        self._log_activity("System", f"✅ LLM configured: {self.llm_model_id}")
        logger.info(f"Configured to use HuggingFace Endpoint for model: {self.llm_model_id}")

    def _setup_agents(self):
        self._log_activity("System", "Initializing agents...")
        self.head_agent = HeadAgent(self)
        self.query_agent = QueryAgent(self)
        self.retriever_agent = RetrieverAgent(self)
        self.reranking_agent = RerankingAgent(self)
        self.answering_agent = AnsweringAgent(self)
        self._log_activity("System", "✅ All agents initialized successfully")
        logger.info("All agents initialized successfully")

    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_new_tokens: int = 1000) -> str:
        try:
            llm = HuggingFaceEndpoint(
                repo_id=self.llm_model_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                # MODIFIED: Removed the token argument. It will be picked up from the environment automatically.
            )

            prompt_content = "\n".join([msg["content"] for msg in messages])
            formatted_prompt = f"[INST] {prompt_content} [/INST]"
            response = llm.invoke(formatted_prompt)
            return response

        except Exception as e:
            logger.error(f"Error calling HuggingFace LLM: {e}")
            return "I apologize, but I'm having trouble processing your request right now."

    def query(self, user_query: str) -> str:
        try:
            return self.head_agent.process_query(user_query)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I apologize, but I encountered an error while processing your question."

# --- Agent classes remain the same ---
# (Rest of the file is identical to the previous version)

class HeadAgent:
    """Orchestrates all other agents"""

    def __init__(self, rag_system: MultiAgentRAGSystem):
        self.rag_system = rag_system

    def process_query(self, user_query: str) -> str:
        """Orchestrate the entire query processing pipeline"""
        self.rag_system._log_activity("Head Agent", f"🚀 Starting query processing: '{user_query[:50]}{'...' if len(user_query) > 50 else ''}'")
        logger.info(f"Head Agent: Processing query: {user_query[:100]}...")

        # Step 1: Query Analysis
        self.rag_system._log_activity("Head Agent", "📋 Step 1: Delegating to Query Agent for analysis")
        query_analysis = self.rag_system.query_agent.analyze_query(user_query)

        # Step 2: Check if query is relevant
        if query_analysis.query_type == QueryType.IRRELEVANT:
            self.rag_system._log_activity("Head Agent", "❌ Query deemed irrelevant - returning guidance message")
            return "I can only answer questions about health plans and Angel One support. How can I help with those topics?"

        self.rag_system._log_activity("Head Agent", f"✅ Query classified as: {query_analysis.query_type.value} (confidence: {query_analysis.confidence:.2f})")

        # Step 3: Retrieve documents
        self.rag_system._log_activity("Head Agent", "📋 Step 2: Delegating to Retriever Agent")
        retrieved_docs = self.rag_system.retriever_agent.retrieve(
            user_query, query_analysis.namespace
        )

        if not retrieved_docs:
            self.rag_system._log_activity("Head Agent", "❌ No documents retrieved - returning no results message")
            return "I could not find any relevant information to answer your question. Please try rephrasing it."

        self.rag_system._log_activity("Head Agent", f"✅ Retrieved {len(retrieved_docs)} documents")

        # Step 4: Rerank documents
        self.rag_system._log_activity("Head Agent", "📋 Step 3: Delegating to Reranking Agent")
        reranked_docs = self.rag_system.reranking_agent.rerank(
            user_query, retrieved_docs
        )

        # Step 5: Generate answer
        self.rag_system._log_activity("Head Agent", "📋 Step 4: Delegating to Answering Agent")
        answer = self.rag_system.answering_agent.generate_answer(
            user_query, reranked_docs, query_analysis
        )

        self.rag_system._log_activity("Head Agent", "🎉 Query processing completed successfully")
        return answer

class QueryAgent:
    """Analyzes queries to determine namespace and relevance"""

    def __init__(self, rag_system: MultiAgentRAGSystem):
        self.rag_system = rag_system

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine type and namespace"""
        self.rag_system._log_activity("Query Agent", f"🔍 Analyzing query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

        prompt = f"""
        Analyze the following user query and determine:
        1. Is it related to health plans/insurance eligibility? (health_plan)
        2. Is it related to Angel One stock broking platform? (angelone_support)
        3. Is it irrelevant to both? (irrelevant)

        Query: "{query}"

        Health plan topics include: insurance eligibility, medical conditions, coverage, plans, health requirements, medical history.
        Angel One topics include: trading, stocks, IPO, mutual funds, account management, charts, compliance, surveillance.

        Respond in JSON format:
        {{
            "query_type": "health_plan|angelone_support|irrelevant",
            "namespace": "all-docx-chunks-cleaned|all-pdf-chunks-cleaned|cleaned-angelone-faqs-chunks|none",
            "confidence": 0.0-1.0,
            "reasoning": "explanation"
        }}
        """

        try:
            self.rag_system._log_activity("Query Agent", "🤖 Calling LLM for query classification")
            response = self.rag_system._call_llm([
                {"role": "system", "content": "You are a query classification expert."},
                {"role": "user", "content": prompt}
            ], temperature=0.1) # Low temperature for classification

            self.rag_system._log_activity("Query Agent", "📝 Parsing LLM response")
            # Parse JSON response
            result = json.loads(response.strip())

            query_type = QueryType(result["query_type"])

            # Map query type to namespace
            if query_type == QueryType.HEALTH_PLAN:
                namespace = "all-docx-chunks-cleaned,all-pdf-chunks-cleaned"
            elif query_type == QueryType.ANGELONE_SUPPORT:
                namespace = "cleaned-angelone-faqs-chunks"
            else:
                namespace = "none"

            self.rag_system._log_activity("Query Agent", f"✅ Classification complete: {query_type.value} → {namespace}")
            self.rag_system._log_activity("Query Agent", f"💭 Reasoning: {result['reasoning'][:100]}{'...' if len(result['reasoning']) > 100 else ''}")

            return QueryAnalysis(
                query_type=query_type,
                namespace=namespace,
                confidence=result["confidence"],
                reasoning=result["reasoning"]
            )

        except Exception as e:
            self.rag_system._log_activity("Query Agent", f"❌ Analysis failed: {str(e)[:50]}...")
            logger.error(f"Error in query analysis: {e}")
            return QueryAnalysis(
                query_type=QueryType.IRRELEVANT,
                namespace="none",
                confidence=0.0,
                reasoning="Analysis failed"
            )

class RetrieverAgent:
    """Retrieves top-10 records from Pinecone"""

    def __init__(self, rag_system: MultiAgentRAGSystem):
        self.rag_system = rag_system

    def retrieve(self, query: str, namespaces: str, top_k: int = 10) -> List[RetrievalResult]:
        """Retrieve relevant documents from Pinecone"""
        self.rag_system._log_activity("Retriever Agent", f"📚 Starting document retrieval from: {namespaces}")

        if namespaces == "none":
            self.rag_system._log_activity("Retriever Agent", "❌ No valid namespaces provided")
            return []

        try:
            # Generate query embedding
            self.rag_system._log_activity("Retriever Agent", "🔢 Generating query embedding")
            query_embedding = self.rag_system.encoder.encode([query]).tolist()[0]

            # Split namespaces if multiple
            namespace_list = [ns.strip() for ns in namespaces.split(",")]
            self.rag_system._log_activity("Retriever Agent", f"🎯 Searching {len(namespace_list)} namespace(s): {', '.join(namespace_list)}")

            all_results = []

            # Query each namespace
            for namespace in namespace_list:
                try:
                    self.rag_system._log_activity("Retriever Agent", f"🔍 Querying namespace: {namespace}")
                    results = self.rag_system.index.query(
                        vector=query_embedding,
                        top_k=top_k // len(namespace_list) + 1,
                        namespace=namespace,
                        include_metadata=True,
                        include_values=False
                    )

                    namespace_results = 0
                    for match in results.matches:
                        all_results.append(RetrievalResult(
                            content=match.metadata.get('content', ''),
                            metadata=match.metadata,
                            score=match.score,
                            namespace=namespace
                        ))
                        namespace_results += 1

                    self.rag_system._log_activity("Retriever Agent", f"✅ Found {namespace_results} documents in {namespace}")

                except Exception as e:
                    self.rag_system._log_activity("Retriever Agent", f"⚠️ Error querying {namespace}: {str(e)[:50]}...")
                    logger.warning(f"Error querying namespace {namespace}: {e}")
                    continue

            # Sort by score and return top 10
            all_results.sort(key=lambda x: x.score, reverse=True)
            final_results = all_results[:top_k]
            
            self.rag_system._log_activity("Retriever Agent", f"🎯 Returning top {len(final_results)} documents (scores: {[f'{r.score:.3f}' for r in final_results[:3]]}...)")
            return final_results

        except Exception as e:
            self.rag_system._log_activity("Retriever Agent", f"❌ Retrieval failed: {str(e)[:50]}...")
            logger.error(f"Error in retrieval: {e}")
            return []

class RerankingAgent:
    """Reranks retrieved documents by relevance"""

    def __init__(self, rag_system: MultiAgentRAGSystem):
        self.rag_system = rag_system

    def rerank(self, query: str, documents: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank documents based on relevance to query"""
        self.rag_system._log_activity("Reranking Agent", f"🎯 Starting reranking of {len(documents)} documents")

        if not documents:
            self.rag_system._log_activity("Reranking Agent", "❌ No documents to rerank")
            return []

        try:
            doc_texts = []
            for i, doc in enumerate(documents):
                doc_texts.append(f"{i+1}. {doc.content[:200]}...")

            self.rag_system._log_activity("Reranking Agent", "🤖 Calling LLM for document reranking")
            prompt = f"""
            Query: "{query}"

            Rank the following documents by relevance to the query (1 = most relevant):

            {chr(10).join(doc_texts)}

            Respond with only the numbers in order of relevance (e.g., "3,1,5,2,4"):
            """

            response = self.rag_system._call_llm([
                {"role": "system", "content": "You are a document ranking expert."},
                {"role": "user", "content": prompt}
            ], temperature=0.1)

            try:
                self.rag_system._log_activity("Reranking Agent", "📝 Parsing reranking results")
                rankings = [int(x.strip()) - 1 for x in response.strip().split(",")]
                reranked = []

                for rank in rankings:
                    if 0 <= rank < len(documents):
                        reranked.append(documents[rank])

                for i, doc in enumerate(documents):
                    if i not in rankings:
                        reranked.append(doc)

                self.rag_system._log_activity("Reranking Agent", f"✅ Reranking complete - new order: {rankings[:5]}...")
                return reranked[:len(documents)]

            except ValueError:
                self.rag_system._log_activity("Reranking Agent", "⚠️ Failed to parse rankings, returning original order")
                return documents

        except Exception as e:
            self.rag_system._log_activity("Reranking Agent", f"❌ Reranking failed: {str(e)[:50]}...")
            logger.error(f"Error in reranking: {e}")
            return documents

class AnsweringAgent:
    """Generates final answers using context and chat history"""

    def __init__(self, rag_system: MultiAgentRAGSystem):
        self.rag_system = rag_system

    def generate_answer(self, query: str, context_docs: List[RetrievalResult],
                       query_analysis: QueryAnalysis) -> str:
        """Generate final answer using context and chat history"""
        self.rag_system._log_activity("Answering Agent", f"💬 Starting answer generation for {query_analysis.query_type.value} query")

        if not context_docs:
            self.rag_system._log_activity("Answering Agent", "❌ No context documents provided")
            return "I could not find any relevant information to answer your question."

        self.rag_system._log_activity("Answering Agent", f"📄 Preparing context from {len(context_docs)} documents")
        context = "\n\n".join([
            f"Source {i+1}: {doc.content}"
            for i, doc in enumerate(context_docs)
        ])

        self.rag_system._log_activity("Answering Agent", "🧠 Retrieving chat history from memory")
        chat_history = self.rag_system.memory.chat_memory.messages
        history_text = ""
        if chat_history:
            history_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in chat_history[-4:]
            ])
            self.rag_system._log_activity("Answering Agent", f"📚 Using {len(chat_history[-4:])} previous messages for context")
        else:
            self.rag_system._log_activity("Answering Agent", "📝 No previous chat history available")

        if query_analysis.query_type == QueryType.HEALTH_PLAN:
            system_prompt = """You are an expert assistant for health insurance. Use only the provided context to answer. If the context is insufficient, say 'I cannot answer based on the provided information.'"""
            self.rag_system._log_activity("Answering Agent", "🏥 Using health insurance expert persona")
        else:
            system_prompt = """You are an expert assistant for Angel One support. Use only the provided context to answer. If the context is insufficient, say 'I cannot answer based on the provided information.'"""
            self.rag_system._log_activity("Answering Agent", "📈 Using Angel One support expert persona")

        prompt = f"""
        {system_prompt}

        Context:
        {context}

        Chat History:
        {history_text}

        User Question: {query}

        Instructions:
        - Answer based ONLY on the provided context.
        - Be helpful and specific.
        - Do not use any information outside of the context block.

        Answer:
        """

        try:
            self.rag_system._log_activity("Answering Agent", "🤖 Calling LLM to generate final answer")
            response = self.rag_system._call_llm([
                {"role": "user", "content": prompt}
            ], temperature=0.5)

            self.rag_system._log_activity("Answering Agent", "💾 Updating conversation memory")
            self.rag_system.memory.chat_memory.add_user_message(query)
            self.rag_system.memory.chat_memory.add_ai_message(response)

            self.rag_system._log_activity("Answering Agent", f"✅ Answer generated successfully ({len(response)} characters)")
            return response.strip()

        except Exception as e:
            self.rag_system._log_activity("Answering Agent", f"❌ Answer generation failed: {str(e)[:50]}...")
            logger.error(f"Error generating answer: {e}")
            return "An error occurred while generating the answer."

def initialize_rag_system(pinecone_api_key: str = None) -> Optional[MultiAgentRAGSystem]:
    """Initialize the multi-agent RAG system"""

    if not pinecone_api_key:
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    # MODIFIED: Check for the standard Hugging Face token name
    if not os.getenv('HUGGING_FACE_HUB_TOKEN'):
        raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable not set")

    try:
        return MultiAgentRAGSystem(pinecone_api_key=pinecone_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return None

if __name__ == "__main__":
    try:
        rag = initialize_rag_system()
        if rag:
            test_queries = [
                "How do I add funds to my Angel One account?",
                "What are the charges for margin pledge?",
                "How to apply for IPO?",
                "What is insurance coverage?",
                "How to contact customer support?"
            ]
            print("Testing RAG System with HuggingFace Endpoint:")
            print("=" * 50)
            for query in test_queries:
                print(f"\nQuery: {query}")
                print("-" * 30)
                response = rag.query(query)
                print(f"Response: {response}")
                print("=" * 50)
    except ValueError as e:
        print(f"Initialization failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")