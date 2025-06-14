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

# Streamlit for secrets management
import streamlit as st

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
                 huggingface_token: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 llm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.pinecone_api_key = pinecone_api_key
        self.huggingface_token = huggingface_token
        self.index_name = "rag-chatbot-index"
        self.embedding_model_name = embedding_model
        self.llm_model_id = llm_model
        self.internal_logs = []

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
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "agent": agent,
            "action": action
        }
        self.internal_logs.append(log_entry)
        logger.info(f"LOG: {agent} - {action}")

    def _setup_pinecone(self):
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {e}")
            raise

    def _setup_embedding_model(self):
        try:
            self.encoder = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def _setup_llm(self):
        if not self.huggingface_token:
            logger.error("Hugging Face token not provided.")
            raise ValueError("HUGGING_FACE_HUB_TOKEN not provided")
        # Set the token as environment variable for HuggingFaceEndpoint
        os.environ["HUGGING_FACE_HUB_TOKEN"] = self.huggingface_token
        logger.info(f"Configured to use HuggingFace Endpoint for model: {self.llm_model_id}")

    def _setup_agents(self):
        self.head_agent = HeadAgent(self)
        self.query_agent = QueryAgent(self)
        self.retriever_agent = RetrieverAgent(self)
        self.reranking_agent = RerankingAgent(self)
        self.answering_agent = AnsweringAgent(self)
        logger.info("All agents initialized successfully")

    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_new_tokens: int = 1000) -> str:
        try:
            llm = HuggingFaceEndpoint(
                repo_id=self.llm_model_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 0.01,
                huggingfacehub_api_token=self.huggingface_token
            )
            prompt_content = "\n".join([msg["content"] for msg in messages])
            formatted_prompt = f"[INST] {prompt_content} [/INST]"
            response = llm.invoke(formatted_prompt)
            return response
        except Exception as e:
            logger.error(f"Error calling HuggingFace LLM: {e}")
            return "I apologize, but I'm having trouble processing your request right now."

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        MODIFIED: This method now returns a dictionary with 'answer' and 'contexts'.
        """
        try:
            self.internal_logs.clear()
            return self.head_agent.process_query(user_query)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question.",
                "contexts": []
            }

class HeadAgent:
    def __init__(self, rag_system: MultiAgentRAGSystem):
        self.rag_system = rag_system

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        MODIFIED: This method now returns a dictionary from the AnsweringAgent.
        """
        self.rag_system._log_activity("Head Agent", f"Received query: '{user_query[:50]}...'")
        self.rag_system._log_activity("Head Agent", "Delegating to Query Agent for analysis.")
        query_analysis = self.rag_system.query_agent.analyze_query(user_query)

        if query_analysis.query_type == QueryType.IRRELEVANT:
            self.rag_system._log_activity("Head Agent", "Query is irrelevant. Halting process.")
            return {
                "answer": "I can only answer questions about health plans and Angel One support. How can I help with those topics?",
                "contexts": []
            }

        self.rag_system._log_activity("Head Agent", f"Query type '{query_analysis.query_type.value}'. Delegating to Retriever Agent.")
        retrieved_docs = self.rag_system.retriever_agent.retrieve(user_query, query_analysis.namespace)

        if not retrieved_docs:
            self.rag_system._log_activity("Head Agent", "No documents found. Halting process.")
            return {
                "answer": "I could not find any relevant information to answer your question. Please try rephrasing it.",
                "contexts": []
            }

        self.rag_system._log_activity("Head Agent", f"Retrieved {len(retrieved_docs)} documents. Delegating to Reranking Agent.")
        reranked_docs = self.rag_system.reranking_agent.rerank(user_query, retrieved_docs)

        self.rag_system._log_activity("Head Agent", f"Reranked to {len(reranked_docs)} documents. Delegating to Answering Agent.")
        # MODIFIED: The result from AnsweringAgent is now returned directly
        result = self.rag_system.answering_agent.generate_answer(user_query, reranked_docs, query_analysis)

        self.rag_system._log_activity("Head Agent", "Process complete. Returning final answer and contexts.")
        return result

class QueryAgent:
    def __init__(self, rag_system: MultiAgentRAGSystem):
        self.rag_system = rag_system

    def analyze_query(self, query: str) -> QueryAnalysis:
        self.rag_system._log_activity("Query Agent", "Analyzing query with enhanced few-shot prompt.")
        prompt = f"""You are an expert query classifier...""" # (keeping previous enhanced prompt)
        try:
            response = self.rag_system._call_llm([{"role": "user", "content": prompt}], temperature=0.0)
            cleaned_response = response.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned_response)
            query_type = QueryType(result["query_type"])

            if query_type == QueryType.HEALTH_PLAN: namespace = "all-docx-chunks-cleaned,all-pdf-chunks-cleaned"
            elif query_type == QueryType.ANGELONE_SUPPORT: namespace = "cleaned-angelone-faqs-chunks"
            else: namespace = "none"

            self.rag_system._log_activity("Query Agent", f"Analysis complete. Type: {query_type.value}.")
            return QueryAnalysis(
                query_type=query_type, namespace=namespace,
                confidence=result["confidence"], reasoning=result["reasoning"]
            )
        except Exception as e:
            self.rag_system._log_activity("Query Agent", f"Analysis failed: {e}")
            return QueryAnalysis(query_type=QueryType.IRRELEVANT, namespace="none", confidence=0.0, reasoning="Analysis failed")

class RetrieverAgent:
    def __init__(self, rag_system: MultiAgentRAGSystem):
        self.rag_system = rag_system

    def retrieve(self, query: str, namespaces: str, top_k: int = 10) -> List[RetrievalResult]:
        self.rag_system._log_activity("Retriever Agent", f"Retrieving top {top_k} docs from namespaces: {namespaces}.")
        if namespaces == "none": return []
        try:
            query_embedding = self.rag_system.encoder.encode([query]).tolist()[0]
            namespace_list = [ns.strip() for ns in namespaces.split(",")]
            all_results = []
            for namespace in namespace_list:
                results = self.rag_system.index.query(
                    vector=query_embedding, top_k=top_k // len(namespace_list) + 1,
                    namespace=namespace, include_metadata=True, include_values=False
                )
                for match in results.matches:
                    all_results.append(RetrievalResult(
                        content=match.metadata.get('content', ''), metadata=match.metadata,
                        score=match.score, namespace=namespace
                    ))
            all_results.sort(key=lambda x: x.score, reverse=True)
            self.rag_system._log_activity("Retriever Agent", f"Found {len(all_results)} total documents.")
            return all_results[:top_k]
        except Exception as e:
            self.rag_system._log_activity("Retriever Agent", f"Retrieval failed: {e}")
            return []

class RerankingAgent:
    def __init__(self, rag_system: MultiAgentRAGSystem):
        self.rag_system = rag_system

    def rerank(self, query: str, documents: List[RetrievalResult]) -> List[RetrievalResult]:
        self.rag_system._log_activity("Reranking Agent", f"Reranking {len(documents)} documents with LLM.")
        if not documents: return []
        try:
            doc_texts = [f"{i+1}. {doc.content[:200]}..." for i, doc in enumerate(documents)]
            prompt = f"Query: \"{query}\"\n\nRank the following documents...:"
            response = self.rag_system._call_llm([{"role": "user", "content": prompt}], temperature=0.1)
            try:
                rankings = [int(x.strip()) - 1 for x in response.strip().split(",")]
                reranked = [documents[rank] for rank in rankings if 0 <= rank < len(documents)]
                for i, doc in enumerate(documents):
                    if i not in rankings: reranked.append(doc)
                self.rag_system._log_activity("Reranking Agent", f"Returning top 5 reranked documents.")
                return reranked[:5]
            except:
                self.rag_system._log_activity("Reranking Agent", "Reranking failed to parse, returning original top 5.")
                return documents[:5]
        except Exception as e:
            self.rag_system._log_activity("Reranking Agent", f"Reranking failed: {e}")
            return documents[:5]

class AnsweringAgent:
    def __init__(self, rag_system: MultiAgentRAGSystem):
        self.rag_system = rag_system

    def generate_answer(self, query: str, context_docs: List[RetrievalResult], query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """
        MODIFIED: This method now returns a dictionary with 'answer' and 'contexts'.
        """
        self.rag_system._log_activity("Answering Agent", "Generating final answer from context.")
        if not context_docs:
            return {
                "answer": "I could not find any relevant information to answer your question.",
                "contexts": []
            }

        context_str_list = [doc.content for doc in context_docs]
        context = "\n\n".join([f"Source {i+1}: {doc}" for i, doc in enumerate(context_str_list)])
        history_text = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" for msg in self.rag_system.memory.chat_memory.messages[-4:]])
        system_prompt = "You are an expert assistant for health insurance..." if query_analysis.query_type == QueryType.HEALTH_PLAN else "You are an expert assistant for Angel One support..."
        prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser Question: {query}\n\nAnswer:"

        try:
            response = self.rag_system._call_llm([{"role": "user", "content": prompt}], temperature=0.5)
            self.rag_system.memory.chat_memory.add_user_message(query)
            self.rag_system.memory.chat_memory.add_ai_message(response)
            self.rag_system._log_activity("Answering Agent", "Answer generated and memory updated.")
            return {
                "answer": response.strip(),
                "contexts": context_str_list
            }
        except Exception as e:
            self.rag_system._log_activity("Answering Agent", f"Answer generation failed: {e}")
            return {
                "answer": "An error occurred while generating the answer.",
                "contexts": context_str_list # Return context even if answer fails
            }

def initialize_rag_system(pinecone_api_key: str = None, huggingface_token: str = None) -> Optional[MultiAgentRAGSystem]:
    # Try to get from Streamlit secrets first, then fall back to provided parameters or environment variables
    try:
        if not pinecone_api_key:
            pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv('PINECONE_API_KEY')
        if not huggingface_token:
            huggingface_token = st.secrets.get("HUGGING_FACE_HUB_TOKEN") or os.getenv('HUGGING_FACE_HUB_TOKEN')
    except Exception as e:
        # Fallback to environment variables if secrets are not available (e.g., running locally)
        logger.info("Streamlit secrets not available, falling back to environment variables")
        if not pinecone_api_key:
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not huggingface_token:
            huggingface_token = os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in secrets or environment variables")
    if not huggingface_token:
        raise ValueError("HUGGING_FACE_HUB_TOKEN not found in secrets or environment variables")
    
    try:
        return MultiAgentRAGSystem(
            pinecone_api_key=pinecone_api_key,
            huggingface_token=huggingface_token
        )
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return None