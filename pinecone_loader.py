import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import uuid
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeDataLoader:
    def __init__(self, api_key: str, index_name: str = "rag-chatbot-index"):
        """
        Initialize Pinecone data loader
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
        """
        self.api_key = api_key
        self.index_name = index_name
        self.pc = Pinecone(api_key=api_key)
        
        # Initialize Hugging Face sentence transformer
        logger.info("Loading Hugging Face sentence transformer model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient model
        self.embedding_dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize or connect to index
        self._setup_index()
    
    def _setup_index(self):
        """Setup Pinecone index"""
        try:
            # Check if index exists
            if self.index_name not in self.pc.list_indexes().names():
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Error setting up Pinecone index: {str(e)}")
            raise
    
    def load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} chunks from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            return []
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using Hugging Face sentence transformer"""
        try:
            logger.info(f"Creating embeddings for {len(texts)} texts...")
            embeddings = self.encoder.encode(texts, show_progress_bar=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return []
    
    def prepare_vectors(self, data: List[Dict[str, Any]], namespace: str) -> List[Dict[str, Any]]:
        """Prepare vectors for Pinecone upload"""
        vectors = []
        texts = [item['content'] for item in data]
        
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        if not embeddings:
            logger.error("Failed to create embeddings")
            return []
        
        # Prepare vectors with metadata
        for i, (item, embedding) in enumerate(zip(data, embeddings)):
            vector_id = f"{namespace}_{uuid.uuid4().hex[:8]}_{i}"
            
            # Prepare metadata
            metadata = item.get('metadata', {})
            metadata['content'] = item['content'][:1000]  # Truncate content for metadata
            metadata['namespace'] = namespace
            metadata['full_content_length'] = len(item['content'])
            
            vectors.append({
                'id': vector_id,
                'values': embedding,
                'metadata': metadata
            })
        
        return vectors
    
    def upload_to_pinecone(self, vectors: List[Dict[str, Any]], namespace: str, batch_size: int = 100):
        """Upload vectors to Pinecone in batches"""
        try:
            logger.info(f"Uploading {len(vectors)} vectors to namespace '{namespace}'...")
            
            # Upload in batches
            for i in tqdm(range(0, len(vectors), batch_size), desc=f"Uploading to {namespace}"):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
            
            logger.info(f"Successfully uploaded {len(vectors)} vectors to namespace '{namespace}'")
            
        except Exception as e:
            logger.error(f"Error uploading to Pinecone: {str(e)}")
            raise
    
    def load_json_to_namespace(self, json_file_path: str, namespace: str):
        """Load a single JSON file to a specific namespace"""
        logger.info(f"Processing {json_file_path} for namespace '{namespace}'...")
        
        # Load data
        data = self.load_json_file(json_file_path)
        if not data:
            logger.warning(f"No data loaded from {json_file_path}")
            return
        
        # Prepare vectors
        vectors = self.prepare_vectors(data, namespace)
        if not vectors:
            logger.warning(f"No vectors prepared from {json_file_path}")
            return
        
        # Upload to Pinecone
        self.upload_to_pinecone(vectors, namespace)
        
        logger.info(f"Completed processing {json_file_path} -> namespace '{namespace}'")
    
    def load_all_jsons_from_directory(self, directory_path: str):
        """Load all JSON files from directory to separate namespaces"""
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory_path}")
            return
        
        json_files = list(directory.glob("*.json"))
        if not json_files:
            logger.warning(f"No JSON files found in {directory_path}")
            return
        
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        # Process each JSON file
        for json_file in json_files:
            # Create namespace from filename (remove .json extension)
            namespace = json_file.stem.replace('_', '-').lower()
            
            try:
                self.load_json_to_namespace(str(json_file), namespace)
            except Exception as e:
                logger.error(f"Failed to process {json_file}: {str(e)}")
                continue
        
        logger.info("Completed loading all JSON files to Pinecone")
    
    def get_index_stats(self):
        """Get statistics about the Pinecone index"""
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Index statistics: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            return None

def main():
    """Main function to load data into Pinecone"""
    # Configuration
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    if not PINECONE_API_KEY:
        logger.error("Please set PINECONE_API_KEY environment variable")
        return
    
    INDEX_NAME = "rag-chatbot-index"
    JSON_DIRECTORY = "jsons_from_sources"
    
    try:
        # Initialize loader
        loader = PineconeDataLoader(PINECONE_API_KEY, INDEX_NAME)
        
        # Load all JSON files
        loader.load_all_jsons_from_directory(JSON_DIRECTORY)
        
        # Get final stats
        loader.get_index_stats()
        
        logger.info("Data loading completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 