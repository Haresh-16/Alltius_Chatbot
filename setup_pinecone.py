#!/usr/bin/env python3
"""
Setup script for loading data into Pinecone
This script helps you configure and run the Pinecone data loader
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment variables and check requirements"""
    print("=== Pinecone Data Loader Setup ===\n")
    
    # Check if API key is set
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable is not set!")
        print("\nTo set your Pinecone API key:")
        print("Windows: set PINECONE_API_KEY=your_api_key_here")
        print("Linux/Mac: export PINECONE_API_KEY=your_api_key_here")
        print("\nOr create a .env file with:")
        print("PINECONE_API_KEY=your_api_key_here")
        return False
    else:
        print(f"✅ PINECONE_API_KEY is set (ends with: ...{api_key[-4:]})")
    
    # Check if JSON directory exists
    json_dir = Path("jsons_from_sources")
    if not json_dir.exists():
        print(f"❌ JSON directory not found: {json_dir}")
        return False
    else:
        json_files = list(json_dir.glob("*.json"))
        print(f"✅ Found {len(json_files)} JSON files in {json_dir}")
        for file in json_files:
            print(f"   - {file.name}")
    
    return True

def show_namespace_mapping():
    """Show how JSON files will be mapped to namespaces"""
    print("\n=== Namespace Mapping ===")
    json_dir = Path("jsons_from_sources")
    json_files = list(json_dir.glob("*.json"))
    
    print("JSON files will be loaded into the following namespaces:")
    for file in json_files:
        namespace = file.stem.replace('_', '-').lower()
        print(f"   {file.name} → namespace: '{namespace}'")

def run_loader():
    """Run the Pinecone data loader"""
    print("\n=== Running Pinecone Data Loader ===")
    
    try:
        from pinecone_loader import PineconeDataLoader
        
        api_key = os.getenv('PINECONE_API_KEY')
        loader = PineconeDataLoader(api_key, "rag-chatbot-index")
        
        # Load all JSON files
        loader.load_all_jsons_from_directory("jsons_from_sources")
        
        # Show final stats
        print("\n=== Final Index Statistics ===")
        loader.get_index_stats()
        
        print("\n✅ Data loading completed successfully!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required packages are installed:")
        print("pip install sentence-transformers pinecone-client")
    except Exception as e:
        print(f"❌ Error during data loading: {e}")

def main():
    """Main setup function"""
    print("Pinecone RAG Chatbot Data Loader")
    print("=" * 40)
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Setup failed. Please fix the issues above and try again.")
        return
    
    # Show namespace mapping
    show_namespace_mapping()
    
    # Ask user if they want to proceed
    print("\n" + "=" * 40)
    response = input("Do you want to proceed with loading data into Pinecone? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        run_loader()
    else:
        print("Setup completed. Run this script again when you're ready to load data.")

if __name__ == "__main__":
    main() 