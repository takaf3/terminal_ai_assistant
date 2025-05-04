import os
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
from pathlib import Path
from .config_util import load_config


class VectorMemory:
    """
    Vector store-based memory implementation using ChromaDB.
    This provides enhanced memory capabilities for the assistant with semantic search.
    """
    
    def __init__(self, debug: bool = False, config_path: Optional[str] = None, 
                 persist_directory: Optional[str] = None):
        """
        Initialize the vector memory store.
        
        Args:
            debug: Enable debug output
            config_path: Path to configuration file
            persist_directory: Directory to persist the ChromaDB data
        """
        # Load configuration
        self.config = load_config(config_path)
        self.debug = debug
        
        # Get persist directory from config or use default in user's home directory
        if persist_directory is None:
            # Use config value if available
            if self.config.get("memory", {}).get("persist_directory"):
                self.persist_directory = self.config["memory"]["persist_directory"]
            else:
                # Default to ~/.terminal_assistant/vector_memory
                home_dir = os.path.expanduser("~")
                self.persist_directory = os.path.join(home_dir, ".terminal_assistant", "vector_memory")
        else:
            self.persist_directory = persist_directory
            
        if self.debug:
            print(f"[DEBUG] Using vector memory persist directory: {self.persist_directory}")
        
        # Ensure the persist directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Get OpenAI API key for embeddings
        self.api_key = os.environ.get("OPENAI_API_KEY", None) or self.config["llm"]["api_key"]
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or add to config.yaml")
        
        # Initialize embeddings
        self.embedding_model = self.config.get("memory", {}).get("embedding_model", "text-embedding-3-small")
        self.embedding_function = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=self.api_key
        )
        
        # Initialize the persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Create or get collections for different memory categories
        self.collections = {
            "facts": self._get_or_create_vector_store("user_facts"),
            "preferences": self._get_or_create_vector_store("user_preferences"),
            "dates": self._get_or_create_vector_store("important_dates"),
            "world_facts": self._get_or_create_vector_store("world_facts"),
            "general": self._get_or_create_vector_store("general_memory")
        }
        
        if self.debug:
            print(f"[DEBUG] VectorMemory initialized with collections: {list(self.collections.keys())}")
    
    def _get_or_create_vector_store(self, collection_name: str) -> Chroma:
        """
        Get or create a vector store collection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            Chroma vector store instance
        """
        # Get or create the collection in ChromaDB
        collection = self.client.get_or_create_collection(collection_name)
        
        # Create a Chroma vector store with the collection
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_fact(self, fact: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a user fact to the vector store.
        
        Args:
            fact: The fact to store
            metadata: Additional metadata
        
        Returns:
            ID of the stored fact
        """
        if metadata is None:
            metadata = {"type": "user_fact"}
        else:
            metadata["type"] = "user_fact"
        
        doc_id = str(uuid.uuid4())
        document = Document(page_content=fact, metadata=metadata)
        
        self.collections["facts"].add_documents([document], ids=[doc_id])
        
        if self.debug:
            print(f"[DEBUG] Added user fact: {fact} with ID: {doc_id}")
        
        return doc_id
    
    def add_preference(self, category: str, key: str, value: str) -> str:
        """
        Add a user preference to the vector store.
        
        Args:
            category: Preference category
            key: Preference key
            value: Preference value
        
        Returns:
            ID of the stored preference
        """
        content = f"User prefers {value} for {key} in category {category}"
        metadata = {
            "type": "preference",
            "category": category,
            "key": key,
            "value": value
        }
        
        doc_id = str(uuid.uuid4())
        document = Document(page_content=content, metadata=metadata)
        
        self.collections["preferences"].add_documents([document], ids=[doc_id])
        
        if self.debug:
            print(f"[DEBUG] Added preference: {category}/{key}={value} with ID: {doc_id}")
        
        return doc_id
    
    def add_date(self, description: str, date: str) -> str:
        """
        Add an important date to the vector store.
        
        Args:
            description: Description of the date
            date: The date string
        
        Returns:
            ID of the stored date
        """
        content = f"{description}: {date}"
        metadata = {
            "type": "important_date",
            "description": description,
            "date": date
        }
        
        doc_id = str(uuid.uuid4())
        document = Document(page_content=content, metadata=metadata)
        
        self.collections["dates"].add_documents([document], ids=[doc_id])
        
        if self.debug:
            print(f"[DEBUG] Added important date: {description} - {date} with ID: {doc_id}")
        
        return doc_id
    
    def add_world_fact(self, fact: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a world fact to the vector store.
        
        Args:
            fact: The fact to store
            metadata: Additional metadata
        
        Returns:
            ID of the stored fact
        """
        if metadata is None:
            metadata = {"type": "world_fact"}
        else:
            metadata["type"] = "world_fact"
        
        doc_id = str(uuid.uuid4())
        document = Document(page_content=fact, metadata=metadata)
        
        self.collections["world_facts"].add_documents([document], ids=[doc_id])
        
        if self.debug:
            print(f"[DEBUG] Added world fact: {fact} with ID: {doc_id}")
        
        return doc_id
    
    def add_general_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a general memory item to the vector store.
        
        Args:
            content: The content to store
            metadata: Additional metadata
        
        Returns:
            ID of the stored memory item
        """
        if metadata is None:
            metadata = {"type": "general"}
        
        doc_id = str(uuid.uuid4())
        document = Document(page_content=content, metadata=metadata)
        
        self.collections["general"].add_documents([document], ids=[doc_id])
        
        if self.debug:
            print(f"[DEBUG] Added general memory: {content} with ID: {doc_id}")
        
        return doc_id
    
    def search(self, query: str, collection_name: Optional[str] = None, 
               k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        Search for relevant memories using semantic search.
        
        Args:
            query: The search query
            collection_name: Name of the collection to search (if None, searches all collections)
            k: Number of results to return
            filter: Filter for metadata
        
        Returns:
            List of (document, score) tuples
        """
        if collection_name is not None and collection_name in self.collections:
            # Search in a specific collection
            collection = self.collections[collection_name]
            results = collection.similarity_search_with_score(query, k=k, filter=filter)
            
            if self.debug:
                print(f"[DEBUG] Searched {collection_name} for '{query}', found {len(results)} results")
            
            return results
        else:
            # Search across all collections and combine results
            all_results = []
            
            for name, collection in self.collections.items():
                results = collection.similarity_search_with_score(query, k=k, filter=filter)
                all_results.extend(results)
            
            # Sort by score and take top k
            all_results.sort(key=lambda x: x[1])
            final_results = all_results[:k]
            
            if self.debug:
                print(f"[DEBUG] Searched all collections for '{query}', found {len(final_results)} results")
            
            return final_results
    
    def get_all_memories(self, collection_name: Optional[str] = None) -> List[Document]:
        """
        Get all memories from a collection or all collections.
        
        Args:
            collection_name: Name of the collection (if None, gets from all collections)
        
        Returns:
            List of documents
        """
        if collection_name is not None and collection_name in self.collections:
            # Return items from a specific collection by using large k
            collection = self.collections[collection_name]
            results = collection.similarity_search("", k=10000)
            return results
        else:
            # Return items from all collections
            all_results = []
            
            for name, collection in self.collections.items():
                results = collection.similarity_search("", k=10000)
                all_results.extend(results)
            
            return all_results
    
    def delete_memory(self, doc_id: str, collection_name: Optional[str] = None) -> bool:
        """
        Delete a memory item by ID.
        
        Args:
            doc_id: ID of the document to delete
            collection_name: Name of the collection (if None, tries to delete from all)
        
        Returns:
            True if deleted successfully, False otherwise
        """
        if collection_name is not None and collection_name in self.collections:
            # Delete from a specific collection
            collection = self.collections[collection_name]
            collection.delete(ids=[doc_id])
            
            if self.debug:
                print(f"[DEBUG] Deleted memory with ID {doc_id} from {collection_name}")
            
            return True
        else:
            # Try to delete from all collections
            deleted = False
            
            for name, collection in self.collections.items():
                try:
                    collection.delete(ids=[doc_id])
                    deleted = True
                    
                    if self.debug:
                        print(f"[DEBUG] Deleted memory with ID {doc_id} from {name}")
                    
                    break  # Document should only be in one collection
                except:
                    continue
            
            return deleted
    
    def update_memory(self, doc_id: str, new_content: str, collection_name: Optional[str] = None, 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a memory item by ID.
        
        Args:
            doc_id: ID of the document to update
            new_content: New content for the document
            collection_name: Name of the collection
            metadata: New metadata for the document
        
        Returns:
            True if updated successfully, False otherwise
        """
        document = Document(page_content=new_content, metadata=metadata or {})
        
        if collection_name is not None and collection_name in self.collections:
            # Update in a specific collection
            collection = self.collections[collection_name]
            try:
                collection.update_document(document_id=doc_id, document=document)
                
                if self.debug:
                    print(f"[DEBUG] Updated memory with ID {doc_id} in {collection_name}")
                
                return True
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] Error updating memory: {str(e)}")
                return False
        else:
            # Try to update in all collections
            updated = False
            
            for name, collection in self.collections.items():
                try:
                    collection.update_document(document_id=doc_id, document=document)
                    updated = True
                    
                    if self.debug:
                        print(f"[DEBUG] Updated memory with ID {doc_id} in {name}")
                    
                    break  # Document should only be in one collection
                except:
                    continue
            
            return updated 