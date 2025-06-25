from typing import Literal, Optional, List, Dict, Any
import os
import uuid
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import time
import json
import chromadb
from prompts import analyze_and_structure_prompt, evolve_neighbors_prompt

# --- UTILITY AND SETUP ---

# It's good practice to place your API key setup in a more robust way,
# but for this script, getting it from the environment is fine.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")

# --- CLASS DEFINITIONS ---

class EmbeddingModel:
    """
    Wraps the SentenceTransformer model for generating embeddings.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> List[float]:
        """Encodes text into a vector embedding."""
        return self.model.encode(text).tolist()

class MemoryNote:
    """
    Represents a single memory unit.
    """
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 timestamp: Optional[str] = None,
                 retrieval_count: Optional[int] = 0,
                 links: Optional[List[str]] = None,
                 context: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 tags: Optional[List[str]] = None
                 ):
        
        self.content: str = content
        self.id: str = id or str(uuid.uuid4())
        self.timestamp: str = timestamp or time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # Metadata and usage stats
        self.retrieval_count: int = retrieval_count or 0
        self.links: List[str] = links or []
        
        # LLM-generated structured data
        self.context: str = context or ""
        self.keywords: List[str] = keywords or []
        self.tags: List[str] = tags or []

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the note object to a dictionary for storage."""
        return {
            "content": self.content,
            "timestamp": self.timestamp,
            "retrieval_count": self.retrieval_count,
            "links": json.dumps(self.links), # Store lists/dicts as JSON strings
            "context": self.context,
            "keywords": json.dumps(self.keywords),
            "tags": json.dumps(self.tags)
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any], note_id: str) -> 'MemoryNote':
        """Deserializes a dictionary into a MemoryNote object."""
        return cls(
            id=note_id,
            content=data.get("content", ""),
            timestamp=data.get("timestamp"),
            retrieval_count=data.get("retrieval_count", 0),
            links=json.loads(data.get("links", "[]")),
            context=data.get("context", ""),
            keywords=json.loads(data.get("keywords", "[]")),
            tags=json.loads(data.get("tags", "[]"))
        )

class MemoryManager:
    """
    Handles interactions with the Large Language Model (LLM).
    This class is responsible for generating the structured metadata for notes.
    """
    def __init__(self, model: str = 'gpt-4o-mini', api_key: Optional[str] = OPENAI_API_KEY):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def analyze_and_structure_note(self, content: str) -> Dict[str, Any]:
        """
        Calls the LLM to generate context, keywords, and tags from the content.
        This corresponds to the "Note Construction" phase in the paper.
        """
        prompt = analyze_and_structure_prompt.format(content=content)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            # The response content is a JSON string, so we parse it.
            structured_data = json.loads(response.choices[0].message.content)
            return {
                "context": structured_data.get("context", ""),
                "keywords": structured_data.get("keywords", []),
                "tags": structured_data.get("tags", [])
            }
        except Exception as e:
            print(f"Error analyzing note content: {e}")
            # Return a default structure on failure
            return {"context": "", "keywords": [], "tags": []}

class MemoryEvolver:
    """
    Handles the "Memory Evolution" logic.
    """
    def __init__(self, manager: MemoryManager):
        self.manager = manager

    def evolve_neighbors(self, new_note: MemoryNote, neighbors: List[MemoryNote], collection: chromadb.Collection) -> None:
        """
        Analyzes a new note and its neighbors, then asks the LLM if and how
        the neighbors should be updated (evolved).
        """
        if not neighbors:
            return
        
        # Update links of full neighbor notes
        for neighbor in neighbors:
            if new_note.id not in neighbor.links:
                neighbor.links.append(new_note.id)
                collection.update(ids=[neighbor.id], metadatas=[neighbor.to_dict()])

        # Create a simplified representation of neighbors for the prompt
        neighbors_context = "\n".join([f"- ID: {n.id}, Context: {n.context}" for n in neighbors])

        prompt = evolve_neighbors_prompt.format(
            new_note_id=new_note.id,
            new_note_context=new_note.context,
            new_note_keywords=new_note.keywords,
            neighbors_context=neighbors_context
        )
        
        try:
            response_str = self.manager.client.chat.completions.create(
                model=self.manager.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            ).choices[0].message.content
            
            evolutions = json.loads(response_str).get("evolutions", [])

            for evolution in evolutions:
                note_id = evolution.get("id")
                if not note_id:
                    continue
                
                # Fetch the full metadata to update it
                original_note_results = collection.get(ids=[note_id])
                if not original_note_results['ids']:
                    print(f"Warning: Could not find neighbor {note_id} to evolve.")
                    continue
                original_note_meta = original_note_results['metadatas'][0]
                
                # Update with new values
                updated_meta = original_note_meta.copy()
                updated_meta['context'] = evolution.get('new_context', original_note_meta.get('context'))
                updated_meta['tags'] = json.dumps(evolution.get('new_tags', json.loads(original_note_meta.get('tags', '[]'))))
                
                collection.update(ids=[note_id], metadatas=[updated_meta])
                print(f"✅ Evolved memory note: {note_id}")

        except Exception as e:
            print(f"Error during memory evolution: {e}")


class MemorySystem:
    """
    The main system orchestrating memory operations, now aligned with the A-Mem paper's flow.
    """
    def __init__(self,
                 manager: Optional[MemoryManager] = None,
                 embedding_model: Optional[EmbeddingModel] = None,
                 client: Optional[chromadb.Client] = None):
        self.manager = manager or MemoryManager()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.evolver = MemoryEvolver(self.manager)
        
        # Using a persistent client is better for real applications
        # self.client = client or chromadb.PersistentClient(path="/path/to/db")
        self.client = client or chromadb.Client() # In-memory for simplicity
        
    def _get_note_embedding_text(self, note: MemoryNote) -> str:
        """
        Creates the combined text used for generating the main embedding.
        """
        return " ".join([note.content, note.context] + note.keywords + note.tags)

    def add_memory(self, content: str, collection_name: str, link_num: int = 5) -> str:
        """
        Adds a memory note to the system, following the 3-stage process.
        """
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            
            # --- 1. Note Construction ---
            # Create the basic note
            note = MemoryNote(content=content)
            # Use LLM to generate structured metadata
            structured_data = self.manager.analyze_and_structure_note(content)
            note.context = structured_data["context"]
            note.keywords = structured_data["keywords"]
            note.tags = structured_data["tags"]

            # --- 2. Link Generation ---
            # Find potential neighbors based on the rich context
            if note.context and collection.count() > 0:
                query_text_for_links = self._get_note_embedding_text(note)
                query_embedding = self.embedding_model.encode(query_text_for_links)
                
                related_notes_result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(link_num, collection.count())
                )
                note.links = related_notes_result['ids'][0] if related_notes_result.get('ids') else []

            # --- Add the new note to the database ---
            embedding_text = self._get_note_embedding_text(note)
            collection.add(
                ids=[note.id],
                documents=[note.content],
                metadatas=[note.to_dict()],
                embeddings=[self.embedding_model.encode(embedding_text)]
            )
            print(f"🟢 Added new memory note: {note.id}")

            # --- 3. Memory Evolution ---
            if note.links:
                # Retrieve the full neighbor notes to pass to the evolver
                neighbor_metas = collection.get(ids=note.links)['metadatas']
                neighbors = [MemoryNote.from_dict(meta, id) for id, meta in zip(note.links, neighbor_metas)]
                self.evolver.evolve_neighbors(note, neighbors, collection)

            return note.id
        except Exception as e:
            raise RuntimeError(f"Failed to add memory to collection '{collection_name}': {e}") from e

    def retrieve_memory(self, query: str, collection_name: str, top_k: int = 5) -> List[MemoryNote]:
        """Retrieves memory notes based on a query."""
        try:
            collection = self.client.get_collection(name=collection_name)
            if collection.count() == 0: return []
            query_embedding = self.embedding_model.encode(query)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection.count())
            )
            
            retrieved_notes = []
            if not results['ids'][0]:
                return []
                
            for i in range(len(results['ids'][0])):
                note_id = results['ids'][0][i]
                metadata = results['metadatas'][0][i]
                
                note = MemoryNote.from_dict(metadata, note_id)
                retrieved_notes.append(note)
                
                # Update retrieval count without overwriting the original note
                new_count = note.retrieval_count + 1
                current_metadata = collection.get(ids=[note.id])['metadatas'][0]
                current_metadata['retrieval_count'] = new_count
                collection.update(ids=[note.id], metadatas=[current_metadata])
                
            return retrieved_notes
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve memory from '{collection_name}': {e}") from e

    def delete_memory(self, note_id: str, collection_name: str) -> None:
        """Deletes a memory note from the system."""
        try:
            collection = self.client.get_collection(name=collection_name)
            collection.delete(ids=[note_id])
            print(f"🔴 Deleted memory note: {note_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to delete memory '{note_id}': {e}") from e

    def print_all_notes_in_collection(self, collection_name: str):
        """
        Helper function to fetch and print all notes for debugging and observation.
        """
        print("\n" + "="*25 + " CURRENT MEMORY STATE " + "="*25)
        try:
            collection = self.client.get_collection(name=collection_name)
            notes = collection.get() # Get all items
            if not notes['ids']:
                print("Memory is empty.")
                return

            for i in range(len(notes['ids'])):
                note = MemoryNote.from_dict(notes['metadatas'][i], notes['ids'][i])
                print(f"\n📝 NOTE ID: {note.id}")
                print(f"   - Timestamp: {note.timestamp}")
                content = note.content.strip().replace('\n', ' ')
                print(f"   - Content: {content}")
                print(f"   - Context: {note.context}")
                print(f"   - Keywords: {note.keywords}")
                print(f"   - Tags: {note.tags}")
                print(f"   - Links: {note.links}")
                print(f"   - Retrieval Count: {note.retrieval_count}")
            print("="*72 + "\n")

        except Exception as e:
            print(f"Could not print notes from '{collection_name}': {e}")


# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    if not OPENAI_API_KEY:
        print("Please set your OPENAI_API_KEY environment variable to run the example.")
    else:
        # 1. Initialize the system and define collection name
        memory_system = MemorySystem()
        collection_name = "ai_history_and_concepts"

        # Clean up previous runs if necessary
        try:
            memory_system.client.delete_collection(name=collection_name)
            print(f"🧹 Cleaned up existing collection: '{collection_name}'")
        except Exception as e:
            print(f"Collection didn't exist (which is fine): {e}")

        # 2. Define a more complex scenario: The evolution of AI
        memories_to_add = [
            # Memory 1: The foundation
            "Alan Turing's 1950 paper 'Computing Machinery and Intelligence' introduced the Turing Test, a test of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human.",
            # Memory 2: The shift to learning
            "Arthur Samuel, an American pioneer in the field of computer gaming and artificial intelligence, coined the term 'Machine Learning' in 1959 while at IBM. He developed a checkers-playing program that could learn from its own mistakes.",
            # Memory 3: The deep learning revolution
            "The concept of deep learning, using neural networks with many layers (deep neural networks), gained massive traction in the 2010s with breakthroughs in computer vision and speech recognition, largely thanks to advancements in GPU hardware.",
            # Memory 4: The generative era
            "Generative Adversarial Networks (GANs), introduced by Ian Goodfellow in 2014, consist of two dueling neural networks, a generator and a discriminator, which are trained together to produce highly realistic synthetic data, such as images or text."
        ]

        # 3. Add memories one by one and observe the evolution
        for i, content in enumerate(memories_to_add):
            print(f"\n\n{'='*30} STEP {i+1}: ADDING NEW MEMORY {'='*30}")
            print(f"INPUT CONTENT: \"{content[:80]}...\"")
            
            memory_system.add_memory(content, collection_name=collection_name, link_num=3)
            
            # Print the state of all memories after each addition
            memory_system.print_all_notes_in_collection(collection_name)
            
            # Adding a small delay to make the output more readable
            time.sleep(2)

        # 4. Final retrieval test
        print(f"\n\n{'='*30} FINAL STEP: RETRIEVAL {'='*30}")
        print("--- Retrieving memories related to 'generative AI models' ---")
        retrieved = memory_system.retrieve_memory(
            query="What are some generative AI models?",
            collection_name=collection_name,
            top_k=2
        )

        if retrieved:
            for note in retrieved:
                print(f"\nRetrieved Note ID: {note.id}")
                print(f"  Content: {note.content.strip()}")
                print(f"  Context: {note.context}")
                print(f"  Retrieval Count: {note.retrieval_count}")
        else:
            print("No memories found for the final query.")
