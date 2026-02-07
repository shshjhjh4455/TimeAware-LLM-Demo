#!/usr/bin/env python3
"""
Memory Retriever - Vector Search for Character Knowledge
Implements 'Dual Timeline Knowledge Retrieval' from paper (Section 2.3)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    Vector-based retrieval of character memories with temporal filtering

    Key Features:
        1. Temporal Filtering: Only retrieves memories before character_period
        2. Semantic Search: Finds memories most relevant to the question
        3. Character-specific: Filters by character name
        4. Efficient: Uses numpy vectorized operations
    """

    def __init__(self, db_dir: str = "knowledge_enriched_rag/databases"):
        """
        Load Memory DB from disk

        Args:
            db_dir: Directory containing memory_vectors.npy and memory_metadata.json
        """
        self.db_dir = Path(db_dir)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Load vectors and metadata
        self._load_database()

    def _load_database(self):
        """Load vector embeddings and metadata"""

        # Load vectors
        vectors_file = self.db_dir / "memory_vectors.npy"
        if not vectors_file.exists():
            raise FileNotFoundError(f"Memory vectors not found: {vectors_file}")

        self.vectors = np.load(vectors_file)
        logger.info(f"✅ Loaded memory vectors: {self.vectors.shape}")

        # Load metadata
        metadata_file = self.db_dir / "memory_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Memory metadata not found: {metadata_file}")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.memories = metadata['memories']
        self.embedding_model = metadata['embedding_model']

        logger.info(f"✅ Loaded {len(self.memories):,} memories")

        # Build index for fast lookup
        self.memory_by_id = {mem['id']: mem for mem in self.memories}

    def search_memories(
        self,
        question: str,
        character: str,
        before_chapter: str,
        top_k: int = 10,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories with temporal and character filtering

        Args:
            question: The question being asked
            character: Character name (e.g., "Harry Potter")
            before_chapter: Character's current position (e.g., "Book1-chapter12")
                           Only memories BEFORE this point are retrieved
            top_k: Number of top memories to return
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of memory dictionaries with similarity scores, sorted by relevance
        """

        # Step 1: Embed question
        query_embedding = self._embed_question(question)

        # Step 2: Temporal + Character filtering
        valid_indices, valid_memories = self._filter_memories(
            character=character,
            before_chapter=before_chapter
        )

        if not valid_indices:
            logger.warning(f"No memories found for {character} before {before_chapter}")
            return []

        # Step 3: Calculate similarities (vectorized)
        valid_vectors = self.vectors[valid_indices]
        similarities = self._cosine_similarity(query_embedding, valid_vectors)

        # Step 4: Filter by threshold and get top-k
        valid_mask = similarities >= min_similarity
        filtered_similarities = similarities[valid_mask]
        filtered_indices = np.array(valid_indices)[valid_mask]
        filtered_memories = [valid_memories[i] for i in range(len(valid_memories)) if valid_mask[i]]

        # Sort by similarity (descending)
        sorted_idx = np.argsort(filtered_similarities)[::-1][:top_k]

        # Build results
        results = []
        for idx in sorted_idx:
            memory = filtered_memories[idx]
            memory_with_score = memory.copy()
            memory_with_score['similarity_score'] = float(filtered_similarities[idx])
            results.append(memory_with_score)

        logger.info(f"🔍 Found {len(results)} relevant memories for {character}")
        if results:
            logger.info(f"   Top similarity: {results[0]['similarity_score']:.3f}")

        return results

    def _embed_question(self, question: str) -> np.ndarray:
        """Generate embedding for question using OpenAI API"""

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[question],
            encoding_format="float"
        )

        return np.array(response.data[0].embedding, dtype=np.float32)

    def _filter_memories(
        self,
        character: str,
        before_chapter: str
    ) -> Tuple[List[int], List[Dict]]:
        """
        Filter memories by character and temporal constraint

        Returns:
            (valid_indices, valid_memories)
        """

        # Parse target chapter for comparison
        target_book, target_chapter = self._parse_chapter_id(before_chapter)

        valid_indices = []
        valid_memories = []

        for mem in self.memories:
            # Filter by character
            if mem['character'] != character:
                continue

            # Filter by temporal constraint (before or at current position)
            if mem['book_num'] < target_book:
                # Earlier book - always valid
                valid_indices.append(mem['id'])
                valid_memories.append(mem)
            elif mem['book_num'] == target_book and mem['chapter_num'] <= target_chapter:
                # Same book, earlier or same chapter
                valid_indices.append(mem['id'])
                valid_memories.append(mem)
            # else: Future memory - skip

        return valid_indices, valid_memories

    def _parse_chapter_id(self, chapter_id: str) -> Tuple[int, int]:
        """
        Parse chapter_id into (book_num, chapter_num)

        Example: "Book1-chapter4" → (1, 4)
        """
        try:
            parts = chapter_id.replace('Book', '').split('-chapter')
            book_num = int(parts[0])
            chapter_num = int(parts[1])
            return book_num, chapter_num
        except:
            logger.warning(f"Failed to parse chapter_id: {chapter_id}")
            return 0, 0

    def _cosine_similarity(
        self,
        query_vec: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and multiple vectors

        Args:
            query_vec: (embedding_dim,)
            vectors: (n_vectors, embedding_dim)

        Returns:
            similarities: (n_vectors,)
        """
        # Normalize
        query_norm = query_vec / np.linalg.norm(query_vec)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        # Dot product = cosine similarity (for normalized vectors)
        similarities = np.dot(vectors_norm, query_norm)

        return similarities

    def get_memory_by_id(self, memory_id: int) -> Dict[str, Any]:
        """Retrieve specific memory by ID"""
        return self.memory_by_id.get(memory_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""

        char_counts = {}
        type_counts = {}

        for mem in self.memories:
            # Character distribution
            char = mem['character']
            char_counts[char] = char_counts.get(char, 0) + 1

            # Type distribution
            mem_type = mem['type']
            type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

        return {
            'total_memories': len(self.memories),
            'characters': char_counts,
            'types': type_counts,
            'embedding_dim': self.vectors.shape[1],
            'database_size_mb': self.vectors.nbytes / 1024 / 1024
        }


def test_memory_retriever():
    """Test Memory Retriever with sample query"""

    logger.info("🧪 Testing Memory Retriever...")

    retriever = MemoryRetriever()

    # Show statistics
    stats = retriever.get_statistics()
    logger.info(f"\n📊 Database Statistics:")
    logger.info(f"   Total memories: {stats['total_memories']:,}")
    logger.info(f"   Characters: {stats['characters']}")
    logger.info(f"   Memory types: {stats['types']}")

    # Test query
    test_cases = [
        {
            'question': "What do you know about Voldemort?",
            'character': "Harry Potter",
            'before_chapter': "Book1-chapter12",
            'top_k': 5
        },
        {
            'question': "How do you feel about Hermione?",
            'character': "Ronald Weasley",
            'before_chapter': "Book1-chapter17",
            'top_k': 3
        }
    ]

    for i, test in enumerate(test_cases, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Test Case {i}:")
        logger.info(f"   Question: {test['question']}")
        logger.info(f"   Character: {test['character']}")
        logger.info(f"   Before: {test['before_chapter']}")

        results = retriever.search_memories(
            question=test['question'],
            character=test['character'],
            before_chapter=test['before_chapter'],
            top_k=test['top_k']
        )

        logger.info(f"\n🔍 Top {len(results)} Relevant Memories:")
        for j, mem in enumerate(results, 1):
            logger.info(f"\n   {j}. [{mem['type']}] ({mem['similarity_score']:.3f})")
            logger.info(f"      Chapter: {mem['chapter_id']} / Scene: {mem['scene_id']}")
            logger.info(f"      Description: {mem['description'][:150]}...")


if __name__ == "__main__":
    test_memory_retriever()
