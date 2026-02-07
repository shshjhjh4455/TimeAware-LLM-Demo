#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector-based Knowledge Retrieval System
Uses semantic similarity between question and character knowledge for optimal selection
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class VectorKnowledgeRetriever:
    """
    Advanced knowledge retrieval using vector similarity instead of rule-based filtering

    Core concept: Embed character knowledge and retrieve most relevant items
    based on semantic similarity with the question
    """

    def __init__(self, database_dir: str = "knowledge_enriched_rag/databases_gpt"):
        self.database_dir = Path(database_dir)
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Initialize knowledge embedding cache
        self.knowledge_embeddings_path = self.database_dir / "knowledge_embeddings.pkl"
        self.knowledge_embeddings = None
        self.knowledge_metadata = None

        logger.info("Initialized VectorKnowledgeRetriever")

        # FIX: Eager loading to prevent race condition in multi-threaded environment
        # Build embeddings once during initialization instead of lazy loading
        self.build_knowledge_embeddings()

    def build_knowledge_embeddings(self, force_rebuild: bool = False):
        """
        Build embeddings for all character knowledge items
        """

        if not force_rebuild and self.knowledge_embeddings_path.exists():
            logger.info("Loading existing knowledge embeddings...")
            # Try npy+json format first (numpy version-compatible)
            npy_path = self.database_dir / "knowledge_embeddings_array.npy"
            meta_json_path = self.database_dir / "knowledge_embeddings_metadata.json"
            if npy_path.exists() and meta_json_path.exists():
                self.knowledge_embeddings = np.load(str(npy_path))
                with open(meta_json_path, 'r') as f:
                    meta_data = json.load(f)
                self.knowledge_metadata = meta_data['metadata']
                logger.info(f"Loaded {len(self.knowledge_metadata)} knowledge embeddings (npy+json)")
                return
            # Fallback to pickle
            with open(self.knowledge_embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.knowledge_embeddings = data['embeddings']
                self.knowledge_metadata = data['metadata']
                logger.info(f"Loaded {len(self.knowledge_metadata)} knowledge embeddings")
                return

        logger.info("Building new knowledge embeddings...")

        # Load character knowledge from JSON
        knowledge_json_path = self.database_dir / "character_knowledge.json"
        with open(knowledge_json_path, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)

        all_knowledge_items = knowledge_data['knowledge_items']

        # Prepare texts for embedding
        knowledge_texts = []
        metadata = []

        for item in all_knowledge_items:
            # Combine description and source text for better representation
            description = item.get('description', '').strip()
            source_text = item.get('source_text', '').strip()

            # Skip items without description
            if not description:
                continue

            # Create comprehensive text for embedding
            if source_text:
                combined_text = f"{description} [Context: {source_text[:500]}]"  # Limit source text
            else:
                combined_text = description

            # Ensure text is not empty and not too long
            combined_text = combined_text.strip()
            if combined_text and len(combined_text) > 10:  # At least 10 characters
                # Limit to reasonable length for embeddings
                if len(combined_text) > 8000:
                    combined_text = combined_text[:8000] + "..."

                knowledge_texts.append(combined_text)
                metadata.append({
                    'chapter_id': item.get('chapter_id', ''),
                    'scene_id': item.get('scene_id', ''),
                    'character': item.get('character', ''),
                    'type': item.get('type', ''),
                    'description': item.get('description', ''),
                    'source_text': item.get('source_text', ''),
                    'scene_title': item.get('scene_title', '')
                })

        # Check if we have valid texts
        if not knowledge_texts:
            logger.warning("No valid knowledge texts found for embedding")
            self.knowledge_embeddings = np.array([])
            self.knowledge_metadata = []
            return

        # Build embeddings in batches
        logger.info(f"Building embeddings for {len(knowledge_texts)} valid knowledge items...")
        embeddings = []

        batch_size = 100
        for i in range(0, len(knowledge_texts), batch_size):
            batch_texts = knowledge_texts[i:i + batch_size]

            # Filter out any remaining empty or invalid texts
            valid_batch_texts = [text for text in batch_texts if text and text.strip() and len(text.strip()) > 5]

            if not valid_batch_texts:
                logger.warning(f"Skipping batch {i//batch_size + 1} - no valid texts")
                continue

            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(knowledge_texts)-1)//batch_size + 1} ({len(valid_batch_texts)} texts)")

            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-large",
                    input=valid_batch_texts,
                    encoding_format="float"
                )

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

            except Exception as e:
                logger.error(f"Embedding batch {i//batch_size + 1} failed: {e}")
                logger.error(f"Sample texts: {[text[:100] for text in valid_batch_texts[:3]]}")
                # Continue with next batch instead of failing completely
                continue

        # Convert to numpy array
        self.knowledge_embeddings = np.array(embeddings)
        self.knowledge_metadata = metadata

        logger.info(f"Created knowledge embeddings: {self.knowledge_embeddings.shape}")

        # Save embeddings cache
        with open(self.knowledge_embeddings_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.knowledge_embeddings,
                'metadata': self.knowledge_metadata,
                'model': 'text-embedding-3-large'
            }, f)

        logger.info(f"Saved knowledge embeddings cache to {self.knowledge_embeddings_path}")

    def retrieve_relevant_knowledge(
        self,
        question: str,
        character: str,
        current_position: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant character knowledge using vector similarity

        Args:
            question: The question being asked
            character: Character name
            current_position: Character's current temporal position (e.g., "Book1-chapter6")
            top_k: Number of most relevant knowledge items to return

        Returns:
            List of most relevant knowledge items
        """

        # FIX: Removed lazy loading check - embeddings now loaded in __init__
        # This prevents race condition in multi-threaded environment

        # Get question embedding
        logger.info(f"Retrieving relevant knowledge for {character} at {current_position}")

        response = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=[question],
            encoding_format="float"
        )
        question_embedding = np.array(response.data[0].embedding)

        # Parse current position for temporal filtering
        current_book, current_chapter = self._parse_book_chapter(current_position)

        # Filter knowledge by character and temporal constraints
        valid_indices = []
        for i, meta in enumerate(self.knowledge_metadata):
            # Must be from the same character
            if meta['character'] != character:
                continue

            # Must be within temporal constraints
            chapter_id = meta['chapter_id']
            if self._is_temporally_valid(chapter_id, current_book, current_chapter):
                valid_indices.append(i)

        if not valid_indices:
            logger.warning(f"No temporally valid knowledge found for {character} at {current_position}")
            return []

        logger.info(f"Found {len(valid_indices)} temporally valid knowledge items for {character}")

        # Calculate similarities for valid knowledge items
        valid_embeddings = self.knowledge_embeddings[valid_indices]
        similarities = np.dot(valid_embeddings, question_embedding) / (
            np.linalg.norm(valid_embeddings, axis=1) * np.linalg.norm(question_embedding)
        )

        # Get top-k most similar
        top_indices = similarities.argsort()[-top_k:][::-1]

        # Prepare results with similarity scores
        results = []
        for idx in top_indices:
            original_idx = valid_indices[idx]
            metadata = self.knowledge_metadata[original_idx]

            # Add similarity score to metadata
            result = metadata.copy()
            result['similarity_score'] = float(similarities[idx])
            results.append(result)

        similarities_list = [f"{r['similarity_score']:.3f}" for r in results[:3]]
        logger.info(f"Retrieved top-{len(results)} knowledge items (similarities: {similarities_list})")

        return results

    def _parse_book_chapter(self, position: str) -> Tuple[int, int]:
        """Parse book-chapter position into integers"""
        try:
            if '-' not in position:
                return (1, 1)

            parts = position.replace('Book', '').split('-')
            book = int(parts[0]) if parts[0].isdigit() else 1

            chapter_part = parts[1] if len(parts) > 1 else 'chapter1'
            chapter = int(chapter_part.replace('chapter', '')) if 'chapter' in chapter_part else 1

            return (book, chapter)
        except:
            return (1, 1)

    def _is_temporally_valid(self, knowledge_chapter_id: str, current_book: int, current_chapter: int) -> bool:
        """Check if knowledge is temporally valid (not from the future)"""
        try:
            if '-' not in knowledge_chapter_id:
                return True

            k_book, k_chapter = self._parse_book_chapter(knowledge_chapter_id)

            # Knowledge is valid if it's from the past or present
            return (k_book < current_book) or (k_book == current_book and k_chapter <= current_chapter)
        except:
            return True

    def build_vector_knowledge_context(
        self,
        relevant_knowledge: List[Dict[str, Any]],
        character: str,
        is_present: bool
    ) -> str:
        """
        Build knowledge context using vector-retrieved items
        Let the character's own voice come through their memories
        """

        if not relevant_knowledge:
            return "Drawing from my memories and experiences:\n(No directly relevant memories come to mind for this question)\n"

        # Separate knowledge by type to avoid duplicates
        absence_items = [k for k in relevant_knowledge if k.get('type') == 'EVENT_ABSENCE']
        direct_knowledge = [k for k in relevant_knowledge if k.get('type') != 'EVENT_ABSENCE']

        context = ""

        # Prioritize direct knowledge (FACT_LEARNED, BELIEF_CHANGED, RELATIONSHIP_UPDATED)
        if direct_knowledge:
            # Group by relevance
            high_relevance = [k for k in direct_knowledge if k.get('similarity_score', 0) > 0.7]
            medium_relevance = [k for k in direct_knowledge if 0.5 < k.get('similarity_score', 0) <= 0.7]

            if high_relevance:
                context += "What I remember most clearly:\n"
                for item in high_relevance[:5]:
                    context += f"- {item['description']}\n"

            if medium_relevance and len(high_relevance) < 5:
                context += "\nOther relevant memories:\n" if context else "What I remember:\n"
                remaining_slots = max(3, 5 - len(high_relevance))
                for item in medium_relevance[:remaining_slots]:
                    context += f"- {item['description']}\n"

        # Add absence items only if character is not present AND we haven't already included them
        if absence_items and not is_present:
            # Only add if we don't have enough direct knowledge, and avoid duplicates
            if len(direct_knowledge) < 3:
                context += "\nWhat I understand from what others have told me:\n" if context else "What I've heard about events I wasn't present for:\n"
                # Limit absence items and ensure no duplicates
                seen_descriptions = set()
                added_count = 0
                for item in absence_items:
                    desc = item['description']
                    # Avoid adding items that are already covered
                    if desc not in seen_descriptions and added_count < 3:
                        context += f"- {desc}\n"
                        seen_descriptions.add(desc)
                        added_count += 1

        return context