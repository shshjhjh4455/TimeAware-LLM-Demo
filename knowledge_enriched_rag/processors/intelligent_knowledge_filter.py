#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Knowledge Filter for TimeAware Character Role-Playing
Filters relevant knowledge instead of using all cumulative knowledge
"""

import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import os

logger = logging.getLogger(__name__)

class IntelligentKnowledgeFilter:
    """
    Filters character knowledge intelligently based on:
    1. Question relevance
    2. Character-specific traits
    3. Temporal context appropriateness
    4. Response pattern requirements
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Character-specific knowledge priorities
        self.character_traits = {
            'Hermione Granger': {
                'book_knowledge': ['Modern Magical History', 'Rise and Fall', 'Great Wizarding Events'],
                'personality_traits': ['studious', 'helpful', 'logical', 'thorough'],
                'speech_patterns': ['honestly', 'really', 'quite', 'terrible', 'fascinating']
            },
            'Ronald Weasley': {
                'family_knowledge': ['mum', 'dad', 'brothers', 'family', 'home'],
                'personality_traits': ['loyal', 'emotional', 'direct', 'empathetic'],
                'speech_patterns': ['blimey', 'reckon', 'right', 'chuffed', 'brilliant', 'mental']
            },
            'Harry Potter': {
                'dursley_knowledge': ['Dursleys', 'aunt', 'uncle', 'Dudley', 'Privet Drive'],
                'personality_traits': ['reflective', 'comparing', 'grateful', 'resilient'],
                'speech_patterns': ['different', 'better', 'never', 'finally', 'brilliant']
            }
        }

    def filter_relevant_knowledge(
        self,
        question: str,
        character: str,
        character_knowledge: List[Dict],
        is_present: bool,
        max_items: int = 15
    ) -> Dict[str, List[Dict]]:
        """
        Filter knowledge to most relevant items for the specific question

        Args:
            question: The question being asked
            character: Character name
            character_knowledge: All available character knowledge
            is_present: Whether character was present in the scene
            max_items: Maximum knowledge items to include

        Returns:
            Dictionary of filtered knowledge by type
        """

        logger.info(f"Filtering knowledge for {character}: {len(character_knowledge)} total → max {max_items}")

        # Group by type
        knowledge_by_type = {
            'FACT_LEARNED': [],
            'BELIEF_CHANGED': [],
            'RELATIONSHIP_UPDATED': [],
            'EVENT_ABSENCE': []
        }

        for item in character_knowledge:
            item_type = item.get('type', 'UNKNOWN')
            if item_type in knowledge_by_type:
                knowledge_by_type[item_type].append(item)

        # Filter strategy depends on presence
        if not is_present:
            return self._filter_for_absent_character(question, character, knowledge_by_type, max_items)
        else:
            return self._filter_for_present_character(question, character, knowledge_by_type, max_items)

    def _filter_for_absent_character(
        self,
        question: str,
        character: str,
        knowledge_by_type: Dict[str, List[Dict]],
        max_items: int
    ) -> Dict[str, List[Dict]]:
        """Filter knowledge for absent character - focus on contextual awareness"""

        filtered = {}

        # For absent characters, priority is:
        # 1. EVENT_ABSENCE (limited - only general awareness)
        # 2. Character-specific contextual knowledge
        # 3. Relevant background knowledge

        # EVENT_ABSENCE: Use sparingly - only general awareness
        absence_items = knowledge_by_type.get('EVENT_ABSENCE', [])
        filtered['EVENT_ABSENCE'] = self._select_general_absence_items(absence_items, max_items=3)

        # Character-specific knowledge that enables contextual response
        contextual_knowledge = []

        # Get character traits for this character
        traits = self.character_traits.get(character, {})

        for k_type in ['FACT_LEARNED', 'BELIEF_CHANGED', 'RELATIONSHIP_UPDATED']:
            items = knowledge_by_type.get(k_type, [])
            relevant_items = self._select_contextual_items(items, question, character, traits)
            contextual_knowledge.extend(relevant_items)
            filtered[k_type] = relevant_items[:max(1, max_items // 4)]  # Limit each type

        logger.info(f"Absent character filtered: EVENT_ABSENCE={len(filtered['EVENT_ABSENCE'])}, "
                   f"Contextual={sum(len(v) for k,v in filtered.items() if k != 'EVENT_ABSENCE')}")

        return filtered

    def _filter_for_present_character(
        self,
        question: str,
        character: str,
        knowledge_by_type: Dict[str, List[Dict]],
        max_items: int
    ) -> Dict[str, List[Dict]]:
        """Filter knowledge for present character - focus on direct experience"""

        filtered = {}

        # For present characters, priority is:
        # 1. Direct facts about the questioned scene/topic
        # 2. Relevant beliefs and relationships
        # 3. Recent experiences that provide context

        # Get character traits
        traits = self.character_traits.get(character, {})

        for k_type in ['FACT_LEARNED', 'BELIEF_CHANGED', 'RELATIONSHIP_UPDATED']:
            items = knowledge_by_type.get(k_type, [])
            relevant_items = self._select_relevant_items(items, question, character, traits)
            filtered[k_type] = relevant_items[:max(2, max_items // 3)]  # Distribute evenly

        # No EVENT_ABSENCE for present characters
        filtered['EVENT_ABSENCE'] = []

        logger.info(f"Present character filtered: "
                   f"FACT={len(filtered['FACT_LEARNED'])}, "
                   f"BELIEF={len(filtered['BELIEF_CHANGED'])}, "
                   f"RELATIONSHIP={len(filtered['RELATIONSHIP_UPDATED'])}")

        return filtered

    def _select_general_absence_items(self, absence_items: List[Dict], max_items: int = 3) -> List[Dict]:
        """Select only the most general EVENT_ABSENCE items to avoid specific details"""

        # Prefer general statements over specific ones
        general_items = []
        for item in absence_items:
            description = item.get('description', '').lower()

            # Avoid overly specific details
            specific_indicators = ['said', 'announced', 'argued', 'when', 'while', 'after', 'before']
            is_specific = any(indicator in description for indicator in specific_indicators)

            if not is_specific or len(general_items) < max_items // 2:
                general_items.append(item)

        return general_items[:max_items]

    def _select_contextual_items(
        self,
        items: List[Dict],
        question: str,
        character: str,
        traits: Dict
    ) -> List[Dict]:
        """Select knowledge items that provide good contextual background"""

        if not items:
            return []

        scored_items = []
        question_lower = question.lower()

        for item in items:
            description = item.get('description', '').lower()
            score = 0

            # Character-specific trait bonus
            if character == 'Hermione Granger':
                # Bonus for book knowledge and academic traits
                book_terms = traits.get('book_knowledge', [])
                if any(term.lower() in description for term in book_terms):
                    score += 3

            elif character == 'Ronald Weasley':
                # Bonus for family and emotional responses
                family_terms = traits.get('family_knowledge', [])
                if any(term in description for term in family_terms):
                    score += 3

            elif character == 'Harry Potter':
                # Bonus for Dursley-related experiences
                dursley_terms = traits.get('dursley_knowledge', [])
                if any(term.lower() in description for term in dursley_terms):
                    score += 3

            # Topic relevance bonus
            topic_keywords = self._extract_topic_keywords(question_lower)
            if any(keyword in description for keyword in topic_keywords):
                score += 2

            # Recency bonus (more recent knowledge is more relevant)
            chapter_id = item.get('chapter_id', '')
            if 'chapter' in chapter_id.lower():
                try:
                    chapter_num = int(chapter_id.split('chapter')[-1])
                    score += min(chapter_num / 10, 1)  # Recent chapters get slight bonus
                except:
                    pass

            scored_items.append((score, item))

        # Sort by score and return top items
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_items[:5]]  # Top 5 contextual items

    def _select_relevant_items(
        self,
        items: List[Dict],
        question: str,
        character: str,
        traits: Dict
    ) -> List[Dict]:
        """Select most relevant knowledge items for present character"""

        if not items:
            return []

        scored_items = []
        question_lower = question.lower()

        for item in items:
            description = item.get('description', '').lower()
            source_text = item.get('source_text', '').lower()
            score = 0

            # Direct relevance to question topic
            topic_keywords = self._extract_topic_keywords(question_lower)
            relevance_score = sum(1 for keyword in topic_keywords if keyword in description or keyword in source_text)
            score += relevance_score * 2

            # Character voice consistency
            speech_patterns = traits.get('speech_patterns', [])
            if any(pattern in description for pattern in speech_patterns):
                score += 1

            # Source text quality (prefer items with good source quotes)
            if source_text and len(source_text) > 20:
                score += 1

            scored_items.append((score, item))

        # Sort by relevance score
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_items[:8]]  # Top 8 relevant items

    def _extract_topic_keywords(self, question: str) -> List[str]:
        """Extract key topic words from the question"""

        # Common topic indicators
        topic_words = []

        # Character names
        characters = ['harry', 'hermione', 'ron', 'dursley', 'aunt', 'uncle', 'dudley']
        topic_words.extend([word for word in characters if word in question])

        # Location/situation words
        locations = ['home', 'house', 'train', 'platform', 'school', 'hogwarts', 'kitchen']
        topic_words.extend([word for word in locations if word in question])

        # Action words
        actions = ['awakened', 'ordered', 'bacon', 'breakfast', 'birthday', 'letter', 'zoo']
        topic_words.extend([word for word in actions if word in question])

        # Emotional context
        emotions = ['feelings', 'felt', 'experience', 'surprised', 'happy', 'upset']
        topic_words.extend([word for word in emotions if word in question])

        return list(set(topic_words))  # Remove duplicates

    def build_filtered_knowledge_context(
        self,
        filtered_knowledge: Dict[str, List[Dict]],
        character: str,
        is_present: bool
    ) -> str:
        """Build the knowledge context string using only filtered items with character enhancement"""

        context = "My relevant knowledge and experiences:\n"

        # Add character-specific background knowledge hints
        if not is_present and character == 'Hermione Granger':
            context += "\nBackground knowledge from my reading:\n"
            context += "- I've read extensively about famous wizards and magical history, including books about Harry Potter's background and upbringing\n"
            context += "- From my studies, I understand how difficult life can be for wizards raised by Muggles who don't understand magic\n"

        elif not is_present and character == 'Ronald Weasley':
            context += "\nFamily perspective:\n"
            context += "- Coming from a wizarding family, I understand the importance of treating everyone with kindness\n"
            context += "- My mum has taught us that family should support each other, not make life harder\n"

        # Add filtered knowledge by type
        facts = filtered_knowledge.get('FACT_LEARNED', [])
        if facts:
            context += "\nRelevant facts:\n"
            for fact in facts:
                context += f"- {fact['description']}\n"

        beliefs = filtered_knowledge.get('BELIEF_CHANGED', [])
        if beliefs:
            context += "\nRelevant thoughts:\n"
            for belief in beliefs:
                context += f"- {belief['description']}\n"

        relationships = filtered_knowledge.get('RELATIONSHIP_UPDATED', [])
        if relationships:
            context += "\nRelevant relationships:\n"
            for rel in relationships:
                context += f"- {rel['description']}\n"

        # For absent characters, add minimal general awareness
        if not is_present:
            absences = filtered_knowledge.get('EVENT_ABSENCE', [])
            if absences:
                context += "\nGeneral awareness:\n"
                for absence in absences[:2]:  # Only 2 most general items
                    context += f"- {absence['description']}\n"

        return context