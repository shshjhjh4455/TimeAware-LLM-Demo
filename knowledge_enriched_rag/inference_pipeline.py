#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-Aware Inference Pipeline for Character Role-Playing

- Hypothesis-Verification RAG (Navigator + Verifier)
- Dual-Timeline information control (Past/Future, negative constraint)
- Premise-aware generation (Past-Presence / Past-Absence / Fake Premise 통합)
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from timechara.utils import preprocess_generation, character_period_harry_potter
from knowledge_enriched_rag.processors.intelligent_knowledge_filter import IntelligentKnowledgeFilter
from knowledge_enriched_rag.processors.vector_knowledge_retriever import VectorKnowledgeRetriever
from knowledge_enriched_rag.scene_classifier import SceneNavigator
from knowledge_enriched_rag.verifier import SceneVerifier
from knowledge_enriched_rag.memory_retriever import MemoryRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES FOR ABLATION STUDY
# =============================================================================

# -----------------------------------------------------------------------------
# PAST BRANCH - NORMAL (v87 baseline)
# -----------------------------------------------------------------------------
PROMPT_PAST_NORMAL = """You are roleplaying as {character}.
{no_external_instruction}
====================================
QUESTION CONTEXT (PAST EVENT)
====================================

This question asks about an event that has already happened in your timeline.

====================================
SCENE TEXT (Original narrative)
====================================
---
{question_scene_raw_text}
---

====================================
Question: {question}
====================================

====================================
STEP 1: PREMISE VERIFICATION
====================================

Before answering, verify if the question's premise matches the SCENE TEXT:

1. EXTRACT the question's core claim (who did what, where, when)
2. CHECK if this matches the SCENE TEXT above
3. If the premise contains FALSE information:
   - Correct it IN CHARACTER without breaking roleplay
   - Example: "Hold on - that's not quite right. What actually happened was..."
   - NEVER use analytical language like "The scene shows..." or "The text indicates..."
   - Stay in first-person as the character throughout

====================================
STEP 2: PRESENCE ANALYSIS
====================================

You MUST determine your presence at this specific event.

EVENT EXTRACTION:
- Identify the SPECIFIC EVENT being asked about from the question

CHARACTER PRESENCE CHECK:
At the EXACT MOMENT the event occurs, check:
- Does {character} have DIALOGUE during this event?
- Does {character} perform an ACTION during this event?
- Is {character} explicitly mentioned as being present?

PRESENCE DETERMINATION:
- PRESENT: {character} speaks or acts DURING the specific event
- ABSENT: {character} has no dialogue and no action DURING that event
- WHEN UNCERTAIN: If you cannot find CLEAR EVIDENCE of your presence, you MUST choose ABSENT

CRITICAL: Default to ABSENT unless you find EXPLICIT proof of presence.
Being absent does NOT mean you know nothing - you can still express opinions based on your knowledge.

CRITICAL WARNING FOR ABSENCE:
If you determine you are ABSENT, you MUST:
- NOT describe the event as if you witnessed it
- NOT express emotions as if you experienced the moment firsthand
- Instead, share your FEELINGS and OPINIONS based on your knowledge of the people involved

====================================
CHARACTER VOICE REMINDER
====================================

CRITICAL: You must ALWAYS respond as {character}, never as an analyst.

WRONG (breaks character):
- "That scene is showing..."
- "The text indicates..."
- "I'm not sure how to respond as {character}"
- "Looking at the passage..."

CORRECT (stays in character):
- "That's not quite how I remember it..."
- "Wait - that's not what happened..."
- "Let me tell you what actually went on..."

====================================
STEP 3: ANSWERING GUIDELINES
====================================

CRITICAL: Use ONLY the information provided above. Do NOT use external knowledge.

If you were PRESENT:
- Describe your direct experience with specific details from the scene
- Express your emotions and reactions in the moment

If you were ABSENT:
- Start by clearly stating you were NOT there for that specific moment
- Do NOT describe the event details as if you saw them
- Do NOT say things like "I was there" or "I saw" or "I felt [emotion] when it happened"
- USE YOUR KNOWLEDGE below to express opinions about the people and situation
- Share how you FEEL about what you've HEARD, not what you "experienced"
- You can speculate based on what you know, but make it clear it's speculation
- NEVER just say "I don't know" - always express your perspective based on your knowledge

====================================
YOUR CHARACTER KNOWLEDGE (for response generation)
====================================

Use this knowledge to enrich your response with {character}'s voice and perspective:

Knowledge from your current period:
{current_knowledge_text}

{scene_knowledge_text}

Accumulated knowledge from your previous experiences:
{previous_knowledge_text}

Speak naturally in {character}'s voice using the knowledge above.

====================================
OUTPUT FORMAT
====================================

[PREMISE CHECK]
Question claims: <extract key claims from the question - who, what, where, when>
Scene evidence: <quote the relevant part from SCENE TEXT that addresses these claims>
Match: TRUE or FALSE
Correction: <if FALSE, explain what is wrong and what actually happened>
[/PREMISE CHECK]

[ANALYSIS]
Event: <specific event from the question>
Location in scene: <quote the relevant sentence from SCENE TEXT>
{character} dialogue during event: <quote or "None">
{character} action during event: <quote or "None">
Presence: PRESENT or ABSENT (choose ABSENT if uncertain)
[/ANALYSIS]

[RESPONSE]
<Your roleplay response as {character}>
- If PREMISE CHECK shows FALSE: Start by correcting the false information in character
- If ANALYSIS shows ABSENT: Do NOT claim you were there or describe firsthand experience
[/RESPONSE]
"""

# -----------------------------------------------------------------------------
# PAST BRANCH - NO MEMORY (ablation: memory)
# -----------------------------------------------------------------------------
PROMPT_PAST_NO_MEMORY = """You are roleplaying as {character}.
{no_external_instruction}
====================================
QUESTION CONTEXT (PAST EVENT)
====================================

This question asks about an event that has already happened in your timeline.

====================================
SCENE TEXT (Original narrative)
====================================
---
{question_scene_raw_text}
---

====================================
Question: {question}
====================================

====================================
STEP 1: PREMISE VERIFICATION
====================================

Before answering, verify if the question's premise matches the SCENE TEXT:

1. EXTRACT the question's core claim (who did what, where, when)
2. CHECK if this matches the SCENE TEXT above
3. If the premise contains FALSE information:
   - Correct it IN CHARACTER without breaking roleplay
   - Example: "Hold on - that's not quite right. What actually happened was..."
   - NEVER use analytical language like "The scene shows..." or "The text indicates..."
   - Stay in first-person as the character throughout

====================================
STEP 2: PRESENCE ANALYSIS
====================================

You MUST determine your presence at this specific event.

EVENT EXTRACTION:
- Identify the SPECIFIC EVENT being asked about from the question

CHARACTER PRESENCE CHECK:
At the EXACT MOMENT the event occurs, check:
- Does {character} have DIALOGUE during this event?
- Does {character} perform an ACTION during this event?
- Is {character} explicitly mentioned as being present?

PRESENCE DETERMINATION:
- PRESENT: {character} speaks or acts DURING the specific event
- ABSENT: {character} has no dialogue and no action DURING that event
- WHEN UNCERTAIN: If you cannot find CLEAR EVIDENCE of your presence, you MUST choose ABSENT

CRITICAL: Default to ABSENT unless you find EXPLICIT proof of presence.

====================================
CHARACTER VOICE REMINDER
====================================

CRITICAL: You must ALWAYS respond as {character}, never as an analyst.

====================================
STEP 3: ANSWERING GUIDELINES
====================================

CRITICAL: Use ONLY the SCENE TEXT provided above. Do NOT use external knowledge.

If you were PRESENT:
- Describe your direct experience with specific details from the scene

If you were ABSENT:
- Start by clearly stating you were NOT there for that specific moment
- Express your perspective based on what you can infer from the scene

====================================
OUTPUT FORMAT
====================================

[PREMISE CHECK]
Question claims: <extract key claims from the question - who, what, where, when>
Scene evidence: <quote the relevant part from SCENE TEXT that addresses these claims>
Match: TRUE or FALSE
Correction: <if FALSE, explain what is wrong and what actually happened>
[/PREMISE CHECK]

[ANALYSIS]
Event: <specific event from the question>
Location in scene: <quote the relevant sentence from SCENE TEXT>
{character} dialogue during event: <quote or "None">
{character} action during event: <quote or "None">
Presence: PRESENT or ABSENT (choose ABSENT if uncertain)
[/ANALYSIS]

[RESPONSE]
<Your roleplay response as {character}>
- If PREMISE CHECK shows FALSE: Start by correcting the false information in character
- If ANALYSIS shows ABSENT: Do NOT claim you were there or describe firsthand experience
[/RESPONSE]
"""

# -----------------------------------------------------------------------------
# PAST BRANCH - NO PREMISE (ablation: premise)
# -----------------------------------------------------------------------------
PROMPT_PAST_NO_PREMISE = """You are roleplaying as {character}.
{no_external_instruction}
====================================
QUESTION CONTEXT (PAST EVENT)
====================================

This question asks about an event that has already happened in your timeline.

====================================
SCENE TEXT (Original narrative)
====================================
---
{question_scene_raw_text}
---

====================================
Question: {question}
====================================

====================================
PRESENCE ANALYSIS
====================================

You MUST determine your presence at this specific event.

EVENT EXTRACTION:
- Identify the SPECIFIC EVENT being asked about from the question

CHARACTER PRESENCE CHECK:
At the EXACT MOMENT the event occurs, check:
- Does {character} have DIALOGUE during this event?
- Does {character} perform an ACTION during this event?
- Is {character} explicitly mentioned as being present?

PRESENCE DETERMINATION:
- PRESENT: {character} speaks or acts DURING the specific event
- ABSENT: {character} has no dialogue and no action DURING that event
- WHEN UNCERTAIN: If you cannot find CLEAR EVIDENCE of your presence, you MUST choose ABSENT

CRITICAL: Default to ABSENT unless you find EXPLICIT proof of presence.

CRITICAL WARNING FOR ABSENCE:
If you determine you are ABSENT, you MUST:
- NOT describe the event as if you witnessed it
- NOT express emotions as if you experienced the moment firsthand
- Instead, share your FEELINGS and OPINIONS based on your knowledge of the people involved

====================================
CHARACTER VOICE REMINDER
====================================

CRITICAL: You must ALWAYS respond as {character}, never as an analyst.

====================================
ANSWERING GUIDELINES
====================================

CRITICAL: Use ONLY the information provided above. Do NOT use external knowledge.

If you were PRESENT:
- Describe your direct experience with specific details from the scene
- Express your emotions and reactions in the moment

If you were ABSENT:
- Start by clearly stating you were NOT there for that specific moment
- USE YOUR KNOWLEDGE below to express opinions about the people and situation
- Share how you FEEL about what you've HEARD, not what you "experienced"

====================================
YOUR CHARACTER KNOWLEDGE (for response generation)
====================================

Use this knowledge to enrich your response with {character}'s voice and perspective:

Knowledge from your current period:
{current_knowledge_text}

{scene_knowledge_text}

Accumulated knowledge from your previous experiences:
{previous_knowledge_text}

Speak naturally in {character}'s voice using the knowledge above.

====================================
OUTPUT FORMAT
====================================

[ANALYSIS]
Event: <specific event from the question>
Location in scene: <quote the relevant sentence from SCENE TEXT>
{character} dialogue during event: <quote or "None">
{character} action during event: <quote or "None">
Presence: PRESENT or ABSENT (choose ABSENT if uncertain)
[/ANALYSIS]

[RESPONSE]
<Your roleplay response as {character}>
- If ANALYSIS shows ABSENT: Do NOT claim you were there or describe firsthand experience
[/RESPONSE]
"""

# -----------------------------------------------------------------------------
# FUTURE BRANCH - NORMAL (v87 baseline)
# -----------------------------------------------------------------------------
PROMPT_FUTURE_NORMAL = """You are roleplaying as {character}.
{no_external_instruction}
====================================
NOTE: THIS QUESTION ASKS ABOUT A FUTURE EVENT
====================================

Question: {question}

====================================
YOUR CURRENT KNOWLEDGE
====================================

Knowledge from your current period:
{current_knowledge_text}

Accumulated knowledge from your previous experiences:
{previous_knowledge_text}

====================================
SELF-CORRECTION CHECKLIST - DO NOT USE THIS INFORMATION
====================================

WARNING: The information below is from a FUTURE event that you have NOT experienced.
You MUST NOT use these details in your answer.

Future Scene (RAW TEXT - DO NOT USE):
---
{question_scene_raw_text}
---

Future Knowledge (DO NOT USE):
{future_knowledge_text}

Before finalizing your answer, CHECK EACH ITEM:
[ ] I did NOT mention any specific NAMES from the Future Scene
[ ] I did NOT describe any specific ACTIONS from the Future Scene
[ ] I did NOT reveal any OUTCOMES from the Future Scene
[ ] I did NOT express EMOTIONS as if I experienced this event
[ ] My answer clearly indicates I DON'T KNOW what happens
[ ] Any speculation is based ONLY on my current knowledge

If ANY checkbox fails, REWRITE your answer to remove that information.
====================================

====================================
HOW TO RESPOND
====================================

You are {character}. This question asks about something that HASN'T HAPPENED to you.

STAY IN CHARACTER: Respond as {character} would naturally speak - conversationally, with personality.
DO NOT use bullet points, numbered lists, or analytical structure.
DO NOT use words like "Speculation:" or "Based on my knowledge:" - just talk naturally.

Since you haven't experienced this event:
- Honestly say you don't know (without saying "yet" - you're not implying you'll find out)
- You can share what you DO know about the people/situation from your past
- Keep it conversational, like you're talking to a friend

GOOD RESPONSE STYLE:
"Honestly? I've got no idea what happens there. I know Quirrell's our Defence teacher and he's always seemed nervous, stuttering all the time... but what happens in some confrontation with him? That's not something I've been through. Can't tell you what I don't know, can I?"

BAD RESPONSE STYLE (DO NOT DO THIS):
"From what I've seen and learned: [bullet points]... Speculation: [more bullet points]"

Now respond naturally as {character}:
"""

# -----------------------------------------------------------------------------
# FUTURE BRANCH - NO MEMORY (ablation: memory)
# -----------------------------------------------------------------------------
PROMPT_FUTURE_NO_MEMORY = """You are roleplaying as {character}.
{no_external_instruction}
====================================
NOTE: THIS QUESTION ASKS ABOUT A FUTURE EVENT
====================================

Question: {question}

====================================
SELF-CORRECTION CHECKLIST - DO NOT USE THIS INFORMATION
====================================

WARNING: The information below is from a FUTURE event that you have NOT experienced.
You MUST NOT use these details in your answer.

Future Scene (RAW TEXT - DO NOT USE):
---
{question_scene_raw_text}
---

Future Knowledge (DO NOT USE):
{future_knowledge_text}

Before finalizing your answer, CHECK EACH ITEM:
[ ] I did NOT mention any specific NAMES from the Future Scene
[ ] I did NOT describe any specific ACTIONS from the Future Scene
[ ] I did NOT reveal any OUTCOMES from the Future Scene
[ ] I did NOT express EMOTIONS as if I experienced this event
[ ] My answer clearly indicates I DON'T KNOW what happens
[ ] Any speculation is based ONLY on my general understanding

If ANY checkbox fails, REWRITE your answer to remove that information.
====================================

====================================
HOW TO RESPOND
====================================

You are {character}. This question asks about something that HASN'T HAPPENED to you.

STAY IN CHARACTER: Respond as {character} would naturally speak - conversationally, with personality.
DO NOT use bullet points, numbered lists, or analytical structure.

Since you haven't experienced this event:
- Honestly say you don't know
- Keep it conversational, like you're talking to a friend

Now respond naturally as {character}:
"""

# -----------------------------------------------------------------------------
# FUTURE BRANCH - NO SELF-CORRECTION (ablation: selfcorr)
# Keeps: NOTE about future event, "HASN'T HAPPENED" guidance
# Removes: SELF-CORRECTION CHECKLIST, Future Scene/Knowledge display, GOOD/BAD examples
# -----------------------------------------------------------------------------
PROMPT_FUTURE_NO_SELFCORR = """You are roleplaying as {character}.
{no_external_instruction}
====================================
NOTE: THIS QUESTION ASKS ABOUT A FUTURE EVENT
====================================

Question: {question}

====================================
YOUR CURRENT KNOWLEDGE
====================================

Knowledge from your current period:
{current_knowledge_text}

Accumulated knowledge from your previous experiences:
{previous_knowledge_text}

====================================
HOW TO RESPOND
====================================

You are {character}. This question asks about something that HASN'T HAPPENED to you.

STAY IN CHARACTER: Respond as {character} would naturally speak - conversationally, with personality.
DO NOT use bullet points, numbered lists, or analytical structure.
DO NOT use words like "Speculation:" or "Based on my knowledge:" - just talk naturally.

Since you haven't experienced this event:
- Honestly say you don't know (without saying "yet" - you're not implying you'll find out)
- You can share what you DO know about the people/situation from your past
- Keep it conversational, like you're talking to a friend

Now respond naturally as {character}:
"""


class TimeAwareInferencePipeline:
    """
    Inference pipeline for time-aware character role-playing.

    설계 포인트:
    - Navigator: 질의 → Top-k 챕터 가설
    - Verifier: 각 챕터의 모든 scene 요약을 보고 1+ scene 선택
    - 선택된 scene들의 요약/원문을 unified question context로 병합
    - Retrieval은 항상 수행하되, Dual-Timeline + negative constraint로
      미래 정보 사용을 프롬프트 레벨에서 강하게 제약
    - Past 질의에 대해서는 프롬프트 내부에서
      Past-Presence / Past-Absence / Fake Premise를 통합적으로 처리
    """

    def __init__(
        self,
        database_dir: str = "knowledge_enriched_rag/databases",
        use_vector_retrieval: bool = True,
        enable_detailed_logging: bool = True,
        device: str = None,
        navigator: 'SceneNavigator' = None,
        top_k_chapters: int = 3,
        ablation: str = None,  # "verifier", "memory", "premise", "selfcorr"
        knowledge_order: str = "chronological",  # "chronological" (default, +2.7% vs similarity) or "similarity"
        memory_top_k: int = 5,  # Number of knowledge items to retrieve from MemoryRetriever
        memory_min_similarity: float = 0.45,  # Minimum similarity threshold for MemoryRetriever
        no_external_knowledge: bool = False,  # If True, add "no external knowledge" instruction to prompt
    ):
        self.database_dir = Path(database_dir)
        self.use_vector_retrieval = use_vector_retrieval
        self.enable_detailed_logging = enable_detailed_logging
        self.device = device if device is not None else "cuda"
        self.ablation = ablation  # Ablation study: which component to disable
        self.knowledge_order = knowledge_order  # Knowledge ordering: "chronological" or "similarity"
        self.memory_top_k = memory_top_k  # Number of knowledge items from MemoryRetriever
        self.memory_min_similarity = memory_min_similarity  # Minimum similarity threshold
        self.no_external_knowledge = no_external_knowledge  # Disable external knowledge usage

        # LLM 설정 (GPT-5 계열)
        self.chat_model = "gpt-5-mini"       # 내부 reasoning, verifier 등
        self.generation_model = "gpt-5-mini" # 최종 응답 생성

        # Navigator Top-k 챕터 개수
        self.top_k_chapters = top_k_chapters

        # 로그 디렉토리
        self.log_dir = Path("pipeline_logs")
        self.log_dir.mkdir(exist_ok=True)

        # OpenAI 클라이언트 (thread-safe 호출에서는 새로 생성)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Navigator (공유 인스턴스 허용, "skip"으로 비활성화 가능)
        if navigator == "skip":
            logger.info("Navigator explicitly disabled (using precomputed hypotheses only)")
            self.navigator = None
        elif navigator is not None:
            logger.info("Using pre-initialized Navigator (shared for parallel processing)")
            self.navigator = navigator
        else:
            logger.info(f"Initializing Navigator (Scene Classifier) on {self.device}")
            self.navigator = SceneNavigator(model_type="full_ft", device=self.device)

        logger.info("Initializing Verifier (Hypothesis Selector)")
        self.verifier = SceneVerifier()

        logger.info("Initializing Memory Retriever (Dual Timeline)")
        self.memory_retriever = MemoryRetriever(db_dir=str(database_dir))

        # Knowledge retrieval
        if use_vector_retrieval:
            logger.info("Initializing Vector-based Knowledge Retrieval")
            self.vector_retriever = VectorKnowledgeRetriever(database_dir)
            self.knowledge_filter = None
        else:
            logger.info("Initializing Rule-based Knowledge Filter")
            self.vector_retriever = None
            self.knowledge_filter = IntelligentKnowledgeFilter()

        # DB 로딩
        logger.info("Loading knowledge databases...")
        self.load_databases()

        # 캐릭터 period → book/chapter 매핑 테이블
        self.character_periods = character_period_harry_potter

    # -------------------------------------------------------------------------
    # DB 로딩
    # -------------------------------------------------------------------------
    def load_databases(self):
        """Load all necessary databases."""

        # 1) scene-level search index (embedding + metadata)
        # Try npy+json format first (numpy version-compatible)
        search_emb_path = self.database_dir / "search_index_embeddings.npy"
        search_meta_path = self.database_dir / "search_index_metadata.json"
        if search_emb_path.exists() and search_meta_path.exists():
            embedding_matrix = np.load(str(search_emb_path))
            with open(search_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.search_index = {
                'embedding_matrix': embedding_matrix,
                'scene_metadata': meta['scene_metadata'],
                'model': meta.get('model', ''),
                'embedding_dim': meta.get('embedding_dim', 3072),
            }
            logger.info(f"Loaded search index with {embedding_matrix.shape[0]} scenes (npy+json)")
        else:
            search_index_path = self.database_dir / "search_index.pkl"
            with open(search_index_path, "rb") as f:
                self.search_index = pickle.load(f)
            logger.info(
                f"Loaded search index with "
                f"{self.search_index['embedding_matrix'].shape[0]} scenes"
            )

        # 2) scenes metadata (요약, 참가자 등)
        scenes_path = self.database_dir / "scenes_vector.json"
        with open(scenes_path, "r", encoding="utf-8") as f:
            self.scenes_data = json.load(f)
            logger.info(f"Loaded {len(self.scenes_data)} scenes metadata")

        # chapter_id → scene index 리스트 (O(1) 접근용 인덱스)
        self.chapter_to_scene_indices: Dict[str, List[int]] = {}
        for i, scene in enumerate(self.scenes_data):
            chapter_id = scene["chapter_id"]
            self.chapter_to_scene_indices.setdefault(chapter_id, []).append(i)

        # 3) character knowledge DB (JSON 기반 인덱스)
        knowledge_json_path = self.database_dir / "character_knowledge.json"
        with open(knowledge_json_path, "r", encoding="utf-8") as f:
            knowledge_data = json.load(f)

        # 두 가지 포맷 지원 (old / optimized)
        if "knowledge_items" in knowledge_data:
            # old flat format
            self.knowledge_items = knowledge_data["knowledge_items"]
            logger.info(
                f"Loaded {len(self.knowledge_items)} knowledge items "
                f"from JSON (old flat format)"
            )

            self.knowledge_index: Dict[str, Dict[str, Dict[str, List[Dict]]]] = {}
            for item in self.knowledge_items:
                char = item["character"]
                chapter = item["chapter_id"]
                scene = item["scene_id"]

                self.knowledge_index.setdefault(char, {})
                self.knowledge_index[char].setdefault(chapter, {})
                self.knowledge_index[char][chapter].setdefault(scene, [])
                self.knowledge_index[char][chapter][scene].append(item)
        else:
            # optimized format: { "Harry Potter": { "Book1-chapter1_Sc1": [items], ... } }
            self.knowledge_index = {}
            total_items = 0

            for char, scenes in knowledge_data.items():
                self.knowledge_index[char] = {}
                for scene_id, items in scenes.items():
                    # e.g., "Book1-chapter1_Sc1" → "Book1-chapter1"
                    chapter_id = "_".join(scene_id.split("_")[:-1])
                    self.knowledge_index[char].setdefault(chapter_id, {})

                    # 각 아이템에 메타데이터 추가 (시간순 정렬을 위해 필요)
                    book_num, chapter_num = self.parse_book_chapter(chapter_id)
                    enriched_items = []
                    for item in items:
                        enriched_item = item.copy()
                        enriched_item['scene_id'] = scene_id
                        enriched_item['chapter_id'] = chapter_id
                        enriched_item['book_num'] = book_num
                        enriched_item['chapter_num'] = chapter_num
                        enriched_items.append(enriched_item)

                    self.knowledge_index[char][chapter_id][scene_id] = enriched_items
                    total_items += len(items)

            logger.info(
                f"Loaded {total_items} knowledge items from JSON (optimized index)"
            )
            self.knowledge_items = []

        logger.info(f"Built knowledge index for {len(self.knowledge_index)} characters")

        # 4) Navigator training 질문 데이터 (디버깅/예시용)
        questions_path = self.database_dir / "training_questions_train.json"
        with open(questions_path, "r", encoding="utf-8") as f:
            self.training_data = json.load(f)
            logger.info(
                f"Loaded {len(self.training_data['questions'])} training questions"
            )

    # -------------------------------------------------------------------------
    # 유틸
    # -------------------------------------------------------------------------
    def _normalize_character_name(self, character: str) -> str:
        """
        Normalize character names to match database format.

        TimeChara validation 데이터와 knowledge DB 간 이름 차이를 보정.
        """
        mapping = {
            "Ronald Weasley": "Ron Weasley",
            "Ron Weasley": "Ron Weasley",
            "Hermione Granger": "Hermione Granger",
            "Harry Potter": "Harry Potter",
        }
        return mapping.get(character, character)

    def parse_book_chapter(self, book_chapter: str) -> Tuple[int, int]:
        """
        "Book1-chapter10" → (1, 10)
        """
        if "-" not in book_chapter:
            return 1, 1

        parts = book_chapter.replace("Book", "").split("-")
        book = int(parts[0]) if parts[0].isdigit() else 1

        chapter_part = parts[1] if len(parts) > 1 else "chapter1"
        if "chapter" in chapter_part:
            chapter = int(chapter_part.replace("chapter", ""))
        else:
            chapter = 1

        return book, chapter

    def _sort_knowledge_chronologically(self, knowledge_items: List[Dict]) -> List[Dict]:
        """
        지식 아이템을 시간순으로 정렬 (book_num, chapter_num, scene_id 순)

        Args:
            knowledge_items: 지식 아이템 리스트

        Returns:
            시간순 정렬된 리스트
        """
        def sort_key(item):
            book_num = item.get('book_num', 0)
            chapter_num = item.get('chapter_num', 0)
            scene_id = item.get('scene_id', '')
            # scene_id에서 숫자 추출 (예: "Book1-chapter2_Sc3" -> 3)
            scene_num = 0
            if '_Sc' in scene_id:
                try:
                    scene_num = int(scene_id.split('_Sc')[-1])
                except ValueError:
                    pass
            return (book_num, chapter_num, scene_num)

        return sorted(knowledge_items, key=sort_key)

    def map_character_period_to_chapter(self, character: str, character_period: str) -> str:
        """
        TimeChara의 period 텍스트를 내부 book/chapter 포맷으로 매핑.

        예:
        "5th-year / on Christmas" → "Book5-chapterX"
        """
        parts = character_period.split("/")
        year = parts[0].strip()

        if len(parts) == 1:
            special_event = "at the end of the scene"
        else:
            special_event = parts[1].strip()

        # year → book number
        book_no = year[0] if year[0].isdigit() else "7"

        # key 정규화
        if special_event == "on the 1st of September":
            key = "on the 1st of september"
        elif special_event == "on Christmas":
            key = "on christmas"
        elif special_event == "on Halloween":
            key = "on halloween"
        elif "end" in special_event:
            key = "at the end of the scene"
        else:
            key = "on the 1st of september"

        # 매핑 테이블에서 챕터 얻기
        if key in self.character_periods:
            chapters = self.character_periods[key]
            for chapter in chapters:
                if chapter.startswith(book_no + "-"):
                    return f"Book{book_no}-chapter{chapter.split('-')[1]}"

        # fallback: 해당 책의 1장
        return f"Book{book_no}-chapter1"

    # -------------------------------------------------------------------------
    # Hypothesis-Verification: Navigator + Verifier
    # -------------------------------------------------------------------------
    def _build_hypotheses_with_all_scenes(
        self,
        hypotheses_raw: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Navigator가 예측한 각 chapter 가설에 대해,
        해당 chapter에 포함된 모든 scene을 붙여서 Verifier 입력 포맷으로 변환.

        설계 의도:
        - "scene embedding으로 top-k scene만 추리는" 게 아니라,
        - "3개 챕터의 모든 scene 요약을 Verifier 프롬프트에 넣고,
           LLM이 1개 이상의 scene을 선택"하는 구조를 반영.
        """
        hypotheses_with_scenes: List[Dict[str, Any]] = []

        for hyp in hypotheses_raw:
            chapter_id = hyp["chapter_id"]
            scene_indices = self.chapter_to_scene_indices.get(chapter_id, [])

            if not scene_indices:
                continue

            hyp_with_scenes = hyp.copy()
            hyp_with_scenes["scenes"] = []

            for idx in scene_indices:
                scene_data = self.scenes_data[idx]
                hyp_with_scenes["scenes"].append(
                    {
                        "scene_id": scene_data["scene_id"],
                        "scene_title": scene_data["scene_title"],
                        "scene_content": scene_data["scene_content"],
                        "text": scene_data.get("scene_content", ""),
                        "detailed_summary": scene_data.get("detailed_summary", ""),
                        "chapter_id": chapter_id,
                    }
                )

            hypotheses_with_scenes.append(hyp_with_scenes)

        return hypotheses_with_scenes

    # -------------------------------------------------------------------------
    # Knowledge Retrieval (Dual-Timeline)
    # -------------------------------------------------------------------------
    def get_cumulative_knowledge(
        self,
        character: str,
        current_position: str,
        question_position: str,
    ) -> List[Dict[str, Any]]:
        """
        캐릭터 현재 위치까지의 모든 character knowledge를 누적.

        - knowledge_index[character][chapter_id][scene_id] 구조를 사용
        - current_position 이전(포함) 챕터의 knowledge만 포함
        """
        current_book, current_chapter = self.parse_book_chapter(current_position)

        cumulative_knowledge: List[Dict[str, Any]] = []
        char_index = self.knowledge_index.get(character, {})

        for chapter_id, scenes in char_index.items():
            if "-" not in chapter_id:
                continue
            kb, kc = self.parse_book_chapter(chapter_id)

            # 현재 위치 이전(포함)만 사용
            if (kb < current_book) or (kb == current_book and kc <= current_chapter):
                for scene_id, items in scenes.items():
                    for item in items:
                        knowledge_item = {
                            "chapter_id": chapter_id,
                            "scene_id": scene_id,
                            "type": item["type"],
                            "description": item["description"],
                            "source_text": item.get("source_text", ""),
                            "scene_title": item.get("scene_title", ""),
                        }
                        if chapter_id == question_position:
                            knowledge_item["is_question_scene"] = True
                        cumulative_knowledge.append(knowledge_item)

        # type별 분포 로그 (3개 타입만 사용)
        by_type = {"FACT_LEARNED": 0, "BELIEF_CHANGED": 0, "RELATIONSHIP_UPDATED": 0}
        for item in cumulative_knowledge:
            t = item.get("type")
            if t in by_type:
                by_type[t] += 1

        logger.info(
            "Knowledge distribution: "
            + ", ".join(f"{k}:{v}" for k, v in by_type.items())
        )
        return cumulative_knowledge

    # -------------------------------------------------------------------------
    # Prompt Building (Dual-Timeline + Premise-aware Generation)
    # -------------------------------------------------------------------------
    def build_inference_prompt(
        self,
        question: str,
        character: str,
        character_period: str,
        question_scene: Dict[str, Any],
        combined_knowledge: List[Dict[str, Any]],
        is_future_question: bool,
        future_knowledge_items: List[Dict[str, Any]],
        premise_verified: bool = True,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Dual-Timeline + Premise-aware generation을 위한 프롬프트 생성.

        - Past/Future 두 분기
        - Past: scene 내부에서 Presence / Fake Premise를 프롬프트 레벨에서 처리
        - Future: future knowledge는 negative constraint 섹션으로만 제공
        - Ablation: 각 ablation별로 별도 템플릿 사용
        """

        # 1) memory_results (벡터 유사도 검색 결과) 직접 사용
        relevant_knowledge = combined_knowledge if combined_knowledge else []

        # 디버그 로깅
        selected_chapters = [k.get("chapter_id", "unknown") for k in relevant_knowledge[:5]]
        logger.info(f"Selected knowledge chapters (top {len(relevant_knowledge)}): {selected_chapters}...")

        # question scene 정보 - raw text 사용 (summary는 환각 가능성 높음)
        question_scene_raw_text = question_scene.get("scene_content", question_scene.get("text", "")) if question_scene else "(Scene text not available)"

        # character_knowledge.json에서 해당 scene의 캐릭터 지식 가져오기
        scene_knowledge_for_char = []
        if self.ablation == "memory":
            # ABLATION: Skip scene-specific knowledge retrieval
            scene_knowledge_for_char = []
        elif question_scene:
            combined_scenes = question_scene.get("combined_from", [question_scene.get("scene_id", "")])
            for sid in combined_scenes:
                if sid:
                    cid = "_".join(sid.split("_")[:-1]) if "_" in sid else question_scene.get("chapter_id", "")
                    items = self.knowledge_index.get(character, {}).get(cid, {}).get(sid, [])
                    scene_knowledge_for_char.extend(items)

        scene_knowledge_text = "\n".join(
            f"- {item['description']}"
            for item in scene_knowledge_for_char
            if "description" in item
        ) if scene_knowledge_for_char else ""

        # 현재 시점 챕터 계산
        current_chapter = self.map_character_period_to_chapter(character, character_period)
        current_book, current_chap = self.parse_book_chapter(current_chapter)

        # 1) current_period_knowledge: 현재 챕터의 모든 지식
        current_period_knowledge = []
        if self.ablation != "memory":
            char_knowledge = self.knowledge_index.get(character, {})
            chapter_data = char_knowledge.get(f"Book{current_book}-chapter{current_chap}", {})
            for scene_id, items in chapter_data.items():
                current_period_knowledge.extend(items)

        # 2) previous_period_knowledge: 벡터 검색 결과 중 이전 챕터 것만
        previous_period_knowledge = []
        if self.ablation != "memory":
            for item in relevant_knowledge:
                chapter_id = item.get("chapter_id", "")
                if chapter_id:
                    kb, kc = self.parse_book_chapter(chapter_id)
                    if not (kb == current_book and kc == current_chap):
                        previous_period_knowledge.append(item)
                else:
                    previous_period_knowledge.append(item)

        # 정렬 적용 (chronological or similarity order)
        if self.knowledge_order == "chronological":
            current_period_knowledge = self._sort_knowledge_chronologically(current_period_knowledge)
            previous_period_knowledge = self._sort_knowledge_chronologically(previous_period_knowledge)
            order_label = "(Chronological order)"
        else:
            order_label = "(Relevance order)"

        # 지식 텍스트 생성
        current_descs = [
            item.get('description', '')
            for item in current_period_knowledge
            if item.get('description', '').strip()
        ]
        current_knowledge_text = f"{order_label}\n" + " -> ".join(current_descs) if current_descs else "(No knowledge from current period)"

        previous_descs = [
            item.get('description', '')
            for item in previous_period_knowledge
            if item.get('description', '').strip()
        ]
        previous_knowledge_text = f"{order_label}\n" + " -> ".join(previous_descs) if previous_descs else "(No accumulated knowledge from previous periods)"

        # No external knowledge instruction
        no_external_instruction = ""
        if self.no_external_knowledge:
            logger.info("[CONFIG] No external knowledge mode enabled")
            no_external_instruction = """
====================================
CRITICAL: NO EXTERNAL KNOWLEDGE
====================================

You MUST answer based ONLY on the provided context.
Do NOT use any external knowledge about Harry Potter books, movies, or any other sources.
If the information is not in the provided context, do not assume or infer from external sources.
"""

        # -------------------------------------------------------------
        # Template Selection based on branch and ablation
        # -------------------------------------------------------------
        if is_future_question:
            # Future knowledge text (for self-correction section)
            future_knowledge_text = ""
            if future_knowledge_items:
                future_descs = [
                    item.get("description", "")
                    for item in future_knowledge_items
                    if "description" in item
                ]
                future_knowledge_text = "\n".join(
                    f"- {d}" for d in future_descs if d.strip()
                ) if future_descs else "(No future knowledge)"
            else:
                future_knowledge_text = "(No future knowledge)"

            # Template selection for Future branch
            if self.ablation == "memory":
                logger.info("[ABLATION] Using PROMPT_FUTURE_NO_MEMORY template")
                template = PROMPT_FUTURE_NO_MEMORY
            elif self.ablation == "selfcorr":
                logger.info("[ABLATION] Using PROMPT_FUTURE_NO_SELFCORR template")
                template = PROMPT_FUTURE_NO_SELFCORR
            else:
                template = PROMPT_FUTURE_NORMAL

            prompt = template.format(
                character=character,
                question=question,
                no_external_instruction=no_external_instruction,
                question_scene_raw_text=question_scene_raw_text,
                current_knowledge_text=current_knowledge_text,
                previous_knowledge_text=previous_knowledge_text,
                future_knowledge_text=future_knowledge_text,
            )
        else:
            # Template selection for Past branch
            if self.ablation == "memory":
                logger.info("[ABLATION] Using PROMPT_PAST_NO_MEMORY template")
                template = PROMPT_PAST_NO_MEMORY
            elif self.ablation == "premise":
                logger.info("[ABLATION] Using PROMPT_PAST_NO_PREMISE template")
                template = PROMPT_PAST_NO_PREMISE
            else:
                template = PROMPT_PAST_NORMAL

            prompt = template.format(
                character=character,
                question=question,
                no_external_instruction=no_external_instruction,
                question_scene_raw_text=question_scene_raw_text,
                current_knowledge_text=current_knowledge_text,
                previous_knowledge_text=previous_knowledge_text,
                scene_knowledge_text=scene_knowledge_text,
            )

        return prompt, relevant_knowledge

    # -------------------------------------------------------------------------
    # Response Generation
    # -------------------------------------------------------------------------
    def generate_response(
        self,
        prompt: str,
        character: str,
    ) -> str:
        """
        실제 LLM 호출로 캐릭터 응답 생성.
        - GPT-5 계열 파라미터 사용 (developer role, max_completion_tokens 등)
        - 간단한 재시도 로직 포함
        """
        max_retries = 5
        base_delay = 2

        for attempt in range(max_retries):
            try:
                # build_inference_prompt에서 완성된 프롬프트를 직접 사용

                thread_safe_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

                if self.generation_model in ["gpt-5", "gpt-5-mini"]:
                    response = thread_safe_client.chat.completions.create(
                        model=self.generation_model,
                        messages=[{"role": "developer", "content": prompt}],
                        max_completion_tokens=8000,
                        verbosity="high",
                        reasoning_effort="minimal",
                    )
                else:
                    response = thread_safe_client.chat.completions.create(
                        model=self.generation_model,
                        messages=[{"role": "system", "content": prompt}],
                        max_completion_tokens=8000,
                    )

                content = response.choices[0].message.content
                if content:
                    return content.strip()

                # empty 응답 → backoff
                import time

                wait = base_delay * (2 ** attempt)
                logger.warning(
                    f"Empty response on attempt {attempt + 1}/{max_retries} for {character}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait} seconds...")
                    time.sleep(wait)
                    continue
                else:
                    logger.error(f"FAILED: Empty response after {max_retries} retries for {character}")
                    return "[GENERATION_FAILED]"

            except Exception as e:
                import time, traceback

                error_type = type(e).__name__
                wait = base_delay * (2 ** attempt)
                if "RateLimit" in error_type or "rate_limit" in str(e).lower():
                    wait *= 2
                    logger.error(
                        f"RATE LIMIT ERROR on attempt {attempt + 1}/{max_retries}: {e}"
                    )
                else:
                    logger.error(
                        f"ERROR ({error_type}) on attempt {attempt + 1}/{max_retries}: {e}"
                    )

                if attempt == 0:
                    logger.error(f"  Character: {character}")
                    logger.error(f"  Prompt: {prompt[:100]}...")
                    logger.error(f"  Model: {self.generation_model}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {wait} seconds...")
                    time.sleep(wait)
                    continue
                else:
                    logger.error("FINAL FAILURE after retries")
                    logger.error(traceback.format_exc())
                    return f"I'm not quite sure how to respond to that as {character}."

        return f"I'm not quite sure how to respond to that as {character}."

    def generate_response_stream(
        self,
        prompt: str,
        character: str,
    ):
        """
        Streaming version of generate_response.
        Yields chunks of the response as they arrive.
        """
        thread_safe_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        try:
            if self.generation_model in ["gpt-5", "gpt-5-mini"]:
                stream = thread_safe_client.chat.completions.create(
                    model=self.generation_model,
                    messages=[{"role": "developer", "content": prompt}],
                    max_completion_tokens=8000,
                    stream=True,
                )
            else:
                stream = thread_safe_client.chat.completions.create(
                    model=self.generation_model,
                    messages=[{"role": "system", "content": prompt}],
                    max_completion_tokens=8000,
                    stream=True,
                )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"I'm not quite sure how to respond to that as {character}."

    # -------------------------------------------------------------------------
    # 메인 파이프라인
    # -------------------------------------------------------------------------
    def process_query(
        self,
        query_data: Dict[str, Any],
        precomputed_hypotheses: Optional[List[Dict[str, Any]]] = None,
        precomputed_verification_result: Optional[Dict[str, Any]] = None,
        skip_generation: bool = False,
    ) -> Dict[str, Any]:
        """
        메인 inference 파이프라인.

        - Navigator → Top-k chapter 가설
        - 각 챕터의 모든 scene 요약을 Verifier에 입력
        - 선택된 scene들의 요약/원문을 unified question context로 병합
        - Dual-Timeline + Premise-aware generation 프롬프트로 최종 LLM 호출
        """
        # 입력 unpack
        question = query_data["question"]
        character_original = query_data["character"]
        character_period = query_data["character_period"]

        character = self._normalize_character_name(character_original)
        logger.info(f"Processing: {character} at {character_period}")
        logger.info(f"Question: {question[:120]}...")

        # Pre-compute character mapping (local dict lookup, instant)
        # Moved here to enable parallel Verifier + Memory execution
        current_book_chapter = self.map_character_period_to_chapter(
            character, character_period
        )
        logger.info(f"Character current position: {current_book_chapter}")

        # ---------------------------------------------------------------------
        # 0. Hypothesis-Verification (Navigator + Verifier)
        # ---------------------------------------------------------------------
        if precomputed_verification_result is not None:
            logger.info("Using precomputed verification result")
            verification_result = precomputed_verification_result

            if verification_result is None:
                raise RuntimeError("precomputed_verification_result is None")

            selected_scenes = verification_result.get("selected_scenes", [])
            if not selected_scenes:
                raise RuntimeError("Verifier returned empty scene list")

            question_period = selected_scenes[0]["chapter_id"]
            hypotheses_raw = []
        else:
            # 0.1 Navigator: Top-k chapter 가설
            if precomputed_hypotheses is not None:
                hypotheses_raw = precomputed_hypotheses
                logger.info(
                    f"Using precomputed hypotheses (len={len(hypotheses_raw)})"
                )
            else:
                k = self.top_k_chapters
                hypotheses_raw = self.navigator.predict_top_k(question, k=k)
                logger.info(f"Navigator generated {len(hypotheses_raw)} hypotheses:")
                for h in hypotheses_raw:
                    logger.info(
                        f"  - {h['chapter_id']} (confidence={h['confidence']:.3f})"
                    )

            # 0.2 각 챕터의 모든 scene 요약을 붙여서 Verifier 입력 포맷 생성
            hypotheses_with_scenes = self._build_hypotheses_with_all_scenes(
                hypotheses_raw
            )
            if not hypotheses_with_scenes:
                logger.error("No hypotheses with scenes found.")
                # fallback: 첫 hypothesis의 chapter_id 사용
                question_period = hypotheses_raw[0]["chapter_id"]
                verification_result = None
                selected_scenes = []
            else:
                # Ablation: skip Verifier, use first scene directly
                if self.ablation == "verifier":
                    # Use first scene from first hypothesis without LLM verification
                    first_hyp = hypotheses_with_scenes[0]
                    selected_scenes = [first_hyp["scenes"][0]] if first_hyp.get("scenes") else []
                    question_period = first_hyp["chapter_id"]
                    verification_result = {
                        "selected_scenes": selected_scenes,
                        "confidence": 1.0,
                        "premise_verified": True,
                        "reasoning": "[ABLATION: Verifier skipped]"
                    }
                    logger.info(f"[ABLATION] Verifier skipped, using first scene: {question_period}")
                else:
                    # 0.3 Verifier + Memory in PARALLEL
                    # Memory only needs question, character, current_book_chapter
                    # (no dependency on Verifier result)
                    from concurrent.futures import ThreadPoolExecutor

                    run_memory = (self.ablation != "memory")

                    def _verifier_task():
                        return self.verifier.verify_hypotheses(
                            question=question,
                            hypotheses=hypotheses_with_scenes,
                            character=character,
                        )

                    def _memory_task():
                        return self.memory_retriever.search_memories(
                            question=question,
                            character=character,
                            before_chapter=current_book_chapter,
                            top_k=self.memory_top_k,
                            min_similarity=self.memory_min_similarity,
                        )

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        verifier_future = executor.submit(_verifier_task)
                        if run_memory:
                            logger.info("Running Verifier + MemoryRetriever in parallel")
                            memory_future = executor.submit(_memory_task)
                        verification_result = verifier_future.result()
                        if run_memory:
                            _parallel_memory_results = memory_future.result()
                        else:
                            _parallel_memory_results = None

                    selected_scenes = verification_result["selected_scenes"]
                    question_period = selected_scenes[0]["chapter_id"]

                    logger.info(
                        f"Verifier selected {len(selected_scenes)} scene(s) from {question_period} "
                        f"(confidence={verification_result['confidence']:.2f})"
                    )
                    logger.info(
                        "  Scenes: " + ", ".join(s["scene_id"] for s in selected_scenes)
                    )
                    logger.info(f"Reasoning: {verification_result['reasoning']}")

        logger.info(f"Final question_period: {question_period}")

        # ---------------------------------------------------------------------
        # 로깅 구조 초기화
        # ---------------------------------------------------------------------
        current_log: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "query": {
                "character": character,
                "character_period": character_period,
                "question": question,
                "question_period": question_period,
                "question_period_source": "predicted",
            },
            "pipeline_steps": {},
        }

        # Extract premise_verified flag from verifier result
        premise_verified = verification_result.get("premise_verified", True) if verification_result else True

        if self.enable_detailed_logging and verification_result:
            current_log["pipeline_steps"]["hypothesis_verification"] = {
                "method": "navigator+verifier_all_scenes",
                "navigator_hypotheses": [
                    {"chapter_id": h["chapter_id"], "confidence": h["confidence"]}
                    for h in hypotheses_raw
                ],
                "verifier_selection": {
                    "selected_chapter": selected_scenes[0]["chapter_id"],
                    "selected_scenes": [s["scene_id"] for s in selected_scenes],
                    "num_scenes_selected": len(selected_scenes),
                    "confidence": verification_result["confidence"],
                    "premise_verified": premise_verified,
                    "reasoning": verification_result["reasoning"],
                },
            }

        # ---------------------------------------------------------------------
        # 1. 캐릭터 현재 위치 매핑 (already computed above for parallelization)
        # ---------------------------------------------------------------------
        if self.enable_detailed_logging:
            current_log["pipeline_steps"]["character_mapping"] = {
                "input_period": character_period,
                "mapped_position": current_book_chapter,
            }

        # ---------------------------------------------------------------------
        # 2. question_scene (선택된 scene들 → unified context)
        # ---------------------------------------------------------------------
        if not verification_result or not selected_scenes:
            raise RuntimeError("Navigator-Verifier pipeline failed to select scenes")

        # 여러 scene의 요약/원문 병합
        combined_summary = "\n\n".join(
            s.get("detailed_summary", "") for s in selected_scenes
        )
        combined_raw_text = "\n\n".join(
            s.get("scene_content", s.get("text", "")) for s in selected_scenes
        )

        question_scene = selected_scenes[0].copy()
        question_scene["detailed_summary"] = combined_summary
        question_scene["scene_content"] = combined_raw_text
        question_scene["text"] = combined_raw_text
        question_scene["combined_from"] = [s["scene_id"] for s in selected_scenes]

        logger.info(
            f"Question scene(s) ready: {question_scene.get('combined_from', [])}"
        )

        if self.enable_detailed_logging:
            current_log["pipeline_steps"]["question_scene_selection"] = {
                "question": question,
                "target_chapter": question_period,
                "selected_scene": {
                    "scene_id": question_scene.get("scene_id"),
                    "scene_title": question_scene.get("scene_title", ""),
                    "summary": question_scene.get("detailed_summary", ""),
                },
            }
            current_log["pipeline_steps"]["question_scene"] = {
                "scene_id": question_scene.get("scene_id", ""),
                "scene_title": question_scene.get("scene_title", ""),
                "summary": question_scene.get("detailed_summary", ""),
            }

        # ---------------------------------------------------------------------
        # 3. Past / Future 판별
        # ---------------------------------------------------------------------
        question_book, question_chap = self.parse_book_chapter(question_period)
        current_book, current_chap = self.parse_book_chapter(current_book_chapter)

        is_future_question = (question_book > current_book) or (
            question_book == current_book and question_chap > current_chap
        )
        logger.info(f"is_future_question = {is_future_question}")

        # ---------------------------------------------------------------------
        # 4. Knowledge Retrieval - Vector similarity search from Memory DB
        #    (Optimized: already ran in parallel with Verifier if available)
        # ---------------------------------------------------------------------
        # Use parallel result if available, otherwise run sequentially
        if '_parallel_memory_results' in dir() and _parallel_memory_results is not None:
            memory_results = _parallel_memory_results
            logger.info(f"MemoryRetriever: using parallel result ({len(memory_results)} memories)")
        elif self.ablation == "memory":
            memory_results = []
            logger.info("[ABLATION] MemoryRetriever skipped, using empty knowledge")
        else:
            logger.info(f"MemoryRetriever: searching similar knowledge (top_k={self.memory_top_k})")
            memory_results = self.memory_retriever.search_memories(
                question=question,
                character=character,
                before_chapter=current_book_chapter,
                top_k=self.memory_top_k,
                min_similarity=self.memory_min_similarity,
            )
            logger.info(f"MemoryRetriever found {len(memory_results)} memories")

        combined_knowledge = memory_results

        if self.enable_detailed_logging:
            knowledge_by_type: Dict[str, List[Dict[str, Any]]] = {}
            for item in memory_results:
                t = item.get("type", "UNKNOWN")
                knowledge_by_type.setdefault(t, []).append(
                    {
                        "source": "memory_retriever",
                        "similarity_score": item.get("similarity_score", 0.0),
                        "chapter_id": item.get("chapter_id", ""),
                        "description": item.get("description", "")[:200],
                    }
                )

            current_log["pipeline_steps"]["knowledge_retrieval"] = {
                "memory_items": len(memory_results),
                "total_items": len(combined_knowledge),
                "distribution": {k: len(v) for k, v in knowledge_by_type.items()},
                "sample_items": {k: v[:3] for k, v in knowledge_by_type.items()},
            }

        # 4-3) Future question: future knowledge 추출 (negative constraint용)
        future_knowledge_items: List[Dict[str, Any]] = []
        if is_future_question and question_scene and self.ablation != "memory":
            char_index = self.knowledge_index.get(character, {})
            future_knowledge_items = (
                char_index
                .get(question_scene["chapter_id"], {})
                .get(question_scene["scene_id"], [])
            )
            logger.info(
                f"Future knowledge items for negative constraint: "
                f"{len(future_knowledge_items)}"
            )

        # ---------------------------------------------------------------------
        # 5. 프롬프트 생성 (Dual-Timeline + Premise-aware)
        # ---------------------------------------------------------------------
        prompt, selected_knowledge = self.build_inference_prompt(
            question=question,
            character=character,
            character_period=character_period,
            question_scene=question_scene,
            combined_knowledge=combined_knowledge,
            is_future_question=is_future_question,
            future_knowledge_items=future_knowledge_items,
            premise_verified=premise_verified,
        )

        if self.enable_detailed_logging:
            current_log["pipeline_steps"]["prompt_building"] = {
                "prompt_length": len(prompt),
                "prompt_full": prompt,
                "selected_knowledge_count": len(selected_knowledge),
                "future_knowledge_items": len(future_knowledge_items)
                if is_future_question
                else 0,
                "premise_verified": premise_verified,
            }

        # ---------------------------------------------------------------------
        # 6. 최종 응답 생성
        # ---------------------------------------------------------------------
        if skip_generation:
            response = ""
        else:
            response = self.generate_response(
                prompt=prompt,
                character=character,
            )

        if self.enable_detailed_logging and response:
            current_log["pipeline_steps"]["response_generation"] = {
                "response": response,
                "response_length": len(response),
            }

            log_filename = (
                f"{character}_{character_period.replace('/', '_').replace(' ', '')}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            log_path = self.log_dir / log_filename
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(current_log, f, indent=2, ensure_ascii=False)
            logger.info(f"Detailed log saved to: {log_path}")

        # ---------------------------------------------------------------------
        # 결과 패키징
        # ---------------------------------------------------------------------
        return {
            "character": character,
            "character_period": character_period,
            "question": question,
            "question_period": question_period,
            "question_period_source": "predicted",
            "verification_result": verification_result,
            "current_position": current_book_chapter,
            "is_future_question": is_future_question,
            "knowledge_items_used": len(combined_knowledge),
            "memory_knowledge_items": len(memory_results),
            "selected_knowledge": selected_knowledge,
            "question_scene": question_scene,
            "prompt": prompt,
            "generated_response": response,
            "answer": response,
            "response": response,
            # Premise validation은 프롬프트 내부에 통합 → 상태만 표시
            "premise_validation": {
                "mode": "integrated_in_prompt",
                "separate_api_used": False,
            },
            "log": current_log,
        }


# 간단한 테스트 함수 (원하면 유지)
def test_inference_pipeline():
    pipeline = TimeAwareInferencePipeline()

    with open("data/test_harrypotter.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    for i, query in enumerate(test_data[:3]):
        print("\n" + "=" * 80)
        print(f"Test {i + 1}: {query['character']} at {query['character_period']}")
        print(f"Question: {query['question'][:120]}...\n")

        result = pipeline.process_query(query)
        print(f"Current Position: {result['current_position']}")
        print(f"Is Future Question: {result['is_future_question']}")
        print(f"Knowledge Items Used: {result['knowledge_items_used']}")
        print(
            "Question Scene:",
            result["question_scene"]["scene_id"]
            if result["question_scene"]
            else "None",
        )
        print("\nGenerated Response:\n", result["generated_response"])
        print("\nGold Response:\n", query.get("gold_response", "N/A"))

        results.append(result)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"knowledge_enriched_rag/test_outputs/inference_test_{ts}.json"
    Path("knowledge_enriched_rag/test_outputs").mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": ts, "test_count": len(results), "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    test_inference_pipeline()
