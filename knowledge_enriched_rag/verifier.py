#!/usr/bin/env python3
"""
Scene Verifier - Hypothesis Selection Module
Implements 'Evidence Gathering & Verification' from paper (Section 2.2)
"""

import json
import logging
from typing import List, Dict, Any, Tuple
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneVerifier:
    """
    Verifies and selects the best scene hypothesis for answering a question

    Purpose:
        Given Top-k chapter hypotheses (from Navigator), retrieves candidate
        scenes and selects the single best scene that provides evidence to
        answer the question.

    This module overcomes the Navigator's imperfect accuracy by:
        1. Expanding search to Top-k candidates (typically k=3)
        2. Using LLM-based semantic comparison
        3. Providing confidence scores and reasoning
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Args:
            model: LLM model for verification (gemini-2.5-flash)
        """
        self.model_name = model
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction="You are a meticulous literary analyst and fact-checker. Your task is to evaluate candidate scenes and select the one that best answers the given question. You MUST respond with valid JSON."
        )

    def verify_hypotheses(
        self,
        question: str,
        hypotheses: List[Dict[str, Any]],
        character: str = None,
        return_reasoning: bool = True,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Select the best hypothesis from multiple candidates with retry logic

        Args:
            question: User's question
            hypotheses: List of candidate hypotheses, each containing:
                {
                    'hypothesis_id': int,
                    'chapter_id': str,
                    'scenes': [  # List of scenes
                        {
                            'scene_id': str,
                            'scene_title': str,
                            'detailed_summary': str,
                            'similarity_score': float
                        }
                    ]
                }
            return_reasoning: If True, return detailed reasoning
            max_retries: Maximum number of API call attempts (default: 3)

        Returns:
            {
                'selected_scenes': [  # List of selected scene objects
                    {
                        'scene_id': str,
                        'chapter_id': str,
                        'scene_title': str,
                        'detailed_summary': str,
                        'scene_content': str,
                        ...
                    }
                ],
                'confidence': float (0.0 to 1.0),
                'reasoning': str (if return_reasoning=True)
            }
            Or None if all retries failed (caller should exclude this sample)
        """

        if not hypotheses:
            raise ValueError("No hypotheses provided")

        # Handle single hypothesis with single scene - return directly
        total_scenes = sum(len(hyp.get('scenes', [])) for hyp in hypotheses)
        if total_scenes == 1:
            hyp = hypotheses[0]
            scene = hyp['scenes'][0]
            return {
                'selected_scenes': [{
                    'scene_id': scene['scene_id'],
                    'chapter_id': hyp['chapter_id'],
                    'scene_title': scene.get('scene_title', ''),
                    'detailed_summary': scene.get('detailed_summary', ''),
                    'scene_content': scene.get('scene_content', ''),
                    'text': scene.get('text', scene.get('scene_content', '')),
                    'participants': scene.get('participants', []),
                    'similarity_score': scene.get('similarity_score', 0.0)
                }],
                'confidence': 1.0,
                'reasoning': "Only one scene available"
            }

        # Build verification prompt and get all_scenes (THREAD-SAFE: local variable)
        prompt, all_scenes = self._build_verification_prompt(question, hypotheses, character)

        # Retry logic
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=1.0,
                        max_output_tokens=4000,
                        response_mime_type="application/json"
                    )
                )

                result_text = response.text

                # DEBUG: Log actual response
                if not result_text or result_text.strip() == "":
                    logger.warning(f"Attempt {attempt+1}: Empty response from API!")
                    logger.warning(f"   Response object: {response}")
                    continue

                # Try to parse JSON
                result = json.loads(result_text)

                # Validate required fields
                selected_indices = result.get('selected_scene_indices')
                if selected_indices is None:
                    logger.warning(f"Attempt {attempt+1}: Missing 'selected_scene_indices', retrying...")
                    continue

                if not isinstance(selected_indices, list) or len(selected_indices) == 0:
                    logger.warning(f"Attempt {attempt+1}: Invalid scene indices format (must be non-empty list), retrying...")
                    continue

                # Validate all indices are valid (THREAD-SAFE: use local all_scenes)
                num_scenes = len(all_scenes)
                invalid_indices = [idx for idx in selected_indices if not isinstance(idx, int) or idx < 0 or idx >= num_scenes]
                if invalid_indices:
                    logger.warning(f"Attempt {attempt+1}: Invalid scene indices {invalid_indices} (total scenes: {num_scenes}), retrying...")
                    continue

                # Valid result found! Extract selected scenes (THREAD-SAFE: use local all_scenes)
                selected_scenes = []
                for idx in selected_indices:
                    scene_info = all_scenes[idx]
                    scene = scene_info['scene']
                    selected_scenes.append({
                        'scene_id': scene['scene_id'],
                        'chapter_id': scene_info['chapter_id'],
                        'scene_title': scene.get('scene_title', ''),
                        'detailed_summary': scene.get('detailed_summary', ''),
                        'scene_content': scene.get('scene_content', ''),
                        'text': scene.get('text', scene.get('scene_content', '')),
                        'participants': scene.get('participants', []),
                        'similarity_score': scene.get('similarity_score', 0.0)
                    })

                # Extract premise_verified flag (defaults to True for backward compatibility)
                premise_verified = result.get('premise_verified', True)

                logger.info(f"✅ Verifier succeeded on attempt {attempt+1}")
                logger.info(f"   Selected {len(selected_scenes)} scene(s): {[s['scene_id'] for s in selected_scenes]}")
                logger.info(f"   Premise verified: {premise_verified}")

                return {
                    'selected_scenes': selected_scenes,
                    'confidence': result.get('confidence', 0.5),
                    'premise_verified': premise_verified,
                    'reasoning': result.get('reasoning', '') if return_reasoning else ''
                }

            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt+1}: JSON parsing failed - {e}")
                last_exception = e
                continue

            except Exception as e:
                logger.warning(f"Attempt {attempt+1}: API/other error - {e}")
                last_exception = e
                continue

        # All retries failed
        logger.error(f"⚠️ Verifier failed after {max_retries} attempts: {last_exception}")

        # Return None to signal caller to EXCLUDE this sample
        return None

    def _build_verification_prompt(
        self,
        question: str,
        hypotheses: List[Dict[str, Any]],
        character: str = None
    ) -> Tuple[str, list]:
        """
        Build structured comparison prompt for LLM

        Returns:
            Tuple of (prompt: str, all_scenes: list)
            - prompt instructs LLM to analyze and select scenes
            - all_scenes is returned for thread-safe access (not stored in self._all_scenes)
        """

        # Flatten all scenes from all hypotheses into single list (THREAD-SAFE: local variable)
        all_scenes = []
        for hyp in hypotheses:
            scenes = hyp.get('scenes', [])  # Get scenes list
            for scene in scenes:
                all_scenes.append({
                    'scene': scene,
                    'chapter_id': hyp['chapter_id']
                })

        # Build candidate descriptions from flattened scenes (V0 ORIGINAL - with metadata)
        candidates_text = []
        for i, item in enumerate(all_scenes):
            scene = item['scene']
            candidate_text = f"""
**CANDIDATE {i}:**
- Chapter: {item['chapter_id']}
- Scene ID: {scene['scene_id']}
- Scene Title: {scene.get('scene_title', 'Untitled')}
- Navigator Similarity Score: {scene.get('similarity_score', 0.0):.3f}

Scene Summary:
{scene['detailed_summary']}
"""
            candidates_text.append(candidate_text)

        candidates_section = "\n".join(candidates_text)

        # Build character context section if character is provided
        character_section = ""
        if character:
            character_section = f"""
**TARGET CHARACTER:**
The answer will be given from **{character}**'s perspective.
- CRITICAL: Prefer scenes where {character} is explicitly present, speaking, or mentioned as participating.
- If {character} is NOT present in any candidate scene, this may indicate the character was ABSENT from the event being asked about.
- Consider: Does the scene show {character} directly experiencing or witnessing the event?
"""

        prompt = f"""You are evaluating {len(all_scenes)} candidate scenes to determine which ones provide the most direct and sufficient evidence to answer the given QUESTION.

**QUESTION:**
"{question}"
{character_section}
**CANDIDATE SCENES:**
{candidates_section}

**INSTRUCTIONS:**

1. PREMISE EXTRACTION (CRITICAL FIRST STEP):
   Extract the CORE CLAIM from the question. The core claim is WHO did WHAT ACTION.
   Example: "Why did Hagrid support Harry's decision to relax?"
            -> Core claim: "Hagrid explicitly supported/encouraged Harry to relax (not work on the egg)"
   Example: "What did Ron say when Harry caught the Snitch?"
            -> Core claim: "Ron spoke during the moment Harry caught the Snitch"

2. PREMISE VERIFICATION (STRICT - BEFORE SCORING):
   For EACH candidate scene, check: Does the scene summary contain EXPLICIT EVIDENCE of the EXACT action claimed?

   STRICT VERIFICATION RULES:
   - The SAME CHARACTER must perform the SAME ACTION as claimed
   - "Being pleased" or "being happy" is NOT the same as "supporting a decision"
   - "Being present" is NOT the same as "actively doing something"
   - If the question says "X supported Y's decision to do Z", verify X explicitly said/did something to support Z
   - If a DIFFERENT character did the action (e.g., Ron instead of Hagrid), the premise is FALSE
   - A scene mentioning the same TOPIC is NOT sufficient; the SPECIFIC ACTION must be verified

   Example of FALSE premise:
   - Question: "Why did Hagrid support Harry's decision to relax?"
   - Scene says: "Hagrid was pleased when Harry said he was doing great"
   - Analysis: Hagrid being pleased != Hagrid supporting relaxation. Premise NOT verified.

3. INDEPENDENT EVALUATION (Rate each candidate separately):
   For EACH CANDIDATE SCENE, assign a relevance score (0-10):
   - 0-3: Premise NOT verified OR not relevant
   - 4-6: Partially relevant but premise verification unclear
   - 7-8: Premise verified AND highly relevant
   - 9-10: Premise verified with complete information

   Evaluate based on:
   - Does the scene VERIFY the core claim from the question?
   - Is this a PRIMARY scene where the claimed action occurs?
   - How completely does the scene answer the question?

4. CONTRASTIVE ANALYSIS:
   - If CANDIDATE 0 has the highest score, explicitly consider: "Why might another candidate be better?"
   - Look for cases where a lower-ranked candidate might have more complete or accurate information
   - Don't automatically choose the first candidate without comparing content quality
   - CRITICAL: Distinguish between "planning/discussing an event" vs "event actually happening"
     * If a scene shows characters PLANNING or DISCUSSING an action, that is NOT the scene where it happens
     * Select the scene where the ACTION in the question ACTUALLY OCCURS, not where it is mentioned or planned

5. FINAL SELECTION:
   Select ONE OR MORE scenes (minimum 1) that together provide complete context:
   - If the question's events span multiple scenes, select all relevant scenes
   - If one scene is sufficient, select only that scene
   - Scene boundaries may be subjective, so consider adjacent/related scenes
   - If NO scene verifies the premise, select the most relevant scene AND set premise_verified to false

6. Provide a confidence score (0.0 to 1.0) and reasoning (2-3 sentences).

**IMPORTANT:**
- ALWAYS verify the premise before selecting scenes
- If the premise cannot be verified in any scene, set premise_verified to false
- Select based on CONTENT RELEVANCE to the question
- Prefer selecting fewer scenes if they provide complete information
- If events/information span scene boundaries, select multiple scenes
- If no candidate seems relevant, select the best available and set confidence < 0.5

**OUTPUT FORMAT (JSON ONLY):**
{{
  "selected_scene_indices": [<list of 0-indexed scene IDs, e.g., [0] or [1, 2]>],
  "confidence": <0.0 to 1.0>,
  "premise_verified": <true if the question's core claim is verified in selected scene(s), false otherwise>,
  "reasoning": "<Brief explanation including: 1) what is the core claim, 2) is it verified, 3) why these scenes were selected>"
}}
"""

        # Return both prompt and all_scenes (THREAD-SAFE: no instance variable)
        return prompt, all_scenes
