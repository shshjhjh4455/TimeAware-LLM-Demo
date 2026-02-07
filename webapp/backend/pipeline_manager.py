"""Singleton manager for TimeAwareInferencePipeline."""

import sys
import time
import json
import logging
import re
from pathlib import Path
from threading import Lock

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.backend.config import PIPELINE_CONFIG

logger = logging.getLogger(__name__)


class PipelineManager:
    """Thread-safe singleton for the inference pipeline."""

    _instance = None
    _lock = Lock()
    _pipeline = None
    _initialized = False
    _init_error = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self):
        """Initialize the pipeline (call once at app startup)."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return
            try:
                logger.info("Initializing TimeAwareInferencePipeline...")
                start = time.time()

                from knowledge_enriched_rag.scene_classifier import SceneNavigator
                from knowledge_enriched_rag.inference_pipeline import TimeAwareInferencePipeline

                navigator = SceneNavigator(
                    model_type="full_ft",
                    device=PIPELINE_CONFIG["device"],
                )

                self._pipeline = TimeAwareInferencePipeline(
                    database_dir=PIPELINE_CONFIG["database_dir"],
                    use_vector_retrieval=PIPELINE_CONFIG["use_vector_retrieval"],
                    enable_detailed_logging=PIPELINE_CONFIG["enable_detailed_logging"],
                    device=PIPELINE_CONFIG["device"],
                    navigator=navigator,
                    top_k_chapters=PIPELINE_CONFIG["top_k_chapters"],
                    knowledge_order=PIPELINE_CONFIG["knowledge_order"],
                    memory_top_k=PIPELINE_CONFIG["memory_top_k"],
                    memory_min_similarity=PIPELINE_CONFIG["memory_min_similarity"],
                    no_external_knowledge=PIPELINE_CONFIG["no_external_knowledge"],
                )

                elapsed = time.time() - start
                logger.info(f"Pipeline initialized in {elapsed:.1f}s")
                self._initialized = True

            except Exception as e:
                logger.error(f"Pipeline initialization failed: {e}", exc_info=True)
                self._init_error = str(e)
                raise

    @property
    def pipeline(self):
        if not self._initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._pipeline

    @property
    def is_ready(self):
        return self._initialized

    @property
    def error(self):
        return self._init_error

    def process_query(self, character, character_period, question):
        """Run the inference pipeline and return structured results."""
        start = time.time()

        query_data = {
            "character": character,
            "character_period": character_period,
            "question": question,
        }

        result = self.pipeline.process_query(query_data)
        elapsed_ms = int((time.time() - start) * 1000)

        # Extract the clean answer from generated_response
        answer = self._extract_response(result.get("generated_response", ""))
        if not answer:
            answer = result.get("answer", result.get("response", ""))

        # Build debug info
        debug = self._build_debug(result, elapsed_ms)

        return {
            "answer": answer,
            "character": character,
            "character_period": character_period,
            "debug": debug,
        }

    def process_query_stream(self, character, character_period, question):
        """
        Streaming version: yields SSE events.
        Phase 1: Run pipeline without generation (Navigator+Verifier+Memory) → emit debug
        Phase 2: Stream LLM response tokens → emit token chunks
        """
        start = time.time()

        query_data = {
            "character": character,
            "character_period": character_period,
            "question": question,
        }

        # Phase 1: Everything except LLM generation
        result = self.pipeline.process_query(query_data, skip_generation=True)
        prep_ms = int((time.time() - start) * 1000)

        # Build debug from Phase 1 results
        debug = self._build_debug(result, prep_ms)

        # Send debug info as first event
        yield f"data: {json.dumps({'type': 'debug', 'debug': debug, 'character': character, 'character_period': character_period}, ensure_ascii=False)}\n\n"

        # Phase 2: Stream LLM response
        # Strategy:
        #   - Buffer until we see [RESPONSE] tag, then stream content after it
        #   - If no structured tags appear in first ~200 chars, stream everything directly
        prompt = result.get("prompt", "")
        full_response = ""
        streaming_mode = "detect"  # "detect" -> "buffered" or "direct"
        sent_length = 0

        for chunk in self.pipeline.generate_response_stream(prompt, character):
            full_response += chunk

            if streaming_mode == "detect":
                # Check if structured tags are appearing
                if "[RESPONSE]" in full_response:
                    streaming_mode = "buffered"
                    after_tag = full_response.split("[RESPONSE]", 1)[1].replace("[/RESPONSE]", "")
                    if after_tag:
                        sent_length = len(after_tag)
                        yield f"data: {json.dumps({'type': 'token', 'content': after_tag}, ensure_ascii=False)}\n\n"
                elif "[ANALYSIS]" in full_response or "[PREMISE" in full_response:
                    # Structured output detected, wait for [RESPONSE]
                    streaming_mode = "buffered"
                elif len(full_response) > 200:
                    # No tags after 200 chars - stream directly
                    streaming_mode = "direct"
                    yield f"data: {json.dumps({'type': 'token', 'content': full_response}, ensure_ascii=False)}\n\n"
                    sent_length = len(full_response)

            elif streaming_mode == "buffered":
                if "[RESPONSE]" in full_response:
                    after_tag = full_response.split("[RESPONSE]", 1)[1].replace("[/RESPONSE]", "")
                    new_content = after_tag[sent_length:]
                    if new_content:
                        sent_length = len(after_tag)
                        yield f"data: {json.dumps({'type': 'token', 'content': new_content}, ensure_ascii=False)}\n\n"

            elif streaming_mode == "direct":
                if chunk:
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

        # If still in detect mode (very short response), send everything
        if streaming_mode == "detect" and full_response:
            yield f"data: {json.dumps({'type': 'token', 'content': full_response.strip()}, ensure_ascii=False)}\n\n"

        # Final event with complete answer and updated timing
        total_ms = int((time.time() - start) * 1000)
        answer = self._extract_response(full_response)
        debug["time_ms"] = total_ms
        debug["generation"] = {
            "response_length": len(full_response),
            "has_response_tag": "[RESPONSE]" in full_response,
            "has_premise_check": "[PREMISE CHECK]" in full_response or "[PREMISE_CHECK]" in full_response,
            "has_analysis": "[ANALYSIS]" in full_response,
        }

        yield f"data: {json.dumps({'type': 'done', 'answer': answer, 'debug': debug}, ensure_ascii=False)}\n\n"

    def _extract_response(self, generated_response):
        """Extract text between [RESPONSE]...[/RESPONSE] tags."""
        if not generated_response:
            return ""
        # Case 1: Both opening and closing tags
        match = re.search(
            r"\[RESPONSE\](.*?)\[/RESPONSE\]", generated_response, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        # Case 2: Opening tag only (closing tag missing - LLM truncation)
        match = re.search(
            r"\[RESPONSE\](.*)", generated_response, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        return generated_response.strip()

    def _build_debug(self, result, elapsed_ms):
        """Build detailed debug information from pipeline result."""
        import torch

        log = result.get("log", {})
        pipeline_steps = log.get("pipeline_steps", {})

        # --- System info ---
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

        debug = {
            "time_ms": elapsed_ms,
            "system": {
                "gpu_available": gpu_available,
                "gpu_name": gpu_name,
                "navigator_type": type(self.pipeline.navigator).__name__,
                "chat_model": self.pipeline.chat_model,
                "generation_model": self.pipeline.generation_model,
                "top_k_chapters": self.pipeline.top_k_chapters,
                "memory_top_k": self.pipeline.memory_top_k,
                "knowledge_order": self.pipeline.knowledge_order,
            },
        }

        # --- Step 0: Navigator ---
        hyp_verification = pipeline_steps.get("hypothesis_verification", {})
        hypotheses = hyp_verification.get("navigator_hypotheses", [])
        nav = self.pipeline.navigator
        debug["navigator"] = {
            "hypotheses": [
                {
                    "chapter_id": h.get("chapter_id", ""),
                    "confidence": round(h.get("confidence", 0), 4),
                }
                for h in hypotheses[:5]
            ],
            "model_type": getattr(nav, 'model_type', 'unknown'),
            "num_classes": getattr(nav, 'num_classes', 0),
            "device": str(getattr(nav, 'device', 'unknown')),
        }

        # --- Step 0: Verifier ---
        verification = result.get("verification_result")
        if verification:
            debug["verifier"] = {
                "selected_scenes": [
                    {
                        "scene_id": s.get("scene_id", ""),
                        "chapter_id": s.get("chapter_id", ""),
                        "scene_title": s.get("scene_title", ""),
                    }
                    for s in verification.get("selected_scenes", [])
                ],
                "confidence": verification.get("confidence", 0),
                "premise_verified": verification.get("premise_verified", None),
                "reasoning": verification.get("reasoning", ""),
            }

        # --- Step 1: Character mapping ---
        char_mapping = pipeline_steps.get("character_mapping", {})
        debug["character_mapping"] = {
            "input_period": char_mapping.get("input_period", result.get("character_period", "")),
            "mapped_position": char_mapping.get("mapped_position", result.get("current_position", "")),
        }

        # --- Step 2: Question scene ---
        scene = result.get("question_scene")
        if scene:
            debug["question_scene"] = {
                "scene_id": scene.get("scene_id", ""),
                "scene_title": scene.get("scene_title", ""),
                "chapter_id": scene.get("chapter_id", ""),
                "combined_from": scene.get("combined_from", []),
            }

        # --- Step 3: Temporal classification ---
        debug["temporal"] = {
            "is_future": result.get("is_future_question", False),
            "question_period": result.get("question_period", ""),
            "current_position": result.get("current_position", ""),
        }

        # --- Step 4: Knowledge retrieval (detailed) ---
        selected = result.get("selected_knowledge", [])
        type_counts = {}
        knowledge_items = []
        for item in selected:
            t = item.get("type", item.get("knowledge_type", "unknown"))
            type_counts[t] = type_counts.get(t, 0) + 1
            knowledge_items.append({
                "type": t,
                "chapter_id": item.get("chapter_id", ""),
                "scene_id": item.get("scene_id", ""),
                "description": item.get("description", ""),
                "similarity": round(item.get("similarity_score", 0), 4),
            })

        # Sort by similarity descending
        knowledge_items.sort(key=lambda x: x["similarity"], reverse=True)

        debug["knowledge"] = {
            "total_count": len(selected),
            "memory_count": result.get("memory_knowledge_items", len(selected)),
            "type_distribution": type_counts,
            "items": knowledge_items,
        }

        # --- Step 5: Prompt stats ---
        prompt_info = pipeline_steps.get("prompt_building", {})
        debug["prompt"] = {
            "length_chars": prompt_info.get("prompt_length", len(result.get("prompt", ""))),
            "future_knowledge_items": prompt_info.get("future_knowledge_items", 0),
            "premise_verified": prompt_info.get("premise_verified", None),
        }

        # --- Step 6: Generation ---
        generated = result.get("generated_response", "")
        debug["generation"] = {
            "response_length": len(generated),
            "has_response_tag": "[RESPONSE]" in generated,
            "has_premise_check": "[PREMISE CHECK]" in generated or "[PREMISE_CHECK]" in generated,
            "has_analysis": "[ANALYSIS]" in generated,
        }

        return debug


# Global singleton
pipeline_manager = PipelineManager()
