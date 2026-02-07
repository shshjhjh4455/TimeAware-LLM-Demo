#!/usr/bin/env python3
"""
Scene Classifier (Navigator) - Top-k Chapter Prediction
Implements 'Hypothesis Generation' from paper (Section 2.1)
"""

import torch
import json
from pathlib import Path
from typing import List, Dict, Any
import logging
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SceneNavigator:
    """
    RoBERTa-based scene classification with Top-k hypothesis generation

    Purpose:
        Predicts Top-k most likely chapters for a given question.
        Handles imperfect accuracy (60-80%) by returning multiple hypotheses
        for downstream Verifier to select from.

    Supports:
        - Full Fine-Tuned models (roberta_200q_best_*.pt)
        - LoRA adapters (for future comparison)
    """

    def __init__(
        self,
        model_type: str = "full_ft",
        model_path: str = None,
        mapping_file: str = "knowledge_enriched_rag/chapter_to_label_mapping_200q.json",
        device: str = None
    ):
        """
        Args:
            model_type: "full_ft" or "lora"
            model_path: Path to model weights (auto-detect if None)
            mapping_file: Chapter-to-label mapping JSON
            device: "cuda" or "cpu" (auto-detect if None)
        """
        self.model_type = model_type
        # Use CPU for now to avoid OOM (LoRA training using GPU)
        self.device = device or "cpu"

        logger.info(f"🚀 Initializing Scene Navigator ({model_type})")
        logger.info(f"   Device: {self.device}")

        # Load tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

        # Load label mapping
        self._load_mapping(mapping_file)

        # Load model
        if model_path is None:
            model_path = self._auto_detect_model()

        self._load_model(model_path)

        logger.info(f"✅ Navigator ready: {self.num_classes} chapters")

    def _load_mapping(self, mapping_file: str):
        """Load chapter-to-label mapping"""

        mapping_path = Path(mapping_file)
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)

        self.chapter_to_label = mapping_data['chapter_to_label']
        self.label_to_chapter = mapping_data['label_to_chapter']
        self.num_classes = len(self.chapter_to_label)

        logger.info(f"📋 Loaded mapping: {self.num_classes} chapters")

    def _auto_detect_model(self) -> str:
        """Auto-detect latest trained model"""

        models_dir = Path("knowledge_enriched_rag/models")

        if self.model_type == "full_ft":
            # Find best full fine-tuned model
            model_files = sorted(models_dir.glob("roberta_200q_best_*.pt"))
            if not model_files:
                raise FileNotFoundError("No Full FT 200q model found")
            model_path = model_files[-1]
            logger.info(f"📁 Auto-detected Full FT model: {model_path.name}")

        elif self.model_type == "lora":
            # Find best LoRA adapter
            lora_dirs = sorted(models_dir.glob("lora_adapters/best_adapter_*"))
            if not lora_dirs:
                raise FileNotFoundError("No LoRA adapter found")
            model_path = lora_dirs[-1]
            logger.info(f"📁 Auto-detected LoRA adapter: {model_path.name}")

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return str(model_path)

    def _load_model(self, model_path: str):
        """Load model weights"""

        if self.model_type == "full_ft":
            # Load full fine-tuned model
            self.model = RobertaForSequenceClassification.from_pretrained(
                'roberta-large',
                num_labels=self.num_classes
            )

            # Load trained weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

            logger.info(f"✅ Loaded Full FT model from: {Path(model_path).name}")

        elif self.model_type == "lora":
            # Load LoRA adapter
            from peft import PeftModel

            # Load base model
            base_model = RobertaForSequenceClassification.from_pretrained(
                'roberta-large',
                num_labels=self.num_classes
            )

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, model_path)

            logger.info(f"✅ Loaded LoRA adapter from: {Path(model_path).name}")

        self.model.to(self.device)
        self.model.eval()

    def predict_top_k(
        self,
        question: str,
        k: int = 3,
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict Top-k most likely chapters for the question

        Args:
            question: User's question
            k: Number of top hypotheses to return
            return_probabilities: Include probability scores

        Returns:
            List of hypotheses sorted by confidence:
            [
                {
                    'hypothesis_id': 0,
                    'chapter_id': 'Book1-chapter4',
                    'confidence': 0.85,
                    'probability': 0.42  (if return_probabilities=True)
                },
                ...
            ]
        """

        # Tokenize
        inputs = self.tokenizer(
            question,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)[0]
        top_k_probs, top_k_indices = torch.topk(probabilities, k)

        # Build hypotheses
        hypotheses = []
        for i, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
            label = idx.item()
            chapter_id = self.label_to_chapter[str(label)]

            hypothesis = {
                'hypothesis_id': i,
                'chapter_id': chapter_id,
                'confidence': float(prob)
            }

            if return_probabilities:
                hypothesis['probability'] = float(prob)

            hypotheses.append(hypothesis)

        return hypotheses

    def predict_single(self, question: str) -> str:
        """
        Predict single most likely chapter (for baseline comparison)

        Returns:
            chapter_id: e.g., "Book1-chapter4"
        """
        hypotheses = self.predict_top_k(question, k=1)
        return hypotheses[0]['chapter_id']

    def batch_predict_top_k(
        self,
        questions: List[str],
        k: int = 3,
        batch_size: int = 32
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch prediction for efficiency

        Args:
            questions: List of questions
            k: Number of top hypotheses per question
            batch_size: Batch size for inference

        Returns:
            List of hypothesis lists (one per question)
        """

        all_hypotheses = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            # Get top-k for each question in batch
            probabilities = torch.softmax(logits, dim=1)

            for probs in probabilities:
                top_k_probs, top_k_indices = torch.topk(probs, k)

                hypotheses = []
                for j, (idx, prob) in enumerate(zip(top_k_indices, top_k_probs)):
                    label = idx.item()
                    chapter_id = self.label_to_chapter[str(label)]

                    hypotheses.append({
                        'hypothesis_id': j,
                        'chapter_id': chapter_id,
                        'confidence': float(prob)
                    })

                all_hypotheses.append(hypotheses)

        return all_hypotheses


def test_navigator():
    """Test Scene Navigator"""

    logger.info("🧪 Testing Scene Navigator...")

    # Initialize with Full FT model
    navigator = SceneNavigator(model_type="full_ft")

    # Test questions
    test_questions = [
        "Were you present when Hagrid told Harry the truth about his parents?",
        "What happened when Harry first met Hermione on the train?",
        "Describe the Sorting Hat ceremony."
    ]

    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Test {i}: {question}")

        # Get Top-3 hypotheses
        hypotheses = navigator.predict_top_k(question, k=3)

        logger.info(f"\n🔍 Top-3 Chapter Hypotheses:")
        for hyp in hypotheses:
            logger.info(f"   {hyp['hypothesis_id']}. {hyp['chapter_id']} "
                       f"(confidence: {hyp['confidence']:.3f})")


if __name__ == "__main__":
    test_navigator()
