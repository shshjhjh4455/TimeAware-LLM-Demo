"""Flask API server for TimeAware character role-playing."""

import sys
import logging
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from webapp.backend.config import (
    FLASK_HOST,
    FLASK_PORT,
    FLASK_DEBUG,
    CHARACTER_CONFIG,
)
from webapp.backend.pipeline_manager import pipeline_manager

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


@app.route("/api/health", methods=["GET"])
def health():
    """Health check + pipeline status."""
    return jsonify({
        "status": "ok" if pipeline_manager.is_ready else "initializing",
        "pipeline_ready": pipeline_manager.is_ready,
        "error": pipeline_manager.error,
    })


@app.route("/api/characters", methods=["GET"])
def characters():
    """Return available characters and their time periods."""
    result = []
    for name, config in CHARACTER_CONFIG.items():
        result.append({
            "name": name,
            "avatar": config["avatar"],
            "periods": config["periods"],
        })
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main inference endpoint."""
    if not pipeline_manager.is_ready:
        return jsonify({"error": "Pipeline is still initializing. Please wait."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    character = data.get("character")
    character_period = data.get("character_period")
    question = data.get("question")
    include_debug = data.get("include_debug", True)

    if not all([character, character_period, question]):
        return jsonify({
            "error": "Missing required fields: character, character_period, question"
        }), 400

    if character not in CHARACTER_CONFIG:
        return jsonify({"error": f"Unknown character: {character}"}), 400

    try:
        result = pipeline_manager.process_query(character, character_period, question)
        if not include_debug:
            result.pop("debug", None)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": f"Pipeline error: {str(e)}"}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Streaming inference endpoint using Server-Sent Events."""
    if not pipeline_manager.is_ready:
        return jsonify({"error": "Pipeline is still initializing. Please wait."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body required"}), 400

    character = data.get("character")
    character_period = data.get("character_period")
    question = data.get("question")

    if not all([character, character_period, question]):
        return jsonify({
            "error": "Missing required fields: character, character_period, question"
        }), 400

    if character not in CHARACTER_CONFIG:
        return jsonify({"error": f"Unknown character: {character}"}), 400

    def generate():
        try:
            for event in pipeline_manager.process_query_stream(
                character, character_period, question
            ):
                yield event
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            import json
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def create_app():
    """Application factory for gunicorn."""
    logger.info("Starting pipeline initialization...")
    pipeline_manager.initialize()
    logger.info("Pipeline ready. Starting Flask server.")
    return app


if __name__ == "__main__":
    pipeline_manager.initialize()
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
