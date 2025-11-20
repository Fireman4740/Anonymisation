"""API Flask routes."""

from flask import request, jsonify
from src.core import anonymize_text


def register_routes(app):
    """Register all API routes."""
    
    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({"status": "ok"})
    
    @app.route("/", methods=["GET"])
    def metadata():
        """API metadata."""
        return jsonify({
            "name": "Anonymization API",
            "version": "2.0",
            "endpoints": {
                "health": "/health",
                "anonymize": "/anonymize"
            }
        })
    
    @app.route("/anonymize", methods=["POST"])
    def anonymize():
        """Anonymize text endpoint."""
        data = request.get_json()
        
        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        try:
            result = anonymize_text(
                value=data["text"],
                scope_id=data.get("scope_id"),
                secret_salt=data.get("secret_salt", "default_secret"),
                level=data.get("level", "L0"),
                ner_results=data.get("ner_results", []),
                openrouter_models=data.get("openrouter_models"),
                overrides=data.get("overrides"),
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
