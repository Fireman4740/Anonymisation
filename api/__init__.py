"""API Flask initialization."""

__all__ = ["create_app"]


def create_app():
    """Create and configure Flask application."""
    from flask import Flask
    from .routes import register_routes
    
    app = Flask(__name__)
    register_routes(app)
    
    return app
