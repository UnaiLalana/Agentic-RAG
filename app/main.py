from flask import Flask
from flask_cors import CORS

from database import init_db


def create_app():
    """Flask application factory."""
    app = Flask(__name__)
    CORS(app)

    # Initialize metadata DB
    init_db()

    # Register blueprints
    from routes.health import health_bp
    from routes.documents import documents_bp
    from routes.query import query_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(query_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
