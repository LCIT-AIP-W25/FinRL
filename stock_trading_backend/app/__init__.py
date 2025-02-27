from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS to allow frontend requests

    from app.routes import stock_bp  # Import the blueprint
    app.register_blueprint(stock_bp)  # Register the blueprint

    return app
