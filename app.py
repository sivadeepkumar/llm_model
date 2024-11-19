from flask import Flask
from dotenv import load_dotenv
from flask_cors import CORS 
import logging
from flask import Flask, request, jsonify
import requests
import os

from ai_models.open_ai.routes import open_ai as open_ai_path
from ai_models.amazon.routes import aws_ai as amazon_path
from ai_models.meta.llama_routes import  llama_ai as meta_path

logging.basicConfig(filename='./logs/base.log', level=logging.INFO)
logger = logging.getLogger(__name__)
ai_app = Flask(__name__)
CORS(ai_app)


ai_app.register_blueprint(amazon_path, url_prefix='/amazon_model')
ai_app.register_blueprint(meta_path, url_prefix='/meta')
ai_app.register_blueprint(open_ai_path, url_prefix='/open_ai_model')


@ai_app.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200




if __name__ == '__main__':
    ai_app.run(port= 5000)



