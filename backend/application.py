from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from cbfs import cbfs  # Import the cbfs class from cbfs.py
from dotenv import load_dotenv
import os
import importlib

# Load environment variables
dotenv_path = 'KEYs.env'
_ = load_dotenv(dotenv_path)
  
app = Flask(__name__, static_folder='../frontend/build', static_url_path='') # Adjust this path as necessary
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize your Langchain-based class
langchain_instance = cbfs()

# Dedicated route for checking API status
@app.route('/api/status')
def api_status():
    return jsonify({"status": "API is working"}), 200

# Route to serve React App's static files in production
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/process', methods=['POST'])
def process_text():
    print("Received request at /process")
    print("Headers:", request.headers)  # Log the headers
    print("Raw Data:", request.data)  # Log the raw request data
    print("JSON Data:", request.json)  # Log the parsed JSON data

    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        if 'question' in data:
            response = langchain_instance.convchain(data['question'])
            print("API:PY:", response)
            print("API:PY:", jsonify(response))
            return jsonify(response), 200
        else:
            return jsonify({"error": "No question provided"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        langchain_instance.clr_history()
        return jsonify({"message": "History cleared"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reload_cbfs', methods=['POST'])
def reload_cbfs():
    try:
        importlib.reload(cbfs)  # Reload the cbfs module
        return jsonify({'message': 'cbfs module reloaded'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Default to 5000 if no environment variable
    app.run(debug=True, host='0.0.0.0', port=port)