from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from cbfs import cbfs  # Import the cbfs class from cbfs.py
from dotenv import load_dotenv
import os

# Load environment variables
dotenv_path = 'KEYs.env'
_ = load_dotenv(dotenv_path)

app = Flask(__name__, static_folder='build')
CORS(app)  # Enable CORS for all routes

# Initialize your Langchain-based class
langchain_instance = cbfs()

# Route to serve React App's static files in production
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
# def serve(path):
#     if os.environ.get('FLASK_ENV') == 'production':
#         if path != "" and os.path.exists(app.static_folder + '/' + path):
#             return send_from_directory(app.static_folder, path)
#         else:
#             return send_from_directory(app.static_folder, 'index.html')
#     else:
#         return "The API is working!"
    
def serve(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        print(app.static_folder)
        return send_from_directory(app.static_folder, path)
    else:
        print(f"Path requested: {path}")
        print(f"Full path: {os.path.join(app.static_folder, path)}")
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/process', methods=['POST'])
def process_text():
    print("Received request at /process")
    print(request.json)

    # print("TEST REQUEST:", request.data) 
    # Receive JSON data from the request
    data = request.json

    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # If 'question' is in the request, send it to the Langchain model
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

# Main entry point
if __name__ == '__main__':
    # On Elastic Beanstalk, 'FLASK_ENV' should be set to 'production'
    # Get the port number from the environment variable
    # Default to 5000 if not available
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True if os.environ.get('FLASK_ENV') == 'production' else False)
