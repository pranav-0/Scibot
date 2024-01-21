from flask import Flask, request, jsonify
from bot_logic import BotLogic

app = Flask(__name__)
bot_logic = BotLogic()

@app.route('/execute', methods=['POST'])
def execute():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Ensure the 'text' key is present in the request
        if 'text' not in data:
            return jsonify({'error': 'Missing "text" parameter'}), 400

        # Get the text from the 'text' parameter
        input_text = data['text']

        # Process the input using BotLogic
        response = bot_logic.generate_response(input_text)

        # Return the response in JSON format
        return jsonify({'response': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=8000)
