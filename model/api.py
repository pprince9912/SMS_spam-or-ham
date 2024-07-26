from flask import Flask, request, jsonify
from model import preprocess_text, log_naive_bayes, word_frequency, class_frequency

app = Flask(__name__)

@app.route('/check', methods=['POST'])
def spam_checker():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON provided'}), 400
        
        input_sms = data.get('sms', '')
        
        if not input_sms:
            return jsonify({'error': 'No SMS provided'}), 400
        
        input_sms = preprocess_text(input_sms)
        prediction = log_naive_bayes(input_sms, word_frequency, class_frequency)
        
        result = "spam" if prediction == 1 else "not spam"
        
        return jsonify({'result': result})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

if __name__ == '__main__':
    app.run(debug=True)
