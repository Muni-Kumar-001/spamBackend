from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load models
model1 = joblib.load(os.path.join(os.path.dirname(__file__), "myModel.pkl"))
model2 = joblib.load(os.path.join(os.path.dirname(__file__), "mySVCModel.pkl"))

@app.route('/api/check_spam', methods=['POST'])
def check_spam():
    try:
        data = request.get_json()

        # Validate input
        if not data or 'rawdata' not in data:
            return jsonify({"error": "Missing 'rawdata' parameter"}), 400

        raw_data = data['rawdata']

        # Model Predictions
        prediction1 = model1.predict([raw_data])[0]
        prediction2 = model2.predict([raw_data])[0]

        # If any model predicts 'spam', return spam; otherwise, return ham
        result = "spam" if prediction1 == "spam" or prediction2 == "spam" else "ham"

        return jsonify({"answer": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
