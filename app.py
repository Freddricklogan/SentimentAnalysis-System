from flask import Flask, render_template, request, jsonify
import sentiment_analyzer

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    try:
        # Get sentiment analysis results
        sentiment, confidence = sentiment_analyzer.analyze(text)

        # Get aspect-based results (if text is long enough)
        aspects = {}
        if len(text.split()) > 10:
            aspects = sentiment_analyzer.extract_aspects(text)

        # Get emotion analysis
        emotions = sentiment_analyzer.detect_emotions(text)

        return jsonify({
            'sentiment': sentiment,
            'confidence': float(confidence),
            'aspects': aspects,
            'emotions': emotions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
