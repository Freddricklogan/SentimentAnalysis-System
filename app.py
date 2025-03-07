from flask import Flask, render_template, request, jsonify
import sentiment_analyzer
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    
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

if __name__ == '__main__':
    app.run(debug=True)
