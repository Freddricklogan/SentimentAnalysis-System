# Sentiment Analyzer - Demonstration Version
# In a real implementation, this would use actual ML models
import random
import re
import json
import os

# Simulate model loading
print("Loading sentiment analysis models...")

def analyze(text):
    """
    Analyze the sentiment of the given text.
    Returns sentiment (positive, negative, neutral) and confidence score.
    
    This is a demo version that uses rules and randomization.
    In production, this would use a trained model like BERT.
    """
    # Simple rule-based demonstration
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'fantastic']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor', 'disappointing']
    
    # Count occurrences of positive and negative words
    positive_count = sum(1 for word in text.lower().split() if word in positive_words)
    negative_count = sum(1 for word in text.lower().split() if word in negative_words)
    
    # Determine sentiment based on word counts
    if positive_count > negative_count:
        # Add some randomness to confidence, but keep it high for positive text
        confidence = 0.7 + (random.random() * 0.25)
        return "positive", confidence
    elif negative_count > positive_count:
        # Add some randomness to confidence, but keep it high for negative text
        confidence = 0.7 + (random.random() * 0.25)
        return "negative", confidence
    else:
        # For neutral text, confidence is lower
        confidence = 0.4 + (random.random() * 0.3)
        return "neutral", confidence

def extract_aspects(text):
    """
    Extract aspects and their sentiments from the text.
    Returns a dictionary of aspects and their sentiments.
    
    This is a demo version that uses rules.
    In production, this would use a trained aspect extraction model.
    """
    # Define some common aspects for demonstration
    aspects = {
        'price': None,
        'quality': None,
        'service': None,
        'delivery': None,
        'usability': None
    }
    
    # Check for aspects in the text and assign sentiments
    for aspect in aspects.keys():
        if aspect in text.lower():
            # Get a window of 5 words around the aspect
            pattern = r'(\w+\s+){0,5}' + aspect + r'(\s+\w+){0,5}'
            match = re.search(pattern, text.lower())
            
            if match:
                aspect_context = match.group(0)
                # Check sentiment in this context
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'best', 'fantastic']
                negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'poor', 'disappointing']
                
                positive_count = sum(1 for word in aspect_context.split() if word in positive_words)
                negative_count = sum(1 for word in aspect_context.split() if word in negative_words)
                
                if positive_count > negative_count:
                    aspects[aspect] = "positive"
                elif negative_count > positive_count:
                    aspects[aspect] = "negative"
                else:
                    aspects[aspect] = "neutral"
    
    # Remove aspects not found in the text
    return {k: v for k, v in aspects.items() if v is not None}

def detect_emotions(text):
    """
    Detect emotions in the text.
    Returns a dictionary of emotions and their intensities.
    
    This is a demo version that uses rules.
    In production, this would use a trained emotion detection model.
    """
    # Define emotion keywords for demonstration
    emotion_keywords = {
        'joy': ['happy', 'joy', 'delighted', 'pleased', 'glad', 'enjoy', 'love'],
        'sadness': ['sad', 'unhappy', 'disappointed', 'upset', 'miserable', 'depressed'],
        'anger': ['angry', 'annoyed', 'frustrated', 'irritated', 'furious', 'outraged'],
        'fear': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous'],
        'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned']
    }
    
    # Initialize emotion intensities
    emotions = {emotion: 0.0 for emotion in emotion_keywords}
    
    # Check for emotion keywords in the text
    text_lower = text.lower()
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Increment emotion intensity for each occurrence
                emotions[emotion] += 0.2
                
    # Normalize emotions to be between 0 and 1
    for emotion in emotions:
        emotions[emotion] = min(emotions[emotion], 1.0)
        # Add a small random factor
        emotions[emotion] = round(emotions[emotion] + (random.random() * 0.1), 2)
    
    return emotions

# Simple test if run directly
if __name__ == "__main__":
    test_text = "I really love this product. The quality is excellent, but the price is a bit high."
    sentiment, confidence = analyze(test_text)
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")
    
    aspects = extract_aspects(test_text)
    print(f"Aspects: {json.dumps(aspects, indent=2)}")
    
    emotions = detect_emotions(test_text)
    print(f"Emotions: {json.dumps(emotions, indent=2)}")
