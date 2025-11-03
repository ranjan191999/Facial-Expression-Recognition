# stage2/generate_reviews.py
"""
Simple synthetic review generator for each emotion label.
Usage:
    from stage2.generate_reviews import generate_review
    reviews = generate_review("happy", k=3)
"""
import random
from typing import List

EMOTION_TO_TEMPLATES = {
    "happy": [
        "Absolutely loved it! The experience made my day.",
        "Super satisfied — would recommend to everyone!",
        "Very happy with the service and results."
    ],
    "sad": [
        "Felt a bit let down; expected more.",
        "Left me feeling disappointed.",
        "Not what I hoped for; quite sad about it."
    ],
    "angry": [
        "Extremely frustrating — this needs to be fixed.",
        "Terrible experience; I wouldn’t use this again.",
        "Angry about the way this was handled."
    ],
    "surprise": [
        "Unexpectedly great — exceeded my expectations!",
        "Pleasant surprise; much better than I thought.",
        "Was not expecting this level of quality — pleasantly surprised."
    ],
    "neutral": [
        "It was okay; nothing special either way.",
        "Average experience with room for improvement.",
        "Neutral — neither good nor bad overall."
    ],
    "fear": [
        "I felt unsure and hesitant during the process.",
        "Made me anxious; not very reassuring.",
        "I was fearful about the safety and reliability."
    ],
    "disgust": [
        "Really off-putting; not for me.",
        "Found the experience unpleasant and distasteful.",
        "Disgusted by the lack of care and quality."
    ],
}

def generate_review(emotion: str, k: int = 1) -> List[str]:
    """
    Return k synthetic reviews for given emotion.
    emotion: string (case-insensitive)
    k: number of samples to return
    """
    emotion_key = emotion.lower()
    bank = EMOTION_TO_TEMPLATES.get(emotion_key, EMOTION_TO_TEMPLATES["neutral"])
    return [random.choice(bank) for _ in range(k)]

if __name__ == "__main__":
    # quick demo
    print(generate_review("happy", k=5))