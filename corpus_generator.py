import json
import re
from collections import defaultdict
from datasets import load_dataset
from tqdm import tqdm
import os

os.environ["HF_TOKEN"] = "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"
# ==========================================================
# Configuration
# ==========================================================

TARGET_COUNTS = {
    "history": 2000,
    "geography": 2000,
    "science": 2000,
    "politics": 2000,
    "culture": 2000,
    "technology": 2000,
    "literature": 2000,
}

MIN_LENGTH = 300
MAX_LENGTH = 2000

OUTPUT_FILE = "french_wikipedia_corpus.jsonl"

# ==========================================================
# Topic Keywords
# ==========================================================

TOPIC_KEYWORDS = {
    "history": [
        "histoire",
        "empire",
        "royaume",
        "guerre",
        "révolution",
        "dynastie",
        "bataille",
        "historique",
        "médiéval",
        "antiquité",
        "colonisation",
        "indépendance"
    ],

    "geography": [
        "géographie",
        "fleuve",
        "rivière",
        "montagne",
        "océan",
        "mer",
        "climat",
        "pays",
        "ville",
        "région",
        "désert",
        "continent"
    ],

    "science": [
        "science",
        "physique",
        "chimie",
        "biologie",
        "mathématiques",
        "astronomie",
        "génétique",
        "écologie",
        "laboratoire",
        "cellule",
        "molécule"
    ],

    "politics": [
        "politique",
        "président",
        "gouvernement",
        "élection",
        "parlement",
        "député",
        "ministre",
        "constitution",
        "parti politique",
        "république"
    ],

    "culture": [
        "culture",
        "tradition",
        "musique",
        "festival",
        "danse",
        "religion",
        "patrimoine",
        "coutume",
        "art",
        "gastronomie"
    ],

    "technology": [
        "technologie",
        "informatique",
        "ordinateur",
        "logiciel",
        "internet",
        "intelligence artificielle",
        "robot",
        "numérique",
        "programmation",
        "réseau"
    ],

    "literature": [
        "littérature",
        "roman",
        "poésie",
        "poète",
        "écrivain",
        "auteur",
        "livre",
        "nouvelle",
        "théâtre",
        "édition"
    ]
}

# ==========================================================
# Helpers
# ==========================================================

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_paragraphs(text):
    paragraphs = []
    for p in text.split("\n"):
        p = clean_text(p)
        if len(p) < MIN_LENGTH:
            continue
        if len(p) > MAX_LENGTH:
            continue

        paragraphs.append(p)

    return paragraphs


def classify_topic(title, paragraph):
    content = (title + " " + paragraph).lower()
    scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in content:
                score += 1
        scores[topic] = score

    best_topic = max(scores, key=scores.get)
    if scores[best_topic] == 0:
        return None
    return best_topic

def topic_completed(samples):
    for topic, target in TARGET_COUNTS.items():
        if len(samples[topic]) < target:
            return False
    return True

# ==========================================================
# Load French Wikipedia
# ==========================================================

print("Loading French Wikipedia...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.fr")
print("Dataset loaded.")

# ==========================================================
# Collect Samples
# ==========================================================

samples = defaultdict(list)


for article in tqdm(wiki['train']):
    if topic_completed(samples):
        break

    title = article["title"]
    text = article["text"]
    paragraphs = split_into_paragraphs(text)

    for paragraph in paragraphs:
        topic = classify_topic(title, paragraph)
        if topic is None:
            continue
        if len(samples[topic]) >= TARGET_COUNTS[topic]:
            continue

        samples[topic].append({
            "topic": topic,
            "title": title,
            "text": paragraph
        })

#---- ENREGISTREMENT DU JEUX DE DONNEES
total = 0
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for topic, entries in samples.items():
        for entry in entries:
            f.write(
                json.dumps(entry, ensure_ascii=False) + "\n"
            )
            total += 1

print("\nDataset saved.")
print(f"Total samples: {total}")