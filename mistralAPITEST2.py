import os
import json
import random
from mistralai.client import Mistral

# Configuration
OUTPUT_FILE = "generation_results_mistral01.jsonl"

# Initialize client once (outside the function)
client = Mistral(api_key="YOUR_MISTRAL_API_KEY_HERE")

SYSTEM_PROMPT = """
    Tu es un expert en pédagogie francophone.
    Ta tâche consiste à générer des questions à partir d'un texte source.

    Règles :
    - Produire uniquement du français.
    - Générer le nombre exact de questions demandé.
    - Mélanger questions ouvertes et questions à choix multiples.
    - Les distracteurs doivent être plausibles.
    - Fournir la réponse correcte.
    - Retourner uniquement un JSON valide.

    Format :

    [
    {
        "question": "...",
        "options": ["...", "...", "..."],
        "answer": "...",
        "category": "multiple-choice"
    },
    {
        "question": "...",
        "answer": "...",
        "category": "open"
    }
    ]
    """

def extract_json(text):
    # """Extract JSON from text, handling potential markdown or extra content."""
    try:
        # Try to find JSON array
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            json_str = text[start:end + 1]
            # Validate it's valid JSON
            json.loads(json_str)
            return json_str
    except json.JSONDecodeError:
        pass
    
    # Try to parse the entire text as JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        return None

def save_example(source_text, teacher_output, typeq, n_questions):
    # """Save generated example to JSONL file."""
    sample = {
        "input": source_text[:500] + "..." if len(source_text) > 500 else source_text,  # Truncate long texts
        "output": teacher_output,
        "type": typeq,
        "n_questions": n_questions
    }

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def generate_questions(source_text, n_questions, typeq):
    # """Generate questions using Mistral API."""
    user_prompt = f"""
        Génère des questions à partir du texte source fourni.
        Source: 
        {source_text}

        Nombre de questions:
        {n_questions}

        Catégories:
        {typeq}        
    """
    
    try:
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            stream=False,
            response_format={"type": "json_object"}
        )
    
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating questions: {e}")
        return None

def main():
    # """Main execution function."""
    # Get corpus from the JSONL file
    ffrench_corpus = "french_wikipedia_corpus.jsonl"
    corpus = []

    with open(ffrench_corpus, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    if "text" in data:
                        corpus.append(data["text"])
    print(f"Loaded {len(corpus)} texts from corpus.")
    
    # Process each text
    for i, text in enumerate(corpus, 1):
        if(i>8966):
            n_questions = random.choice([3, 5, 8, 10, 15])
            typeq = random.choice(["open", "multiple-choice", "open,multiple-choice"])
            
            print(f"\nProcessing text {i}/{len(corpus)}")
            print(f"Generating {n_questions} questions of type: {typeq}")
            
            # Generate questions
            response = generate_questions(text, n_questions, typeq)
            
            if response is None:
                print("Failed to generate questions, skipping...")
                continue
            
            # Extract JSON from response
            json_output = extract_json(response)
            
            if json_output is None:
                print("Invalid JSON response:")
                print(response[:500] + "..." if len(response) > 500 else response)
                print("Skipping...")
                continue
            
            # Save the example
            save_example(text, json_output, typeq, n_questions)
            print(f"Saved successfully!")

if __name__ == "__main__":
    main()