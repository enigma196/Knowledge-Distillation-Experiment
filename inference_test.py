import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the directory where you saved the merged model
model_path = "qwen2-0.5b-distilled-final-4004"

# 1. Load the merged model and tokenizer
print("Loading merged model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    dtype=torch.float16,  # Use float16 for faster inference
    device_map="auto"           # Automatically uses GPU if available
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Define your test prompt
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

USER_PROMPT = f"""
    Génère 6 questions à partir du texte suivant.

    Texte :
    Bologhine possédait toutes les villes du Maghreb, il avait pour ordre de tuer tous les Zénètes, de ramasser l'impôt des Berbères sous l'emprise de l'épée. Ceci provoqua une marche de contestation de la part des autres tribus. Les Kutama devinrent jaloux des Zirides et la guerre éclata entre les deux tribus ; Mila et Sétif furent rasées par les Zirides. Les Omeyyades acceptèrent enfin d'aider les Zénètes à reconquérir leurs territoires, en particulier des Maghraoua. Bologhine ibn Ziri rebroussa chemin en voyant toute l'armée des Zénètes venue d'Andalousie par voie maritime qui s'installa à Ceuta. En 983, Bologhine ibn Ziri mourut. S'ensuivit une longue période de défaite pour les Zirides. Les Maghraouas regagnèrent leurs territoires et leur souveraineté dans le Maghreb central et dans l'Ouest grâce à Ziri Ibn Attia issue des Maghraouas. Toutes les villes du Centre jusqu'à Tanger redevinrent des villes Zénètes, y compris Alger.

    Catégories :
    multiple-choice, open
    """

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": USER_PROMPT}
]

# The tokenizer will format this into the exact ChatML structure the model expects
formatted_prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False,            # Returns a plain string instead of token IDs (good for checking)
    add_generation_prompt=True # Appends '<|im_start|>assistant\n' to tell the model it's its turn to talk
)

# 3. Format the prompt (If you used a specific chat template during training, use it here)
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)


# 4. Generate the response
# print("\nGenerating response...\n" + "="*40)
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024, 
        temperature=0.3, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# 5. Print the output
input_length = inputs.input_ids.shape[-1]
print('----------- RESPONSE -------')
response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
print(response)
# print("="*40)