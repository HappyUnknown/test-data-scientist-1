import json
import random
from typing import List, Dict

# --- Configuration ---
# Your 10+ canonical animal classes used in your CV model
ANIMAL_CLASSES: List[str] = [
    'lion', 'tiger', 'leopard', 'elephant', 'bear',
    'cat', 'dog', 'cow', 'horse', 'zebra', 'penguin', 'snake',
    'polar bear', 'black panther' # Added multi-word animals for better coverage
]

# Common sentence templates a user might use
SENTENCE_TEMPLATES: List[str] = [
    "I see a {} in the picture.",
    "The photo shows a {}.",
    "There is definitely a {} visible.",
    "Can you verify if this is a {}?",
    "I think I see a {} in the photo.",
    "A {} is present here.",
    "Is this a {}?",
    "A {} is running in the field.",
    "The {} is sleeping by the water.",
    "I saw a huge {} near the tree.",
]

# Common context words that should NOT be classified as animals (Negative Samples/Distractors)
CONTEXT_WORDS: List[str] = [
    'field', 'mountain', 'water', 'sky', 'tree', 'bush',
    'rock', 'house', 'car', 'person', 'boat', 'cloud', 'grass',
    'seem', 'now', 'bow', 'mat', 'sleep'
]

# Templates for negative samples (sentences with no animal, only scenery/objects)
NEGATIVE_SENTENCE_TEMPLATES: List[str] = [
    "The picture shows a {} and a {}.",
    "I can only see the {} and the {} in the photo.",
    "Is this a photo of a {} near a {}?",
    "The setting is a {} with some {}.",
    "A {} is visible next to a large {}.",
]

OUTPUT_FILE: str = "configs/ner_training_data_enhanced.json"
NUM_SAMPLES_PER_CLASS: int = 150 # Reduced to balance with negative samples
NUM_NEGATIVE_SAMPLES: int = 500 # A good ratio of negative samples is essential

# --- Generator Function ---
def generate_ner_data_enhanced(
    classes: List[str],
    templates: List[str],
    neg_templates: List[str],
    scenery: List[str],
    num_pos_samples: int,
    num_neg_samples: int
) -> List[Dict]:
    """Generates synthetic NER data, including positive and negative samples."""

    # Define tags: 0=O (Outside), 1=B-ANIMAL (Begin ANIMAL), 2=I-ANIMAL (Inside ANIMAL)
    tag_map = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
    dataset = []

    # Helper to apply simple space tokenization and tag
    def tokenize_and_tag(sentence: str, animal: str = None) -> (List[str], List[int]):
        # Simple tokenization by space, removing trailing punctuation for simplicity
        tokens = [t.strip('.,?!') for t in sentence.split() if t.strip('.,?!')]
        ner_tags = [tag_map["O"]] * len(tokens)

        if animal:
            animal_tokens = animal.split()
            
            # Simple check for a match between the sentence tokens and the animal name tokens
            # This is a naive search but works for this synthetic data
            for i, token in enumerate(tokens):
                if token == animal_tokens[0]:
                    # Check if the subsequent tokens match the rest of the animal name
                    match = True
                    for j in range(1, len(animal_tokens)):
                        if i + j >= len(tokens) or tokens[i+j] != animal_tokens[j]:
                            match = False
                            break
                    
                    if match:
                        # Apply BIO tags
                        ner_tags[i] = tag_map["B-ANIMAL"]
                        for j in range(1, len(animal_tokens)):
                            ner_tags[i + j] = tag_map["I-ANIMAL"]
                        break # Tagged the animal, stop searching

        return tokens, ner_tags

    ## 1. Generate Positive Samples (with animal name) ##
    for animal in classes:
        for _ in range(num_pos_samples):
            # Select a template and replace the placeholder
            template = random.choice(templates)
            sentence = template.format(animal)
            
            # Tokenize and tag
            tokens, ner_tags = tokenize_and_tag(sentence, animal)
            
            # Add to dataset only if the animal was successfully tagged
            if tag_map["B-ANIMAL"] in ner_tags:
                dataset.append({"id": str(len(dataset)), "tokens": tokens, "ner_tags": ner_tags})
    
    ## 2. Generate Negative Samples (no animal name) ##
    for _ in range(num_neg_samples):
        # Select a negative template and two random scenery words
        neg_template = random.choice(neg_templates)
        s1, s2 = random.sample(scenery, 2)
        
        # Create the negative sentence
        sentence = neg_template.format(s1, s2)
        
        # Tokenize and tag (no animal to tag, so all tags will be 'O')
        tokens, ner_tags = tokenize_and_tag(sentence)
        
        # Add to dataset
        dataset.append({"id": str(len(dataset)), "tokens": tokens, "ner_tags": ner_tags})
        
    random.shuffle(dataset)
    return dataset

# --- Execution ---
if __name__ == "__main__":
    print(f"Generating enhanced NER dataset with {len(ANIMAL_CLASSES)} classes...")
    data = generate_ner_data_enhanced(
        ANIMAL_CLASSES, 
        SENTENCE_TEMPLATES, 
        NEGATIVE_SENTENCE_TEMPLATES, 
        CONTEXT_WORDS, 
        NUM_SAMPLES_PER_CLASS, 
        NUM_NEGATIVE_SAMPLES
    )
    
    # Save to JSON file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Successfully generated {len(data)} total samples and saved to {OUTPUT_FILE}")
    print(f"Dataset now includes both positive and negative samples for better discrimination.")