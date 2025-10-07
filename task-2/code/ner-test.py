import sys
import json
import os
import numpy as np 
from transformers import pipeline
import torch # Needed for device detection
from typing import List, Dict, Any

# configuration
MODEL_DIR = "../dataset/ner_animal_model" 

# global variable to hold the pipeline instance
ner_pipeline = None

def log(message):
    """Writes a log message to standard error."""
    print(message, file=sys.stderr, flush=True)

def json_default_serializer(obj):
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

# core logic \|/

def load_ner_pipeline(model_path: str):
    """Loads the fine-tuned NER model/pipeline from the specified path."""
    global ner_pipeline
    
    if ner_pipeline:
        return ner_pipeline
        
    log(f"[LOG] Attempting to load NER model from: {model_path}") 
    
    if not os.path.exists(model_path):
        log(f"[CRITICAL ERROR] Model directory not found: {model_path}")
        sys.exit(1)
        
    try:
        # 0: GPU, -1: CPU
        try:
            # safely attempt to determine device
            device = 0 if torch.cuda.is_available() else -1
        except Exception:
             # fallback if torch or CUDA check fails
             device = -1 
        
        ner_pipeline = pipeline(
            "ner",
            model=model_path,
            tokenizer=model_path,
            grouped_entities=True, # ensure the pipeline groups tokens into single words
            device=device
        )
        log("[LOG] NER pipeline loaded successfully.") 
        return ner_pipeline
        
    except Exception as e:
        log(f"[CRITICAL ERROR] Failed to initialize NER pipeline: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)


def classify_text(text_input: str, ner_pipeline: Any) -> List[Dict[str, Any]]:
    """Runs the NER model pipeline on the text input and processes the result."""
    log(f"[LOG] Analyzing text with model.")
    
    # run the classification
    results = ner_pipeline(text_input) 
    
    # process and filter the results to match the expected format
    processed_results = []
    for entity in results:
        processed_results.append({
            "entity": entity.get('entity_group', entity.get('entity')), 
            "word": entity['word'].strip(),
            "score": float(entity['score']), 
            "start": entity['start'],
            "end": entity['end']
        })
        
    return processed_results 

def main():
    log("[LOG] Script execution started (NER Mode).")
    
    # input argument quantitative restriction
    if len(sys.argv) != 2:
        log("\n[ERROR] Invalid number of arguments.")
        log("Usage: python ner-test.py \"<TEXT_TO_ANALYZE>\"")
        sys.exit(1)

    text_input = sys.argv[1]
    log(f"[LOG] Arguments received: Text='{text_input}'")

    try:
        # loading NER pipeline
        ner_pipeline_instance = load_ner_pipeline(MODEL_DIR)
        
        # classification
        final_result_data = classify_text(text_input, ner_pipeline_instance) 

        # output result through STDOUT buffer
        print(json.dumps(final_result_data, default=json_default_serializer), end='', flush=True) 
        
        log("[LOG] Final result written to STDOUT. Script exiting successfully.")
        
    except Exception as e:
        log(f"[CRITICAL ERROR] An unexpected error occurred in main: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()