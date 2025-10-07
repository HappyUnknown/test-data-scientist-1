import subprocess
import sys
import json
import os
import re
from typing import List, Dict, Any

SCRIPT_NAME = "ner-test.py" 
CV_SCRIPT_NAME = "cv-test.py"

def run_ner_script(text_input: str) -> Any:
    """
    Executes the NER script (SCRIPT_NAME) and captures its JSON result.
    
    Args:
        text_input: The text to pass as the FIRST command-line argument.
        image_path: The image path is kept here for function signature consistency 
                    but is NOT passed to the NER script.
        
    Returns:
        The captured and parsed JSON result (list/dict) or None on failure.
    """
    
    print(f"🚀 Starting script '{SCRIPT_NAME}' with input: '{text_input}'")
    print("-" * 50)
    
    try:
        # only pass text_input to the NER subprocess
        command = [sys.executable, SCRIPT_NAME, text_input] 

        # executing external script
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE, # output logs
            stderr=subprocess.PIPE, # output exceptions
            text=True # decode stdout/stderr as text
        )

        # printing logs to separate stream (crucial for debugging)
        if result.stderr:
            print(result.stderr, file=sys.stderr, end='') 

        # saving stdout clean from control characters
        raw_output = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', result.stdout).strip()
        
        # json-decoding of a given class
        try:
            if not raw_output:
                # showing the input text again to help diagnose a model miss
                print(f"[WARNING] Script succeeded but returned empty output on STDOUT for input: '{text_input}'.")
                return None
            
            # find the first index of '{' and '['
            idx_curly = raw_output.find('{')
            idx_square = raw_output.find('[')
            
            # collect all valid starting indices
            valid_starts = [i for i in [idx_curly, idx_square] if i != -1] # -1 instead of "not found"
            
            if valid_starts:
                # take the earliest starting point
                json_start = min(valid_starts)
            else:
                json_start = -1

            if json_start > 0:
                print(f"[WARNING] Trimming {json_start} leading characters from output. Found JSON at index {json_start}", file=sys.stderr)
                raw_output = raw_output[json_start:]
            elif json_start == -1:
                # if no json start is found anywhere, raise an error
                raise json.JSONDecodeError(f"Output does not contain starting JSON marker. Raw output: {raw_output}", raw_output, 0)

            captured_result = json.loads(raw_output)
            return captured_result
            
        except json.JSONDecodeError as json_e:
            # this catches the case where the output wasn't valid json
            print(f"[ERROR] Failed to decode JSON result. JSONDecodeError: {json_e}", file=sys.stderr)
            print(f"Raw output causing error:\n{raw_output}", file=sys.stderr)
            return None

    except subprocess.CalledProcessError as e:
        # error for subprocess cannot be called
        print("\n" + "❌" * 25, file=sys.stderr)
        print(f"SCRIPT FAILED (Exit Code {e.returncode})", file=sys.stderr)
        # print the logs captured from the subprocess's STDERR on failure
        if e.stderr:
            print("--- Subprocess STDERR Logs ---", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        print(f"A non-zero exit code was returned by {SCRIPT_NAME}.", file=sys.stderr)
        print("❌" * 25, file=sys.stderr)
        return None
        
    except FileNotFoundError:
        # wrong path error procession
        print(f"\n[FATAL ERROR] Python executable or script '{SCRIPT_NAME}' not found.", file=sys.stderr)
        return "NER Failed (Script Not Found)"


def run_cv_script(image_path: str) -> str:
    """
    Executes CV_SCRIPT_NAME to get the classification result.
    
    Args:
        image_path: The path to the image file to be classified.
        
    Returns:
        The classified class name as a string, or an error message.
    """
    
    print(f"\n📸 Starting CV script '{CV_SCRIPT_NAME}' with image: '{image_path}'")
    print("=" * 50)
    
    try:
        # CV script only needs the image path from terminal
        command = [sys.executable, CV_SCRIPT_NAME, image_path]
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # print the log messages captured on STDERR
        if result.stderr:
            print(result.stderr, file=sys.stderr, end='') 

        # the CV script only returns a single raw string
        raw_output = result.stdout.strip()
        if not raw_output:
            print("[CV WARNING] Script succeeded but returned empty output.")
            return "Unknown" # default class if classification fails silently
        return raw_output.strip()

    except subprocess.CalledProcessError as e:
        # error for subprocess cannot be called
        print("\n" + "❌" * 25, file=sys.stderr)
        print(f"CV SCRIPT FAILED (Exit Code {e.returncode})", file=sys.stderr)
        if e.stderr:
            print("--- Subprocess STDERR Logs ---", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        print("❌" * 25, file=sys.stderr)
        return "Classification Failed"
        
    except FileNotFoundError:
        # wrong path error procession
        print(f"\n[FATAL ERROR] CV script '{CV_SCRIPT_NAME}' not found.", file=sys.stderr)
        return "Classification Failed (Script Not Found)"

if __name__ == "__main__":
    
    # required arguments are script name + 2 other arguments, with total 3
    if len(sys.argv) < 3:
        print(f"Error: Missing required arguments.")
        print(f"Usage: python {sys.argv[0]} \"<TEXT_INPUT>\" \"<IMAGE_PATH>\"")
        sys.exit(1)


    # preparing arguments
    INPUT_TEXT = sys.argv[1] 
    IMAGE_PATH = sys.argv[2] 
    
    # check subscript file existance
    if not os.path.exists(SCRIPT_NAME):
        print(f"Error: The NER script '{SCRIPT_NAME}' was not found.")
        sys.exit(1)
    if not os.path.exists(CV_SCRIPT_NAME):
        print(f"Error: The CV script '{CV_SCRIPT_NAME}' was not found.")
        sys.exit(1)

    # image classifcation result obtained from the function
    classified_class = run_cv_script(IMAGE_PATH) 
    
    # same is for text classifier - class obtained from function
    final_data = run_ner_script(INPUT_TEXT) 
    print(f"final_data: {final_data}")

    # NER conditional result log
    print("\n" + "-"*50)
    if final_data is not None:
        print("✅ MAIN SCRIPT: Successfully captured and processed NER data.")
        
        print("\n--- Captured NER Subprocess Result ---")
        # conditional printing for several data types
        if isinstance(final_data, dict) or isinstance(final_data, list):
            print("Captured Result (Python Object/JSON):")
            print(json.dumps(final_data, indent=4)) # unparse + print of NER classification data
        else:
            print(f"Captured Raw String: {final_data}")    
    else:
        print("⚠️ MAIN SCRIPT: Execution failed or returned no NER data.")
    print("="*50)

    # model result comparison
    print("\n" + "-"*50)
    print("🔬 FINAL CROSS-MODEL COMPARISON 🔬")
    print("-" * 50)
    
    # getting sole classnames from NER result
    ner_words: List[str] = []
    if final_data and (isinstance(final_data, list) or isinstance(final_data, dict)):
        data_list = final_data if isinstance(final_data, list) else [final_data]
        
        # extract the 'word' value for every entity 
        ner_words = [
            el['word'].lower() 
            for el in data_list 
            if isinstance(el, dict) and 'word' in el
        ]

    # matching cases of CV and NER to lower 
    cv_class_lower = classified_class.lower()
    print(f"CV Classification Class: {cv_class_lower}")

    # determine the comparison result
    comparison_result = "MISMATCH"
    # if negative keywords present
    if "failed" in cv_class_lower or "unknown" in cv_class_lower:
        # CV classification considered as failed
        print("RESULT: Cannot compare. CV Classification failed or returned unknown.")
        comparison_result = "CV_FAIL"
    # if loaded ner_words list is empty
    elif not ner_words:
        # NER classification considered as failed
        print("RESULT: Cannot compare. NER model found no entities in the text.")
        comparison_result = "NER_FAIL"
    # if any match for CV class found in NER list 
    elif any(cv_class_lower == word for word in ner_words):
        # log sucess
        print(f"✅ EXACT MATCH FOUND: Image class '{classified_class}' matches an entity in the text.")
        comparison_result = "MATCH"
    # if CV class contained in some NER list element or NER class in some CV class
    elif any(cv_class_lower in word for word in ner_words) or any(word in cv_class_lower for word in ner_words):
        # log relative sucess
        print(f"✅ BROAD MATCH FOUND: Image class ('{classified_class}') is broadly mentioned (e.g., 'dog' vs 'sheepdog').")
        comparison_result = "MATCH"
    else:
    # if no entries among forseen conditions - it is a mismatch
        print(f"❌ MISMATCH: Classified image ('{classified_class}') not found in NER entities.")
        
    
    print("-" * 50)
    print(f"Input Text Entities: {ner_words if ner_words else 'None Found'}")
    print(f"Image Classification: {classified_class}")
    print(f"Final Status: {comparison_result}")

    sys.exit(0)