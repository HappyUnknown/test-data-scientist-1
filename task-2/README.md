# Test task 2

### About
###### This project implements an integrated pipeline to perform Named Entity Recognition (NER) on text and Computer Vision (CV) classification on an image, and then compares the results to find matches. The core script acts as an orchestrator, running two external Python scripts (ner-test.py and cv-test.py) as sub-processes and analyzing their output for conceptual alignment.

### Task and respective file name 
- Jupyter notebook with exploratory data analysis of your dataset => analysis.ipynb
- Parametrized train and inference .py files for the NER model => ner-train.py, ner-test.py
- Parametrized train and inference .py files for the Image Classification model => cv-train.py, cv-test.py
- Python script for the entire pipeline that takes 2 inputs (text and image) and provides 1 boolean value as an output => main.py


### Features
**Cross-Modal Verification**: Checks if the primary object class detected in an image (CV output) is mentioned as an entity within the accompanying text (NER output).
**Named Entity Recognition (NER)**: Executes an external script (ner-test.py) to extract and classify entities (e.g., names, locations, objects) from a given text input. Expected NER output is a JSON structure.
**Computer Vision (CV) Analysis**: Executes an external script (__cv-test.py__) to classify the content of a given image file. Expected CV output is a single classification string.
**Robust Logging**: Detailed logging to STDOUT and STDERR for easy debugging of subprocess failures, JSON parsing issues, and final match status.

### Requirements

Before running the application, ensure you have the following installed:
- Python 3.8+
- pip (Python package installer)

*⚠️ The project relies on the successful execution of two separate scripts: __ner-test.py__ and __cv-test.py__. These scripts must be present in the project root and configured to run your specific models.*

### Setup

*Process has already been described in __setup.ipynb__*
*Execute commands from __code__ directory*

###### 1. Download image dataset
Unzip this archive: https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/fk29shm2kn-2.zip

###### 2. Install requirements
`pip install -r requirements.txt`

###### 3. Generate NER dataset
`python .\ner-gen.py`

###### 4. Train NER model
`python .\ner-train.py ".\configs\ner_training_data_enhanced.json" "..\dataset\ner_animal_model"`

###### 5. Test NER performance
`python .\ner-test.py "Seems that I saw a lion"`

###### 6. Train CV model
`python .\cv-train.py ..\dataset\images ..\dataset\cv_animal_model`

###### 7. Test CV performance
`python .\cv-test.py "..\dataset\images\Sheep\test\images\sheep-closeup-eating-grass.jpg"`

###### 8. Run main.py to interact with the whole system
`python .\main.py "Seems that I saw a lion" "..\dataset\images\Sheep\train\images\Sheep (20).jpg"`

### Usage
*The main script must be called with two required arguments: the text to analyze and the path to the image file.*
Example:
Assume you have an image of a dog saved at `..\dataset\images\Sheep\test\images\sheep-closeup-eating-grass.jpg`

### Expected Output & Verification Status
The script will first execute cv-test.py, then ner-test.py, and finally log the cross-modal comparison:

| Status Code | Description |
| :--- | :--- |
| `MATCH` | The CV class (e.g., “dog”) was found as an entity in the NER text output. |
| `MISMATCH` | The CV class was successfully found, but no matching entity was found in the text. |
| `CV_FAIL` | The CV classification failed or returned “Unknown”. |
| `NER_FAIL` | The NER model found no entities in the text to compare against. |

### Project Structure
~~~
│   README.md
│
├───code
│   │   animal-image-dataset.zip
│   │   cv-test.py
│   │   cv-train.py
│   │   main-prev.py
│   │   main.ipynb
│   │   main.py
│   │   ner-gen.py
│   │   ner-test.py
│   │   ner-train.py
│   │   requirements.txt
│   │   setup.ipynb
│   │
│   ├───.ipynb_checkpoints
│   │       cv-test-checkpoint.py
│   │       cv-train-checkpoint.py
│   │       main-checkpoint.ipynb
│   │       main-checkpoint.py
│   │       main-prev-checkpoint.py
│   │       ner-gen-checkpoint.py
│   │       ner-test-checkpoint.py
│   │       ner-train-checkpoint.py
│   │
│   ├───configs
│   │       ner_training_data_enhanced.json
│   │
│   └───logs
│           events.out.tfevents.1759503621.DESKTOP-4IDAF0Q.22872.0
│           events.out.tfevents.1759503796.DESKTOP-4IDAF0Q.4704.0
│           events.out.tfevents.1759580730.DESKTOP-4IDAF0Q.9712.0
│           events.out.tfevents.1759746525.DESKTOP-4IDAF0Q.20008.0
│
└───dataset
    ├───cv_animal_model
    │       class_names.txt
    │       resnet18_best.pth
    │
    ├───images
    │   ├───Cat
    │   │   │   data.yaml
    │   │   │   README.dataset.txt
    │   │   │   README.roboflow.txt
    │   │   │
    │   │   └───train
    │   ├───Cow
    │   │   │   data.yaml
    │   │   │   README.dataset.txt
    │   │   │   README.roboflow.txt
    │   │   │
    │   │   └───train
    │   ├───Deer
    │   │   └───train
    │   ├───Dog
    │   │   │   data.yaml
    │   │   │   README.dataset.txt
    │   │   │   README.roboflow.txt
    │   │   │
    │   │   └───train
    │   ├───Goat
    │   │   │   data.yaml
    │   │   │   README.dataset.txt
    │   │   │   README.roboflow.txt
    │   │   │
    │   │   └───train
    │   ├───Hen
    │   │   │   data.yaml
    │   │   │   README.dataset.txt
    │   │   │   README.roboflow.txt
    │   │   │
    │   │   └───train
    │   ├───NightVision
    │   │   │   data.yaml
    │   │   │   README.dataset.txt
    │   │   │   README.roboflow.txt
    │   │   │
    │   │   └───train
    │   ├───Rabbit
    │   │   │   data.yaml
    │   │   │   README.dataset.txt
    │   │   │   README.roboflow.txt
    │   │   │
    │   │   └───train
    │   └───Sheep
    │       │   data.yaml
    │       │   README.dataset.txt
    │       │   README.roboflow.txt
    │       │
    │       ├───test
    │       └───train
    └───ner_animal_model
        │   config.json
        │   model.safetensors
        │   special_tokens_map.json
        │   tokenizer.json
        │   tokenizer_config.json
        │   training_args.bin
        │   vocab.txt
        │
        ├───checkpoint-147
        │       config.json
        │       model.safetensors
        │       optimizer.pt
        │       rng_state.pth
        │       scheduler.pt
        │       special_tokens_map.json
        │       tokenizer.json
        │       tokenizer_config.json
        │       trainer_state.json
        │       training_args.bin
        │       vocab.txt
        │
        ├───checkpoint-294
        │       config.json
        │       model.safetensors
        │       optimizer.pt
        │       rng_state.pth
        │       scheduler.pt
        │       special_tokens_map.json
        │       tokenizer.json
        │       tokenizer_config.json
        │       trainer_state.json
        │       training_args.bin
        │       vocab.txt
        │
        ├───checkpoint-441
        │       config.json
        │       model.safetensors
        │       optimizer.pt
        │       rng_state.pth
        │       scheduler.pt
        │       special_tokens_map.json
        │       tokenizer.json
        │       tokenizer_config.json
        │       trainer_state.json
        │       training_args.bin
        │       vocab.txt
        │
        └───logs
                ...
~~~