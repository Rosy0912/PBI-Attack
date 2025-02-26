### 1. Prepare the Environment

##### 1.1. Install the Required Packages

```bash
pip install -r requirements.txt
```

##### 1.2. Modify the Configuration File

(Using Minigpt-4 as an example)

Edit the fields in the `/eval_configs/minigpt4_eval.yaml` file and set the path to the model directory.

##### 1.3. API Configuration

Modify the API configuration in both `utils.py` and `/ConfigForJailBreak/jailbreak_config.py`.

##### 1.4. Download minigpt4

Download the `minigpt4` folder from the following link:

XXXX-4

### 2. Train the Prior Model

Run `image_init.py`, select the desired parameters, and generate the adversarial images.

Run `test_text.py` to obtain the adversarial text suffixes.

Note: Please ensure that you modify the data paths and other parameters before running the scripts.

### 3. Jailbreak Attack

##### 3.1. White-box Attack

Run `main.py` to perform the white-box interactive attack.

##### 3.2. Evaluation

Modify the path to choose the test dataset, then run `evaluate.py` for evaluation.

##### 3.3. Black-box Attack

Configure the port and API_KEY in `optimize.py`, then run it to perform the black-box attack.

Note: Please ensure that you modify the data paths and other parameters before running the scripts.

### 4. Web Visualization

Run `demo.py` and open the specified URL in your browser to visualize the results of the attack.