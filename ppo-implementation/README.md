# PPO Vanilla Implementation

This implementation can be run in two ways: **Local Machine** or **Google Colab**. Follow the steps below for your chosen environment.

---

## 1. Clone the Repository

- Clone the GitHub repository.
- Open the `PPO_vanilla.ipynb` notebook.

> **Note:** If running locally, ensure that your conda environment is using Python version **>= 3.9**.

---

## 2. Environment Setup

### Google Colab

- **Folder Path:**  
  Modify the `folder_path` variable to point to the appropriate folder in your Google Drive.

- **Configuration:**  
  Set the `running_on_colab` variable to `True`.

- **Requirements:**  
  When running the `requirements.txt` file, you might be prompted to restart the session. Once restarted, click **Run all cells** to proceed.

### Local Machine

- **Configuration:**  
  Set the `running_on_colab` variable to `False`.

- **Directory Change:**  
  Execute `%cd "local/folder/path"` (replace `"local/folder/path"` with the actual path of the folder that contains the repository files).

- **Package Updates:**  
  Even if a restart isn’t explicitly prompted after updating packages, please restart your session once to ensure that all updates are applied.

---

## 3. Running the Notebook

- **Execute All Cells:**  
  After the environment is properly set up, run all cells in the notebook.
- **Experiment:**  
  Feel free to play around with the parameters and experiment.

---

## 4. Performance Note

- **Timesteps Warning:**  
  Running cells with **1,000,000 timesteps** may take approximately **5-10 minutes each** depending on your hardware.
  For testing functionality, consider reducing the number of timesteps.

---


## 4. Delta Implementation
Open the `DeltaImplementation.ipynb` notebook.
- 

> **Note:** Separate files specifically for the delta implementation should be run instead - network_attn.py, ppo_attn.py, network_act.py. To run experiment 1, ppo_attn.py was run in conjunction with network_attn.py with the use 'use_attention' parameter enabled. To run experiment 2, ppo_attn.py was run in conjunction with network_act.py with the 'use_attention' parameter disabled. The notebook runs these experiments accordingly. 

---

Enjoy experimenting with the PPO implementation!
