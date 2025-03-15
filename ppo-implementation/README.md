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
  Running cells with **1,000,000 timesteps** may take approximately **5-10 minutes each**.  
  For testing functionality, consider reducing the number of timesteps.

---

Enjoy experimenting with the PPO implementation!
