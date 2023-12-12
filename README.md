# LogBERT
This document provides instructions for setting up and running two Python scripts evaluate log level quality.

## Installation

#### Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

#### Setting Up the Environment
1. **Create a Virtual Environment**:
    - Open a terminal (Command Prompt or Bash).
    - Navigate to the project directory.
    - Run the following command to create a virtual environment:
      ```bash
      python -m venv venv
      ```
    - This will create a new directory `venv` in your project folder.

2. **Activate the Virtual Environment**:
    - On Windows, run:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS or Linux, run:
      ```bash
      source venv/bin/activate
      ```

3. **Install Dependencies**:
    - Ensure your virtual environment is active.
    - Install required packages by running:
      ```bash
      pip install pandas tensorflow transformers sklearn numpy
      ```

#### Running the Scripts
1. **Log Level Classification**:
    - Ensure the virtual environment is active and you are in the project directory.
    - Run the script using:
      ```bash
      python log_level.py
      ```

2. **Linguistic Structure Classification**:
    - Follow the same steps as for Script 1, but use `linguistic.py`:
      ```bash
      python linguistic.py
      ```

#### Deactivating the Virtual Environment
- Once you are done, you can deactivate the virtual environment by running:
  ```bash
  deactivate
  ```

## PDF Files
This repository includes two PDF files that complement the Python scripts and provide additional context and information about the research and methods used in this project.

1. **Research Paper (research_paper.pdf)**:
   - This file contains the in-depth research paper associated with the project.
   - It provides a detailed explanation of the methodologies, data analysis, results, and conclusions.
   - Ideal for readers who are looking for a comprehensive understanding of the research work.

2. **Overview (overview.pdf)**:
   - This file is a slide deck that offers a simple introduction to the topic.
   - It summarizes the key points and findings of the research paper.
   - Perfect for those who need a quick understanding of the project's objectives and outcomes.



