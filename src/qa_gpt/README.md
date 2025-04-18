# Installation
1. add `./src//qa_gpt/chat/private_keys.py`
2. put pdf materials in a folder
3. python3.12 -m venv py_env
4. source py_env/bin/activate
5. python3.12 -m pip install -r requirements/requirments.txt

# Run
1. `streamlit run ./src//qa_gpt/script/QA_ui.py`

# Object design
1. DatabaseController
    * Interact with static storage.
    * Like `.pkl` file as simple local db
2. MaterialController
    * Depend on `DatabaseController`
    * Fetch materials like `pdf` files.
    * Generate materials like `output json` files.

3. PreprocessController
    * Preprocess input text
4. QAController
    * Depend on `PreprocessController`
    * Interact with GPT api

# Testing
1. Install test dependencies:
   ```bash
   pip install pytest
   ```

2. Run tests:
   ```bash
   # Run all tests
   python -m pytest src//qa_gpt/tests/ -v

   # Run specific test file
   python -m pytest src//qa_gpt/tests/material_controller_test.py -v

   # Run specific test
   python -m pytest src//qa_gpt/tests/material_controller_test.py -v -k "test_material_controller_initialization"

   # Run tests with print statements visible
   python -m pytest src//qa_gpt/tests/material_controller_test.py -v -s
   ```

3. Useful pytest options:
   - `-v`: Verbose output
   - `-s`: Show print statements
   - `-x`: Stop on first failure
   - `--pdb`: Drop into debugger on failures
   - `--tb=short`: Shorter traceback format
   - `--cov=src`: Generate coverage report (requires pytest-cov)

