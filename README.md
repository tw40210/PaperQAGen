# PaperQAGen

# Installation
1. pip install -r requirements/requirments.txt
2. (Develop) pre-commit install
3. touch src/qa_gpt/chat/private_keys.py  (put your openai key here as "openapi_key = {your openai key}")

# Launch
1. streamlit run ./src/qa_gpt/script/QA_ui.py

# TODO
1. Use rag to get relative indexes for summaries and question generation.
2. Include corresponding tables with the selected indexes
3. Daily upload limit
4. Leave comments on questions


# Future roadmap
1. Allow users to customize the question sets and summaries. (Adding, updating, modifying, deleting)
