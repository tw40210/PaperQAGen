# PaperQAGen

# Usage

1. Upload a paper pdf file and the system will automatically process it.
2. Select a specific summary and get flash ideas.
3. Select a question set according to your interest to testify your understanding.
4. Operations to execute like leaving comments or deleting a paper material.

![image](https://github.com/user-attachments/assets/764b40f2-5fc1-4845-83f8-1e6c62719a84)
![image](https://github.com/user-attachments/assets/01d3724c-e414-4da0-9d9c-a3884747be42)


# Installation

```
conda create -n PaperQAGen python=3.12.10
conda activate PaperQAGen
```

```
pip install -r requirements/requirements.txt
```

```
(Develop) pre-commit install
```

```
touch src/qa_gpt/chat/private_keys.py  (put your openai key here as "openapi_key = {your openai key}")
```

MinerU - PDF processor
```
pip install -U magic-pdf[full]==1.3.3 -i https://mirrors.aliyun.com/pypi/simple
```

Torch and CUDA (supporting 11.8/12.4/12.6)
```
pip install --force-reinstall torch "numpy<=2.1.1" --index-url https://download.pytorch.org/whl/cu124
```

Faiss
```
conda install -c conda-forge faiss-gpu
```

Model Download
```
python download_models.py
```

The Python script will automatically download the model files and set up the model directory in the configuration file. The configuration file can be found in the user directory, and its filename is magic-pdf.json.

On Windows, the user directory is "C:\Users\YourUsername"

On Linux, the user directory is "/home/YourUsername"

To enable CUDA, modify the "device-mode" value in the magic-pdf.json configuration file located in the user directory:

json
```
{
    "device-mode": "cuda"
}
```



# Launch
1. python -m streamlit run ./src/qa_gpt/script/QA_ui.py  (default port 8501)

# TODO
1. Use rag to get relative indexes for summaries and question generation.
2. Include corresponding tables with the selected indexes
3. Daily upload limit
4. Leave comments on questions


# Future roadmap
1. Allow users to customize the question sets and summaries. (Adding, updating, modifying, deleting)
