# PaperQAGen

# Installation

```
conda create -n PaperQAGen python=3.12.10
conda activate PaperQAGen
```

```
pip install -r requirements/requirments.txt
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

模型下载

python download_models.py
python脚本会自动下载模型文件并配置好配置文件中的模型目录，配置文件可以在用户目录中找到，文件名为magic-pdf.json

windows的【用户目录】为 "C:\Users\用户名", linux【用户目录】为 "/home/用户名"

修改【用户目录】配置文件magic-pdf.json中"device-mode"的值来启用CUDA

{
    "device-mode":"cuda"
}




# Launch
1. streamlit run ./src/qa_gpt/script/QA_ui.py

# TODO
1. Use rag to get relative indexes for summaries and question generation.
2. Include corresponding tables with the selected indexes
3. Daily upload limit
4. Leave comments on questions


# Future roadmap
1. Allow users to customize the question sets and summaries. (Adding, updating, modifying, deleting)
