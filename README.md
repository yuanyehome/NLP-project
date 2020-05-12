# AI引论大作业——对话系统

## requirements

- pytorch1.0+
- jieba
- tqdm
- pandas

## 数据来源

- 数据来自[中文数据集](https://github.com/codemayq/chinese_chatbot_corpus)
- 完整模型文件和预处理文件[下载链接](https://disk.pku.edu.cn:443/link/D1AD87E72C443AE147298AAD71FD6863)


## 目录结构

- `src/seq2seq`目录下为一个seq2seq模型的生成模型，运行`python main.py`即可交互操作；
- `src/transformer/*`，`src/train_transformer.py`，`src/transformer_main.py`为transformer模型主要文件；
