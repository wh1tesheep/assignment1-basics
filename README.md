# CS336 2025 年春季作业 1：基础知识

有关本次作业的完整说明，请参阅作业讲义：
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

如果你发现作业讲义或代码中存在任何问题，欢迎提交 GitHub Issue，或创建包含修复内容的 Pull Request。

## 环境配置

### 环境
我们使用 `uv` 管理环境，以确保环境可复现、可移植且易于使用。
建议按照[这里](https://github.com/astral-sh/uv#installation)的说明安装 `uv`，也可以运行 `pip install uv` 或 `brew install uv`。
建议阅读[这里](https://docs.astral.sh/uv/guides/projects/#managing-dependencies)的 `uv` 项目管理简介（你不会后悔的！）。

现在，你可以使用以下命令运行仓库中的任意代码：
```sh
uv run <python_file_path>
```
`uv` 会在需要时自动解析并激活环境。

### 运行单元测试


```sh
uv run pytest
```

最初，所有测试都应因抛出 `NotImplementedError` 而失败。
要将你的实现接入测试，请完成 [./tests/adapters.py](./tests/adapters.py) 中的函数。

### 下载数据
下载 TinyStories 数据以及 OpenWebText 的一个子样本：

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
