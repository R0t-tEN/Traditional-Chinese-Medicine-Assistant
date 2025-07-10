我用夸克网盘分享了「t_out_model.pth」，点击链接即可保存。
# 链接：https://pan.quark.cn/s/6eae371e3c82
提取码：gd2F
# 数据来源：https://aistudio.baidu.com/datasetdetail/246739
# 分享者：梦泽ovo

以上为模型训练素材和用一个用ResNet-18训练了25轮的pth文件


# 以下为 app_run_with_Django_final_edition的介绍
# 百草学堂 - 智能中草药识别与学习平台

“百草学堂”是一个基于 PyTorch 和 Django 的现代化中草药学习平台。它集成了三大核心功能：一个可供查阅的**药材知识库**，一个先进的**AI智能识别工具**，以及一个寓教于乐的**“看图猜药材”趣味游戏**。


## ✨ 主要特性

-   **药材知识库**：以图文并茂的卡片形式展示已收录的中草药，支持关键词搜索。
-   **智能识别**：通过上传图片，利用深度学习模型快速识别中草药种类。
-   **趣味游戏**：通过“看图猜药材”的游戏模式，在娱乐中巩固中草药知识，并提供实时计分板。
-   **后台驱动**：知识库和游戏题库的内容均可通过强大的 Django Admin 后台进行添加、编辑和管理，无需修改代码。
-   **多模型支持**：后端识别引擎支持 `ResNet-18` 和 `DenseNet-121` 等多种模型，开发者可按需配置切换。
-   **可扩展的训练流程**：提供完整的训练脚本，方便开发者使用自定义数据集进行模型训练。

---

## 🚀 1. 快速上手指南

本指南将帮助你快速在本地运行“百草学堂”应用。

## 快速上手 (直接使用预训练模型)

如果你只想快速体验应用，并且项目已包含训练好的模型文件 (`.pth` 和 `.json`)，请按以下步骤操作。
注：本文件因为大小限制原因只包含了.json文件，需要手动补充.pth文件到app\app_run_with_Django\herb_project\ml_assets中，并命名为resnet18_herb_model.pth
（此处默认使用resnet18，如需要更改，请见readme_for_developer）


### 步骤 1.1: 环境配置

在开始之前，请确保你的系统已安装 Python (推荐 3.8+)。我们强烈建议使用虚拟环境。

1.  **创建并激活虚拟环境** (以 `conda` 为例):
    ```bash
    # 创建一个名为 herb_env 的新环境
    conda create -n herb_env python=3.9
    # 激活环境
    conda activate herb_env
    ```
2.  **进入 Django 项目目录**:
    ```bash
    cd app_run_with_Django
    ```

3.  **一键安装所有依赖**:
    在项目根目录下（app_run_with_Django_final_edition\app_run_with_Django） ，运行以下命令来安装所有必需的库：
    ```bash
    pip install -r requirements.txt
    ```

### 步骤 1.2: 启动应用



1.  **初始化数据库**:
    这是首次运行时必须的步骤，它会创建应用所需的数据库表。
    在app_run_with_Django_final_edition\app_run_with_Django\herb_project下运行
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

2.  **启动 Web 服务器**:
    ```bash
    python manage.py runserver
    ```

3.  **访问应用**:
    当终端显示服务器成功启动后，打开浏览器访问 **`http://127.0.0.1:8000/`**。你将看到“药材目录”页面。点击导航栏即可在“药材目录”、“智能识别”和“趣味游戏”之间切换。

---

## 📚 2. 内容管理员指南

本应用的所有核心内容（药材目录、游戏题库）都通过后台进行管理。

### 步骤 2.1: 创建管理员账号

如果你还没有管理员账号，请在终端中运行（确保位于 `herb_project` 目录）:
```bash
python manage.py createsuperuser
```
然后按照提示设置用户名和密码。

目前自带一个管理员账号
账号：rotten
密码：nettor


### 步骤 2.2: 登录后台

1.  确保你的 Web 服务器正在运行。
2.  访问后台地址：**`http://127.0.0.1:8000/admin/`**
3.  使用你创建的管理员账号登录。

### 步骤 2.3: 管理内容

登录后，你将看到两个主要的管理模块：

-   **中草药 (Herbs)**:
    -   点击“增加”来添加新的药材到**知识库**中。
    -   **拼音名称**: 必须与训练模型时使用的文件夹名一致，这是AI识别结果与目录信息关联的关键。
    -   **中文名称/介绍/示例图片**: 这些信息将直接显示在“药材目录”页面。

-   **游戏图片 (Game Images)**:
    -   点击“增加”来为**“看图猜药材”游戏**添加新题目。
    -   **关联药材**: 从下拉菜单中选择这张图片对应的正确答案（数据来源于你已添加的“中草药”列表）。
    -   **游戏图片**: 上传用于游戏的图片。你可以为同一种药材上传多张不同的图片，增加游戏的多样性。

---




## 3. 开发者指南：切换模型

本项目支持在 `ResNet-18` 和 `DenseNet-121` 之间切换。切换操作对最终用户透明，由开发者在后端完成。

### 步骤 3.1: 确保模型已训练

在切换前，请确保你想要使用的模型已经训练完毕。如果对应的模型文件不存在，请先参考 **[章节 4](#4-高级指南重新训练模型)** 进行训练。

### 步骤 3.2: 修改配置文件

1.  打开核心配置文件：
    `herb_project/ml_assets/config.py`

2.  找到 `ACTIVE_MODEL` 变量，将其值修改为你想要的模型名称。
    ```python
    # herb_project/ml_assets/config.py

    # 可选项: 'resnet18', 'densenet121'
    ACTIVE_MODEL = 'densenet121' # <-- 修改这里
    ```

### 步骤 3.3: 重启服务

**重要**：每次修改完配置文件后，都需要**重启 Django 服务器**才能让改动生效。在运行服务器的终端中按 `Ctrl+C` 停止服务，然后重新运行 `python manage.py runserver`。

---

## 4. 高级指南：重新训练模型

你可以使用自己的数据集来训练一个全新的模型，或对现有模型进行微调。

### 步骤 4.1: 准备数据集

1.  在项目根目录 (`total_code/`)下，创建一个名为 `dataset` 的文件夹。
2.  在 `dataset` 文件夹内，创建 `train` 和 `val` 两个子文件夹，分别用于存放训练集和验证集。
3.  在 `train` 和 `val` 文件夹内，为每一种中草药创建一个以其**拼音**命名的文件夹（例如 `gouqizi`, `jinyinhua`）。
4.  将对应类别的图片放入相应的拼音文件夹中。

最终目录结构应如下所示：

```
dataset/
├── train/
│   ├── gouqizi/
│   │   ├── 001.jpg
│   │   └── 002.jpg
│   └── jinyinhua/
│       ├── 001.jpg
│       └── 002.jpg
└── val/
    ├── gouqizi/
    │   ├── 003.jpg
    │   └── 004.jpg
    └── jinyinhua/
        ├── 003.jpg
        └── 004.jpg
```

### 步骤 4.2: 执行训练命令

在项目根目录 (`total_code/`)下，使用 `train.py` 脚本并指定要训练的模型。

-   **训练 ResNet-18**:
    ```bash
    python train.py --model resnet18
    ```

-   **训练 DenseNet-121**:
    ```bash
    python train.py --model densenet121
    ```

训练脚本会自动读取 `dataset` 文件夹中的数据，开始训练。训练完成后，最新的模型文件（如 `resnet18_herb_model.pth`）和类别文件 (`class_names.json`) 将保存在 `herb_project/ml_assets/` 目录下，并覆盖旧文件。

### 步骤 4.3: 加载新模型

训练完成后，请参考 **[章节 3](#3-开发者指南切换模型)** 的步骤来激活并加载你刚刚训练好的新模型。

---

## 5. 项目日志

项目运行期间的所有重要信息（如模型加载、图片识别等）都会被记录下来。

-   **控制台输出**：简化的日志信息会实时显示在运行服务器的终端中。
-   **日志文件**：更详细的日志保存在 `herb_project/logs/herb_project.log` 文件中，方便进行问题排查和分析。




# 如果已经有了训练好的pth文件，使用方法：
# 🚀 模型部署操作步骤

---

### **步骤 1: 🧩 确定你的 `.pth` 文件对应的模型架构**

**这是最重要的一步。**

> 你必须知道你这个 `.pth` 文件是用哪种模型训练的。
>
> -   它是用 `ResNet-18` 训练的吗？
> -   还是用 `DenseNet-121` 训练的？
>
> 如果把 `ResNet-18` 的权重加载到一个 `DenseNet-121` 的模型结构里（反之亦然），程序会因为网络层结构不匹配而立即报错崩溃。

---

### **步骤 2: 📁 重命名并放置 `.pth` 文件**

根据你在步骤 1 中确定的模型架构，将你的 `.pth` 文件重命名为项目约定的名称，并把它放到正确的文件夹里。

-   **目标文件夹**: `herb_project/ml_assets/`

-   **如果你的模型是 `ResNet-18`**:
    将你的 `.pth` 文件（例如叫 `my_final_model.pth`）重命名为 `resnet18_herb_model.pth`。

-   **如果你的模型是 `DenseNet-121`**:
    将你的 `.pth` 文件重命名为 `densenet121_herb_model.pth`。

然后，将这个重命名后的文件移动或复制到 `herb_project/ml_assets/` 文件夹下。

---

### **步骤 3: ✍️ 准备并放置 `class_names.json` 文件**

> 模型权重文件 (`.pth`) 自身并不包含类别的名称信息。它只知道输出一个包含163个数字的向量。`class_names.json` 文件就是用来告诉程序，这163个数字按顺序分别对应哪个中草药（比如 `ajiao`, `aiye`...）。

**⚠️注意：这个 `class_names.json` 文件必须与你的 `.pth` 文件在训练时使用的类别和顺序完全匹配！**

1.  找到与你的 `.pth` 文件配套的 `class_names.json` 文件。
2.  将这个 `class_names.json` 文件也移动或复制到 `herb_project/ml_assets/` 文件夹下。

> #### **如果你没有 `class_names.json` 文件怎么办？**
> 你必须根据你当初训练该 `.pth` 文件时所用的数据集来重新生成它。你可以使用我们之前讨论过的 `generate_classes.py` 脚本，将它指向你原来的数据集文件夹，来生成一个与训练时顺序一致的 `class_names.json` 文件。

---

### **步骤 4: ⚙️ 修改配置文件以激活模型**

现在，文件已经各就各位了，最后一步是告诉 Django 应用去加载它们。

1.  打开配置文件：`herb_project/ml_assets/config.py`
2.  找到 `ACTIVE_MODEL` 变量。
3.  根据你在步骤 1 和 2 中准备的文件，修改它的值。

    -   如果你准备的是 `resnet18_herb_model.pth`，就设置为:
        ```python
        ACTIVE_MODEL = 'resnet18'
        ```

    -   如果你准备的是 `densenet121_herb_model.pth`，就设置为:
        ```python
        ACTIVE_MODEL = 'densenet121'
        ```

---

### **步骤 5: ▶️ 启动应用**

所有配置都已完成！现在你可以像普通用户一样启动 Web 应用了。

1.  进入 `herb_project` 目录：
    ```bash
    cd herb_project
    ```

2.  启动服务器：
    ```bash
    python manage.py runserver
    ```

> 服务器启动时，它会读取 `config.py`，发现 `ACTIVE_MODEL` 被设置了，然后就会去 `ml_assets` 文件夹加载你刚刚放进去的 `.pth` 文件和 `class_names.json` 文件。

#
#
#
#




# 以下为app_run_with_Gradio 使用方法：
在虚拟环境中运行 app.py即可
