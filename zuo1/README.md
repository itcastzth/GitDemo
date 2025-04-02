# 文本分类项目

## 代码核心功能说明

### 算法基础
本项目采用多项式朴素贝叶斯分类器（Multinomial Naive Bayes）进行文本分类。朴素贝叶斯分类器基于贝叶斯定理，假设特征之间相互独立。在文本分类任务中，每个词被视为一个特征，假设每个词的出现与其他词的出现相互独立。

#### 贝叶斯定理
贝叶斯定理的公式为：
\[ P(C|D) = \frac{P(D|C) \cdot P(C)}{P(D)} \]
其中：
- \( P(C|D) \) 是给定数据 \( D \) 的情况下类别 \( C \) 的后验概率。
- \( P(D|C) \) 是给定类别 \( C \) 的情况下数据 \( D \) 的似然概率。
- \( P(C) \) 是类别 \( C \) 的先验概率。
- \( P(D) \) 是数据 \( D \) 的边缘概率。

在文本分类中，我们通过计算每个类别的后验概率，选择概率最大的类别作为预测结果。

#### 特征独立性假设
多项式朴素贝叶斯假设每个特征（词）在给定类别的情况下是相互独立的。这意味着：
\[ P(D|C) = \prod_{i=1}^{n} P(t_i|C) \]
其中 \( t_i \) 是文本中的第 \( i \) 个词，\( n \) 是文本中词的总数。

#### 应用形式
在邮件分类中，我们通过计算每封邮件属于“垃圾邮件”或“普通邮件”的后验概率，选择概率较大的类别作为预测结果。具体步骤如下：
1. 计算每个类别的先验概率 \( P(C) \)。
2. 计算每个词在每个类别中的条件概率 \( P(t_i|C) \)。
3. 使用贝叶斯定理计算每封邮件属于每个类别的后验概率。
4. 选择后验概率最大的类别作为预测结果。

### 数据处理流程
1. **数据加载**：
   - 从指定目录加载文本文件，每个文件被视为一个样本。
   - 支持多种文件格式（如 `.txt`）。

2. **分词处理**：
   - 使用 `jieba` 或其他分词工具将文本分割成单词或短语。
   - 示例代码：
     ```python
     import jieba
     text = "这是一段示例文本，用于分词处理。"
     words = jieba.lcut(text)
     print(words)  # 输出：['这', '是', '一段', '示例', '文本', '，', '用于', '分词', '处理', '。']
     ```

3. **停用词过滤**：
   - 移除常见的停用词（如“的”“了”“是”等），以减少噪声。
   - 使用预定义的停用词列表：
     ```python
     stopwords = set(line.strip() for line in open('stopwords.txt', 'r', encoding='utf-8').readlines())
     filtered_words = [word for word in words if word not in stopwords]
     ```

4. **特征提取**：
   - 将文本转换为数值特征，支持两种方法：
     - **高频词特征**：统计每个词的出现频率。
     - **TF-IDF加权特征**：使用TF-IDF（词频-逆文档频率）对特征进行加权。

### 特征构建过程
1. **高频词特征**：
   - **数学表达**：词频（Term Frequency, TF）。
     \[ TF(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中的出现次数}}{\text{文档 } d \text{ 中的总词数}} \]
   - **实现**：使用 `CountVectorizer` 提取词频特征。
     ```python
     from sklearn.feature_extraction.text import CountVectorizer
     vectorizer = CountVectorizer(stop_words='english')
     X = vectorizer.fit_transform(texts)
     ```

2. **TF-IDF加权特征**：
   - **数学表达**：
     \[ TF-IDF(t, d) = TF(t, d) \cdot IDF(t) \]
     其中：
     - \( TF(t, d) \) 是词 \( t \) 在文档 \( d \) 中的词频。
     - \( IDF(t) = \log\left(\frac{N}{DF(t)}\right) \)，其中 \( N \) 是文档总数，\( DF(t) \) 是包含词 \( t \) 的文档数。
   - **实现**：使用 `TfidfVectorizer` 提取 TF-IDF 特征。
     ```python
     from sklearn.feature_extraction.text import TfidfVectorizer
     vectorizer = TfidfVectorizer(stop_words='english')
     X = vectorizer.fit_transform(texts)
     ```

### 特征选择方法对比
| 特征选择方法 | 优点 | 缺点 | 适用场景 |
|-------------|------|------|----------|
| 高频词特征  | 简单高效，计算成本低 | 可能包含大量噪声词 | 数据量较小或特征维度较低时 |
| TF-IDF      | 能有效衡量词的重要性，减少噪声 | 计算成本较高 | 数据量较大或特征维度较高时 |

## 高频词/TF-IDF两种特征模式的切换方法
特征选择方法支持参数化切换，通过修改代码中的 `method` 参数即可选择不同的特征提取方式：
```python
def create_features(data, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        vectorizer = CountVectorizer(stop_words='english')
    features = vectorizer.fit_transform(data)
    return features

# 使用示例
method = 'tfidf'  # 或者 'high_freq'
X = create_features(texts, method=method)
```

## 项目结构
- `classify.py`：主程序文件，实现文本分类功能。
- `README.md`：项目说明文档。
- `naive_bayes_model.pkl`：训练好的模型文件。
- `邮件_files`：数据集目录，包含训练和测试数据。
- `stopwords.txt`：停用词列表文件。

## 项目运行
1. 将 `邮件_files.zip` 解压到项目目录。
2. 运行 `classify.py`：
   ```bash
   python classify.py
   ```
3. 查看运行结果，包括分类评估报告。

## 选做功能
1. **样本平衡处理**：
   - 使用 SMOTE（Synthetic Minority Over-sampling Technique）对训练集进行过采样，缓解类别不平衡问题。
   - 实现代码：
     ```python
     from imblearn.over_sampling import SMOTE
     smote = SMOTE()
     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
     ```

2. **模型评估指标**：
   - 输出包含精度（Precision）、召回率（Recall）和 F1 值的分类评估报告。
   - 实现代码：
     ```python
     from sklearn.metrics import classification_report
     print(classification_report(y_test, y_pred))
     ```
