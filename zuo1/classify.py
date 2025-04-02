import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import joblib
from imblearn.over_sampling import SMOTE
from collections import Counter

# 增强版数据加载函数（支持文件名动态分类）
def load_data(data_dir):
    texts = []
    labels = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            # 通过文件名自动分类（示例逻辑：文件名包含'spam'则标记为垃圾）
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                    if text:
                        texts.append(text)
                        # 动态分配标签（根据文件名特征）
                        if 'spam' in filename.lower():  # 文件名包含spam则标记为垃圾
                            labels.append('spam')
                        else:
                            labels.append('ham')
            except Exception as e:
                print(f"读取文件 {file_path} 出错: {str(e)}")
    
    print(f"加载成功: {len(texts)} 条文本")
    print(f"类别分布: {Counter(labels)}")
    return texts, labels

# 特征工程函数
def create_features(data, method='tfidf'):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # 包含二元语法
        max_features=5000    # 限制特征维度
    )
    return vectorizer.fit_transform(data)

if __name__ == "__main__":
    data_dir = r'C:\Users\86134\PycharmProjects\NLP\files' 
    
    # 加载数据
    texts, labels = load_data(data_dir)
    
    # 数据验证
    if len(set(labels)) < 2:
        print(f"错误：需要至少两个类别，当前检测到 {set(labels)}")
        exit(1)
        
    # 特征提取
    X = create_features(texts)
    
    # 数据拆分（增加分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, 
        test_size=0.2, 
        stratify=labels,  # 保持类别分布
        random_state=42
    )
    
    # 样本平衡处理
    print("\n训练集原始分布:", Counter(y_train))
    if len(Counter(y_train)) > 1:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("过采样后分布:", Counter(y_train))
    
    # 模型训练
    model = MultinomialNB(alpha=0.1)  # 增加平滑系数
    model.fit(X_train, y_train)
    
    # 预测与评估
    y_pred = model.predict(X_test)
    
    # 完整评估报告
    print("\n" + "="*50)
    print("分类评估报告：")
    print(classification_report(y_test, y_pred, digits=4))
    
    # 核心指标提取
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n核心指标汇总：")
    print(f"加权精确率: {precision:.4f}")
    print(f"加权召回率: {recall:.4f}")
    print(f"加权F1值: {f1:.4f}")
    
    # 模型保存
    joblib.dump(model, 'spam_classifier.pkl')
    print("\n模型已保存为 spam_classifier.pkl")