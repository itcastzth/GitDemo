# RNN  

1. 两个模型的核心差异体现在什么机制上？  
   A. 字符编码方式不同  
   B. 是否考虑国家信息作为生成条件  
   C. RNN单元类型不同（GRU/LSTM）  
   D. 损失函数计算方式不同  
   **答案：C. RNN单元类型不同（GRU/LSTM）**  

2. 在条件生成模型（Model2_Conditioned_Surname_Generation）中，国家信息通过什么方式影响生成过程？  
   A. 作为额外的输入特征拼接  
   B. 作为GRU的初始隐藏状态  
   C. 作为注意力机制的key  
   D. 作为输出层的偏置项  
   **答案：A. 作为额外的输入特征拼接**  

3. 文件2中新增的nation_emb层的主要作用是：  
   `self.nation_emb = nn.Embedding(num_nationalities, rnn_hidden_size)`  
   A. 将字符索引映射为稠密向量  
   B. 将国家标签转换为隐藏状态初始化向量  
   C. 生成姓氏的长度控制参数  
   D. 计算交叉熵损失的辅助参数  
   **答案：B. 将国家标签转换为隐藏状态初始化向量**  

4. 对比两个文件的sample_from_model函数，文件2新增了哪个关键参数？  
   A. temperature  
   B. nationalities  
   C. device  
   D. max_length  
   **答案：B. nationalities**

# 截图一
<img width="648" alt="2f6a5fb56d55c3413df3762ae0d8d10" src="https://github.com/user-attachments/assets/06ce5746-6ae5-491c-8b52-3c89eb0ce673" />


# 截图二
<img width="610" alt="9133ae256115995374613545466c861" src="https://github.com/user-attachments/assets/b82ab685-6393-45af-89ff-d553cab304c7" />


# 截图三
<img width="608" alt="443d862b6d0969b54450821a306735d" src="https://github.com/user-attachments/assets/8bdaca8f-2363-48c8-9956-1ed84911296d" />
