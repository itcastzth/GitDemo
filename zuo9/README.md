# RNN  

1. 两个模型的核心差异体现在什么机制上？  
   A. 字符编码方式不同  
   B. 是否考虑国家信息作为生成条件  
   C. RNN单元类型不同（GRU/LSTM）  
   D. 损失函数计算方式不同  
   **答案： B. 是否考虑国家信息作为生成条件**  

2. 在条件生成模型（Model2_Conditioned_Surname_Generation）中，国家信息通过什么方式影响生成过程？  
   A. 作为额外的输入特征拼接  
   B. 作为GRU的初始隐藏状态  
   C. 作为注意力机制的key  
   D. 作为输出层的偏置项  
   **答案： B. 作为GRU的初始隐藏状态**  

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

# 截图一数据预处理
<img width="229" alt="01e5eb57df61431057ed631e7454d78" src="https://github.com/user-attachments/assets/b489b51e-bbb7-40d3-8019-f84b27ee440f" />



# 截图二模型结构
<img width="542" alt="a4aa274501c64b4fb1019b50a4db1b0" src="https://github.com/user-attachments/assets/74b14edc-386e-4f71-95fd-d70beb398ff7" />

# 截图三 RNN处理
<img width="465" alt="5dd2300ee3e29339423c78096ff3106" src="https://github.com/user-attachments/assets/638e16ae-dc4b-4106-b893-4ad5ec413c80" />


#  截图四损失值
<img width="578" alt="9b25c1d573315c4673f3843217f998e" src="https://github.com/user-attachments/assets/2861158e-5f9e-41ff-a410-1a19dba9fcc9" />


# 最后
<img width="554" alt="3296e0048d54ce41a0fadfe6f9b5c6a" src="https://github.com/user-attachments/assets/67533d38-7195-4581-9d04-4079a9f1ae33" />


<img width="339" alt="4fe1663fb5d82da2cf705ebeff8750d" src="https://github.com/user-attachments/assets/11d58060-b6fd-4890-874e-c1a770448510" />


<img width="610" alt="9133ae256115995374613545466c861" src="https://github.com/user-attachments/assets/b82ab685-6393-45af-89ff-d553cab304c7" />



<img width="608" alt="443d862b6d0969b54450821a306735d" src="https://github.com/user-attachments/assets/8bdaca8f-2363-48c8-9956-1ed84911296d" />
