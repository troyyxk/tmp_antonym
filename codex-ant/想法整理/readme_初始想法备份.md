# 反义词 encoder
- 同意相反，而不是无关
- 有没有专门做反义词词的encoder？我说的是good vs bad, left vs right这种。我问这个问题就是因为感觉大多数encoder做的都是relevancy，不相关的词汇会离得比较远，但是反义词的vector很近

杜老师，我感觉有一个可以做的小idea。目前没有反义词编码器，当前的encoder针对good vs bad, left vs right因为在句中出现的地点类似，所以vector space距离也是接近的。我在想专门做一个反义词encoder，可以做一个类似MoE的格式，由两个encoder一起组成一个反义词encoder。
组成反义词需要有两特点，一个是high relevancy，另一个就是semantic opposite。那是否可以一个encoder的vector space是越相关越近，一个的是semantic越相反越远。两个都达到一个threshold的时候才判断反义词。单独relevancy的问题是good 和 bad接近但是不知道是否相反；单独反义词，我判断可能会出现不相关的词但是距离很远。杜老师觉得可行吗。
- 感觉不能只输出是否是反义词，需要输出一个 distance
- 最好通过同一个encoder就能用
    - 不然和 Siamese 有啥区别
## 研究一下这些和我想做的有没有重合
- https://aclanthology.org/P19-1319/
- antonym-sensitive embedding 
    - https://aclanthology.org/K15-1026.pdf?utm_source=chatgpt.com
- Integrating Distributional Lexical Contrast into Word Embeddings for Antonym-Synonym Distinction
    - https://arxiv.org/abs/1605.07766
- Bhav-Net: Knowledge Transfer for Cross-Lingual Antonym vs Synonym Distinction via Dual-Space Graph Transformers
    - https://www.arxiv.org/abs/2508.15792

Retrofitting / Counter-fitting (Mrkšić et al., NAACL 2016):

## 反义词的定义是啥
- 语义基础：必须在同一个“维度”上
- 逻辑关系：必须是对立或排斥的
- 语法属性：词性必须一致
