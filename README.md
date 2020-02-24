# boyu-platform-pytorch-lessons
伯禹平台 pytorch 课程

课程目录
 1）线性回归
 线性回归的基本要素
 模型
 数据集
 损失函数
 优化函数 - 随机梯度下降
 矢量计算
 线性回归模型从零开始的实现
 生成数据集
 使用图像来展示生成的数据
 读取数据集
 初始化模型参数
 定义模型
 定义损失函数
 定义优化函数
 训练
 线性回归模型使用pytorch的简洁实现
 生成数据集
 读取数据集
 定义模型
 初始化模型参数
 定义损失函数
 定义优化函数
 训练
 两种实现方式的比较
 
 2）多层感知机
 多层感知机的基本知识
 隐藏层
 表达公式
 激活函数
 ReLU函数
 Sigmoid函数
 tanh函数
 关于激活函数的选择
 多层感知机
 多层感知机从零开始的实现
 获取训练集
 定义模型参数
 定义激活函数
 定义网络
 定义损失函数
 训练
 多层感知机pytorch实现
 初始化模型和各个参数
 训练
 
 3）softmax和分类模型
 softmax的基本概念
 交叉熵损失函数
 模型训练和预测
 获取Fashion-MNIST训练集和读取数据
 get dataset
 softmax从零开始的实现
 获取训练集数据和测试集数据
 模型参数初始化
 对多维Tensor按维度操作
 定义softmax操作
 softmax回归模型
 定义损失函数
 定义准确率
 训练模型
 模型预测
 softmax的简洁实现
 初始化参数和获取数据
 定义网络模型
 初始化模型参数
 定义损失函数
 定义优化函数
 训练
 
 4）文本预处理
 读入文本
 分词
 建立字典
 将词转为索引
 用现有工具进行分词
 
 5）语言模型
 语言模型
 n元语法
 语言模型数据集
 读取数据集
 建立字符索引
 时序数据的采样
 随机采样
 相邻采样
 
 6）循环神经网络
 循环神经网络的构造
 从零开始实现循环神经网络
 one-hot向量
 初始化模型参数
 定义模型
 裁剪梯度
 定义预测函数
 困惑度
 定义模型训练函数
 训练模型并创作歌词
 循环神经网络的简介实现
 定义模型
 
 7）循环神经网络进阶
 GRU
载入数据集
初始化参数
GRU模型
训练模型
简洁实现
LSTM
初始化参数
LSTM模型
训练模型
简洁实现
深度循环神经网络
双向循环神经网络

8）机器翻译及相关技术
机器翻译和数据集
数据预处理
分词
建立词典
载入数据集
Encoder-Decoder
Sequence to Sequence模型
模型：
具体结构：
Encoder
Decoder
损失函数
训练
测试
Beam Search

9）注意力机制与Seq2seq模型
注意力机制
注意力机制框架
Softmax屏蔽
点积注意力
测试
多层感知机注意力
测试
总结
引入注意力机制的Seq2seq模型
解码器
训练
训练和预测

10）Transformer
Transformer
多头注意力层
基于位置的前馈网络
Add and Norm
位置编码
测试
编码器
解码器
训练

11）优化与深度学习
优化与深度学习
优化与估计
优化在深度学习中的挑战
局部最小值
鞍点
梯度消失
凸性 （Convexity）
基础
集合
函数
Jensen 不等式
性质
无局部最小值
与凸集的关系
凸函数与二阶导数
限制条件
拉格朗日乘子法
惩罚项
投影

12）梯度下降
梯度下降
一维梯度下降
学习率
局部极小值
多维梯度下降
自适应方法
牛顿法
收敛性分析
预处理 （Heissan阵辅助梯度下降）
梯度下降与线性搜索（共轭梯度法）
随机梯度下降
随机梯度下降参数更新
动态学习率
小批量随机梯度下降
读取数据
从零开始实现
简洁实现

13）优化算法
11.6 Momentum
An ill-conditioned Problem
Maximum Learning Rate
Supp: Preconditioning
Solution to ill-condition
Momentum Algorithm
Exponential Moving Average
Supp
由指数加权移动平均理解动量法
Implement
Pytorch Class
11.7 AdaGrad
Algorithm
Feature
Implement
Pytorch Class
11.8 RMSProp
Algorithm
Implement
Pytorch Class
11.9 AdaDelta
Algorithm
Implement
Pytorch Class
11.10 Adam
Algorithm
Implement
Pytorch Class

