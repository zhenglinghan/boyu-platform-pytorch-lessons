softmax和分类模型
内容包含：

softmax回归的基本概念
如何获取Fashion-MNIST数据集和读取数据
softmax回归模型的从零开始实现，实现一个对Fashion-MNIST训练集中的图像数据进行分类的模型
使用pytorch重新实现softmax回归模型
softmax的基本概念
分类问题
一个简单的图像分类问题，输入图像的高和宽均为2像素，色彩为灰度。
图像中的4像素分别记为 x1,x2,x3,x4 。
假设真实标签为狗、猫或者鸡，这些标签对应的离散值为 y1,y2,y3 。
我们通常使用离散的数值来表示类别，例如 y1=1,y2=2,y3=3 。

权重矢量
o1=x1w11+x2w21+x3w31+x4w41+b1
 
o2=x1w12+x2w22+x3w32+x4w42+b2
 
o3=x1w13+x2w23+x3w33+x4w43+b3
 
神经网络图
下图用神经网络图描绘了上面的计算。softmax回归同线性回归一样，也是一个单层神经网络。由于每个输出 o1,o2,o3 的计算都要依赖于所有的输入 x1,x2,x3,x4 ，softmax回归的输出层也是一个全连接层。
Image Name

softmax回归是一个单层神经网络
 
既然分类问题需要得到离散的预测输出，一个简单的办法是将输出值 oi 当作预测类别是 i 的置信度，并将值最大的输出所对应的类作为预测输出，即输出  argmaxioi 。例如，如果 o1,o2,o3 分别为 0.1,10,0.1 ，由于 o2 最大，那么预测类别为2，其代表猫。

输出问题
直接使用输出层的输出有两个问题：
一方面，由于输出层的输出值的范围不确定，我们难以直观上判断这些值的意义。例如，刚才举的例子中的输出值10表示“很置信”图像类别为猫，因为该输出值是其他两类的输出值的100倍。但如果 o1=o3=103 ，那么输出值10却又表示图像类别为猫的概率很低。
另一方面，由于真实标签是离散值，这些离散值与不确定范围的输出值之间的误差难以衡量。
softmax运算符（softmax operator）解决了以上两个问题。它通过下式将输出值变换成值为正且和为1的概率分布：

y^1,y^2,y^3=softmax(o1,o2,o3)
 
其中

y^1=exp(o1)∑3i=1exp(oi),y^2=exp(o2)∑3i=1exp(oi),y^3=exp(o3)∑3i=1exp(oi).
 
容易看出 y^1+y^2+y^3=1 且 0≤y^1,y^2,y^3≤1 ，因此 y^1,y^2,y^3 是一个合法的概率分布。这时候，如果 y^2=0.8 ，不管 y^1 和 y^3 的值是多少，我们都知道图像类别为猫的概率是80%。此外，我们注意到

argmaxioi=argmaxiy^i
 
因此softmax运算不改变预测类别输出。

计算效率
单样本矢量计算表达式
为了提高计算效率，我们可以将单样本分类通过矢量计算来表达。在上面的图像分类问题中，假设softmax回归的权重和偏差参数分别为
W=⎡⎣⎢⎢⎢w11w21w31w41w12w22w32w42w13w23w33w43⎤⎦⎥⎥⎥,b=[b1b2b3],
 
设高和宽分别为2个像素的图像样本 i 的特征为

x(i)=[x(i)1x(i)2x(i)3x(i)4],
 
输出层的输出为

o(i)=[o(i)1o(i)2o(i)3],
 
预测为狗、猫或鸡的概率分布为

y^(i)=[y^(i)1y^(i)2y^(i)3].
 
softmax回归对样本 i 分类的矢量计算表达式为

o(i)y^(i)=x(i)W+b,=softmax(o(i)).
 
小批量矢量计算表达式
为了进一步提升计算效率，我们通常对小批量数据做矢量计算。广义上讲，给定一个小批量样本，其批量大小为 n ，输入个数（特征数）为 d ，输出个数（类别数）为 q 。设批量特征为 X∈Rn×d 。假设softmax回归的权重和偏差参数分别为 W∈Rd×q 和 b∈R1×q 。softmax回归的矢量计算表达式为
OY^=XW+b,=softmax(O),
 
其中的加法运算使用了广播机制， O,Y^∈Rn×q 且这两个矩阵的第 i 行分别为样本 i 的输出 o(i) 和概率分布 y^(i) 。

交叉熵损失函数
对于样本 i ，我们构造向量 y(i)∈Rq  ，使其第 y(i) （样本 i 类别的离散数值）个元素为1，其余为0。这样我们的训练目标可以设为使预测概率分布 y^(i) 尽可能接近真实的标签概率分布 y(i) 。

平方损失估计
Loss=|y^(i)−y(i)|2/2
 
然而，想要预测分类结果正确，我们其实并不需要预测概率完全等于标签概率。例如，在图像分类的例子里，如果 y(i)=3 ，那么我们只需要 y^(i)3 比其他两个预测值 y^(i)1 和 y^(i)2 大就行了。即使 y^(i)3 值为0.6，不管其他两个预测值为多少，类别预测均正确。而平方损失则过于严格，例如 y^(i)1=y^(i)2=0.2 比 y^(i)1=0,y^(i)2=0.4 的损失要小很多，虽然两者都有同样正确的分类预测结果。

改善上述问题的一个方法是使用更适合衡量两个概率分布差异的测量函数。其中，交叉熵（cross entropy）是一个常用的衡量方法：

H(y(i),y^(i))=−∑j=1qy(i)jlogy^(i)j,
 
其中带下标的 y(i)j 是向量 y(i) 中非0即1的元素，需要注意将它与样本 i 类别的离散数值，即不带下标的 y(i) 区分。在上式中，我们知道向量 y(i) 中只有第 y(i) 个元素 y(i)y(i) 为1，其余全为0，于是 H(y(i),y^(i))=−logy^(i)y(i) 。也就是说，交叉熵只关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。当然，遇到一个样本有多个标签时，例如图像里含有不止一个物体时，我们并不能做这一步简化。但即便对于这种情况，交叉熵同样只关心对图像中出现的物体类别的预测概率。

假设训练数据集的样本数为 n ，交叉熵损失函数定义为
ℓ(Θ)=1n∑i=1nH(y(i),y^(i)),
 
其中 Θ 代表模型参数。同样地，如果每个样本只有一个标签，那么交叉熵损失可以简写成 ℓ(Θ)=−(1/n)∑ni=1logy^(i)y(i) 。从另一个角度来看，我们知道最小化 ℓ(Θ) 等价于最大化 exp(−nℓ(Θ))=∏ni=1y^(i)y(i) ，即最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。

模型训练和预测
在训练好softmax回归模型后，给定任一样本特征，就可以预测每个输出类别的概率。通常，我们把预测概率最大的类别作为输出类别。如果它与真实类别（标签）一致，说明这次预测是正确的。在3.6节的实验中，我们将使用准确率（accuracy）来评价模型的表现。它等于正确预测数量与总预测数量之比。

获取Fashion-MNIST训练集和读取数据
在介绍softmax回归的实现前我们先引入一个多类图像分类数据集。它将在后面的章节中被多次使用，以方便我们观察比较算法之间在模型精度和计算效率上的区别。图像分类数据集中最常用的是手写数字识别数据集MNIST[1]。但大部分模型在MNIST上的分类精度都超过了95%。为了更直观地观察算法之间的差异，我们将使用一个图像内容更加复杂的数据集Fashion-MNIST[2]。

我这里我们会使用torchvision包，它是服务于PyTorch深度学习框架的，主要用来构建计算机视觉模型。torchvision主要由以下几部分构成：

torchvision.datasets: 一些加载数据的函数及常用的数据集接口；
torchvision.models: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
torchvision.transforms: 常用的图片变换，例如裁剪、旋转等；
torchvision.utils: 其他的一些有用的方法。
# import needed package
%matplotlib inline
from IPython import display
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import time

import sys
sys.path.append("/home/kesci/input")
import d2lzh1981 as d2l

print(torch.__version__)
print(torchvision.__version__)
1.3.0
0.4.1a0+d94043a
get dataset
mnist_train = torchvision.datasets.FashionMNIST(root='/home/kesci/input/FashionMNIST2065', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='/home/kesci/input/FashionMNIST2065', train=False, download=True, transform=transforms.ToTensor())
class torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None, download=False)

root（string）– 数据集的根目录，其中存放processed/training.pt和processed/test.pt文件。
train（bool, 可选）– 如果设置为True，从training.pt创建数据集，否则从test.pt创建。
download（bool, 可选）– 如果设置为True，从互联网下载数据并放到root文件夹下。如果root目录下已经存在数据，不会再次下载。
transform（可被调用 , 可选）– 一种函数或变换，输入PIL图片，返回变换之后的数据。如：transforms.RandomCrop。
target_transform（可被调用 , 可选）– 一种函数或变换，输入目标，进行变换。
# show result 
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))
<class 'torchvision.datasets.mnist.FashionMNIST'>
60000 10000
# 我们可以通过下标来访问任意一个样本
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width
torch.Size([1, 28, 28]) 9
如果不做变换输入的数据是图像，我们可以看一下图片的类型参数：

mnist_PIL = torchvision.datasets.FashionMNIST(root='/home/kesci/input/FashionMNIST2065', train=True, download=True)
PIL_feature, label = mnist_PIL[0]
print(PIL_feature)
<PIL.Image.Image image mode=L size=28x28 at 0x7F54A41612E8>
# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0]) # 将第i个feature加到X中
    y.append(mnist_train[i][1]) # 将第i个label加到y中
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 读取数据
batch_size = 256
num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
4.95 sec
softmax从零开始的实现
import torch
import torchvision
import numpy as np
import sys
sys.path.append("/home/kesci/input")
import d2lzh1981 as d2l

print(torch.__version__)
print(torchvision.__version__)
1.3.0
0.4.1a0+d94043a
获取训练集数据和测试集数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='/home/kesci/input/FashionMNIST2065')
模型参数初始化
num_inputs = 784
print(28*28)
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
784
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)
对多维Tensor按维度操作
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  # dim为0，按照相同的列求和，并在结果中保留列特征
print(X.sum(dim=1, keepdim=True))  # dim为1，按照相同的行求和，并在结果中保留行特征
print(X.sum(dim=0, keepdim=False)) # dim为0，按照相同的列求和，不在结果中保留列特征
print(X.sum(dim=1, keepdim=False)) # dim为1，按照相同的行求和，不在结果中保留行特征
tensor([[5, 7, 9]])
tensor([[ 6],
        [15]])
tensor([5, 7, 9])
tensor([ 6, 15])
定义softmax操作
y^j=exp(oj)∑3i=1exp(oi)
 
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # print("X size is ", X_exp.size())
    # print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制
X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, '\n', X_prob.sum(dim=1))
tensor([[0.2253, 0.1823, 0.1943, 0.2275, 0.1706],
        [0.1588, 0.2409, 0.2310, 0.1670, 0.2024]]) 
 tensor([1.0000, 1.0000])
softmax回归模型
o(i)y^(i)=x(i)W+b,=softmax(o(i)).
 
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
定义损失函数
H(y(i),y^(i))=−∑j=1qy(i)jlogy^(i)j,
 
ℓ(Θ)=1n∑i=1nH(y(i),y^(i)),
 
ℓ(Θ)=−(1/n)∑i=1nlogy^(i)y(i)
 
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))
tensor([[0.1000],
        [0.5000]])
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
定义准确率
我们模型训练完了进行模型预测的时候，会用到我们这里定义的准确率。

def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()
print(accuracy(y_hat, y))
0.5
# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n
print(evaluate_accuracy(test_iter, net))
0.1445
训练模型
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step() 
            
            
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
epoch 1, loss 0.7851, train acc 0.750, test acc 0.791
epoch 2, loss 0.5704, train acc 0.814, test acc 0.810
epoch 3, loss 0.5258, train acc 0.825, test acc 0.819
epoch 4, loss 0.5014, train acc 0.832, test acc 0.824
epoch 5, loss 0.4865, train acc 0.836, test acc 0.827
模型预测
现在我们的模型训练完了，可以进行一下预测，我们的这个模型训练的到底准确不准确。 现在就可以演示如何对图像进行分类了。给定一系列图像（第三行图像输出），我们比较一下它们的真实标签（第一行文本输出）和模型预测结果（第二行文本输出）。

X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])

softmax的简洁实现
# 加载各种包或者模块
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("/home/kesci/input")
import d2lzh1981 as d2l

print(torch.__version__)
1.3.0
初始化参数和获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='/home/kesci/input/FashionMNIST2065')
定义网络模型
num_inputs = 784
num_outputs = 10

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x): # x 的形状: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y
    
# net = LinearNet(num_inputs, num_outputs)

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x 的形状: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

from collections import OrderedDict
net = nn.Sequential(
        # FlattenLayer(),
        # LinearNet(num_inputs, num_outputs) 
        OrderedDict([
           ('flatten', FlattenLayer()),
           ('linear', nn.Linear(num_inputs, num_outputs))]) # 或者写成我们自己定义的 LinearNet(num_inputs, num_outputs) 也可以
        )
初始化模型参数
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
Parameter containing:
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)
定义损失函数
loss = nn.CrossEntropyLoss() # 下面是他的函数原型
# class torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
定义优化函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1) # 下面是函数原型
# class torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)
训练
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
epoch 1, loss 0.0031, train acc 0.751, test acc 0.795
epoch 2, loss 0.0022, train acc 0.813, test acc 0.809
epoch 3, loss 0.0021, train acc 0.825, test acc 0.806
epoch 4, loss 0.0020, train acc 0.833, test acc 0.813
epoch 5, loss 0.0019, train acc 0.837, test acc 0.822