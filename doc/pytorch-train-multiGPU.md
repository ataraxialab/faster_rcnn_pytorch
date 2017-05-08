# Pytorch 多GPU训练小结

## 需求

利用多卡进行训练不同的图片。

针对每个GPU，进行一轮计算，步骤如下：

1. 输入模型数据和图片

2. 计算前向

3. 梯度计算。

计算结束以后，同步在不同GPU上面的模型数据。然后进行第二轮的计算。



## 方案

## 1. DataParallel

这个类功能是在执行前向时候会把 **前向函数的参数** 自动分成多份，平均按照参数指定的GPU上，同时把模型复制到每个GPU上面，然后在过多GPU上面执行前向函数。

### 结论

DataParallel只做到了前向的多GPU功能。无法实现需求。



## 2. 多任务

### 设计

#### 框架

![multi-gpu-arch](doc/multi-gpu-arch.png)

#### tainner流程

![trainer](doc/trainer-flow.png)

#### parameter updater 流程

![](doc/param-updater.png)



### 2.1 多线程

trainer/parameter updater 分别用一个线程去执行。

#### 结论

在做 `backward` 的时候，除了第一个线程执行成功以外，另一个线程会抛出参数不在同一个GPU上面的问题。这个问题需要在cuda/cpp代码层面去查找原因。无法实现需求。

### 2.2 多进程

trainer/parameter updater 分别用一个进程去执行。

#### 结论

pytorch的多进程，只有在python3+的版本上面才支持。由于，faster-rcnn的代码是基于python2，改动量比较大，未实验。

## 总结

pytorch是一个比较完整的DL库，相对简单的DL框架。在上面改多卡，需要对pytorch实现比较了解才行。



## PS

多GPU的使用方式除了每个GPU计算单独模型以外，每次迭代计算中不同的计算也可以放到多个GPU中计算，这样也是能够加快训练速度的。难点在于，控制在不同GPU上面计算的代码逻辑。