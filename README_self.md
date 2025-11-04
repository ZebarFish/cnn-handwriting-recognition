# cnn-handwriting-recognition

#### 介绍
使用卷积神经网络模型识别手写数字 
> 学习《深度学习入门》一书后，整理代码添加注释方便后续使用

#### 目录说明
- main 模型结构等核心代码
- common 模型用到公用功能
- dataset 训练用数据集

#### 如何运行
- 安装python
- 使用 MNIST 手写数字图像集，网址http://yann.lecun.com/exdb/mnist/
- 如果需要训练数据，可以运行train_deepnet.py，会自动下载训练数据
- 如果需要统计准确率以及查看没有识别的图片，可以运行misclassified_mnist.py
- 如果需要预测数据，运行awesome_net.py，里面会随机取一张mnist的测试图片，关闭图片后会打印出预测数字


下面给出详尽但简洁的理论说明，说明训练时每一步真正做了什么、forward 为什么要“记录参数”、反向如何计算梯度、以及参数如何被更新。结合你工程中的函数/变量名，会更容易对应代码理解。

1) 训练循环的总体流程（高层）
- 从训练集随机抽取一个 mini-batch（大小 = mini_batch_size）。  
- 对这个 mini-batch 做前向传播（forward）：得到网络输出 y，并由损失层计算损失 L = loss(y, t)。前向过程中各层会把下一步反向需要的中间量保存到层对象中。  
- 做反向传播（backward）：从损失对输出的导数开始，沿网络逆序传播，计算每个可训练参数的梯度（dW, db 等）。  
- 用优化器（optimizer.update）根据这些梯度更新参数（self.params）。  
- 记录当前 batch 的损失；按 epoch 周期评估训练/测试集的准确率并记录用于监控。

2) 为什么 forward 要“记录参数 / 中间值”
- 反向传播本质上是链式法则（chain rule）。要计算某一层参数的梯度，需要该层前向时的输入、以及来自上层的梯度。  
- 因此每个层的 forward 会保存“反向需要”的中间数据（示例）：
  - Convolution 保存：输入 x、im2col 展开结果 col、col_W（权重重排） —— 用于计算 dW 和恢复 dx（用 col2im）。  
  - Affine 保存：展平后的输入 x（或 original_x_shape） —— 用于计算 dW = x^T · dout。  
  - Relu 保存：mask（正/负的位置） —— 用于在 backward 中屏蔽负值位置的梯度。  
  - Pooling 保存：最大值对应的索引（arg_max）或 mask —— 用于把上层的梯度分配回正确的位置。  
  - Dropout 保存：随机 mask（训练时） —— backward 时用相同 mask 屏蔽梯度。  
  - SoftmaxWithLoss 保存：softmax 输出 y 和真值 t —— backward 时直接计算 dL/dy。  
- 如果不保存这些中间量，就无法在 backward 精确计算每个参数的导数。

3) 反向传播（backprop）在数学上的直观（链式法则）
- 假设网络的计算是 x → f1 → f2 → … → fn → loss。损失 L 对某一中间量 z 的导数由链式法则给出： dL/dz = (dL/dout_of_next) * (dout_of_next/dz)。  
- 在代码中，这表现为：dout = last_layer.backward(1)；然后依次 dout = layer.backward(dout)。每个 layer.backward 返回的是该层输入的梯度（供上一层继续使用），并同时把该层参数的梯度存到 layer.dW / layer.db。

4) 梯度如何汇总并与参数对应（你代码中的实现）
- network.gradient(x, t) 会：
  - 先 forward 计算 loss（self.loss）。
  - 再从 self.last_layer.backward 开始逆序调用各层的 backward。
  - 然后从有参数的层（卷积或仿射层）读取 layer.dW, layer.db，并把它们按命名规则装到 grads 字典里（'W1','b1',...）。这些 key 与 self.params 中的 key 一一对应。
- Trainer 调用：grads = network.gradient(...); self.optimizer.update(self.network.params, grads)
  - 这样 optimizer 会直接修改 network.params 中的 ndarray（in-place 或替换），更新后的 params 又会被下一次 forward 使用。

5) 优化器的工作（以常见的SGD和Adam为例）
- SGD（随机梯度下降）基本更新： param := param - lr * grad  
  - lr 是学习率（optimizer_param 中的 lr）。
- 带动量（Momentum）的 SGD：会把历史梯度的指数加权平均纳入更新，能加速并抑制震荡。  
- Adam：每个参数维护一阶矩估计（动量 m）和二阶矩估计（v，近似方差），并作偏差修正后用自适应学习率更新：  
  - m_t = beta1*m_{t-1} + (1-beta1)*g  
  - v_t = beta2*v_{t-1} + (1-beta2)*g^2  
  - param := param - lr * (m_t_hat / (sqrt(v_t_hat)+eps))
- 优化器把 grads 里的信息映射到 params 并实际改变权重。

6) Mini-batch 的作用与统计性质
- 用 mini-batch 而不是整批（full-batch）计算梯度，带来两个效果：
  - 计算效率更高（每步处理较少样本），可在内存/速度之间折衷。  
  - 使梯度有噪声（stochastic），这种噪声通常帮助逃离鞍点/局部最小值，但过大噪声会妨碍收敛。
- epoch 定义为“遍历全部训练数据一次”。代码通过 iter_per_epoch 控制何时统计一次训练/测试精度。

7) 正则化与稳定化手段（你网络中使用或可用的）
- 权重初始化（He 初始化，wight_init_scales = sqrt(2/pre_node_nums））：为 ReLU 设计，能使激活值/梯度在网络初始时尺度稳定，减小梯度消失/爆炸。  
- Dropout：训练时随机丢弃神经元，减轻过拟合；推理时不开启（train_flg=False）。  
- BatchNorm（在本项目层实现中存在但网络未必使用）：在训练中标准化批次输入，能加速收敛并允许更大 learning rate。  
- (常见但代码中未显式) L2 权重衰减（weight decay）：在损失上加项 λ/2 * ||W||^2，相当于更新时在梯度上加 λ*W。

8) 损失函数与 softmax 的细节
- 最后层 SoftmaxWithLoss 同时计算 softmax 输出 y 和交叉熵损失： L = -∑ t_i log y_i / N。  
- softmax 的数值稳定实现通常会在 exponent 前减 max 值，避免 exp 溢出。交叉熵 + softmax 在 backward 上简化为 (y - t)/N（当 t 为 one-hot 时），这使得反向起点计算高效。

9) 数值与调参要点（常见实践与检测）
- 学习率 lr 对训练影响最大：太大发散，太小收敛慢或陷入次优。用 scheduler、Adam、或 lr search 常见。  
- 监控 loss 曲线和训练/测试 accuracy：出现训练 acc 很高而测试 acc 低 => 过拟合（增强正则或减小模型）；两者都低 => 欠拟合或 lr 太小。  
- 观察梯度范数与权重范数：极小梯度可能是梯度消失，极大梯度会导致不稳定（考虑梯度裁剪）。  
- 可视化第一层卷积核（filter）和中间特征图以验证学习到的特征是否合理。  
- 若网络不收敛：检查数据归一化、权重初始化、loss 实现、学习率、batch_size、以及是否正确匹配参数形状（如 W7 的展平维度）。

10) 训练的“真实意义”
- 训练的核心是：通过最小化经验风险（在训练集上的平均损失），基于梯度信息调节参数，使模型在分布相似的数据上做出更准确的预测。  
- mini-batch SGD + 合适优化器是实际求解高维非凸优化问题的有效方法；前向计算给出预测与损失，反向计算给出损失对每个参数的敏感度（梯度），优化器则把敏感度转化为参数更新。

11) 与你项目代码的关键对应点（快速映射）
- Trainer.train_step():
  - grads = self.network.gradient(x_batch, t_batch)  <- forward + backward，返回 grads 字典  
  - self.optimizer.update(self.network.params, grads) <- 使用 optimizer 根据 grads 更新 params  
  - loss = self.network.loss(x_batch, t_batch) <- 再次前向算 loss（可以省略重复计算以加速，但现在代码先更新再算 loss）  
- DeepConvNet.predict/loss/gradient：
  - predict 调用每个 layer.forward；forward 内保存中间量。  
  - gradient 先调用 loss（触发 forward），再 last_layer.backward + 逆序 layers.backward，最后从指定 layer 索引读取 dW/db 组成 grads。

- 流程图

输入: X  (1, 28, 28)

-> [Conv1 W1] (16 filters, 3x3, pad=1, stride=1)
   输出: (16, 28, 28)
   forward 保存: x, col, col_W
   backward 计算: dW1, db1 -> 返回 dx

-> [ReLU]
   输出同上
   forward 保存: mask
   backward: 用 mask 屏蔽梯度 -> 返回 dx

-> [Conv2 W2] (16, 3x3, pad=1)
   输出: (16, 28, 28)
   保存: x, col, col_W
   backward: dW2, db2

-> [ReLU]
-> [Pool (2x2, s=2)]
   输出: (16, 14, 14)
   保存: x, arg_max（或 mask）
   backward: 将上层梯度分配回 arg_max -> 返回 dx

-> [Conv3 W3] (32, 3x3, pad=1)
   输出: (32, 14, 14)
   保存: x, col, col_W
   backward: dW3, db3

-> [ReLU]

-> [Conv4 W4] (32, 3x3, pad=2)  ← 注意 pad=2 导致尺寸变化
   输出: (32, 16, 16)
   保存: x, col, col_W
   backward: dW4, db4

-> [ReLU]
-> [Pool (2x2, s=2)]
   输出: (32, 8, 8)

-> [Conv5 W5] (64, 3x3, pad=1)
   输出: (64, 8, 8)
   保存: x, col, col_W
   backward: dW5, db5

-> [ReLU]

-> [Conv6 W6] (64, 3x3, pad=1)
   输出: (64, 8, 8)
   保存: x, col, col_W
   backward: dW6, db6

-> [ReLU]
-> [Pool (2x2, s=2)]
   输出: (64, 4, 4)

-> flatten -> 向量长度 64*4*4 = 1024

-> [Affine W7] (1024 -> hidden_size)
   输出: (hidden_size)  （代码中 hidden_size 默认 50）
   保存: original_x_shape, x（展平）
   backward: dW7, db7

-> [ReLU]
-> [Dropout (0.5)]
   forward 保存: mask（训练时）
   backward: 用相同 mask 屏蔽梯度

-> [Affine W8] (hidden_size -> output_size(10))
   输出: (10)
   保存: original_x_shape, x（展平）
   backward: dW8, db8

-> [Dropout]
-> [SoftmaxWithLoss]
   forward 计算 softmax 输出 y 和交叉熵损失 L，保存 y,t
   backward 返回 dL/dy = (y - t)/N（作为反向起点）

最终输出: 预测概率 / 损失

参数层索引（对应 deep_convnet.layers 列表）：
- layers[0]  = Conv1  (W1, b1)
- layers[2]  = Conv2  (W2, b2)
- layers[5]  = Conv3  (W3, b3)
- layers[7]  = Conv4  (W4, b4)
- layers[10] = Conv5  (W5, b5)
- layers[12] = Conv6  (W6, b6)
- layers[15] = Affine1 (W7, b7)
- layers[18] = Affine2 (W8, b8)

训练时的真实流程（对应代码）
1. 随机选 mini-batch x_batch, t_batch。
2. forward: 逐层调用 layer.forward(...)，每层保存反向需要的中间量（见上）。最终由 SoftmaxWithLoss 计算损失 L 并保存 y,t。
3. backward: dout = last_layer.backward(1)；然后按 layers 逆序调用 layer.backward(dout)，每个有参数的层会计算并保存它们的参数梯度 dW、db。
4. 收集 grads = {'W1':..., 'b1':..., ..., 'W8':..., 'b8':...}，交给 optimizer.update(self.params, grads) 做参数更新（如 SGD/Adam）。
5. 重复直到训练结束。


箭头说明：
- forward → 保存中间量（forward 为后续 backward 提供必要信息）
- backward ← 使用保存的中间量计算梯度并传递给上一层

用途提示（快速定位）：
- 若想查看第 k 个卷积层学到的 filter，可可视化 params['Wk']（对应 layers 中的 W）
- 若要调试反向，打印 layer.dW, layer.db 或每层返回的 dx
