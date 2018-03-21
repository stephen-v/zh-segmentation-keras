# zh-segmentation-keras
Chinese segmentation simple by keras 

## requirements
* keras==2.1.14
* pandas 
* numpy

# 基于双向BiLstm以及HMM模型的中文分词详解及源码
> 在自然语言处理中（NLP，Natural Language ProcessingNLP，Natural Language Processing），分词是一个较为简单也基础的基本技术。常用的分词方法包括这两种：**基于字典的机械分词** 和 **基于统计序列标注的分词**。对于基于字典的机械分词本文不再赘述，可看[字典分词方法](https://spaces.ac.cn/archives/3908 "字典分词方法")。在本文中主要讲解基于深度学习的分词方法及原理，包括一下几个步骤：`1标注序列`，`2双向LSTM网络预测标签`，`3Viterbi算法求解最优路径`

## 1 标注序列
中文分词的第一步便是标注字，字标注是通过给句子中每个字打上标签的思路来进行分词，比如之前提到过的，通过4标签来进行标注`（single，单字成词；begin，多字词的开头；middle，三字以上词语的中间部分；end，多字词的结尾。均只取第一个字母。）`，这样，“为人民服务”就可以标注为“sbebe”了。4标注不是唯一的标注方式，类似地还有6标注，理论上来说，标注越多会越精细，理论上来说效果也越好，但标注太多也可能存在样本不足的问题，一般常用的就是4标注和6标注。前面已经提到过，字标注是通过给句子中每个字打上标签的思路来进行分词，比如之前提到过的，通过4标签来进行标注（single，单字成词；begin，多字词的开头；middle，三字以上词语的中间部分；end，多字词的结尾。均只取第一个字母。），这样，“为人民服务”就可以标注为“sbebe”了。4标注不是唯一的标注方式，类似地还有6标注，理论上来说，标注越多会越精细，理论上来说效果也越好，但标注太多也可能存在样本不足的问题，一般常用的就是4标注和6标注。标注实例如下：

```
人/b  们/e  常/s  说/s  生/b  活/e  是/s  一/s  部/s  教/b  科/m  书/e 
```

## 2 训练网络
这里所指的网络主要是指神经网络，再细化一点就是双向LSTM(长短时记忆网络)，双向LSTM是LSTM的改进版，LSTM是RNN的改进版。因此，首先需要理解RNN。

RNN的意思是，为了预测最后的结果，我先用第一个词预测，当然，只用第一个预测的预测结果肯定不精确，我把这个结果作为特征，跟第二词一起，来预测结果；接着，我用这个新的预测结果结合第三词，来作新的预测；然后重复这个过程；直到最后一个词。这样，如果输入有n个词，那么我们事实上对结果作了n次预测，给出了n个预测序列。整个过程中，模型共享一组参数。因此，RNN降低了模型的参数数目，防止了过拟合，同时，它生来就是为处理序列问题而设计的，因此，特别适合处理序列问题。**循环神经网络原理见下图：**

![2018-03-20-11-47-27](http://qiniu.xdpie.com/2018-03-20-11-47-27.png)

LSTM对RNN做了改进，使得能够捕捉更长距离的信息。但是不管是LSTM还是RNN，都有一个问题，它是从左往右推进的，因此后面的词会比前面的词更重要，但是对于分词这个任务来说是不妥的，因为句子各个字应该是平权的。因此出现了双向LSTM，它从左到右做一次LSTM，然后从右到左做一次LSTM，然后把两次结果组合起来。

在分词中，LSTM可以根据输入序列输出一个序列，这个序列考虑了上下文的联系，因此，可以给每个输出序列接一个softmax分类器，来预测每个标签的概率。基于这个序列到序列的思路，我们就可以直接预测句子的标签。假设每次输入$y_1$-$y_n$由下图所示每个输入所对应的标签为$x_1$-$x_n$。再抽象一点用$ x_{ij} $表示状态$x_i$的第j个可能值。

![2018-03-20-11-48-06](http://qiniu.xdpie.com/2018-03-20-11-48-06.png)

最终输出结果串联起来形成如下图所示的网络


![2018-03-20-11-49-50](http://qiniu.xdpie.com/2018-03-20-11-49-50.png)

图中从第一个可能标签到最后一个可能标签的任何一条路径都可能产生一个新的序列，每条路径可能性不一样，我们需要做的是找出可能的路径。由于路径非常多，因此采用穷举是非常耗时的，因此引入Viterbi算法。

## 3 Viterbi算法求解最优路径

维特比算法是一个特殊但应用最广的动态规划算法，利用动态规划，可以解决任何一个图中的最短路径问题。而维特比算法是针对一个特殊的图——篱笆网络的有向图（Lattice )的最短路径问题而提出的。

而维特比算法的精髓就是，既然知道到第i列所有节点Xi{j=123…}的最短路径，那么到第i+1列节点的最短路径就等于到第i列j个节点的最短路径+第i列j个节点到第i+1列各个节点的距离的最小值，关于维特比算法的详细可以[点击](https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95 "点击")

## 4 keras代码讲解

使用Keras构建bilstm网络，在keras中已经预置了网络模型，只需要调用相应的函数就可以了。需要注意的是，对于每一句话会转换为词向量（Embedding）如下图所示：

![2018-03-20-11-49-20](http://qiniu.xdpie.com/2018-03-20-11-49-20.png)

`embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True)(sequence)`并将不足的补零。

**创建网络**

```python
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model


def create_model(maxlen, chars, word_size):
    """

    :param maxlen:
    :param chars:
    :param word_size:
    :return:
    """
    sequence = Input(shape=(maxlen,), dtype='int32')
    embedded = Embedding(len(chars) + 1, word_size, input_length=maxlen, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
    model = Model(input=sequence, output=output)
    return model

```

**训练数据**

```python
# -*- coding:utf-8 -*-

import re
import numpy as np
import pandas as pd

# 设计模型
word_size = 128
maxlen = 32

with open('data/msr_train.txt', 'rb') as inp:
    texts = inp.read().decode('gbk')
s = texts.split('\r\n')  # 根据换行切分


def clean(s):  # 整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s


s = u''.join(map(clean, s))
s = re.split(u'[，。！？、]/[bems]', s)

data = []  # 生成训练样本
label = []


def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:, 0]), list(s[:, 1])


for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])

d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
d = d[d['data'].apply(len) <= maxlen]
d.index = range(len(d))
tag = pd.Series({'s': 0, 'b': 1, 'm': 2, 'e': 3, 'x': 4})

chars = []  # 统计所有字，跟每个字编号
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars) + 1)

# 保存数据
import pickle

with open('model/chars.pkl', 'wb') as outp:
    pickle.dump(chars, outp)
print('** Finished saving the data.')

# 生成适合模型输入的格式
from keras.utils import np_utils

d['x'] = d['data'].apply(lambda x: np.array(list(chars[x]) + [0] * (maxlen - len(x))))


def trans_one(x):
    _ = map(lambda y: np_utils.to_categorical(y, 5), tag[x].reshape((-1, 1)))
    _ = list(_)
    _.extend([np.array([[0, 0, 0, 0, 1]])] * (maxlen - len(x)))
    return np.array(_)


d['y'] = d['label'].apply(trans_one)

import lstm_model

model = lstm_model.create_model(maxlen, chars, word_size)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 1024
history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1, maxlen, 5)), batch_size=batch_size,
                    nb_epoch=20, verbose=2)
model.save('model/model.h5')
```

![2018-03-20-13-21-20](http://qiniu.xdpie.com/2018-03-20-13-21-20.png)![2018-03-20-13-21-20](http://qiniu.xdpie.com/2018-03-20-13-21-20.png)

**1080显卡训练每次需要耗时44s,训练20个epoch后精度达到95%**

**测试**

```python
import pickle
import lstm_model
import pandas as pd

with open('model/chars.pkl', 'rb') as inp:
    chars = pickle.load(inp)
word_size = 128
maxlen = 32

model = lstm_model.create_model(maxlen, chars, word_size)
model.load_weights('model/model.h5', by_name=True)

import re
import numpy as np

# 转移概率，单纯用了等概率
zy = {'be': 0.5,
      'bm': 0.5,
      'eb': 0.5,
      'es': 0.5,
      'me': 0.5,
      'mm': 0.5,
      'sb': 0.5,
      'ss': 0.5
      }

zy = {i: np.log(zy[i]) for i in zy.keys()}


def viterbi(nodes):
    paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}  # 第一层，只有两个节点
    for layer in range(1, len(nodes)):  # 后面的每一层
        paths_ = paths.copy()  # 先保存上一层的路径
        # node_now 为本层节点， node_last 为上层节点
        paths = {}  # 清空 path
        for node_now in nodes[layer].keys():
            # 对于本层的每个节点，找出最短路径
            sub_paths = {}
            # 上一层的每个节点到本层节点的连接
            for path_last in paths_.keys():
                if path_last[-1] + node_now in zy.keys():  # 若转移概率不为 0
                    sub_paths[path_last + node_now] = paths_[path_last] + nodes[layer][node_now] + zy[
                        path_last[-1] + node_now]
            # 最短路径,即概率最大的那个
            sr_subpaths = pd.Series(sub_paths)
            sr_subpaths = sr_subpaths.sort_values()  # 升序排序
            node_subpath = sr_subpaths.index[-1]  # 最短路径
            node_value = sr_subpaths[-1]  # 最短路径对应的值
            # 把 node_now 的最短路径添加到 paths 中
            paths[node_subpath] = node_value
    # 所有层求完后，找出最后一层中各个节点的路径最短的路径
    sr_paths = pd.Series(paths)
    sr_paths = sr_paths.sort_values()  # 按照升序排序
    return sr_paths.index[-1]  # 返回最短路径（概率值最大的路径）


def simple_cut(s):
    if s:
        r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int)) + [0] * (maxlen - len(s))]),
                          verbose=False)[
                0][:len(s)]
        r = np.log(r)
        print(r)
        nodes = [dict(zip(['s', 'b', 'm', 'e'], i[:4])) for i in r]
        t = viterbi(nodes)
        words = []
        for i in range(len(s)):
            if t[i] in ['s', 'b']:
                words.append(s[i])
            else:
                words[-1] += s[i]
        return words
    else:
        return []


not_cuts = re.compile(u'([\da-zA-Z ]+)|[。，、？！\.\?,!]')


def cut_word(s):
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result


print(cut_word('深度学习主要是特征学习'))

```

结果：

`['深度', '学习', '主要', '是', '特征', '学习']`

## 最后
本例子使用 Bi-directional LSTM 来完成了序列标注的问题。本例中展示的是一个分词任务，但是还有其他的序列标注问题都是可以通过这样一个架构来实现的，比如 POS（词性标注）、NER（命名实体识别）等。在本例中，单从分类准确率来看的话差不多到 95% 了，还是可以的。可是最后的分词效果还不是非常好，但也勉强能达到实用的水平。

## 源代码地址

https://github.com/stephen-v/zh-segmentation-keras


