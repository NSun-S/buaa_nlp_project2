## 深度学习与自然语言处理第二次作业

<p align='right'> <strong>SY2106318 孙旭东 </strong></p>

**详细技术报告见**[SY2106318-深度学习和自然语言处理第一次作业](https://github.com/NSun-S/buaa_nlp_project1/raw/main/SY2106318-深度学习和自然语言处理第一次作业.pdf)

### 程序简介

#### 程序运行

修改参数设置部分

```python
real_param = {'s1': 0.2, 's2': 0.3, 'p': 0.7, 'q': 0.9, 'r': 0.4}
fake_param = {'s1': 0.1, 's2': 0.6, 'p': 0.5, 'q': 0.5, 'r': 0.3}
data_scale = 10000
per_size = 5
epoch = 20
```

real_param为数据生产时依赖的真实参数，其中，s1, s2为第一种和第二种硬币在袋子中所占的比例，p、q、r分别代表三种硬币掷出正面的概率；fake_param为假定的各个参数的初始值；data_scale为要生成的投掷硬币的组数；在实验中我们设置每组投掷的是同一硬币，per_scale为每组投掷结果中硬币投掷的次数；epoch为迭代进行的次数。

运行main.py即可得到结果展示部分对应内容。

#### 主要模块介绍

```python
def make_data():
    """
    根据真实参数(real_param)和数据规模(data_scale)生成数据
    :return: 生成的01序列
    """
    data = []
    for i in range(data_scale):
        which_icon = random.random()
        if which_icon < real_param['s1']:
            data.append([])
            for j in range(per_size):
                if random.random() < real_param['p']:
                    data[i].append(1)
                else:
                    data[i].append(0)
        elif which_icon < real_param['s1'] + real_param['s2']:
            data.append([])
            for j in range(per_size):
                if random.random() < real_param['q']:
                    data[i].append(1)
                else:
                    data[i].append(0)
        else:
            data.append([])
            for j in range(per_size):
                if random.random() < real_param['r']:
                    data[i].append(1)
                else:
                    data[i].append(0)
    return data
```

这一函数用于根据real_param、data_scale和per_size生成数据，即题目中要求的01序列，由于按照组进行生成，最终data的形状为data_scale*per_size。

```python
def em_single(x):
    """
    进行一次EM操作
    :param x: 数据
    :return: 更新后的fake_param
    """
    u1_list = []
    u2_list = []
    s1 = fake_param['s1']
    s2 = fake_param['s2']
    p = fake_param['p']
    q = fake_param['q']
    r = fake_param['r']
    for xi in x:
        num_1 = sum(xi)
        num_0 = len(xi) - num_1
        u1_i = (s1*math.pow(p, num_1)*math.pow(1-p, num_0))/((s1*math.pow(p, num_1)*math.pow(1-p, num_0)) +
                                                        (s2 * math.pow(q, num_1) * math.pow(1 - q, num_0)) +
                                                        ((1-s1-s2) * math.pow(r, num_1) * math.pow(1 - r, num_0)))
        u2_i = (s2 * math.pow(q, num_1) * math.pow(1 - q, num_0)) / ((s1 * math.pow(p, num_1) * math.pow(1 - p, num_0)) +
                                                                  (s2 * math.pow(q, num_1) * math.pow(1 - q, num_0)) +
                                                                  ((1 - s1 - s2) * math.pow(r, num_1) * math.pow(1 - r,num_0)))
        u1_list.append(u1_i)
        u2_list.append(u2_i)

    new_s1 = sum(u1_list)/len(x)
    new_s2 = sum(u2_list)/len(x)
    new_p = sum(u1_list[i]*sum(x[i])/len(x[i]) for i in range(len(x)))/sum(u1_list)
    new_q = sum(u2_list[i]*sum(x[i])/len(x[i]) for i in range(len(x)))/sum(u2_list)
    new_r = sum((1-u1_list[i]-u2_list[i])*sum(x[i])/len(x[i]) for i in range(len(x)))/sum((1-u1_list[i]-u2_list[i]) for i in range(len(x)))
    return new_s1, new_s2, new_p, new_q, new_r
```

其余代码见本仓库main.py。

### 结果展示



### 结论

1. 

### 参考文档

[EM算法 三硬币模型](https://www.cnblogs.com/huangshansan/p/10588318.html)

[EM算法-硬币实验的理解](https://blog.csdn.net/hunk954/article/details/103309468)