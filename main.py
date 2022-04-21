import numpy as np
import math
import random
import matplotlib.pyplot as plt


# s1,s2,p,q,r
real_param = {'s1': 0.2, 's2': 0.3, 'p': 0.7, 'q': 0.9, 'r': 0.4}
fake_param = {'s1': 0.1, 's2': 0.6, 'p': 0.5, 'q': 0.5, 'r': 0.3}
data_scale = 1
per_size = 1000
epoch = 20


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
    # print('s1:{:.4f}, s2:{:.4f}, p:{:.4f}, q:{:.4f}, r:{:.4f}'
    #       .format(s1, s2, p, q, r))
    for xi in x:
        num_1 = sum(xi)
        num_0 = len(xi) - num_1
        u1_i = (s1*math.pow(p, num_1)*math.pow(1-p, num_0))/((s1*math.pow(p, num_1)*math.pow(1-p, num_0)) +
                                                        (s2 * math.pow(q, num_1) * math.pow(1 - q, num_0)) +
                                                        ((1-s1-s2) * math.pow(r, num_1) * math.pow(1 - r, num_0)))
        # print(s1*math.pow(p, xi)*math.pow(1-p, 1-xi), (s2 * math.pow(q, xi) * math.pow(1 - q, 1 - xi)), ((1-s1-s2) * math.pow(r, xi) * math.pow(1 - r, 1 - xi)))
        u2_i = (s2 * math.pow(q, num_1) * math.pow(1 - q, num_0)) / ((s1 * math.pow(p, num_1) * math.pow(1 - p, num_0)) +
                                                                  (s2 * math.pow(q, num_1) * math.pow(1 - q, num_0)) +
                                                                  ((1 - s1 - s2) * math.pow(r, num_1) * math.pow(1 - r,num_0)))
        u1_list.append(u1_i)
        u2_list.append(u2_i)
    # print(u1_list)
    # print(u2_list)
    new_s1 = sum(u1_list)/len(x)
    new_s2 = sum(u2_list)/len(x)
    new_p = sum(u1_list[i]*sum(x[i])/len(x[i]) for i in range(len(x)))/sum(u1_list)
    new_q = sum(u2_list[i]*sum(x[i])/len(x[i]) for i in range(len(x)))/sum(u2_list)
    new_r = sum((1-u1_list[i]-u2_list[i])*sum(x[i])/len(x[i]) for i in range(len(x)))/sum((1-u1_list[i]-u2_list[i]) for i in range(len(x)))
    return new_s1, new_s2, new_p, new_q, new_r


def em(x, times):
    """
    进行em算法的多次迭代
    :param x: 数据
    :param times: 迭代次数
    :return:
    """
    i = 0
    z = [[fake_param['s1'], fake_param['s2'], 1- fake_param['s1'] - fake_param['s2']]]
    theta = [[fake_param['p'], fake_param['q'], fake_param['r']]]
    while i < times:
        fake_param['s1'], fake_param['s2'], fake_param['p'], fake_param['q'], fake_param['r'] = em_single(x)
        z.append([fake_param['s1'], fake_param['s2'], 1- fake_param['s1'] - fake_param['s2']])
        theta.append([fake_param['p'], fake_param['q'], fake_param['r']])
        i += 1
        if i % 1 == 0:
            print('s1:{:.4f}, s2:{:.4f}, p:{:.4f}, q:{:.4f}, r:{:.4f}'
                  .format(fake_param['s1'], fake_param['s2'], fake_param['p'], fake_param['q'], fake_param['r']))
    draw(np.array(z), np.array(theta))


def draw(z, theta):
    """
    绘制图像
    :param z:参数z历次迭代后取值
    :param theta: 参数theta历次迭代后取值
    :return:
    """
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.subplot(121)
    plt.plot(z[:, 0], label='s1')
    plt.plot(z[:, 1], label='s2')
    plt.plot(z[:, 2], label='s3')
    real_z = [real_param['s1'], real_param['s2'], 1 - real_param['s1'] - real_param['s2']]
    for t in real_z:
        plt.hlines(t, 0, len(z), linestyles='dashed')
    plt.legend()
    plt.title('隐参数随迭代次数变化')

    plt.subplot(122)
    plt.plot(theta[:, 0], label='p')
    plt.plot(theta[:, 1], label='q')
    plt.plot(theta[:, 2], label='r')
    real_theta = [real_param['p'], real_param['q'], real_param['r']]
    for t in real_theta:
        plt.hlines(t, 0, len(theta), linestyles='dashed')
    plt.legend()
    plt.title('各硬币掷正面概率随迭代次数变化')

    plt.suptitle(r'$z_{init}$=' + f'{z[0]}'+r'  $\theta_{init}$=' + f'{theta[0]}')

    # plt.show()
    plt.savefig(f'{data_scale}-{per_size}+z-{z[0]}+theta-{theta[0]}.png', dpi=300)


if __name__ == '__main__':
    data = make_data()
    # print(data)
    em(data, epoch)

