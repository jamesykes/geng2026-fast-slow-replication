import numpy as np
# import math
# import time
import os
import random

'''
-----------------初始条件和参数------------------------
'''
# 实验参数
Tau = 2  # 选择强度
Eta = 0.4  # 学习率
# Two States Two Actions
b = 1.2
r = 0.1
r_p = [1, 0, r, r]  # 收益矩阵-SH
r_d = [1, -r, b, 0]  # 收益矩阵-PD
possible_reward = [-r, 0, r, 1, b]
# r_ac = [1, 0, -r]
# r_ad = [0, r, b]
u = [[1, 0, r, r], [1, -r, b, 0]]
space = 0.01  # 坐标刻度间隔
t_max = 200
'''联合分布'''
# Qc_num = round((max(r_ac) - min(r_ac)) / space) + 1
# Qd_num = round((max(r_ad) - min(r_ad)) / space) + 1
Qc_num = round((max(possible_reward) - min(possible_reward)) / space) + 1
Qd_num = round((max(possible_reward) - min(possible_reward)) / space) + 1
game_num = 2
'''
---------------离散化坐标轴----------------------
'''


def appro(number):
    # 往前后寻找相邻刻度
    point_len = len(str(space).split(".")[1])
    f = 10 ** point_len
    f1 = 0.1 ** point_len
    left = np.around(number, point_len)
    right = left

    while True:
        if int(np.around(right * f)) % int(np.around(space * f)) == 0:
            right = int(np.around(right * f))
            break
        right += f1
    while True:
        if int(np.around(left * f)) % int(np.around(space * f)) == 0:
            left = int(np.around(left * f))
            break
        left -= f1

    if abs(left / f - number) <= abs(right / f - number):
        return left / f
    else:
        return right / f


# 迭代参数
T = 0  # 时间步
ave_qc = 0  # 平均qc
ave_qd = 0  # 平均qd
ave_xc = 0  # 平均c策略
ave_xd = 0  # 平均d策略
ave_fp = 0  # 表示繁荣状态的博弈类型(game1)在所有交互中所占的比例
"""
---------------------计算功能函数------------------------
"""


# def init_p():
#     p_init = np.zeros((Qc_num, Qd_num, game_num, Qc_num, Qd_num))
#     qc_init = 0
#     qd_init = 0
#     game_init = 0
#     p_init[round((qc_init - min(r_ac)) / space)][round((qd_init - min(r_ad)) / space)][game_init][
#         round((qc_init - min(r_ac)) / space)][
#         round((qd_init - min(r_ad)) / space)] = 1.0
#     return p_init

# Beta分布
def init_p():
    p_init = np.zeros((Qc_num, Qd_num, game_num, Qc_num, Qd_num))
    qc_beta1 = 20
    qc_beta2 = 80
    qd_beta1 = 80
    qd_beta2 = 20
    r_min = min(possible_reward)
    r_max = max(possible_reward)
    sample_n = Qc_num ** 2 * 10
    P = {}
    for i in range(Qc_num):
        for j in range(Qd_num):
            qc = appro(i * space + min(possible_reward))
            qd = appro(j * space + min(possible_reward))
            P[(qc, qd)] = 0
    for i in range(sample_n):
        qc = r_min + (r_max - r_min) * random.betavariate(qc_beta1, qc_beta2)
        qd = r_min + (r_max - r_min) * random.betavariate(qd_beta1, qd_beta2)
        qc = appro(qc)
        qd = appro(qd)
        P[(qc, qd)] += 1
    for i in range(Qc_num):
        for j in range(Qd_num):
            qc = appro(i * space + min(possible_reward))
            qd = appro(j * space + min(possible_reward))
            P[(qc, qd)] /= sample_n
    for i in range(Qc_num):
        for j in range(Qd_num):
            qc1 = appro(i * space + min(possible_reward))
            qd1 = appro(j * space + min(possible_reward))
            if P[(qc1, qd1)] > 0:
                for g in range(game_num):
                    for k in range(Qc_num):
                        for m in range(Qd_num):
                            qc2 = appro(k * space + min(possible_reward))
                            qd2 = appro(m * space + min(possible_reward))
                            if P[(qc2, qd2)] > 0:
                                p_init[i][j][g][k][m] = P[(qc1, qd1)] * P[(qc2, qd2)] * 0.5
    return p_init


# def init_delta_q():
#     q=np.zeros((Qc_num,Qd_num))
#     return q

'''初始化存储联合分布概率以及Q值变化量的数组'''
p = init_p()
# print('initialization finish')
q_c = np.zeros((Qc_num, Qd_num))
q_d = np.zeros((Qc_num, Qd_num))


# delta_q=init_delta_q()

# 函数功能：求Q的期望与策略X的期望
def expected():
    global ave_qc, ave_qd
    global ave_xc, ave_xd, ave_fp
    ave_qc = 0
    ave_qd = 0
    ave_xc = 0
    ave_xd = 0
    ave_fp = 0
    ''' Calculation '''
    for i in range(Qc_num):
        for j in range(Qd_num):
            if np.sum(p[i][j]) > 0:
                # qc = i * space + min(r_ac)
                # qd = j * space + min(r_ad)
                qc = i * space + min(possible_reward)
                qd = j * space + min(possible_reward)
                ave_qc += qc * np.sum(p[i][j])
                ave_qd += qd * np.sum(p[i][j])
                ave_fp += np.sum(p[i][j][0])
                temp = np.e ** (Tau * qc) + np.e ** (Tau * qd)
                ave_xc += (np.e ** (Tau * qc) / temp) * np.sum(p[i][j])
                ave_xd += (np.e ** (Tau * qd) / temp) * np.sum(p[i][j])
    sum_p = np.sum(p)
    ave_qc /= sum_p
    ave_qd /= sum_p
    ave_xc /= sum_p
    ave_xd /= sum_p
    ave_fp /= sum_p


# 函数功能：计算(qc,qd)处采取不同动作对应的Q值变化量
def delta():
    global q_c, q_d
    q_c = np.zeros((Qc_num, Qd_num))
    q_d = np.zeros((Qc_num, Qd_num))
    for i in range(Qc_num):
        for j in range(Qd_num):
            mp = np.sum(p[i][j])
            if mp > 0:
                payoff_c = 0
                payoff_d = 0
                for g in range(game_num):
                    for k in range(Qc_num):
                        for m in range(Qd_num):
                            # qc = k * space + min(r_ac)
                            # qd = m * space + min(r_ad)
                            qc = k * space + min(possible_reward)
                            qd = m * space + min(possible_reward)
                            xc = np.e ** (Tau * qc) / (np.e ** (Tau * qc) + np.e ** (Tau * qd))
                            xd = np.e ** (Tau * qd) / (np.e ** (Tau * qc) + np.e ** (Tau * qd))
                            payoff_c += (p[i][j][g][k][m] / mp) * (xc * u[g][0] + xd * u[g][1])
                            payoff_d += (p[i][j][g][k][m] / mp) * (xc * u[g][2] + xd * u[g][3])
                # q_c[i][j] = Eta * (payoff_c - (i * space + min(r_ac)))
                # q_d[i][j] = Eta * (payoff_d - (j * space + min(r_ad)))
                q_c[i][j] = Eta * (payoff_c - (i * space + min(possible_reward)))
                q_d[i][j] = Eta * (payoff_d - (j * space + min(possible_reward)))


# 函数功能：计算P值变化量，并更新P值
# 参数：    无
# 返回值：   无
def delta_p():
    global p
    temp_p = np.zeros((Qc_num, Qd_num, game_num, Qc_num, Qd_num))
    for i in range(Qc_num):
        for j in range(Qd_num):
            if np.sum(p[i][j]) > 0:
                for g in range(game_num):
                    for k in range(Qc_num):
                        for m in range(Qd_num):
                            if p[i][j][g][k][m] > 0:
                                # qc1 = i * space + min(r_ac)
                                # qd1 = j * space + min(r_ad)
                                # qc2 = k * space + min(r_ac)
                                # qd2 = m * space + min(r_ad)
                                qc1 = i * space + min(possible_reward)
                                qd1 = j * space + min(possible_reward)
                                qc2 = k * space + min(possible_reward)
                                qd2 = m * space + min(possible_reward)
                                xc1 = np.e ** (Tau * qc1) / (np.e ** (Tau * qc1) + np.e ** (Tau * qd1))
                                xd1 = np.e ** (Tau * qd1) / (np.e ** (Tau * qc1) + np.e ** (Tau * qd1))
                                xc2 = np.e ** (Tau * qc2) / (np.e ** (Tau * qc2) + np.e ** (Tau * qd2))
                                xd2 = np.e ** (Tau * qd2) / (np.e ** (Tau * qc2) + np.e ** (Tau * qd2))
                                new_qc1 = appro(qc1 + q_c[i][j])
                                new_qd1 = appro(qd1 + q_d[i][j])
                                new_qc2 = appro(qc2 + q_c[k][m])
                                new_qd2 = appro(qd2 + q_d[k][m])
                                new_i = round((new_qc1 - min(possible_reward)) / space)
                                new_j = round((new_qd1 - min(possible_reward)) / space)
                                new_k = round((new_qc2 - min(possible_reward)) / space)
                                new_m = round((new_qd2 - min(possible_reward)) / space)
                                # new_i = round((new_qc1 - min(r_ac)) / space)
                                # new_j = round((new_qd1 - min(r_ad)) / space)
                                # new_k = round((new_qc2 - min(r_ac)) / space)
                                # new_m = round((new_qd2 - min(r_ad)) / space)
                                # 博弈转移规则决定g的值
                                if g == 0:
                                    temp_p[new_i][j][0][new_k][m] += p[i][j][g][k][m] * xc1 * xc2
                                    temp_p[new_i][j][0][k][new_m] += p[i][j][g][k][m] * xc1 * xd2
                                    temp_p[i][new_j][0][new_k][m] += p[i][j][g][k][m] * xd1 * xc2
                                    temp_p[i][new_j][1][k][new_m] += p[i][j][g][k][m] * xd1 * xd2
                                else:
                                    temp_p[new_i][j][0][new_k][m] += p[i][j][g][k][m] * xc1 * xc2
                                    temp_p[new_i][j][1][k][new_m] += p[i][j][g][k][m] * xc1 * xd2
                                    temp_p[i][new_j][1][new_k][m] += p[i][j][g][k][m] * xd1 * xc2
                                    temp_p[i][new_j][1][k][new_m] += p[i][j][g][k][m] * xd1 * xd2
    p = temp_p


# 函数功能：保存每个时刻的值
# 参数：    无
# 返回值：  无
# path = 'Tau=' + str(Tau) + '_Eta=' + str(Eta) + '_b=' + str(b) + '_r=' + str(r) + '_rebuttal'
# if not os.path.exists(path):
#     os.mkdir(path)

path = 'Tau=' + str(Tau) + '_Eta=' + str(Eta) + '_b=' + str(b) + '_r=' + str(r) + '_rebuttal_beta(20,80,80,20)_random'
if not os.path.exists(path):
    os.mkdir(path)


# path = 'Tau=' + str(Tau) + '_Eta=' + str(Eta) + '_bp=' + str(b_p) + '_bd=' + str(b_d) + '_c=' + str(c) + '_case1_0.01'
# if not os.path.exists(path):
#     os.mkdir(path)


def save_data():
    # -----------------计算平均Q值、平均策略X、各博弈类型比例以及保存-------------------
    with open(path + '/ts_results.csv', "a") as file:
        file.writelines([str(T) + ',', str(ave_qc) + ',', str(ave_qd) + ',', str(ave_xc) + ',',
                         str(ave_xd) + ',', str(ave_fp) + '\n'])


# -----------------Q值分布-------------------
# def save_distr():
#     write_temp = []
#     for i in range(Qc_num):
#         for j in range(Qd_num):
#             qc = i * space + min(r_ac)
#             qd = j * space + min(r_ad)
#             write_temp.append([qc, qd, np.sum(p[i][j])])
#     np.savetxt(path + '/' + str(T) + '_p.csv', write_temp, delimiter = ',', fmt = '%.6f')


'''
开始迭代
'''
while T <= t_max:
    # t0 = time.time()
    expected()
    print('Step: %d Q(C):%f Q(D):%f' % (T, ave_qc, ave_qd))
    delta()
    save_data()  # 记录下T时刻的数据
    # if T == 5 or T == 10 or T == 50 or T == 100:
    #     save_distr()
    delta_p()  # 更新——Q值的密度分布P
    # t1 = time.time()
    # print(t1 - t0)
    T += 1  # 时间刻度往后移
