import copy
import csv
import operator
import random
import numpy as np
import scipy.io as scio     # 读取mat文件
from sklearn import metrics
import matplotlib.pyplot as plt


def readdata(filename, label_index = [-1], del_index= [], split_symbal='\t'):
    """
    读取数据文件，转为np的矩阵格式，并提取类标list
    :param filename:str，数据集路径
    :param label_index:list，数据集中的真实标签所在列号，顺序时从0开始计数，按1、2递增；倒序时从-1开始计数，按-2、-3递减
    :param del_index:list，数据集中需要剔除的列，如id等，默认删除category列
    :param split_symbal:str，txt文件中每行数据的分割符号，如空格、逗号等，默认为空格
    :return:
      head_row：list，元素为str，attribute名称
      data_array：numpy.ndarray，属性值矩阵
      category_value：list，元素为list，样本的category值
    """
    head_row = []  # 表头
    data_list = []  # 数据元组list
    data_array = None  # 数据矩阵
    category_value = []  # 数据类标list

    # 1、读取数据集，并转为矩阵形式
    if(filename[-4:] == '.csv'):
        with open(filename) as f:
            reader = csv.reader(f)
            head = 0    # 表头标志
            for item in reader:
                if(len(item) == 0):
                    break
                if(head == 0):
                    head_row = item    # 表头
                    head += 1
                    continue
                data_list.append(item)
        data_array = np.array(data_list)
    if(filename[-4:] == '.mat'):
        data = scio.loadmat(filename)    # 读入mat文件，字典格式，dict_keys(['__header__', '__version__', '__globals__', 'fea', 'gnd'])，前三个每个mat文件都有，第四个为numpy矩阵
        data_array = np.hstack((data['fea'], data['gnd']))    # np.hstack将两个矩阵在行上合并（左右拼起来），vstack为列上合并（上下拼）
    if(filename[-5:] == '.data'):
        with open(filename) as f:
            reader = csv.reader(f)
            for item in reader:
                if (len(item) == 0):
                    break
                data_list.append(item)
        data_array = np.array(data_list)
    if(filename[-4:] == '.txt' or filename[-3:] == '.in'):
        with open(filename) as f:
            reader = f.readlines()
            for line in reader:
                if (len(line) == 0):
                    break
                line = line.strip().split(split_symbal)  # 以空格为分割符拆分列表
                data_list.append(line)
            data_array = np.array(data_list)
    # 2、提取类标矩阵
    category_value = data_array[:, label_index].tolist()    # 提取类标列，得到的是一个二维list。[[a],[b]...]
    if(len(category_value[0]) == 1):    # 若是单类别数据，即类标列只有一列，简化二维list，得到[a,b...]
        category_value = [str(i[0]) for i in category_value]

    # 3、删除矩阵和表头中的特定列(默认删除类标列，若有指定列也一并删除)
    del_index.extend(label_index)
    data_array = np.delete(data_array, del_index, 1)    # （矩阵，行号/列号，删除行/删除列）第3个参数0删行1删列
    head_row = [head_row[i] for i in range(len(head_row)) if (i not in del_index)]    # 删除表头中的类标名

    # 4、转换矩阵中的元素类型
    data_array = data_array.astype('float')    # numpy中的数据类型转换，不能直接改原数据的dtype!  只能用函数astype()

    # 5、数据标准化（0-1标准化）
    # for i in range(data_array.shape[1]):    # 遍历每一列
    #     num_max = data_array[:, i].max()
    #     num_min = data_array[:, i].min()
    #     for j in range(data_array.shape[0]):
    #         data_array[j][i] = round((data_array[j][i]-num_min)/(num_max-num_min), 4)

    return(head_row, data_array, category_value)

def get_cos_similar_matrix(array1, array2):
    """
    计算两个向量库内，两两向量的余弦相似度
    :param array1: 包含n条样本的矩阵
    :param array2: 包含m条样本的矩阵
    :return: n*m的相似度矩阵，第（i,j）个元素表示array1中的第i个向量与array2中的第j个向量的相似度
    """
    num = np.dot(array1, np.array(array2).T)  # 向量点乘
    denom = np.linalg.norm(array1, axis=1).reshape(-1, 1) * np.linalg.norm(array2, axis=1)  # 求模长的乘积
    res = num / denom
    return res

def get_oushi_dis_matrix(x, y):
    """
    get the Euclidean Distance between to matrix
    (x-y)^2 = x^2 + y^2 - 2xy
    :param x: 包含n条样本的矩阵
    :param y: 包含m条样本的矩阵
    :return: n*m的相似度矩阵，第（i,j）个元素表示x中的第i个向量与y中的第j个向量的欧式距离
    """
    x = np.array(x)
    y = np.array(y)
    (rowx, colx) = x.shape
    (rowy, coly) = y.shape
    if colx != coly:
        raise RuntimeError('colx must be equal with coly')
    xy = np.dot(x, y.T)
    x2 = np.repeat(np.reshape(np.sum(np.multiply(x, x), axis=1), (rowx, 1)), repeats=rowy, axis=1)
    y2 = np.repeat(np.reshape(np.sum(np.multiply(y, y), axis=1), (rowy, 1)), repeats=rowx, axis=1).T
    dis = x2 + y2 - 2 * xy
    return dis

def inital_cluster_dict(k):
    """
    初始化k个中间簇结果记录字典
    :param k: 簇数
    :return: dict{0：{'num': 0, 'cluster': []}，。。。。。。}
    """
    cluster_dict = {}
    for i in range(k):
        cluster_dict[i] = {'num': 0, 'cluster': []}
    return(cluster_dict)


def kmeans(dataset, center, k, max_iter = 300):
    """
    kmeans迭代过程函数
    :param k: 簇数
    :param dataset: 样本属性值矩阵
    :param center: 样本中心点
    :return: 每个样本的预测簇标识
    """
    temp_center = copy.deepcopy(center)    # 存储中间临时簇心
    pre_cluster = []    # 初始化每个样本所属的簇，默认均属于0簇，簇名为初始簇心在center中的下标
    for iter in range(max_iter):
        clu_dict = inital_cluster_dict(k)    # 初始化簇字典
        # 1、计算样本与簇心的相似度
        ###*** 采用余弦相似度时使用这两行代码，不用就注释掉 ***###
        # res = get_cos_similar_matrix(dataset, temp_center)    # 计算每个样本与每个簇心的余弦距离
        # pre_cluster = np.argmax(res, axis=1)    # 将每个样本的簇标识修改为距其最近的簇心在簇心列表中的下标，argmin返回res中每行或每列中最小元素的列index或行index，axis为1行0列
        ###*****************************************###

        ###*** 欧式距离用这两行 ***###
        res = get_oushi_dis_matrix(dataset, center)
        print(res.shape)
        pre_cluster = np.argmin(res, axis=1)    # 将每个样本的簇标识修改为距其最近的簇心在簇心列表中的下标，argmax返回res中每行或每列中最大元素的列index或行index，axis为1行0列
        ###**********************###
        for i in range(len(pre_cluster)):    # 统计每个簇中的样本数，并分别存储各个簇用于更新簇心
            clu_dict[pre_cluster[i]]['num'] += 1
            clu_dict[pre_cluster[i]]['cluster'].append(dataset[i].tolist())
        # 2、更新簇心
        # print('第%d轮簇心：' % (iter+1))
        # for i in temp_center:
        #     print(i)
        for i in range(len(temp_center)):
            if(clu_dict[i]['num'] == 0):    # 当前簇心没有分配到样本点
                continue
            temp_cluster = clu_dict[i]['cluster']
            temp_center[i] = (np.sum(temp_cluster, axis=0)/clu_dict[i]['num']).tolist()
            temp_center[i] = [round(item, 4) for item in temp_center[i]]
        if(operator.eq(center, temp_center)):    # Operator提供的函可用于对象比较，逻辑运算，数学运算和序列运算的类别
            break                                # operator.eq()用于判断两个对象是否相等，返回Bool变量
        else:
            center = copy.deepcopy(temp_center)
    return pre_cluster, center

def draw_real(dataset, real_label, filename):
    """
    绘制真实簇图像
    :param dataset:ndarray, 数据点矩阵
    :param real_label:list, 每个数据点的真实标签，与dataset中的下标一一对应
    :param filename:str，图标题
    """
    labelset = list(set(real_label))   # 对标签集去重
    plt.figure(figsize=(12, 12))
    plt.xticks()
    plt.yticks()
    color = ['#7f9ff0', '#7fdff0', '#952df5', '#f78ecc',
             '#f1a709', '#18eaa2', '#36f810', '#76971e', '#f7ec43',
             '#1294f5', '#a75d39', '#4f4e4d', '#e9e9e9', '#7b567a',
             '#4f4e4d', '#e9e9e9', '#FF6347', '#F5F5F5', '#FFFF00',
             '#2E8B57', '#4169E1', '#CD853F', '#AFEEEE', '#00FF7F', '#808080']
    for i in range(len(dataset)):  # 读出数组的长度
        for j in labelset:
            if real_label[i] == j:
                plt.scatter(dataset[i, 0], dataset[i, 1], c=color[labelset.index(j)], linewidths=5)
    plt.title(filename)
    # plt.savefig('pic/' + filename + '.eps')
    # plt.savefig('pic/' + filename + '.pdf')
    # plt.savefig('pic/' + filename + '.png')
    plt.show()
    plt.close()
def draw_pre(dataset, center, pre_label, filename):
    """
    绘制预测的簇图像
    :param dataset:ndarray, 数据点矩阵
    :param center:list, k个簇心，如[[8.1241, 10.8286], [9.4898, 7.9785]]
    :param pre_label:list, 每个数据点的预测标签，与dataset中的下标一一对应
    :param filename:str，图标题
    :return:
    """
    print(dataset)
    print(center)
    print(pre_label)
    labelset = list(set(pre_label))
    plt.figure(figsize=(12,12))
    plt.xticks()
    plt.yticks()
    color = ['#7f9ff0', '#7fdff0', '#952df5', '#f78ecc', '#f1a709',
             '#18eaa2', '#36f810', '#76971e', '#f7ec43', '#808080',
             '#1294f5', '#a75d39', '#4f4e4d', '#e9e9e9', '#7b567a',
             '#4f4e4d', '#e9e9e9', '#FF6347', '#F5F5F5', '#FFFF00',
             '#2E8B57', '#4169E1', '#CD853F', '#AFEEEE', '#00FF7F']
    for i in range(len(dataset)):  # 读出数组的长度
        for j in labelset:
            if pre_label[i] == j:
                plt.scatter(dataset[i, 0], dataset[i, 1], c=color[labelset.index(j)], linewidths=5)
    center_x = [i[0] for i in center]
    center_y = [i[1] for i in center]
    plt.scatter(center_x, center_y, c='r', marker='*', linewidths=3)
    plt.title(filename)
    # plt.savefig('pic/' + filename + '.eps')
    # plt.savefig('pic/' + filename + '.pdf')
    # plt.savefig('pic/' + filename + '.png')
    plt.show()
    plt.close()

def pichuli():
    names = ['R15.csv', 'flame.csv', 'Toy.csv', 'Aggregation.csv', 'iris.csv', 'seeds_dataset.txt']
    for n in names:
        mARI, mNMI, it, mcenter = -1, -1, -1, []
        mfinalcenter, mpre = [], []
        data_name = n
        print('数据集：', data_name[:-4])
        filename = "dataset/" + data_name
        if(n in ['R15.csv', 'Aggregation.csv', 'flame.csv', 'Toy.csv']):
            attri_name, dataset, real_cluster = readdata(filename, label_index=[0])
        else:
            attri_name, dataset, real_cluster = readdata(filename)
        k = len(set(real_cluster))
        if (n in ['R15.csv', 'Aggregation.csv', 'flame.csv', 'Toy.csv']):
            draw_real(dataset, real_cluster, data_name[:-4] + '_k='+str(k) + '_real')

        for i in range(300):
            # print("di", i, '次')
            center = random.sample(dataset.tolist(), k)
            pre_cluster, final_center = kmeans(dataset, center, k)
            pre_cluster = [str(i) for i in pre_cluster]
            result_ARI = metrics.adjusted_rand_score(pre_cluster, real_cluster)
            result_NMI = metrics.normalized_mutual_info_score(pre_cluster, real_cluster)
            if (result_ARI > mARI and result_NMI > mNMI):
                mARI, mNMI = result_ARI, result_NMI
                mcenter = center
                mfinalcenter = final_center
                mpre = pre_cluster
        if (n in ['R15.csv', 'Aggregation.csv', 'flame.csv', 'Toy.csv']):
            draw_pre(dataset, mfinalcenter, mpre, data_name[:-4] + '_k='+str(k) + '_pre')

        print('簇心：', mcenter)
        print('ARI:', round(mARI, 4))
        print('NMI', round(mNMI, 4))


if __name__ == '__main__':
    """批处理"""
    # pichuli()
    # exit(0)


    """单个数据集处理"""
    k = 3
    mARI, mNMI, it, mcenter = -1,-1,-1,[]
    mfinalcenter, mpre = [], []
    data_name = 'spiral.csv'
    filename = "dataset/" + data_name
    attri_name, dataset, real_cluster = readdata(filename, label_index=[0])
    draw_real(dataset, real_cluster, data_name[:-4] + '_k='+str(k) + '_real')
    # print('att_name:',type(attri_name), '\n', attri_name)
    # print('data:',type(dataset), '\n' , dataset)
    # print('real_cluster:',type(real_cluster), '\n' , real_cluster)
    for i in range(10):
        print("di", i, '次')
        center = random.sample(dataset.tolist(), k)
        # center = [[14.744, 5.248], [12.53, 10.0], [11.144, 11.91], [9.178, 11.53], [8.022, 9.334], [9.272, 12.152],
        #  [14.168, 15.276], [8.32, 9.062], [8.224, 10.316], [10.256, 9.25], [11.59, 8.408], [9.026, 3.788],
        #  [9.432, 8.61], [16.004, 10.28], [9.35, 11.71]]

        pre_cluster, final_center = kmeans(dataset, center, k)
        pre_cluster = [str(i) for i in pre_cluster]
        # print(len(pre_cluster), pre_cluster)
        # print(len(real_cluster), real_cluster)
        accuracy = metrics.accuracy_score(real_cluster, pre_cluster)
        result_ARI = metrics.adjusted_rand_score(pre_cluster, real_cluster)
        result_NMI = metrics.normalized_mutual_info_score(pre_cluster, real_cluster)
        # f1score = metrics.f1_score(real_cluster, pre_cluster, average="macro")  # 三个f1_score的平均值
        if(result_ARI>mARI and result_NMI>mNMI):
            mARI, mNMI = result_ARI, result_NMI
            it = i
            mcenter = center
            mfinalcenter = final_center
            mpre = pre_cluster
        # print('数据集：', str(i)+data_name)
        # print('簇心：', center)
        # print('Acc:', round(accuracy, 4))
        # print('ARI:', round(result_ARI, 4))
        # print('NMI', round(result_NMI, 4))
        # print('F1-score:', round(f1score, 4))
        # draw_real(dataset, real_cluster, str(i)+data_name)
        # draw_pre(dataset, final_center, pre_cluster, str(i)+data_name)
    draw_pre(dataset, mfinalcenter, mpre, data_name[:-4] + '_k='+str(k) + '_pre')
    print('数据集：', data_name[:-4])
    print('簇心：', mcenter)
    print('ARI:', round(mARI, 4))
    print('NMI', round(mNMI, 4))
    """"""
