import os
import random
import shutil

'''
将数据集，按比例划分为训练集和测试机
target_path: 原始数据集的路径，每个子文件夹代表一个类别
dataset_path: 划分后的数据集的路径，train和test两个文件夹，每个子文件夹代表一个类别
split_rate: 训练集和测试集的划分比例，训练集：测试集 = 9 : 1
'''

target_path = "E:\Project_Study\Pytorch_Study\Study_Project\data\classify"
dataset_path = './data/classfiy'

#训练集和测试的划分比例，训练集：测试集 = 9 : 1
split_rate = 0.3
def mkfile(path):
    if not os.path.exists(path):
        os.makedirs(path)

#获取文件夹下所有子文件夹的名称
target_class=[cla for cla in os.listdir(target_path) if os.path.isdir(os.path.join(target_path,cla))]

#创建train文件，并由类名创建子文件夹
mkfile(os.path.join(dataset_path,'train'))
#创建test文件，并由类名创建子文件夹
mkfile(os.path.join(dataset_path,'test'))

for cla in target_class:
    mkfile(os.path.join(dataset_path,'train',cla))
    mkfile(os.path.join(dataset_path, 'test', cla))

#遍历文件夹下所有图片，并按比例划分为训练集和测试集
for cla in target_class:
    cla_path=os.path.join(target_path,cla)#获取每个子文件夹的绝对路径
    img_name_list=os.listdir(cla_path)#获取每个子文件夹下的图片
    img_num=len(img_name_list)#获取每个子文件夹下的图片数量
    eval_index=random.sample(img_name_list,k=int(img_num*split_rate))#随机选择测试集图片的索引
    for index,img_name in enumerate(img_name_list):
        #按照比例划分测试集和训练集
        #将图片移动到test文件夹
        if img_name in eval_index:
            img_src=os.path.join(cla_path,img_name)#获取图片的绝对路径
            img_dst=os.path.join(dataset_path,'test',cla,img_name)#获取图片的目标路径
            shutil.copy(img_src,img_dst)#拷贝图片到测试集文件夹
        #其余的图片放入train文件夹
        else:
            img_src=os.path.join(cla_path,img_name)
            img_dst=os.path.join(dataset_path,'train',cla,img_name)
            shutil.copy(img_src,img_dst)#拷贝图片到训练集文件夹

        print("\r[{}] processing [{}/{}]".format(cla,index+1,img_num),end="")
    print()

print("Data partition finished!")

