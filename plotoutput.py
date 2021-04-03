import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False
from matplotlib.font_manager import FontProperties
import numpy as np
file= r'rockyou-train_CDN.log'
#file= r'rockyou-train_longlong.log'
iters=[]
lossD=[]
lossG=[]
npass=[]
ispass=[]
diff=[]
validatingScore=[]
with open(file,'r',encoding='utf-8') as f:
    for i in f:
        i=i[:-1]
        i=i.split('\t')
        iter_=int(i[0].split(':')[1])
        if iter_ <= 400000:
            if iter_ % 100 ==0:
                iters.append(float(i[0].split(':')[1]))
                lossD.append(float(i[1].split(':')[1]))
                lossG.append(float(i[2].split(':')[1]))
                npass.append(float(i[3].split(':')[1]))
                ispass.append(float(i[4].split(':')[1]))
                diff.append(float(i[5].split(':')[1]))
                validatingScore.append(float(i[6].split(':')[1]))



Chinesefont = FontProperties(fname=r".\SimSun.ttc", size=10.5)
Englishfont = FontProperties(fname=r".\tnr.TTF", size=10.5)

def LossD():
    plt.figure(figsize=[7.5, 5])
    plt.plot(iters, lossD)
    plt.xticks(fontproperties=Chinesefont)
    plt.yticks(fontproperties=Chinesefont)
    plt.xlabel("训练迭代轮数", fontproperties=Chinesefont)  # 设置x轴名称
    plt.ylabel("判别器损失", fontproperties=Chinesefont)  # 设置y轴名称
    plt.show()


def LossG():
    plt.figure(figsize=[7.5, 5])
    plt.plot(iters,lossG)
    plt.xticks(fontproperties=Chinesefont)
    plt.yticks(fontproperties=Chinesefont)
    plt.xlabel("训练迭代轮数", fontproperties=Chinesefont)  # 设置x轴名称
    plt.ylabel("生成器损失", fontproperties=Chinesefont)  # 设置y轴名称
    plt.show()

def validating_score():
    plt.figure(figsize=[7.5, 5])
    plt.plot(iters,validatingScore)
    plt.xticks(fontproperties=Chinesefont)
    plt.yticks(fontproperties=Chinesefont)
    plt.xlabel("训练迭代轮数", fontproperties=Chinesefont)  # 设置x轴名称
    plt.ylabel("验证集一个批处理单位口令平均得分", fontproperties=Chinesefont)  # 设置y轴名称
    plt.show()

def training_score():
    plt.figure(figsize=[7.5, 5])
    plt.plot(iters,ispass)
    plt.xticks(fontproperties=Chinesefont)
    plt.yticks(fontproperties=Chinesefont)
    plt.xlabel("训练迭代轮数", fontproperties=Chinesefont)  # 设置x轴名称
    plt.ylabel("训练集一个批处理单位口令平均得分", fontproperties=Chinesefont)  # 设置y轴名称
    plt.show()

def G_score():
    plt.figure(figsize=[7.5, 5])
    plt.plot(iters,npass)
    plt.xticks(fontproperties=Chinesefont)
    plt.yticks(fontproperties=Chinesefont)
    plt.xlabel("训练迭代轮数", fontproperties=Chinesefont)  # 设置x轴名称
    plt.ylabel("生成器生成一个批处理单位口令平均得分", fontproperties=Chinesefont)  # 设置y轴名称
    plt.show()

def diffs():
    plt.figure(figsize=[7.5, 5])
    plt.plot(iters,diff)
    plt.xticks(fontproperties=Chinesefont)
    plt.yticks(fontproperties=Chinesefont)
    plt.xlabel("训练迭代轮数", fontproperties=Chinesefont)  # 设置x轴名称
    plt.ylabel("生成器生成两批口令间差异", fontproperties=Chinesefont)  # 设置y轴名称
    plt.show()

validating_score()
