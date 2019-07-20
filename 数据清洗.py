import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#导入数据
df=pd.read_csv(".\cs-training.csv")
print(df.head(5))

print(df.columns)

# columns转化为中文
states={"Unnamed: 0":"用户ID",
        "SeriousDlqin2yrs":"好坏客户",
        "RevolvingUtilizationOfUnsecuredLines":"可用额度比值",
        "age":"年龄",
        "NumberOfTime30-59DaysPastDueNotWorse":"逾期30-59天笔数",
        "DebtRatio":"负债率",
        "MonthlyIncome":"月收入",
        "NumberOfOpenCreditLinesAndLoans":"信贷数量",
        "NumberOfTimes90DaysLate":"逾期90天笔数",
        "NumberRealEstateLoansOrLines":"固定资产贷款量",
        "NumberOfTime60-89DaysPastDueNotWorse":"逾期60-89天笔数",
        "NumberOfDependents":"家属数量"}
df.rename(columns=states,inplace=True)

print(df)
# df.info()看看有没有缺失值
df.info()
# 月收入缺失比较多，不能直接删除，利用填充平均值的方法进行补充，家属数量的缺失比较少，直接删除
df=df.fillna({"月收入":df["月收入"].mean()})
df1=df.dropna()
# 异常值处理，使用箱型图查看各特征的异常值
x1=df1["可用额度比值"]
x2=df1["负债率"]
fig=plt.figure(1)
ax=fig.add_subplot(111)
ax.boxplot([x1,x2])
ax.set_xticklabels(["可用额度比值","负债率"])
plt.show()

plt.rcParams["font.sans-serif"]='SimHei'
x3=df1["年龄"]
fig=plt.figure(2)
ax1=fig.add_subplot(111)
ax1.boxplot(x3)
ax1.set_xticklabels("年龄")
plt.show()

x4=df1["逾期30-59天笔数"]
x5=df1["逾期60-89天笔数"]
x6=df1["逾期90天笔数"]
fig=plt.figure(3)
ax=fig.add_subplot(111)
ax.boxplot([x4,x5,x6])
ax.set_xticklabels(["逾期30-59天笔数","逾期60-89天笔数","逾期90天笔数"])
plt.show()

x7=df1["信贷数量"]
x8=df1["固定资产贷款量"]
fig=plt.figure(4)
ax=fig.add_subplot(111)
ax.boxplot([x7,x8])
ax.set_xticklabels(["信贷数量","固定资产贷款量"])
plt.show()

#除去异常值
df1=df1[df1["可用额度比值"]<=1]
df1=df1[df1["年龄"]>0]
df1=df1[df1["逾期30-59天笔数"]<80]
df1=df1[df1["固定资产贷款量"]<50]

# 查看好坏客户的整体情况
grouped=df1["用户ID"].groupby(df1["好坏客户"]).count()
print("坏客户占比：{:.2%}".format(grouped[1]/grouped[0]))
grouped.plot(kind="bar")
plt.show()

# 查看各个年龄段好坏客户的整体情况
age_cut=pd.cut(df1["年龄"],5)
age_cut_grouped=df1["好坏客户"].groupby(age_cut).count()
age_cut_grouped1=df1["好坏客户"].groupby(age_cut).sum()
df2=pd.merge(pd.DataFrame(age_cut_grouped), pd.DataFrame(age_cut_grouped1),right_index=True,left_index=True)
df2.rename(columns={"好坏客户_x":"好客户","好坏客户_y":"坏客户"},inplace=True)
df2.insert(2,"坏客户率",df2["坏客户"]/df2["好客户"])
print("age_cut:{}".format(df2))
ax1=df2[["好客户","坏客户"]].plot.bar()
ax1.set_xticklabels(df2.index,rotation=15)
ax1.set_ylabel("客户数")
ax1.set_title("年龄与好坏客户数分布图")
plt.show()

# 查看坏客户率随年龄的变化趋势
ax11=df2["坏客户率"].plot()
ax11.set_ylabel("坏客户率")
ax11.set_title("坏客户率随年龄的变化趋势图")
plt.show()

# 查看月收入和好坏客户的整体情况的关系
cut_bins=[0,5000,10000,15000,20000,100000]
month_cut=pd.cut(df1["月收入"],cut_bins)
month_cut_grouped=df1["好坏客户"].groupby(month_cut).count()
month_cut_grouped1=df1["好坏客户"].groupby(month_cut).sum()
df3=pd.merge(pd.DataFrame(month_cut_grouped), pd.DataFrame(month_cut_grouped1),right_index=True,left_index=True)
df3.rename(columns={"好坏客户_x":"好客户","好坏客户_y":"坏客户"},inplace=True)
df3.insert(2,"坏客户率",df3["坏客户"]/df3["好客户"])
ax23=df3[["好客户","坏客户"]].plot.bar()
ax23.set_xticklabels(df3.index,rotation=15)
ax23.set_ylabel("客户数")
ax23.set_title("好坏客户数与月收入关系")
plt.show()

# 月收入和坏客率的趋势
ax231=df3["坏客户率"].plot()
ax231.set_ylabel("坏客户率")
ax231.set_title("月收入与坏客户率关系")
plt.show()


# 查看家属数量和好坏客户的整体情况的关系
cut_bins=[0,2,4,20]
family_cut=pd.cut(df1["家属数量"],cut_bins)
family_cut_grouped=df1["好坏客户"].groupby(family_cut).count()
family_cut_grouped1=df1["好坏客户"].groupby(family_cut).sum()
df4=pd.merge(pd.DataFrame(family_cut_grouped), pd.DataFrame(family_cut_grouped1),right_index=True,left_index=True)
df4.rename(columns={"好坏客户_x":"好客户","好坏客户_y":"坏客户"},inplace=True)
df4.insert(2,"坏客户率",df4["坏客户"]/df4["好客户"])
ax24=df4[["好客户","坏客户"]].plot.bar()
ax24.set_xticklabels(df4.index,rotation=15)
ax24.set_ylabel("客户数")
ax24.set_title("好坏客户数与家属数量关系")
plt.show()

# 坏客率的家属数量关系趋势图
ax241=df4["坏客户率"].plot()
ax241.set_ylabel("坏客户率")
ax241.set_title("坏客户率与家属数量的关系")
plt.show()

# 绘制热力图，查看各个特征之间的关系
plt.rcParams["font.sans-serif"]='SimHei'
plt.rcParams['axes.unicode_minus'] = False
corr = df1.corr()#计算各变量的相关性系数
xticks = list(corr.index)#x轴标签
yticks = list(corr.index)#y轴标签
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap="rainbow",ax=ax1,linewidths=.5, annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
ax1.set_xticklabels(xticks, rotation=35, fontsize=10)
ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
plt.show()






cut1=pd.qcut(df1["可用额度比值"],4,labels=False)
print('1222222{}'.format(cut1))
cut2=pd.qcut(df1["年龄"],8,labels=False)
bins3=[-1,0,1,3,5,13]
cut3=pd.cut(df1["逾期30-59天笔数"],bins3,labels=False)
print('1222222{}'.format(cut3))
cut4=pd.qcut(df1["负债率"],3,labels=False)
cut5=pd.qcut(df1["月收入"],4,labels=False)
cut6=pd.qcut(df1["信贷数量"],4,labels=False)
bins7=[-1, 0, 1, 3,5, 20]
cut7=pd.cut(df1["逾期90天笔数"],bins7,labels=False)
bins8=[-1, 0,1,2, 3, 33]
cut8=pd.cut(df1["固定资产贷款量"],bins8,labels=False)
bins9=[-1, 0, 1, 3, 12]
cut9=pd.cut(df1["逾期60-89天笔数"],bins9,labels=False)
bins10=[-1, 0, 1, 2, 3, 5, 21]
cut10=pd.cut(df1["家属数量"],bins10,labels=False)


#好坏客户比率
rate=df1["好坏客户"].sum()/(df1["好坏客户"].count()-df1["好坏客户"].sum())

#定义woe计算函数
def get_woe_data(cut):
    grouped=df1["好坏客户"].groupby(cut,as_index = True).value_counts()
    woe=np.log(pd.DataFrame(grouped).unstack().iloc[:,1]/pd.DataFrame(grouped).unstack().iloc[:,0]/rate)#计算每个分组的woe值
    return woe
cut1_woe=get_woe_data(cut1)
cut2_woe=get_woe_data(cut2)
cut3_woe=get_woe_data(cut3)
cut4_woe=get_woe_data(cut4)
cut5_woe=get_woe_data(cut5)
cut6_woe=get_woe_data(cut6)
cut7_woe=get_woe_data(cut7)
cut8_woe=get_woe_data(cut8)
cut9_woe=get_woe_data(cut9)
cut10_woe=get_woe_data(cut10)
print(cut1_woe)
print(type(cut1_woe))


#定义IV值计算函数
def get_IV_data(cut,cut_woe):
    grouped=df1["好坏客户"].groupby(cut,as_index = True).value_counts()
    cut_IV=((pd.DataFrame(grouped).unstack().iloc[:,1]/df1["好坏客户"].sum()-pd.DataFrame(grouped).unstack().iloc[:,0]/(df1["好坏客户"].count()-df1["好坏客户"].sum()))*cut_woe).sum()
    return cut_IV

#计算各分组的IV值
cut1_IV=get_IV_data(cut1,cut1_woe)
cut2_IV=get_IV_data(cut2,cut2_woe)
cut3_IV=get_IV_data(cut3,cut3_woe)
cut4_IV=get_IV_data(cut4,cut4_woe)
cut5_IV=get_IV_data(cut5,cut5_woe)
cut6_IV=get_IV_data(cut6,cut6_woe)
cut7_IV=get_IV_data(cut7,cut7_woe)
cut8_IV=get_IV_data(cut8,cut8_woe)
cut9_IV=get_IV_data(cut9,cut9_woe)
cut10_IV=get_IV_data(cut10,cut10_woe)

#各组的IV值可视化
df_IV=pd.DataFrame([cut1_IV,cut2_IV,cut3_IV,cut4_IV,cut5_IV,cut6_IV,cut7_IV,cut8_IV,cut9_IV,cut10_IV],index=df1.columns[2:])
df_IV.plot(kind="bar")
# for a,b in zip(range(10),df3.values):
#     plt.text(a,b,'%.2f' % b, ha='center', va= 'bottom',fontsize=9)
plt.show()


#定义一个替换函数
def replace_data(cut,cut_woe):
    a=[]
    for i in cut.unique():
        a.append(i)
        a.sort()
    for m in range(len(a)):
        cut.replace(a[m],cut_woe.values[m],inplace=True)
    return cut

#进行替换
df_new = df1.copy()
df_new["可用额度比值"]=replace_data(cut1,cut1_woe)
df_new["年龄"]=replace_data(cut2,cut2_woe)
df_new["逾期30-59天笔数"]=replace_data(cut3,cut3_woe)
df_new["逾期90天笔数"]=replace_data(cut7,cut7_woe)
df_new["逾期60-89天笔数"]=replace_data(cut9,cut9_woe)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

x=df_new.iloc[:,2:]
y=df_new.iloc[:,1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

#模型训练
model=LogisticRegression()
clf=model.fit(x_train,y_train)
print("测试成绩:{}".format(clf.score(x_test,y_test)))
y_pred=clf.predict(x_test)
y_pred1=clf.decision_function(x_test)

#绘制ROC曲线以及计算AUC值
fpr, tpr, threshold = roc_curve(y_test, y_pred1)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange',
          label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_curve')
plt.legend(loc="lower right")
plt.show()