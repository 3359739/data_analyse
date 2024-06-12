"""
@big_name:data_teacher	
@file_name:topic	
@data:2024/6/6	
@developers:handsome_lxh
"""
import numpy as np
import pandas as pd
def one():
    # 把列表转换成Series
    data=['语文','数学','英语']
    new_substance=pd.Series(data)
    print(new_substance)
# one()
def two():
    # 把字典转换成Series
    data={'语文':90,'数学':85,'英语':95}
    new_substance=pd.Series(data)
    print(data)
# two()
def three():
    # 把Series转换成字典
    data=['语文','数学','英语']
    new_substance=pd.Series(data)
    print(new_substance.to_dict())
    # 把Series转换成列表
    print(new_substance.tolist())
# three()
def four():
    # 创建一个空的Series，指定索引
    data=pd.Series([1,2,3],index=['语文','数学','英语'])
    #Series创建dataframe
    new_=pd.DataFrame(data=data,columns=['lbiao'])
    print(new_)
# four()
def five():
    # 为Series添加元素
    data=pd.Series([1,2,3],index=['语文','数学','英语'])
    data=data._append(pd.Series([4,5,6],index=['物理','化学','生物']))
    print(data)
# five()
def six():

    data = pd.Series([1, 2, 3], index=['语文', '数学', '英语'])
    # series转换成dataframe
    data=data.to_frame()
    print(data)
    # series转换成dataframe
    data=data.reset_index()
    data.columns=['weixing','www']
    print(data)

# six()
def seven():
    data=pd.DataFrame({
        "nema":["lxh","lxh1","lxh2"],
        "sum":['ddddd','dddd','ddd'],
        'ooo':['www','www','www']
    })
    # 字典创建dataframe
    print(data)
    # 为dataframe设置索引
    data=data.set_index('nema')
    print(data)
# seven()
def eight():
    # 生成日期
    data=pd.date_range(start='20220101',periods=32)
    # 安年来生成
    data=pd.date_range(start='2022',end='2023',freq='H')#W-MON按周生成  小时H
    # 用日期生成DataFrame
    year=pd.date_range(start='20220101',periods=52)
    df=pd.DataFrame(data={'year':year})
    # 显示一年的第几天dayofyear,day显示一个月的第几天
    df['day']=df['year'].dt.dayofyear
    print(df)
# eight()
def nine():
    df=pd.DataFrame({
        'A':[1,0,2,3],
        'B':[1,4,5,6]
    })
    # 查看表的基础信息
    print(df.info())
    # 查看基础统计
    print(df.describe())
    # 统计每列值出现的次数
    print(df['A'].value_counts())
    print(df['A'].argmin())
    print(df.loc[df['A'].argmin()])
# nine()
def ten():
    # 删除列
    df=pd.DataFrame({
        'A':[1,0,2,3,4],
        'B':[1,4,5,6,None]
    })
    df1=df.drop('A',axis=1)
    print(df1)
    # 删除行
    df2=df.drop(0)
    print(df2)
    # 删除重复值
    df3=df.drop_duplicates()
    print(df3)
    # 删除空值
    df3=df.dropna()
    print(df3)
# ten()
def eleven():
    data=pd.read_csv('./data_table/Telco-Customer-Churn.csv')
    # 空值为True
    print(data.isnull())
    # 汇总每列的空值
    print(data.isnull().sum())
# eleven()
def twelve():
    # 读取数据
    data = pd.read_csv('./data_table/Telco-Customer-Churn.csv')
    # 打印前5行数据
    print(data.head())
    # 将所有数值型列之外的列过滤掉
    numeric_data = data.select_dtypes(include=[float, int])
    # 如果存在 'TotalCharges' 列，将其转换为数值型
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        numeric_data = data.select_dtypes(include=[float, int])
    # 计算相关矩阵
    corr_matrix = numeric_data.corr()
    # 打印相关矩阵
    #                 SeniorCitizen    tenure  MonthlyCharges  TotalCharges
    # SeniorCitizen        1.000000  0.016567        0.220173      0.102411
    # tenure               0.016567  1.000000        0.247900      0.825880
    # MonthlyCharges       0.220173  0.247900        1.000000      0.651065
    # TotalCharges         0.102411  0.825880        0.651065      1.000000
    # 解读意思是相关性接近1表示强相关，-1表示负相关，0表示不相关
    print(corr_matrix)
    # 随机采样
    print(data.sample(10))
# twelve()
import random
def thirteen():
    # 截断
    data1 =pd.Series(np.random.rand(20))
    data2 =pd.Series(np.random.randn(20))
    data=pd.concat([data1,data2],axis=1)
    data.columns=["data1","data2"]
    data['data3']=data['data2'].clip(-1.0,1.0)
    print(data)
    # 小于-1.0为-1.0，大于1.0为1.0，在-1.0到1.0之间不变
    #        data1     data2     data3
    # 0   0.770460 -0.272335 -0.272335
    # 1   0.119513 -2.178805 -1.000000
    # 2   0.676456  0.633410  0.633410
    # 3   0.473857  1.173449  1.000000
    # 4   0.150302  1.431688  1.000000
    # 5   0.154232  0.216329  0.216329
    print(data['data1'].nlargest(5))
    # 最大的5个数
    print(data['data1'].nsmallest(5))
    # 最小的5个数
    print(data['data1'].cumsum())
    # 最大值
    print('-------------------------------------------------------------------------')
    print(data['data1'].max())
    print(data['data1'].cummax())
    print(data['data1'].cummin())
    # 中位数
    print(data['data1'].median())
thirteen()
