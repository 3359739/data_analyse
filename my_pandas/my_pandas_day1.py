"""
@big_name:data_teacher	
@file_name:my_pandas_day1	
@data:2024/5/31	
@developers:handsome_lxh
"""
import time

import pandas as pd

import matplotlib.pyplot as plt
def search_fun():
    # 读取表pd.read_类型()
    substance = pd.read_csv('data_table/beijing_tianqi_2017-2019.csv')
    # 设置索引
    substance.set_index('ymd', inplace=True)
    # 查询方法
    print(substance.loc["2017-01-01":"2017-01-31", :][substance['aqiLevel'] > 4])
    print(substance.loc["2017-01-01":"2017-01-31", :][substance['aqiLevel'] > 4]['aqiLevel'])


def data_statistics():
    # 读取表pd.read_类型()
    substance = pd.read_csv('data_table/beijing_tianqi_2017-2019.csv')
    # 设置索引
    substance.set_index('ymd', inplace=True)
    # 统计方法
    substance['yWendu'] = substance['yWendu'].str.replace("℃", "").astype('int32')
    # 提取所有数字列的统计结果
    print(substance.describe())
    # 去从
    print(substance['fengxiang'].unique())
    # 统计出现次数
    print(substance['fengxiang'].value_counts())


# data_statistics()
def fun(x):
    if x['yWendu'] > 20:
        return '高温'
    elif x['bWendu'] < -2:
        return '低温'
    else:
        return '常温'


def increase_list():
    # 读取表pd.read_类型()
    substance = pd.read_csv('data_table/beijing_tianqi_2017-2019.csv')
    # 设置索引
    substance.set_index('ymd', inplace=True)
    # 统计方法
    substance['yWendu'] = substance['yWendu'].str.replace("℃", "").astype('int32')
    substance['bWendu'] = substance['bWendu'].str.replace("℃", "").astype('int32')
    # 直接赋值
    substance['wendu_increase'] = substance['yWendu'] - substance['bWendu']
    print(substance)
    # apply
    substance['wendu_increase'] = substance.apply(fun, axis=1)
    print(substance.loc[substance['wendu_increase'] == '低温'])
    print(substance['wendu_increase'].value_counts())
    # assgin多行添加
    print(substance.assign(
        xing_wendu=lambda x: x['yWendu'] * 15,
        xing_wedu2=lambda x: x['bWendu'] * 5
    ))
    # 多行直接赋值
    substance['xing_list'] = ''
    substance.loc[substance['yWendu'] > 10, 'xing_list'] = '高温'
    substance.loc[substance['bWendu'] < -2, 'xing_list'] = '低温'
    substance.to_csv("sss.csv")


def lack_data():
    # 读取表pd.read_类型()
    substance = pd.read_csv('data_table/sss.csv')
    # 设置索引
    substance.set_index('ymd', inplace=True)
    # notnull无值为false
    print(substance['xing_list'].notnull())
    # istnull无值为True
    print(substance['xing_list'].isnull())
    # 直接删除空值
    # axis为那列表删除或则按行删除 how all或则any 全部为空删除有all 有一个为空删除any
    substance.dropna(axis=0, how='any')
    # 空的直接设置为指定的值
    substance.fillna({"xing_list": "成为"}, inplace=True)
    print(substance)
    substance.to_csv("data_table/sss.csv")


# 过滤和报警
def filter_alarm():
    # 读取表pd.read_类型()
    substance = pd.read_csv('data_table/beijing_tianqi_2017-2019.csv')
    # 设置索引
    # substance.set_index('ymd', inplace=True)
    # 统计方法
    substance['yWendu'] = substance['yWendu'].str.replace("℃", "").astype('int32')
    substance['bWendu'] = substance['bWendu'].str.replace("℃", "").astype('int32')
    # 过滤出只要2017-05的
    # print(substance.loc[substance['ymd'].str.startswith('2017-05'), :])
    # 排序ascending为False 降序 True为升序
    print(substance['ymd'].sort_values(ascending=False))
    # 多列排序按前后排
    sorted_substance = substance.sort_values(by=['yWendu', 'ymd'], ascending=False)
    print(sorted_substance)
def string_dispose():
    # 读取表pd.read_类型()
    substance = pd.read_csv('data_table/beijing_tianqi_2017-2019.csv')
    # 设置索引
    # substance.set_index('ymd', inplace=True)
    # 处理字符串
    substance['yWendu'] = substance['yWendu'].str.replace("℃", "").astype('int32')
import numpy as np
def aixs_and_index_understanding():
    # 再进行聚合操作时和执行删除时的aisx是相反的
    substance = pd.DataFrame(
        np.arange(12).reshape(3, 4),
        columns=['a', 'b', 'c', 'd']
    )
    print(substance)
    print(substance.mean(axis=1))
    print(substance.mean(axis=0))
    # 设置索引回原来列回不存在
    # drop=False为保留
    substance.set_index('a', inplace=True, drop=False)
    print(substance)
# aixs_and_index_understanding()
def relevancy_table():
    my_user1 = pd.DataFrame({
        "propre": [10, 11, 12, 13, 14],
        "age": [19, 12, 32, 45, 65],
    })
    my_user2 = pd.DataFrame({
        "propre": [10, 11, 12, 13, 14],
        "names": ["name1", "name2", "name3", "name4", "name5"],
    })
    # 一对一合表根据on=propre
    result = pd.merge(my_user1, my_user2, on='propre')
    print(result)
    # 一对多会增加数据量出现乘法
    my_user3 = pd.DataFrame({
        "propre": [10, 10, 10, 15, 13],
        "fun": ["篮球", "足球", "乒乓球", "羽毛球", "台球"]
    })
    my_user4 = pd.DataFrame({
        "propre": [10, 11, 12, 13, 14],
        "names": ["语文45", "数学55", "英语1", "语文4", "数学5"],
    })
    result = pd.merge(my_user3, my_user4, on='propre')
    print(result)
    # 合并concat
    df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3'],
                        'E': ['E0', 'E1', 'E2', 'E3']
                        })
    df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                        'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7'],
                        'F': ['F4', 'F5', 'F6', 'F7']
                        })
    # 直接向上插ignore_index=True忽略原来索引
    df3=pd.concat([df1, df2],ignore_index=True)
    print(df3)
    # 使用join = inner过滤掉不匹配的列
    df4=pd.concat([df1, df2], ignore_index=True, join="inner")
    print(df4)
    # 添加Series
    s = pd.Series([1, 2, 3, 4], name='P')
    print(pd.concat([df1, s.to_frame()], axis=1))
import os
def split_merge():
    my_path='./data_table/'
    my_mkdir_create=f"{my_path}/split"
    if not os.path.exists(my_mkdir_create):
        os.mkdir(my_mkdir_create)

    substance=pd.read_excel('./data_table/crazyant_blog_articles_source.xlsx')
    substance.set_index('id',inplace=True)
    yes=substance.shape[0]
    user_names = ["xiao_shuai", "xiao_wang", "xiao_ming", "xiao_lei", "xiao_bo"]
    v1=yes//4
    print(v1)
    list_my=[]
    for i in range(0,258,64):
        substance_new=substance.iloc[i:i+86]
        list_my.append(substance_new)
    for i,j in zip(list_my,user_names):
        i.to_excel(f"{my_mkdir_create}/{j}.xlsx")
    # merge
    list_my=[]
    for i in user_names:
        list_xing=pd.read_excel(f'{my_mkdir_create}/{i}.xlsx')
        list_my.append(list_xing)
    df_merge=pd.concat(list_my)
    df_merge.to_excel(f"{my_mkdir_create}/merge.xlsx",index=False)
# split_merge()
def index_advanced():
    substance=pd.read_excel('./data_table/互联网公司股票.xlsx')
    goshi=substance['公司'].unique()#获取一个有多少家公司
    # print(substance)
    cpoy=substance.groupby(by=['公司','日期'])['收盘'].mean()
    # print(cpoy.unstack())#把2级索引边成列
    # print(cpoy.reset_index())#重置索引
    # (cpoy['BABA'] > 166)
    #     po=po.loc[(po.index>"2017")&(po.index<"2022"),:]
    print(cpoy.loc[(cpoy>160),:])
# index_advanced()
def time_index():
    # 对时间进行设置
    substance=pd.read_csv('./data_table/beijing_tianqi_2017-2019.csv')
    substance['ymd']=pd.to_datetime(substance['ymd'])
    substance.set_index('ymd',inplace=True)
    print(substance.loc['2018-01':"2018-02"])
    print(substance.loc['2018-01':"2018-01-08"])
    # 对缺失值进行处理
    df=pd.DataFrame({
        'ymd':['2019-01-01','2019-01-02','2019-01-04','2019-01-05'],
        'pv':[19,65,42,56],
        'nv':[22,56,44,55]
    })
    xing_df=df.set_index(pd.to_datetime(df['ymd']))
    xing_df.drop('ymd',axis=1,inplace=True)
    # 这里重新生成完整的索引
    xing=pd.date_range(start='2019-01-01',end='2019-01-05')
    # 然后这里替换调之前datafrom的索引把值设置为0
    my_time=xing_df.reindex(xing,fill_value=0)
    # 生成图表
    # my_time.plot()
    # plt.show()

    # 重新采样
    print(xing_df)
    # 传D是按天进行重新采样
    print(xing_df.resample('D').sum())
# time_index()
# pandas怎样实现Excel的vlookup并且在指定列后面输出
def vlookup():
     user_name1=pd.read_excel('./data_table/学生信息表.xlsx',sheet_name='Sheet1')
     user_name2=pd.read_excel('./data_table/学生成绩表.xlsx',sheet_name='Sheet1')
     user_detailed=user_name1.loc[:,['学号','姓名','性别']]
     user_new1=user_name2.merge(user_detailed,left_on='学号',right_on='学号')
     user_new=user_new1.columns.tolist()
     for name in ["姓名", "性别"][::-1]:
         user_new.remove(name)
         user_new.insert(user_new.index("学号") + 1, name)
     # reindex重新创建列索引
     xing_colu=user_new1.reindex(columns=user_new)
     print(xing_colu)
# vlookup()
from pyecharts.charts import Bar
from pyecharts import options as opts
def pyechars_photo():
    substance=pd.read_excel('./data_table/互联网公司股票.xlsx')
    substance.set_index(pd.to_datetime(substance['日期']),inplace=True)
    substance.drop('日期',axis=1,inplace=True)
    substance=substance.groupby('公司').mean()
    my_bar=Bar()
    my_bar.add_yaxis("开盘价",substance['开盘'].round(2).tolist())
    my_bar.add_yaxis("收盘价",substance['收盘'].round(2).tolist())
    my_bar.add_xaxis(substance.index.tolist())
    my_bar.set_global_opts(title_opts=opts.TitleOpts(title="股票开盘和收盘价",subtitle="数据来源：东方财富网"))
    my_bar.render("./data_table/html/股票开盘和收盘价.html")
# pyechars_photo()
def machine_learn():
     # 机器学习泰坦林肯号预测存活
     alive=pd.read_csv('./data_table/titanic_train.csv')
     feature_cols = ['Pclass', 'Parch']
     X = alive.loc[:, feature_cols]
     #训练特征
     y = alive.Survived
     # 预测目标
     from sklearn.ensemble import GradientBoostingClassifier
     clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                      max_depth=1, random_state=0).fit(X, y)
     # 训练模型
     clf.score(X, y)
     X.drop_duplicates().sort_values(by=["Pclass", "Parch"])
     print(clf.predict([[2, 4]]))
     print(clf.predict_proba([[2, 4]]))
# machine_learn()
def log_details():
    data_dir = "./data_table/log"  # 定义日志文件所在的目录
    list_file = []  # 用于存储每个日志文件读取的数据
    # 遍历目录中的所有文件
    for i in os.listdir(data_dir):
        # 读取每个文件的内容并添加到列表中
        list_file.append(pd.read_csv(f"{data_dir}/{i}", sep=" ", header=None,  on_bad_lines='skip'))
    # 合并所有读取的数据到一个DataFrame中
    new_table = pd.concat(list_file).copy()
    # 打印合并后的数据表
    new_table=new_table[[0, 3, 6, 9]].copy()
    # 重新设置表头
    new_table.columns = ["ip", "stime", "status", "client"]
    # print(new_table.head())
    # 查看
    new_table["is_spider"] = new_table["client"].str.lower().str.contains("spider")
    df_spider = new_table["is_spider"].value_counts()
    bar = (
        Bar()
        .add_xaxis([str(x) for x in df_spider.index])
        .add_yaxis("是否Spider", df_spider.values.tolist())
        .set_global_opts(title_opts=opts.TitleOpts(title="爬虫访问量占比"))
    )
    bar.render()
# log_details()

def one_list_duo():
    # 一行内容拆分出来
    # explode被内容为列表的列调用，该列的每个元素都会被拆分出来
    substance=pd.read_csv('./data_table/movies.csv')
    substance['genres']=substance['genres'].map(lambda x:x.split('|'))
    new_substance=substance.explode('genres').copy()
    new_substance['genres'].value_counts().plot.bar()
    plt.show()
    my_bar=Bar()
    my_bar.add_yaxis("电影类型",new_substance['genres'].value_counts().tolist())
    my_bar.add_xaxis(new_substance['genres'].value_counts().index.tolist())
    my_bar.set_global_opts(
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45))
    )
    my_bar.render("./data_table/html/电影类型.html")

one_list_duo()

def to_compare_huan_compare():
    # 同比：今年1月份-去年1月份/去年1月份
    # 同比：今年1月份-今年2月份/今年2月份
    substance=pd.read_csv('./data_table/beijing_tianqi_2017-2019.csv',index_col='ymd',parse_dates=True)
    substance['bWendu']=substance['bWendu'].str.replace('℃','').astype('int32')
    substance['yWendu']=substance['yWendu'].str.replace('℃','').astype('int32')
    # 对月份进行重新采样
    substance=substance[['bWendu']].resample('M').mean().copy()
    # 进行重新排序
    substance.sort_values('bWendu',inplace=True,ascending=True)
    substance=substance.iloc[1:55]
    # substance.plot()
    # plt.show()
    substance['tobi']=substance['bWendu'].pct_change(periods=1)
    substance['huanbi']=substance['bWendu'].pct_change(periods=12)
    print(substance)
    # 顺便提一下isin包含指某列表()指某列只要包含的例如df['ddd'].isin(1,2,3)表示ddd列表只要ddd
# to_compare_huan_compare()
import pymysql as sql
from sqlalchemy import create_engine
def pandas_sql():
    # 读取 Excel 文件中的数据
    subs = pd.read_excel('./data_table/crazyant_blog_articles_source.xlsx')

    # 创建数据库连接
    engine = create_engine('mysql+mysqlconnector://root:123456@localhost:3306/pandas_base')

    # 将数据写入 MySQL 数据库
    subs.to_sql('pandas_base', con=engine, if_exists='replace', index=False)
    # 执行 SQL 查询并打印结果
    result = pd.read_sql('SELECT * FROM pandas_base', con=engine)
    engine.dispose()
# pandas_sql()
import requests
from bs4 import BeautifulSoup
import re


def my_jiebe_english():
    df_dict = pd.read_csv("./data_table/ecdict.csv")#读取词典
    my_list=df_dict[['word', 'translation']]#取出单词和解释两列表
    url = "https://pandas.pydata.org/docs/user_guide/indexing.html"
    html_cont = requests.get(url).text
    soup = BeautifulSoup(html_cont)
    html_text = soup.get_text()#对爬取出来的文本进行提取
    word_list = re.split("""[ ,.\(\)/\n|\-:=\$\["']""", html_text)#过滤掉特殊字符
    with open("./data_table/stop_words.txt") as fin:#读取停用词
        stop_words = set(fin.read().split("\n"))#消除重复的停用词
    word_list_clean = []
    for word in word_list:
        word = str(word).lower().strip()
        # 过滤掉空词、数字、单个字符的词、停用词
        if not word or word.isnumeric() or len(word) <= 1 or word in stop_words:
            continue
        word_list_clean.append(word)#把提取出来的单词重新创建一个dataframe
    df_words = pd.DataFrame({
        "word": word_list_clean
    })
    print(df_words)
    # 对 DataFrame df_words 进行以下操作
    df_words = (
        df_words
        # 按 "word" 列进行分组，然后对 "word" 列进行聚合操作
        .groupby("word")["word"]
        # 计算每个组的大小，并命名聚合结果列为 "count"
        .agg(count="size")
        # 将聚合结果重置索引，使 "word" 成为一列，而不是索引
        .reset_index()
        # 按 "count" 列进行排序，降序排列（即出现频率最高的词排在最前）
        .sort_values(by="count", ascending=False)
    )

    df_merge = pd.merge(
        left=df_dict,
        right=df_words,
        left_on="word",
        right_on="word"
    )
    # 然后对其进行合并
    print(df_merge[['word','translation']])
# my_jiebe_english()


def xingjiqi():
    substance=pd.read_csv('./data_table/beijing_tianqi_2017-2019.csv', encoding='utf-8')
    substance.set_index(pd.to_datetime(substance['ymd']),inplace=True)
    substance.drop('ymd',axis=1,inplace=True)
    list_month=substance.index.month
    print(list_month.unique())

# xingjiqi()
def change(data):
    data['姓名'],data['性别'],data['年龄'],data['地区']=data['数据'].split(":")
    return data
def one_change_really():
    substance=pd.read_excel('./data_table/学生数据表.xlsx')
    substance_new=substance.apply(change,axis=1)
    substance_new.drop('数据',axis=1,inplace=True)
    print(substance_new)
# one_change_really()
def change_index():
    substance=pd.read_excel('./data_table/学生数据表.xlsx')
    substance.set_index('学号',inplace=True)
    # 把行索引和列索引相互转换transpose和T是一样作用
    print(substance.T)
    blit=substance.transpose()
    bi=['S003', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009',
       'S010', 'S011', 'S012', 'S013', 'S014', 'S015', 'S016', 'S017', 'S018',
       'S019', 'S020', 'S021', 'S022', 'S023', 'S024']
    print(blit[bi])
# change_index()
def compute(x):
    wc=x['bWendu']-x['yWendu']
    mean=(x['bWendu']+x['yWendu'])/2
    return wc,mean
def details_apply():
    substance=pd.read_csv('./data_table/beijing_tianqi_2017-2019.csv')
    substance['bWendu']=substance['bWendu'].apply(lambda x:int(x.replace('℃','')))
    substance['yWendu']=substance['yWendu'].apply(lambda x:int(x.replace('℃','')))
    substance[['pwd','wc']]=substance.apply(compute,axis=1,result_type='expand')
    print(substance.head())
    list_my=substance.columns.tolist()
    list_my.remove('pwd')
    list_my.remove('wc')
    list_my.insert(list_my.index('yWendu')+1,'pwd')
    list_my.insert(list_my.index('yWendu')+1,'wc')
    print(substance[list_my])
# details_apply()

def query_details():
    substance=pd.read_csv('./data_table/beijing_tianqi_2017-2019.csv')
    substance['bWendu']=substance['bWendu'].apply(lambda x:int(x.replace('℃','')))
    substance['yWendu']=substance['yWendu'].apply(lambda x:int(x.replace('℃','')))
    data=5
    print(substance.query('bWendu == 5' ))
    # y引用外部值要加@
    print(substance.query('bWendu == @data' ))

# query_details()