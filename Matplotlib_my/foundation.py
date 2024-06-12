"""
@big_name:data_teacher	
@file_name:foundation	
@data:2024/6/12	
@developers:handsome_lxh
"""
import matplotlib.pyplot as plt
import numpy as np


def day1():
    #  基础画图
    x = np.linspace(1, 10, 100)
    plt.plot(x, x ** 2)
    plt.show()


def day2():
    # 基础画图
    x = np.linspace(1, 10, 100)
    plt.plot(x, x ** 2, 'r--')  # 基础配置
    plt.show()


def day3():
    # 画布配置
    # figsize: 画布大小，宽高
    # dpi ：分辨率
    # facecolor: 背景颜色
    plt.figure(figsize=(5, 3), dpi=1000, facecolor='g')
    x = np.linspace(-1, 1, 100)
    x = x ** 2
    plt.plot(x, 'r--')
    plt.grid()
    plt.show()


def day4():
    # 绘制多条线
    plt.figure(figsize=(5, 3), dpi=1000, facecolor='g')
    x = np.linspace(-1, 1, 100)
    x1 = x ** 2
    plt.plot(x, 'r--')
    plt.plot(np.sin(x1), 'g--')
    plt.grid()
    plt.show()


def day5():
    # 2行2列
    fig = plt.figure(figsize=(8, 5))

    x = np.linspace(-np.pi, np.pi, 30)
    y = np.sin(x)

    # 子图1
    axes1 = plt.subplot(221)  # 2行2列的第1个子视图
    axes1.plot(x, y)
    axes1.set_title('子图1')
    # 子图2
    axes2 = plt.subplot(222)  # 2行2列的第2个子视图
    axes2.plot(x, y)
    axes2.set_title('子图2')
    # 子图3
    axes3 = plt.subplot(2, 2, 3)  # 2行2列的第3个子视图
    axes3.plot(x, y)
    axes3.set_title('子图3')
    # 子图4
    axes4 = plt.subplot(2, 2, 4)  # 2行2列的第4个子视图
    axes4.plot(x, y)
    axes4.set_title('子图4')
    # 自动调整布局
    fig.tight_layout()
    plt.show()


def day6():
    plt.figure(figsize=(8, 5))

    x = np.linspace(-np.pi, np.pi, 30)
    y = np.sin(x)

    # 子图1
    axes1 = plt.subplot(2, 2, 1)
    axes1.plot(x, y, color='red')

    # 子图2
    axes2 = plt.subplot(2, 2, 2)
    lines = axes2.plot(x, y)
    lines[0].set_marker('*')  # 点的样式

    # 子图3
    axes3 = plt.subplot(2, 1, 2)  # 2行1列的第2行
    axes3.plot(x, np.sin(x * x))
    plt.show()


def day7():
    plt.figure(figsize=(8, 5))
    x = np.linspace(0, 10, 100)
    # 图1
    axes1 = plt.gca()  # 获取当前轴域
    axes1.plot(x, np.exp(x), color='red')
    axes1.set_xlabel('time')
    axes1.set_ylabel('exp', color='red')
    axes1.tick_params(axis='y', labelcolor='red')
    # 图2
    axes2 = axes1.twinx()  # 和图1共享x轴
    axes2.set_ylabel('sin', color='blue')
    axes2.plot(x, np.sin(x), color='blue')
    axes2.tick_params(axis='y', labelcolor='blue')

    plt.tight_layout()
    plt.show()


def day8():
    # 图形绘制
    fig = plt.figure(figsize=(8, 5))

    x = np.linspace(0, 2 * np.pi)
    plt.plot(x, np.sin(x))  # 正弦曲线
    plt.plot(x, np.cos(x))  # 余弦曲线

    # 图例
    plt.legend(['Sin', 'Cos'],
               fontsize=18,
               ncol=2,  # 显示成几列
               loc='upper center',

               )
    plt.show()


def day9():
    fig = plt.figure(figsize=(8, 5))
    x = np.linspace(0, 2 * np.pi, 20)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # c : color 线颜色
    # marker: 标记的样式或点的样式
    # mfc: marker face color 标记的背景颜色
    # ls : line style 线的样式
    # lw: line width 线的宽度
    # label: 线标签（图例中显示）
    plt.plot(x, y1, c='r', marker='o', ls='--', lw=1, label='sinx', mfc='y', )

    plt.plot(x, y2, c='b', marker='*', ls='-', lw=2, label='cosx', mfc='g', )

    plt.plot(x, y1 - y2, c='y', marker='^', ls='-', lw=3, label='sinx-cosx', mfc='b', alpha=0.5)

    plt.plot(x, y1 + y2, c='orange', marker='>', ls='-.', lw=4, label='sinx+cosx',
             mfc='y',
             markersize=10,  # 点大小
             markeredgecolor='green',  # 点边缘颜色
             markeredgewidth=2  # 点边缘宽度
             )

    # 图例
    plt.legend()
    plt.show()

    # 图形绘制
    x = np.linspace(0, 10)
    y = np.sin(x)
    plt.plot(x, y)

    # 设置x轴y轴刻度
    plt.xticks(np.arange(0, 11, 1))
    plt.yticks([-1, 0, 1])
    plt.show()


def day10():
    # 图形绘制
    x = np.linspace(0, 10)
    y = np.sin(x)
    plt.plot(x, y)

    # 设置x轴y轴刻度标签
    plt.yticks(ticks=[-1, 0, 1],  # 刻度值
               labels=['min', '0', 'max'],  # 刻度值对应的标签名（显示）
               fontsize=20,  # 文字大小
               ha='right',  # 水平对齐方式
               color='blue'  # 颜色
               )
    plt.xticks(ticks=np.arange(0, 11, 1), fontsize=20, color='red')
    plt.show()


def day11():
    # 图形绘制
    x = np.linspace(0, 10)
    y = np.sin(x)
    plt.plot(x, y)

    # 图的标题
    # fontsize : 标题大小
    # loc：标题位置
    plt.title('sin曲线', fontsize=20, loc='center')
    # 父标题
    plt.suptitle('父标题',
                 y=1.1,  # 位置
                 fontsize=30  # 文字大小
                 )

    # 网格线
    # ls: line style 网格线样式
    # lw：line width  网格线宽度
    # c: color 网格线颜色
    # axis：画哪个轴的网格线，默认x轴和y轴都画
    plt.grid(ls='--', lw=0.5, c='gray', axis='y')

    plt.show()

    # 图形绘制
    x = np.linspace(0, 10)
    y = np.sin(x)
    plt.plot(x, y)

    # 坐标轴标签
    plt.xlabel('y=sin(x)',
               fontsize=20,  # 文字大小
               rotation=0,  # 旋转角度
               )
    plt.ylabel('y=sin(x)',
               rotation=90,  # 旋转角度
               horizontalalignment='right',  # 水平对齐方式
               fontsize=20
               )

    # 标题
    plt.title('正弦曲线')
    plt.show()

    plt.figure(figsize=(8, 5))

    x = np.linspace(0, 10, 10)
    y = np.array([60, 30, 20, 90, 40, 60, 50, 80, 70, 30])
    plt.plot(x, y, ls='--', marker='o')

    # 文字
    for a, b in zip(x, y):
        # 画文本
        plt.text(
            x=a + 0.3,  # x坐标
            y=b + 0.5,  # y坐标
            s=b,  # 文字内容
            ha='center',  # 水平居中
            va='center',  # 垂直居中
            fontsize=14,  # 文字大小
            color='r'  # 文字颜色
        )

    plt.show()


def day12():
    plt.figure(figsize=(8, 5))
    x = np.linspace(0, 10, 10)
    y = np.array([60, 30, 20, 90, 40, 60, 50, 80, 70, 30])
    plt.plot(x, y, ls='--', marker='o')
    # 注释（标注）
    plt.annotate(
        text='最高销量',  # 标注的内容
        xy=(3, 90),  # 标注的坐标点
        xytext=(1, 80),  # 标注的内容的坐标点
        # 箭头样式
        arrowprops={
            'width': 2,  # 箭头线的宽度
            'headwidth': 8,  # 箭头头部的宽度
            'facecolor': 'blue'  # 箭头的背景颜色
        }
    )
    plt.show()

if __name__ == '__main__':
    # pass
    # day1()
    # day2()
    # day3()
    # day4()
    # day5()
    # day6()
    # day7()
    # day8()
    # day9()
    # day10()
    # day11()
    day12()
