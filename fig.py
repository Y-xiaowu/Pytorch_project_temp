# import matplotlib.pyplot as plt  # 导入画图的包matplotlib
# import numpy as np
# from scipy.optimize import curve_fit  # 导入拟合曲线方程的包

# # 先生成一群点，把点画出来
# fig, ax = plt.subplots()
# x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]).astype(np.float64)  # 点的横坐标，放在一个数组里面
# y = np.square(x).astype(np.float64)  # y=x^2,根据x生成y的坐标，我这里用的二次曲线方程
# plt.ylim(-5, 35)  # 坐标系y轴范围
# ax.scatter(x, y)  # 画成散点图
#
# # 忽略除以0的报错
# np.seterr(divide='ignore', invalid='ignore')
#
#
# # 二阶曲线方程
# def func_2(x, a, b, c):
#     return a * np.power(x, 2) + b * x + c
#
#
# # 鼠标点击事件  函数里面又绑定了一个鼠标移动事件，所以生成的效果是鼠标按下并且移动的时候
# def on_button_press(event):
#     fig.canvas.mpl_connect('motion_notify_event', on_button_move)
#
#
# # on_button_move 鼠标移动事件
# def on_button_move(event, y=y):
#     if event.button == 2:  # 1、2、3分别代表鼠标的左键、中键、右键，我这里用的是鼠标中键，根据自己的喜好选择吧
#         x_mouse, y_mouse = event.xdata, event.ydata  # 拿到鼠标当前的横纵坐标
#
#         ind = []  # 这里生成一个列表存储一下要移动的那个点
#         # 计算一下鼠标的位置和图上点的位置距离，如果距离很近就移动图上那个点
#         for i in range(len(x)):
#             # 计算一下距离 图上每个点都和鼠标计算一下距离
#             d = np.sqrt((x_mouse - x[i]) ** 2 + (y_mouse - y[i]) ** 2)
#             if d < 0.8:  # 这里设置一个阈值，如果距离很近，就把它添加到那个列表中去
#                 ind.append(i)
#
#         if ind:  # 如果ind里面有元素，说明当前鼠标的位置距离图上的一个点很近
#             # 通过索引ind[0]去改变当前这个点的坐标，新坐标是当前鼠标的横纵坐标（这样给人的感觉就是这个点跟着鼠标动了）
#             y[ind[0]] = y_mouse
#             x[ind[0]] = x_mouse
#
#             # 然后根据所有点拟合出来一个二次方程曲线
#             popt2, pcov2 = curve_fit(func_2, x, y)
#             a2 = popt2[0]
#             b2 = popt2[1]
#             c2 = popt2[2]
#             yvals2 = func_2(x, a2, b2, c2)
#
#             # 拟合好了以后把曲线画出来
#             ax.cla()
#             plt.ylim(-5, 35)
#             ax.scatter(x, y)
#             ax.plot(x, yvals2)
#             fig.canvas.draw_idle()  # 重新绘制整个图表，所以看到的就是鼠标移动点然后曲线也跟着在变动
#
#
# # 鼠标释放事件，鼠标松开的时候，就把上面鼠标点击并且移动的关系解绑  这样鼠标松开的时候 就不会拖动点了
# def on_button_release(event):
#     fig.canvas.mpl_disconnect(fig.canvas.mpl_connect('motion_notify_event', on_button_move))  # 鼠标释放事件
#
#
# fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)  # 取消默认快捷键的注册，
# fig.canvas.mpl_connect('button_press_event', on_button_press)  # 鼠标点击事件
# fig.canvas.mpl_connect('button_release_event', on_button_release)  # 鼠标松开
# plt.show()  # 显示图像



import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 设置字体
# 注意：这里使用了系统默认的 'SimHei' 字体，这是一个常用的中文字体。
# 如果需要使用其他字体，请确保该字体已安装在系统中，并且路径正确。
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

a1 = -27 / 2450
h1 = 90
k1 = 90

a = -1 / 1250
h = 70
k = 95

# 生成数据
x = np.linspace(10, 120, 110)
y1 = a1 * (x - h1)**2 + k1
y2 = a * (x - h)**2 + k

# 创建图形和轴对象
plt.figure(figsize=(12, 6))

# 绘制第一条曲线
plt.plot(x, y1, label='异步一体机', color='blue')

# 绘制第二条曲线
plt.plot(x, y2, label='永磁一体机', color='red')

# 添加图例
#plt.legend()

# 添加标题
plt.title('效率对比图')

# 设置 y 轴的刻度位置和标签
plt.yticks(np.arange(0, 110, 10))


# 设置 x 轴的刻度位置和标签
plt.xticks(np.arange(0, 140, 20))

# 设置 x 轴和 y 轴的范围
plt.xlim(0, 140)
plt.ylim(10, 110)


# 添加坐标轴标签
plt.xlabel('负载率（%）')
plt.ylabel('效率（%）')
plt.legend()
# 添加网格线
plt.grid(True)
plt.savefig("parabola_high_res.png", dpi=600)

# 显示图形
plt.show()
