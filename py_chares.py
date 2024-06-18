"""
@big_name:data_teacher	
@file_name:py_chares	
@data:2024/6/18	
@developers:handsome_lxh
"""
from pyecharts.charts import Line,Timeline
from pyecharts import options as opts

time_my=Timeline()
time_my.add_schema(play_interval=1000)
# 准备数据
x_data = ["2013", "2014", "2015", "2016", "2017"]
y_data = [5, 9, 19, 30, 40]
for i in range(len(x_data)):
    line = (
        Line()
        .add_xaxis(x_data[i])
        .add_yaxis(
            series_name=f"{x_data[i]}",
            y_axis=[y_data[i]],
        )
    )
    time_my.add(line, x_data[i])

time_my.render("时间线折线图.html")