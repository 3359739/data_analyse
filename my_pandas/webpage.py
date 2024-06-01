"""
@big_name:data_teacher	
@file_name:webpage	
@data:2024/6/2	
@developers:handsome_lxh
"""
from flask import Flask,render_template
import pandas as pd
from pyecharts.charts import Bar
from pyecharts import options as opts
app=Flask(__name__,template_folder="./templates")
@app.route("/")
def flask_pandas():
    xing=pd.read_excel('./data_table/学生信息表.xlsx')
    table_html=xing.to_html()
    return render_template('index.html',table_html=table_html)

if __name__ == '__main__':
    app.run(debug=True)