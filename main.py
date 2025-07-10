"""
股票分析与预测系统

这是一个基于PyQt5和深度学习的股票数据分析和预测系统，主要功能包括：
1. 获取和显示股票实时行情数据
2. 查询和可视化股票历史数据
3. 使用LSTM深度学习模型进行股票价格预测

本系统利用AkShare库获取股票数据，使用Matplotlib进行数据可视化，
并基于TensorFlow/Keras构建LSTM模型进行时间序列预测。

作者：厚西梅
日期：2025/7/8
"""

import sys
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QLineEdit, QPushButton, QTextEdit,
                            QTabWidget, QComboBox, QDateEdit, QSpinBox, QMessageBox)
from PyQt5.QtCore import Qt, QDate
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import akshare as ak
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import math
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
# 设置matplotlib中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class MplCanvas(FigureCanvas):
    """
    自定义的Matplotlib画布基类
    
    这个类继承自FigureCanvasQTAgg，用于在PyQt5应用中嵌入Matplotlib图表。
    作为所有画布类的基类，提供了基本的图形初始化功能。
    
    参数:
        parent (QWidget): 父级窗口部件
        width (int): 画布宽度，默认5
        height (int): 画布高度，默认4
        dpi (int): 每英寸点数，默认100
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()
class RealTimeCanvas(MplCanvas):
    """
    实时股票数据可视化画布
    
    用于展示股票的实时数据，以柱状图形式显示各项指标，
    并支持鼠标悬停显示详细数据。
    
    继承自MplCanvas类。
    """
    def __init__(self, *args, **kwargs):
        super(RealTimeCanvas, self).__init__(*args, **kwargs)
        self.data = None
        self.annotation = None
        # 连接鼠标事件
        self.mpl_connect("motion_notify_event", self.on_hover)

    def plot_data(self, data, stock_code, stock_name=""):
        """
        绘制实时股票数据柱状图
        
        参数:
            data (dict): 包含要绘制的股票数据指标
            stock_code (str): 股票代码
            stock_name (str): 股票名称，默认为空字符串
        """
        self.axes.clear()
        self.data = data
        bars = self.axes.bar(data.keys(), data.values())

        # 为正值和负值设置不同颜色
        for i, bar in enumerate(bars):
            if list(data.values())[i] >= 0:
                bar.set_color('red')  # 中国股市红色表示上涨
            else:
                bar.set_color('green')  # 绿色表示下跌

        title = f'股票 {stock_code} {stock_name} 实时数据'
        self.axes.set_title(title, fontsize=15)
        self.axes.set_ylabel('数值')
        plt.setp(self.axes.get_xticklabels(), rotation=45)
        
        # 创建注释对象，初始不可见
        self.annotation = self.axes.annotate(
            '', xy=(0, 0), xytext=(10, 10),
            textcoords='offset points',
            bbox={'boxstyle': 'round', 'fc': 'yellow', 'alpha': 0.7},
            arrowprops={'arrowstyle': '->'},
            visible=False
        )
        
        self.fig.tight_layout()
        self.draw()
        
    def on_hover(self, event):
        """
        鼠标悬停事件处理函数
        
        当鼠标悬停在柱状图上时，显示该数据点的详细信息。
        
        参数:
            event: Matplotlib鼠标事件对象
        """        
        if event.inaxes == self.axes and self.data is not None:
            # 获取鼠标位置对应的数据点
            for i, label in enumerate(self.data.keys()):
                # 获取x坐标位置
                x = i
                # 获取y值
                value = list(self.data.values())[i]
                
                # 检查鼠标是否在柱形上
                if abs(event.xdata - x) < 0.4:  # 柱形宽度通常为0.8
                    # 更新注释内容和位置
                    self.annotation.xy = (x, value)
                    self.annotation.set_text(f"{label}: {value:.2f}")
                    self.annotation.set_visible(True)
                    self.draw_idle()
                    return
            
            # 如果鼠标不在任何柱形上，隐藏注释
            self.annotation.set_visible(False)
            self.draw_idle()
class HistoricalCanvas(FigureCanvas):
    """
    股票历史数据可视化画布
    
    用于展示股票的历史交易数据，包括四个子图：
    1. 收盘价走势图
    2. 成交量柱状图
    3. 日涨跌幅图
    4. 价格高低范围图
    
    支持鼠标悬停查看具体数据点信息。
    """
    def __init__(self, parent=None, width=10, height=8, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(HistoricalCanvas, self).__init__(self.fig)
        self.stock_data = None
        self.annotations = []
        # 连接鼠标事件
        self.mpl_connect("motion_notify_event", self.on_hover)

    def plot_data(self, stock_hist_df):
        """
        绘制股票历史数据的多个图表
        
        参数:
            stock_hist_df (DataFrame): 包含股票历史数据的DataFrame对象
        """
        self.fig.clear()
        self.stock_data = stock_hist_df
        # 清空之前的注释列表
        self.annotations = []

        # 绘制收盘价走势
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax1.plot(stock_hist_df['日期'], stock_hist_df['收盘'], 'r-')
        ax1.set_title('收盘价走势')
        plt.setp(ax1.get_xticklabels(), rotation=45)
        # 创建注释对象
        annotation1 = ax1.annotate(
            '', xy=(0, 0), xytext=(10, 10),
            textcoords='offset points',
            bbox={'boxstyle': 'round', 'fc': 'yellow', 'alpha': 0.7},
            visible=False
        )
        self.annotations.append((ax1, annotation1, '收盘'))

        # 绘制成交量
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax2.bar(stock_hist_df['日期'], stock_hist_df['成交量'], color='blue')
        ax2.set_title('成交量')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        # 创建注释对象
        annotation2 = ax2.annotate(
            '', xy=(0, 0), xytext=(10, 10),
            textcoords='offset points',
            bbox={'boxstyle': 'round', 'fc': 'yellow', 'alpha': 0.7},
            visible=False
        )
        self.annotations.append((ax2, annotation2, '成交量'))

        # 绘制涨跌幅
        ax3 = self.fig.add_subplot(2, 2, 3)
        change_pct = ((stock_hist_df['收盘'] - stock_hist_df['开盘']) / stock_hist_df['开盘']) * 100
        colors = ['red' if x >= 0 else 'green' for x in change_pct]
        ax3.bar(stock_hist_df['日期'], change_pct, color=colors)
        ax3.set_title('日涨跌幅 (%)')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        # 创建注释对象
        annotation3 = ax3.annotate(
            '', xy=(0, 0), xytext=(10, 10),
            textcoords='offset points',
            bbox={'boxstyle': 'round', 'fc': 'yellow', 'alpha': 0.7},
            visible=False
        )
        self.annotations.append((ax3, annotation3, '涨跌幅'))

        # K线图 (简化版)
        ax4 = self.fig.add_subplot(2, 2, 4)
        ax4.plot(stock_hist_df['日期'], stock_hist_df['收盘'], 'r-')
        ax4.fill_between(stock_hist_df['日期'], stock_hist_df['最高'], stock_hist_df['最低'], color='gray', alpha=0.3)
        ax4.set_title('价格范围 (高-低)')
        plt.setp(ax4.get_xticklabels(), rotation=45)
        # 创建注释对象
        annotation4 = ax4.annotate(
            '', xy=(0, 0), xytext=(10, 10),
            textcoords='offset points',
            bbox={'boxstyle': 'round', 'fc': 'yellow', 'alpha': 0.7},
            visible=False
        )
        self.annotations.append((ax4, annotation4, '价格范围'))

        self.fig.tight_layout()
        self.draw()
        
    def on_hover(self, event):
        """
        鼠标悬停事件处理函数
        
        当鼠标悬停在任意子图上时，显示对应数据点的详细信息。
        不同的子图会显示不同类型的数据信息。
        
        参数:
            event: Matplotlib鼠标事件对象
        """        
        if self.stock_data is None or len(self.stock_data) == 0:
            return
            
        # 获取鼠标位置
        for ax, annotation, data_type in self.annotations:
            if event.inaxes == ax:
                # 找到最接近的数据点
                dates = self.stock_data['日期'].values
                if len(dates) == 0:
                    continue
                    
                # 转换x坐标为日期索引
                xdata = event.xdata
                # 找到最近的日期索引
                idx = min(max(0, int(xdata + 0.5)), len(dates) - 1)
                if idx < 0 or idx >= len(dates):
                    continue
                    
                # 获取对应日期和数据
                date = dates[idx]
                row = self.stock_data.loc[self.stock_data['日期'] == date].iloc[0] if date in self.stock_data['日期'].values else None
                
                if row is not None:
                    # 根据子图类型显示不同的信息
                    if data_type == '收盘':
                        value = row['收盘']
                        y_pos = value
                        text = f"日期: {date}\n收盘价: {value:.2f}\n开盘价: {row['开盘']:.2f}"
                    elif data_type == '成交量':
                        value = row['成交量']
                        y_pos = value
                        text = f"日期: {date}\n成交量: {value}\n成交额: {row['成交额'] if '成交额' in row else 'N/A'}"
                    elif data_type == '涨跌幅':
                        change = ((row['收盘'] - row['开盘']) / row['开盘']) * 100
                        y_pos = change
                        text = f"日期: {date}\n涨跌幅: {change:.2f}%\n收盘: {row['收盘']:.2f}, 开盘: {row['开盘']:.2f}"
                    else:  # 价格范围
                        value = row['收盘']
                        y_pos = value
                        text = f"日期: {date}\n收盘: {row['收盘']:.2f}\n最高: {row['最高']:.2f}\n最低: {row['最低']:.2f}"
                    
                    # 更新注释内容和位置
                    x_idx = list(dates).index(date) if date in dates else 0
                    annotation.xy = (x_idx, y_pos)
                    annotation.set_text(text)
                    annotation.set_visible(True)
                    self.draw_idle()
                    return
            
            # 如果鼠标不在任何子图内，隐藏所有注释
            annotation.set_visible(False)
        
        self.draw_idle()
class PredictionCanvas(MplCanvas):
    """
    股票价格预测结果可视化画布
    
    用于展示股票价格的历史数据、训练预测、测试预测和未来预测结果。
    包括四条曲线：
    1. 历史收盘价
    2. 训练集预测结果
    3. 测试集预测结果
    4. 未来价格预测
    
    支持鼠标悬停查看具体数据点信息。
    
    继承自MplCanvas类。
    """
    def __init__(self, *args, **kwargs):
        super(PredictionCanvas, self).__init__(*args, **kwargs)
        self.hist_data = None
        self.train_data = None
        self.test_data = None
        self.future_data = None
        self.annotation = None
        # 连接鼠标事件
        self.mpl_connect("motion_notify_event", self.on_hover)

    def plot_prediction(self, stock_hist_df, train_plot, test_plot, future_plot):
        """
        绘制股票价格预测结果图
        
        参数:
            stock_hist_df (DataFrame): 包含股票历史数据的DataFrame对象
            train_plot (ndarray): 训练数据预测结果数组
            test_plot (ndarray): 测试数据预测结果数组
            future_plot (ndarray): 未来价格预测结果数组
        """
        self.axes.clear()
        self.hist_data = stock_hist_df['收盘'].values
        self.train_data = train_plot
        self.test_data = test_plot
        self.future_data = future_plot

        # 绘制历史数据
        self.axes.plot(self.hist_data, 'b', label='历史收盘价')

        # 绘制训练预测
        self.axes.plot(self.train_data, 'g--', label='训练预测')

        # 绘制测试预测
        self.axes.plot(self.test_data, 'r--', label='测试预测')

        # 绘制未来预测
        self.axes.plot(self.future_data, 'y--', label='未来预测')

        self.axes.set_title('股票价格预测')
        self.axes.set_xlabel('时间')
        self.axes.set_ylabel('价格')
        self.axes.legend()
        
        # 创建注释对象，初始不可见
        self.annotation = self.axes.annotate(
            '', xy=(0, 0), xytext=(10, 10),
            textcoords='offset points',
            bbox={'boxstyle': 'round', 'fc': 'yellow', 'alpha': 0.7},
            arrowprops={'arrowstyle': '->'},
            visible=False
        )
        
        self.fig.tight_layout()
        self.draw()
        
    def on_hover(self, event):
        """
        鼠标悬停事件处理函数
        
        当鼠标悬停在预测图上时，显示该数据点的详细信息，包括实际价格和各类预测值。
        对于未来预测部分，显示预测天数和预测价格。
        
        参数:
            event: Matplotlib鼠标事件对象
        """        
        if event.inaxes == self.axes and self.hist_data is not None:
            # 获取鼠标位置对应的数据点索引
            x_idx = int(round(event.xdata))
            
            # 检查是否在有效范围内
            if 0 <= x_idx < len(self.hist_data):
                # 获取对应点的各种数据
                hist_value = self.hist_data[x_idx] if x_idx < len(self.hist_data) else None
                train_value = self.train_data[x_idx] if x_idx < len(self.train_data) and not np.isnan(self.train_data[x_idx]) else None
                test_value = self.test_data[x_idx] if x_idx < len(self.test_data) and not np.isnan(self.test_data[x_idx]) else None
                
                # 构建显示文本
                text = f"数据点: {x_idx}\n"
                if hist_value is not None:
                    text += f"实际价格: {hist_value:.2f}\n"
                if train_value is not None:
                    text += f"训练预测: {float(train_value):.2f}\n"
                if test_value is not None:
                    text += f"测试预测: {float(test_value):.2f}"
                
                # 设置显示位置 - 使用实际值的y坐标
                y_pos = hist_value if hist_value is not None else (
                        train_value if train_value is not None else (
                        test_value if test_value is not None else 0))
                
                # 更新注释
                self.annotation.xy = (x_idx, y_pos)
                self.annotation.set_text(text)
                self.annotation.set_visible(True)
                self.draw_idle()
                return
            # 检查是否在未来预测部分
            elif len(self.hist_data) <= x_idx < len(self.future_data):
                future_value = self.future_data[x_idx]
                if not np.isnan(future_value):
                    future_day = x_idx - len(self.hist_data) + 1
                    text = f"未来第{future_day}天\n预测价格: {float(future_value):.2f}"
                    
                    # 更新注释
                    self.annotation.xy = (x_idx, future_value)
                    self.annotation.set_text(text)
                    self.annotation.set_visible(True)
                    self.draw_idle()
                    return
            
            # 如果鼠标不在任何有效数据点上，隐藏注释
            self.annotation.set_visible(False)
            self.draw_idle()
class StockApp(QMainWindow):
    """
    股票分析与预测系统主应用窗口
    
    这是应用程序的主窗口类，包含以下主要功能：
    1. 股票搜索和基本信息查询
    2. 实时行情数据展示
    3. 历史交易数据可视化分析
    4. 基于LSTM的股票价格预测
    
    界面采用标签页设计，分为实时行情、历史数据和股价预测三个主要模块。
    """
    def __init__(self):
        """
        初始化StockApp主窗口
        
        设置窗口标题、大小，初始化UI组件和数据变量
        """
        super().__init__()
        self.setWindowTitle("股票分析与预测系统")
        self.setGeometry(100, 100, 1200, 800)

        # 创建主窗口部件和布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)

        # 添加搜索栏
        self.create_search_bar()

        # 创建标签页控件
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # 添加实时数据标签页
        self.realtime_tab = QWidget()
        self.tabs.addTab(self.realtime_tab, "实时行情")
        self.setup_realtime_tab()

        # 添加历史数据标签页
        self.historical_tab = QWidget()
        self.tabs.addTab(self.historical_tab, "历史数据")
        self.setup_historical_tab()

        # 添加预测标签页
        self.prediction_tab = QWidget()
        self.tabs.addTab(self.prediction_tab, "股价预测")
        self.setup_prediction_tab()

        # 初始变量
        self.stock_code = ""  # 股票代码
        self.target_stock = None  # 目标股票数据
        self.stock_hist_df = None  # 历史数据DataFrame
        self.future_predictions = None  # 未来价格预测结果
    def create_search_bar(self):
        search_layout = QHBoxLayout()

        self.stock_label = QLabel("股票代码:")
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("输入股票代码，如：688729")
        self.search_button = QPushButton("查询")
        self.search_button.clicked.connect(self.search_stock)

        search_layout.addWidget(self.stock_label)
        search_layout.addWidget(self.stock_input)
        search_layout.addWidget(self.search_button)

        self.main_layout.addLayout(search_layout)

    def setup_realtime_tab(self):
        layout = QVBoxLayout(self.realtime_tab)

        # 添加实时数据画布
        self.realtime_canvas = RealTimeCanvas(self, width=10, height=6)
        layout.addWidget(self.realtime_canvas)

        # 添加信息显示区域
        self.realtime_info = QTextEdit()
        self.realtime_info.setReadOnly(True)
        self.realtime_info.setMaximumHeight(150)
        layout.addWidget(self.realtime_info)

    def setup_historical_tab(self):
        layout = QVBoxLayout(self.historical_tab)

        # 控制面板
        control_layout = QHBoxLayout()

        # 日期范围选择
        self.start_date_label = QLabel("开始日期:")
        self.start_date = QDateEdit()
        self.start_date.setDate(QDate.currentDate().addMonths(-6))
        self.start_date.setCalendarPopup(True)

        self.end_date_label = QLabel("结束日期:")
        self.end_date = QDateEdit()
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setCalendarPopup(True)

        self.update_hist_button = QPushButton("更新数据")
        self.update_hist_button.clicked.connect(self.update_historical_data)

        control_layout.addWidget(self.start_date_label)
        control_layout.addWidget(self.start_date)
        control_layout.addWidget(self.end_date_label)
        control_layout.addWidget(self.end_date)
        control_layout.addWidget(self.update_hist_button)

        layout.addLayout(control_layout)

        # 添加历史数据画布
        self.historical_canvas = HistoricalCanvas(self)
        layout.addWidget(self.historical_canvas)

    def setup_prediction_tab(self):
        layout = QVBoxLayout(self.prediction_tab)

        # 控制面板
        control_layout = QHBoxLayout()

        self.epochs_label = QLabel("训练轮数:")
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 100)
        self.epochs_input.setValue(10)

        self.future_days_label = QLabel("预测天数:")
        self.future_days_input = QSpinBox()
        self.future_days_input.setRange(1, 365)
        self.future_days_input.setValue(30)

        self.train_button = QPushButton("训练模型并预测")
        self.train_button.clicked.connect(self.train_and_predict)

        control_layout.addWidget(self.epochs_label)
        control_layout.addWidget(self.epochs_input)
        control_layout.addWidget(self.future_days_label)
        control_layout.addWidget(self.future_days_input)
        control_layout.addWidget(self.train_button)

        layout.addLayout(control_layout)

        # 添加绘图区域
        plot_layout = QHBoxLayout()

        # 添加预测画布
        self.prediction_canvas = PredictionCanvas(self, width=10, height=6)
        plot_layout.addWidget(self.prediction_canvas, 2)

        # 添加预测结果文本区域
        self.prediction_results = QTextEdit()
        self.prediction_results.setReadOnly(True)
        plot_layout.addWidget(self.prediction_results, 1)

        layout.addLayout(plot_layout)

    def search_stock(self):
        """搜索股票并更新所有相关数据"""
        self.stock_code = self.stock_input.text().strip()

        if not self.stock_code:
            QMessageBox.warning(self, "警告", "请输入股票代码")
            return

        try:
            # 获取股票数据
            self.realtime_info.setText(f"正在获取股票代码 {self.stock_code} 的数据...")
            QApplication.processEvents()  # 更新UI

            # 获取A股实时行情数据
            stock_df = ak.stock_sh_a_spot_em()

            # 筛选特定股票
            self.target_stock = stock_df[stock_df['代码'] == self.stock_code]

            if self.target_stock.empty:
                stock_df = ak.stock_sz_a_spot_em()  # 尝试深交所
                self.target_stock = stock_df[stock_df['代码'] == self.stock_code]

            if self.target_stock.empty:
                self.realtime_info.setText(f"未找到股票代码为 {self.stock_code} 的数据")
                return

            # 更新实时数据显示
            self.update_realtime_data()

            # 获取历史数据
            start_date = self.start_date.date().toString("yyyyMMdd")
            end_date = self.end_date.date().toString("yyyyMMdd")

            self.stock_hist_df = ak.stock_zh_a_hist(symbol=self.stock_code, period="daily", 
                                                  start_date=start_date, end_date=end_date, 
                                                  adjust="")

            # 更新历史数据显示
            self.update_historical_view()

            # 切换到实时数据标签页
            self.tabs.setCurrentIndex(0)

        except Exception as e:
            self.realtime_info.setText(f"获取数据时出错: {str(e)}")

    def update_realtime_data(self):
        """更新实时数据显示"""
        if self.target_stock is None or self.target_stock.empty:
            return

        # 提取股票名称
        stock_name = self.target_stock['名称'].values[0] if '名称' in self.target_stock.columns else ""

        # 提取所需数据
        data = {
            '最新价': self.target_stock['最新价'].values[0],
            '涨跌幅': self.target_stock['涨跌幅'].values[0],
            '涨跌额': self.target_stock['涨跌额'].values[0],
            '5分钟涨跌': self.target_stock['涨速'].values[0],
            '60日涨跌幅': self.target_stock['60日涨跌幅'].values[0] if '60日涨跌幅' in self.target_stock.columns else 0
        }

        # 更新画布
        self.realtime_canvas.plot_data(data, self.stock_code, stock_name)

        # 更新信息文本
        info_text = f"股票代码: {self.stock_code}\n"
        info_text += f"股票名称: {stock_name}\n"
        info_text += f"交易日期: {self.target_stock['日期'].values[0] if '日期' in self.target_stock.columns else '今日'}\n"
      
        info_text += f"最高价: {self.target_stock['最高'].values[0]}\n"
        info_text += f"最低价: {self.target_stock['最低'].values[0]}\n"
        info_text += f"成交量(手): {self.target_stock['成交量'].values[0]}\n"
        info_text += f"成交额(万元): {self.target_stock['成交额'].values[0]}\n"

        self.realtime_info.setText(info_text)

    def update_historical_data(self):
        """根据用户选择的日期范围更新历史数据"""
        if not self.stock_code:
            QMessageBox.warning(self, "警告", "请先搜索股票代码")
            return

        try:
            start_date = self.start_date.date().toString("yyyyMMdd")
            end_date = self.end_date.date().toString("yyyyMMdd")

            self.stock_hist_df = ak.stock_zh_a_hist(symbol=self.stock_code, period="daily", 
                                                  start_date=start_date, end_date=end_date, 
                                                  adjust="")

            self.update_historical_view()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取历史数据时出错: {str(e)}")

    def update_historical_view(self):
        """更新历史数据视图"""
        if self.stock_hist_df is None or self.stock_hist_df.empty:
            return

        # 更新历史数据画布
        self.historical_canvas.plot_data(self.stock_hist_df)

        # 切换到历史数据标签页
        self.tabs.setCurrentIndex(1)

    def train_and_predict(self):
        """训练LSTM模型并进行预测"""
        if self.stock_hist_df is None or self.stock_hist_df.empty:
            QMessageBox.warning(self, "警告", "请先获取股票历史数据")
            return
    
        # 新增数据量验证（50个交易日）
        if len(self.stock_hist_df) < 50:
            QMessageBox.warning(self, "警告", "历史数据不足，至少需要50个交易日的数据")
            return
    
        try:
            epochs = self.epochs_input.value()
            future_days = self.future_days_input.value()

            self.prediction_results.setText("开始训练LSTM模型...\n这可能需要一些时间，请耐心等待。")
            QApplication.processEvents()  # 更新UI

            # 准备数据 - 使用收盘价
            data = self.stock_hist_df['收盘'].values.reshape(-1, 1)

            # 数据标准化
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # 创建训练数据集 - 修复版本
            def create_dataset(dataset, time_step=1):
                X, y = [], []
                for i in range(time_step, len(dataset)):
                    X.append(dataset[i-time_step:i, 0])
                    y.append(dataset[i, 0])
                return np.array(X), np.array(y)

            # 动态设置time_step，确保有足够的训练数据
            time_step = min(10, len(scaled_data) // 5)  # 确保至少有足够的样本
            if time_step < 3:
                time_step = 3

            # 确保训练集大小合理
            train_size = max(int(len(scaled_data) * 0.8), time_step + 10)
            train_size = min(train_size, len(scaled_data) - time_step - 1)
            
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size-time_step:]  # 保证测试数据有重叠以创建序列

            X_train, y_train = create_dataset(train_data, time_step)
            X_test, y_test = create_dataset(test_data, time_step)

            # 检查数据形状
            if len(X_train) == 0 or len(X_test) == 0:
                QMessageBox.warning(self, "警告", "训练数据创建失败，请尝试增加历史数据时间范围")
                return

            self.prediction_results.append(f"训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}")

            # 重塑输入数据为LSTM预期的格式
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # 创建更稳定的LSTM模型
            model = Sequential()
            model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(16))
            model.add(Dense(1))
            
            # 使用更稳定的优化器设置
            from tensorflow.keras.optimizers import Adam
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

            # 训练模型
            self.prediction_results.append("\n模型开始训练...")
            QApplication.processEvents()
            
            try:
                # 添加early stopping来避免过拟合
                from tensorflow.keras.callbacks import EarlyStopping
                early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                
                history = model.fit(X_train, y_train,
                                  batch_size=min(32, len(X_train)),
                                  epochs=epochs,
                                  validation_split=0.1,
                                  callbacks=[early_stopping],
                                  verbose=0)
                
                self.prediction_results.append(f"训练完成！最终损失: {history.history['loss'][-1]:.6f}")
                
            except Exception as train_error:
                self.prediction_results.append(f"训练过程出错: {str(train_error)}")
                QMessageBox.critical(self, "训练错误", f"模型训练失败: {str(train_error)}")
                return

            # 进行预测
            self.prediction_results.append("开始预测...")
            QApplication.processEvents()
            
            try:
                train_predict = model.predict(X_train, verbose=0)
                test_predict = model.predict(X_test, verbose=0)
                
                # 反向转换预测结果
                train_predict = scaler.inverse_transform(train_predict)
                test_predict = scaler.inverse_transform(test_predict)
                y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
                y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                
            except Exception as pred_error:
                self.prediction_results.append(f"预测过程出错: {str(pred_error)}")
                QMessageBox.critical(self, "预测错误", f"预测失败: {str(pred_error)}")
                return

            # 计算均方根误差 (RMSE)
            import math
            from sklearn.metrics import mean_squared_error
            train_rmse = math.sqrt(mean_squared_error(y_train_actual, train_predict))
            test_rmse = math.sqrt(mean_squared_error(y_test_actual, test_predict))

            self.prediction_results.append(f"训练数据RMSE: {train_rmse:.2f}")
            self.prediction_results.append(f"测试数据RMSE: {test_rmse:.2f}")

            # 预测未来价格
            try:
                # 确保有足够的数据进行未来预测
                if len(scaled_data) >= time_step:
                    last_data = scaled_data[-time_step:].reshape(1, time_step, 1)
                    
                    future_predictions = []
                    current_batch = last_data.copy()

                    for i in range(future_days):
                        current_pred = model.predict(current_batch, verbose=0)[0]
                        future_predictions.append(current_pred[0])
                        
                        # 更新批次数据
                        new_batch = np.append(current_batch[:,1:,:], 
                                            current_pred.reshape(1, 1, 1), axis=1)
                        current_batch = new_batch

                    # 反向转换未来预测结果
                    future_predictions = scaler.inverse_transform(
                        np.array(future_predictions).reshape(-1, 1))

                    self.prediction_results.append(f"\n未来{future_days}天价格预测:")
                    for i, price in enumerate(future_predictions):
                        self.prediction_results.append(f"第 {i+1} 天: {price[0]:.2f}")

                    self.future_predictions = future_predictions
                    
                else:
                    self.prediction_results.append("数据不足，无法进行未来预测")
                    
            except Exception as future_error:
                self.prediction_results.append(f"未来预测出错: {str(future_error)}")

            # 可视化预测结果
            # 创建历史数据的完整数组
            hist_actual = scaler.inverse_transform(scaled_data)

            # 创建训练预测数组
            train_plot = np.empty_like(hist_actual)
            train_plot[:] = np.nan
            train_plot[time_step:len(train_predict)+time_step] = train_predict

            # 创建测试预测数组 - 修复形状不匹配问题
            test_plot = np.empty_like(hist_actual)
            test_plot[:] = np.nan
            # 计算正确的开始索引
            test_start_idx = len(train_predict) + time_step
            # 确保不超出数组范围
            test_end_idx = min(test_start_idx + len(test_predict), len(hist_actual))
            # 只放入可用范围内的数据
            test_plot[test_start_idx:test_end_idx] = test_predict[:test_end_idx-test_start_idx]

            # 创建未来预测数组
            future_plot = np.empty((len(hist_actual) + future_days, 1))
            future_plot[:] = np.nan
            future_plot[:len(hist_actual)] = hist_actual
            future_plot[len(hist_actual):] = future_predictions

            # 更新预测画布
            self.prediction_canvas.plot_prediction(self.stock_hist_df, train_plot, test_plot, future_plot)

            # 显示预测结果
            self.prediction_results.append("未来价格预测:")
            for i, price in enumerate(future_predictions):
                self.prediction_results.append(f"第 {i+1} 天: {price[0]:.2f}")

            # 保存预测结果
            self.future_predictions = future_predictions

            # 切换到预测标签页
            self.tabs.setCurrentIndex(2)

        except Exception as e:
            self.prediction_results.append(f"\n预测过程中出错: {str(e)}")

# 启动应用
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StockApp()
    window.show()
    sys.exit(app.exec_())