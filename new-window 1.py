# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: yolov5-jungong
File Name: window.py.py
Author: chenming
Create Date: 2021/11/8
Description：图形化界面，保留图片检测功能
-------------------------------------------------
"""
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import argparse
import os
import sys
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import os.path as osp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, LoadImages
from utils.general import (LOGGER, check_file, check_img_size, colorstr,
                           non_max_suppression, scale_coords, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Target detection system')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/OIP.jpg"))
        # 图片检测配置
        self.detect_size = 320
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        self.model = self.model_load(weights="runs2/train/exp_big/weights/best.pt", device=self.device)
        self.initUI()

    @torch.no_grad()
    def model_load(self, weights="", device='', half=False, dnn=False):
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        model.stride, model.names, model.pt = model.stride, model.names, model.pt
        if device.type != 'cpu':
            model(torch.zeros(1, 3, *[self.detect_size, self.detect_size]).to(device).type_as(
                next(model.model.parameters())))
        print("模型加载完成!")
        return model

    def initUI(self):
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 16)

        # 图片检测界面
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片识别功能")
        img_detection_title.setFont(font_title)

        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/UP.png"))
        self.right_img.setPixmap(QPixmap("images/UI/Right.png"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)

        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)

        # 样式设置
        button_style = """
        QPushButton{color:white; background-color:rgb(48,124,208); border:2px; border-radius:5px; 
                    padding:5px 5px; margin:5px 5px;}
        QPushButton:hover{background-color: rgb(2,110,180);}
        """
        up_img_button.setStyleSheet(button_style)
        det_img_button.setStyleSheet(button_style)

        # 布局组合
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # 关于界面（保留）
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用目标检测系统')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_layout.addWidget(about_title)
        about_widget.setLayout(about_layout)

        # 添加标签页
        self.addTab(img_detection_widget, '图片检测')
        self.addTab(about_widget, '关于')
        self.setTabIcon(0, QIcon('images/UI/OIP.jpg'))
        self.setTabIcon(1, QIcon('images/UI/OIP.jpg'))

    def upload_img(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, '选择图片', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = fileName
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            self.right_img.setPixmap(QPixmap("images/UI/Right.png"))

    def detect_img(self):
        if not self.img2predict:
            QMessageBox.warning(self, "警告", "请先上传图片")
            return

        source = self.img2predict
        dataset = LoadImages(source, img_size=[self.detect_size, self.detect_size], stride=self.model.stride,
                             auto=self.model.pt)

        for path, im, im0s, _, _ in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.float() / 255.0
            if len(im.shape) == 3:
                im = im[None]

            pred = self.model(im)
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

            for _, det in enumerate(pred):
                annotator = Annotator(im0s.copy(), line_width=3, example=str(self.model.names))
                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.model.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

                im0 = annotator.result()
                resize_scale = self.output_size / im0.shape[0]
                im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result.jpg", im0)
                self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出', "确定要退出吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())