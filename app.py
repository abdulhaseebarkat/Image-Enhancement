import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QComboBox, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from PIL import Image

class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision Assignment")
        self.setGeometry(100, 100, 1200, 800)
        
        self.original_image = None
        self.filtered_image = None
        
        self.initUI()
        
    def initUI(self):
        # Main widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        
        # Layouts
        self.main_layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.control_layout = QHBoxLayout()
        
        # Heading
        self.heading_label = QLabel("Computer Vision Assignment", self)
        self.heading_label.setAlignment(Qt.AlignCenter)
        self.heading_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        self.main_layout.addWidget(self.heading_label)
        
        # Original image display
        self.panel_original = QLabel(self)
        self.panel_original.setAlignment(Qt.AlignCenter)
        self.panel_original.setStyleSheet("border: 1px solid #ccc; padding: 10px;")
        self.image_layout.addWidget(self.panel_original)
        
        # Filtered image display
        self.panel_filtered = QLabel(self)
        self.panel_filtered.setAlignment(Qt.AlignCenter)
        self.panel_filtered.setStyleSheet("border: 1px solid #ccc; padding: 10px;")
        self.image_layout.addWidget(self.panel_filtered)
        
        self.main_layout.addLayout(self.image_layout)
        
        # Load Image button
        self.btn_load = QPushButton("Load Image", self)
        self.btn_load.setStyleSheet("padding: 10px; font-size: 14px;")
        self.btn_load.clicked.connect(self.load_image)
        self.control_layout.addWidget(self.btn_load)
        
        # Filter selection dropdown
        self.filter_type = QComboBox(self)
        self.filter_type.addItems(["Gaussian Blur", "Butterworth Lowpass Filter", "Laplacian Highpass Filter", "Histogram Matching"])
        self.filter_type.setStyleSheet("padding: 10px; font-size: 14px;")
        self.control_layout.addWidget(self.filter_type)
        
        # Apply Filter button
        self.btn_apply = QPushButton("Apply Filter", self)
        self.btn_apply.setStyleSheet("padding: 10px; font-size: 14px;")
        self.btn_apply.clicked.connect(self.apply_filter)
        self.control_layout.addWidget(self.btn_apply)
        
        self.main_layout.addLayout(self.control_layout)
        self.main_widget.setLayout(self.main_layout)
        
        # Set margins and spacing for layouts
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.control_layout.setSpacing(20)
        self.image_layout.setSpacing(20)
        
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.panel_original)
        
    def display_image(self, image, panel):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        panel.setPixmap(pixmap.scaled(panel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
    def apply_filter(self):
        if self.original_image is None:
            QMessageBox.critical(self, "Error", "Please load an image first.")
            return
        
        filter_type = self.filter_type.currentText()
        
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        if filter_type == "Gaussian Blur":
            filtered_image = self.apply_gaussian_blur(gray_image)
        elif filter_type == "Butterworth Lowpass Filter":
            filtered_image = self.apply_butterworth_lowpass_filter(gray_image)
        elif filter_type == "Laplacian Highpass Filter":
            filtered_image = self.apply_laplacian_filter(gray_image)
        elif filter_type == "Histogram Matching":
            template_path, _ = QFileDialog.getOpenFileName(self, "Open Template Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
            if template_path:
                template_image = cv2.imread(template_path)
                filtered_image = self.histogram_matching(self.original_image, template_image)
            else:
                QMessageBox.critical(self, "Error", "Please select a template image for histogram matching.")
                return
        else:
            QMessageBox.critical(self, "Error", "Unknown filter type.")
            return
        
        self.filtered_image = filtered_image
        self.display_image(filtered_image, self.panel_filtered)
        
    def apply_gaussian_blur(self, image, kernel_size=5):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def apply_butterworth_lowpass_filter(self, image, d0=15, n=1):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)

        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2

        mask = np.zeros((rows, cols), np.float32)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                mask[i, j] = 1 / (1 + (d / d0) ** (2 * n))

        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return np.uint8(img_back)

    def apply_laplacian_filter(self, image):
        laplacian_filtered = cv2.Laplacian(image, cv2.CV_64F)
        laplacian_filtered = np.uint8(np.absolute(laplacian_filtered))
        return laplacian_filtered

    def histogram_matching(self, source, template):
        source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
        template_hist, _ = np.histogram(template.flatten(), 256, [0, 256])

        s_cum_hist = np.cumsum(source_hist) / source.size
        t_cum_hist = np.cumsum(template_hist) / template.size

        s_values = np.zeros_like(source)
        for i in range(256):
            idx = np.abs(s_cum_hist[i] - t_cum_hist).argmin()
            s_values[source == i] = idx
        matched = s_values

        return matched

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = ImageProcessorApp()
    mainWin.show()
    sys.exit(app.exec_())
