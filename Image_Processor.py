import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                            QHBoxLayout, QVBoxLayout, QWidget, QFileDialog, 
                            QMessageBox, QSplitter, QGroupBox, QSizePolicy)
from PyQt5.QtGui import QPixmap, QColor, QPalette
from PyQt5.QtCore import Qt, QSize
import numpy as np
from PIL import Image
import torch
from colorizers import ImageColorizer, load_image, preprocess_image, postprocess_tensor
import cv2

from pathlib import Path

project_root = str(Path(__file__).parent)
gfpgan_path = os.path.join(project_root, 'GFPGAN')

sys.path.insert(0, project_root)
sys.path.insert(0, gfpgan_path)

try:
    from gfpgan import GFPGANer  
except ImportError as e:
    print("Error")
    
    sys.exit(1)

class ImageProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Image Processor")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize models
        self.colorizer = ImageColorizer().eval()
        if torch.cuda.is_available():
            self.colorizer.cuda()
            
        try:
            self.face_enhancer = GFPGANer(
                model_path='GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth',
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize GFPGAN: {str(e)}")
        
        # Setup UI
        self.init_ui()
        self.apply_styles()
        
        # Current images
        self.current_image_path = None
        self.processed_image = None
    
    def init_ui(self):
        # Main widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - image display (70%)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Image splitter
        self.image_splitter = QSplitter(Qt.Horizontal)
        left_layout.addWidget(self.image_splitter)
        
        # Original image panel
        self.original_group = QGroupBox("Original Image")
        original_layout = QVBoxLayout()
        self.original_group.setLayout(original_layout)
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(600, 500)
        original_layout.addWidget(self.original_label)
        self.image_splitter.addWidget(self.original_group)
        
        # Processed image panel
        self.processed_group = QGroupBox("Processed Image")
        processed_layout = QVBoxLayout()
        self.processed_group.setLayout(processed_layout)
        self.processed_label = QLabel()
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setMinimumSize(600, 500)
        processed_layout.addWidget(self.processed_label)
        self.image_splitter.addWidget(self.processed_group)
        
        # Right panel - controls (30%)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        # File operations group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        file_layout.setContentsMargins(5, 10, 5, 5)
        
        # Select image button (green)
        self.select_btn = self.create_styled_button("Select Image", "#4CAF50", width=180)
        self.select_btn.clicked.connect(self.load_image)
        file_layout.addWidget(self.select_btn)
        
        # Save button (blue)
        self.save_btn = self.create_styled_button("Save Processed Image", "#2196F3", width=180)
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        file_layout.addWidget(self.save_btn)
        
        right_layout.addWidget(file_group)
        right_layout.addSpacing(5)
        
        # Processing options group (3 options as requested)
        process_group = QGroupBox("Processing Options")
        process_layout = QVBoxLayout()
        process_group.setLayout(process_layout)
        process_layout.setContentsMargins(5, 10, 5, 5)
        
        # 1. Enhance Image Quality
        self.option1 = self.create_styled_button("1. Enhance Image Quality", "#2196F3", width=180)
        self.option1.clicked.connect(lambda: self.process_image(1))
        self.option1.setEnabled(False)
        process_layout.addWidget(self.option1)
        
        # 2. Colorize B&W Image
        self.option2 = self.create_styled_button("2. Colorize B&W Image", "#2196F3", width=180)
        self.option2.clicked.connect(lambda: self.process_image(2))
        self.option2.setEnabled(False)
        process_layout.addWidget(self.option2)
        
        # 3. Colorize & Enhance
        self.option3 = self.create_styled_button("3. Colorize & Enhance", "#2196F3", width=180)
        self.option3.clicked.connect(lambda: self.process_image(3))
        self.option3.setEnabled(False)
        process_layout.addWidget(self.option3)
        
        right_layout.addWidget(process_group)
        right_layout.addStretch()
        
        # Reset button (red)
        self.reset_btn = self.create_styled_button("Reset All", "#F44336", width=180)
        self.reset_btn.clicked.connect(self.reset_app)
        self.reset_btn.setEnabled(False)
        right_layout.addWidget(self.reset_btn)
        
        main_layout.addWidget(left_panel, 70)
        main_layout.addWidget(right_panel, 30)
    
    def create_styled_button(self, text, color, width=None):
        """Create styled button with full text visibility"""
        btn = QPushButton(text)
        btn.setMinimumHeight(40)
        if width:
            btn.setFixedWidth(width)
        btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: bold;
                border: none;
                font-size: 12px;
                min-width: {width}px;
            }}
            QPushButton:hover {{
                background-color: {self.darken_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(color, 40)};
            }}
            QPushButton:disabled {{
                background-color: #E0E0E0;
                color: #9E9E9E;
            }}
        """)
        return btn
    
    def darken_color(self, hex_color, percent=20):
        """Helper to darken colors for hover effect"""
        color = QColor(hex_color)
        return color.darker(100 + percent).name()
    
    def apply_styles(self):
        """Apply consistent styling"""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #dddddd;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QLabel {
                border: 1px solid #aaaaaa;
                background-color: #f0f0f0;
            }
        """)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        self.setPalette(palette)
    
    def load_image(self):
        """Load image file"""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if path:
            self.current_image_path = path
            self.display_image(path, self.original_label)
            self.toggle_buttons(True)
            self.save_btn.setEnabled(False)
            self.reset_btn.setEnabled(True)
            self.statusBar().showMessage(f"Loaded: {os.path.basename(path)}", 3000)
    
    def toggle_buttons(self, enabled):
        """Toggle processing buttons"""
        self.option1.setEnabled(enabled)
        self.option2.setEnabled(enabled)
        self.option3.setEnabled(enabled)
    
    def display_image(self, path, label):
        """Display image in label"""
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                label.width(), label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled)
    
    def process_image(self, option):
        """Process image based on option"""
        if not self.current_image_path:
            QMessageBox.warning(self, "Error", "No image selected!")
            return
        
        try:
            self.statusBar().showMessage(f"Processing option {option}...")
            QApplication.processEvents()
            
            processed = None
            
            if option == 1:  # Enhance Image Quality
                img = cv2.imread(self.current_image_path)
                if img is None:
                    raise ValueError("Failed to read image file")
                
                if self.face_enhancer is None:
                    raise ValueError("GFPGAN model not initialized")
                
                _, _, enhanced = self.face_enhancer.enhance(img, has_aligned=False)
                if enhanced is None:
                    raise ValueError("Enhancement failed")
                
                processed = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                message = "Image enhancement complete"
                
            elif option == 2:  # Colorize B&W Image
                original_img = load_image(self.current_image_path)
                original_tensor, resized_tensor = preprocess_image(original_img)
                if torch.cuda.is_available():
                    resized_tensor = resized_tensor.cuda()
                colorized = postprocess_tensor(original_tensor, self.colorizer(resized_tensor).cpu())
                processed = (colorized * 255).astype(np.uint8)
                message = "Colorization complete"
                
            elif option == 3:  # Colorize & Enhance
                # First colorize
                original_img = load_image(self.current_image_path)
                original_tensor, resized_tensor = preprocess_image(original_img)
                if torch.cuda.is_available():
                    resized_tensor = resized_tensor.cuda()
                colorized = postprocess_tensor(original_tensor, self.colorizer(resized_tensor).cpu())
                colorized_img = (colorized * 255).astype(np.uint8)
                
                # Then enhance
                if self.face_enhancer is None:
                    raise ValueError("GFPGAN model not initialized")
                
                # Convert to BGR for GFPGAN
                colorized_bgr = cv2.cvtColor(colorized_img, cv2.COLOR_RGB2BGR)
                _, _, enhanced = self.face_enhancer.enhance(colorized_bgr, has_aligned=False)
                if enhanced is None:
                    raise ValueError("Enhancement failed after colorization")
                
                processed = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                message = "Colorization and enhancement complete"
            
            self.processed_image = processed
            temp_path = "temp_output.png"
            Image.fromarray(processed).save(temp_path)
            self.display_image(temp_path, self.processed_label)
            os.remove(temp_path)
            
            self.save_btn.setEnabled(True)
            self.statusBar().showMessage(message, 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
            self.statusBar().showMessage("Processing failed!", 3000)
    
    def save_image(self):
        """Save processed image"""
        if self.processed_image is None:
            QMessageBox.warning(self, "Error", "No processed image to save!")
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;Bitmap Image (*.bmp)"
        )
        if path:
            try:
                Image.fromarray(self.processed_image).save(path)
                self.statusBar().showMessage(f"Image saved to {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")
    
    def reset_app(self):
        """Reset application state"""
        self.current_image_path = None
        self.processed_image = None
        self.original_label.clear()
        self.processed_label.clear()
        self.toggle_buttons(False)
        self.save_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.statusBar().showMessage("Application reset", 2000)
    
    def resizeEvent(self, event):
        if self.current_image_path:
            self.display_image(self.current_image_path, self.original_label)
        super().resizeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec_())