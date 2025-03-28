# Advanced Image Processor

## Description
Advanced Image Processor is an intuitive GUI application for enhancing and colorizing images using advanced deep learning techniques. Built with Python and PyQt, it leverages GFPGAN to enhance image quality and colorize black-and-white images seamlessly.

## Features
- **Enhance Image Quality** with GFPGAN
- **Colorize Black & White images** using AI
- Combined option for **colorization and quality enhancement**

---

## Installation & Setup

### Step 1: Clone the repository

```bash
git clone <your-repo-url>
cd <repo-name>
```

### Step 2: Clone GFPGAN

```bash
git clone https://github.com/TencentARC/GFPGAN.git
cd GFPGAN
```

### Step 2: Download the Pretrained Model

```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models
```

### Step 2: Install dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the application with the following command:

```bash
python Image_Processor.py
```

---

## Application Screenshots

### Main Interface
![App Interface](<path-to-app-screenshot>)

### Processing Options Demonstrations:

- **Enhance Image Quality**
  - ![Enhance Quality](path/to/enhance_image_quality.png)

- **Colorize B&W Image**
  - ![Colorize Image](path/to/colorize_bw_image.png)

- **Colorize & Enhance Image**
  - ![Colorize and Enhance](path/to/colorize_enhance_image.png)

---

## Usage

Run the application:

```bash
python Image_Processor.py
```

Use the provided graphical interface to select images and apply processing options.

---

## Folder Structure

```
├── colorizers
├── GFPGAN
├── imgs
└── Image_Processor.py
```

---

## Dependencies

- PyQt5
- PyTorch
- OpenCV
- Pillow
- GFPGAN


