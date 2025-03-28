# Advanced Image Processor

## Description
Advanced Image Processor is an intuitive GUI application for enhancing and colorizing images using advanced deep learning techniques. Built with Python and PyQt, it leverages GFPGAN to enhance image quality and colorize black-and-white images seamlessly.

### Important: App won't work unless you complete the setup and install all required models.








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
![image (1)](https://github.com/user-attachments/assets/653ad3c4-2502-425d-b09e-18aa66c95673)


### Processing Options Demonstrations:

- **Enhance Image Quality**
  - ![image (2)](https://github.com/user-attachments/assets/4c50cc6b-2079-4f33-8645-9ce962c1537e)

- **Colorize B&W Image**
  - ![image (3)](https://github.com/user-attachments/assets/1d4b559f-ab8d-4dc8-98b8-07c6e9a46678)

- **Colorize & Enhance Image**
  - ![image (4)](https://github.com/user-attachments/assets/4c473c8e-0238-48b3-876d-2a38ca9757ea)

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


