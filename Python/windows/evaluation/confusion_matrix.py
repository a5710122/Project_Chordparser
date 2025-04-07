import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from configs.settings import OUTPUT_CLASSES
import numpy as np

def plot_and_save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """
    สร้างและบันทึก Confusion Matrix
    :param y_true: ค่าจริง (เช่น ['C', 'D', ...])
    :param y_pred: ค่าที่โมเดลทำนาย
    :param filename: ชื่อไฟล์ที่จะบันทึก
    """
    # สร้าง Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=OUTPUT_CLASSES)
    
    # ปรับแต่งการแสดงผล
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=OUTPUT_CLASSES
    )
    
    # ปรับแต่งรูปให้สวยงาม
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix - โน้ตดนตรี", pad=20, fontsize=16)
    plt.xlabel("Predicted Label", labelpad=10)
    plt.ylabel("True Label", labelpad=10)
    
    # บันทึกไฟล์
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"บันทึก Confusion Matrix เรียบร้อยที่ {filename}")

def calculate_metrics(y_true, y_pred):
    """คำนวณค่าสถิติ"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }