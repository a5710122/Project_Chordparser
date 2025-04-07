import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from configs.settings import OUTPUT_CLASSES

def plot_confusion_matrix(y_true: list, y_pred: list, title: str = "Confusion Matrix"):
    """สร้างและแสดงภาพ confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=OUTPUT_CLASSES)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=OUTPUT_CLASSES)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # บันทึกไฟล์
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    return "confusion_matrix.png"  # คืน path ของไฟล์ภาพ