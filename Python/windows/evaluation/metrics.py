from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from configs.settings import OUTPUT_CLASSES

def calculate_metrics(y_true: list, y_pred: list) -> dict:
    """คำนวณค่าสถิติประสิทธิภาพ"""
    metrics = {
        'accuracy': np.mean(np.array(y_true) == np.array(y_pred)),
        'precision': precision_score(y_true, y_pred, average='weighted', labels=OUTPUT_CLASSES),
        'recall': recall_score(y_true, y_pred, average='weighted', labels=OUTPUT_CLASSES),
        'f1_score': f1_score(y_true, y_pred, average='weighted', labels=OUTPUT_CLASSES)
    }
    
    # คำนวณแยกแต่ละคลาส
    class_metrics = {}
    for i, cls in enumerate(OUTPUT_CLASSES):
        class_metrics[cls] = {
            'precision': precision_score(y_true, y_pred, labels=[cls], average='micro'),
            'recall': recall_score(y_true, y_pred, labels=[cls], average='micro'),
            'f1': f1_score(y_true, y_pred, labels=[cls], average='micro')
        }
    
    metrics['class_wise'] = class_metrics
    return metrics