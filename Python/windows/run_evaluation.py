from tests.test_synthetic import test_synthetic_notes
from evaluation.metrics import calculate_metrics
from evaluation.confusion_matrix import plot_confusion_matrix

# ทดสอบกับเสียงสังเคราะห์
results = test_synthetic_notes("models/trained.tflite")

# เตรียมข้อมูลสำหรับ evaluation
y_true = []
y_pred = []
for note, data in results.items():
    y_true.append(note)
    y_pred.append(data['predicted'])

# คำนวณและแสดงผล
metrics = calculate_metrics(y_true, y_pred)
print("=== Metrics ===")
print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"F1 Score: {metrics['f1_score']:.2%}")

# สร้าง confusion matrix
plot_confusion_matrix(y_true, y_pred, "Note Classification Performance")