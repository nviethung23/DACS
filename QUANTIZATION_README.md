# 🚀 DDoS Detection Model - Quantization với TensorFlow Lite

## 📋 Tổng quan

Dự án này áp dụng **Quantization** sử dụng **TensorFlow Lite** để chuyển đổi mô hình DDoS Detection sang dạng **INT8**, giúp:
- ⚡ **Tăng tốc độ xử lý** (inference)
- 📉 **Giảm kích thước mô hình** 
- 🔋 **Tiết kiệm năng lượng** (đặc biệt quan trọng cho edge devices)
- 📱 **Tối ưu cho mobile/embedded systems**

## 🎯 Lợi ích của Quantization

### 1. **Giảm kích thước mô hình**
- Chuyển từ float32 (32-bit) sang int8 (8-bit)
- Giảm kích thước file ~75%
- Tiết kiệm bộ nhớ lưu trữ

### 2. **Tăng tốc độ inference**
- Tối ưu hóa cho CPU/GPU
- Giảm thời gian xử lý
- Phù hợp cho real-time applications

### 3. **Tiết kiệm năng lượng**
- Giảm power consumption
- Phù hợp cho IoT devices
- Tối ưu cho edge computing

## 📁 Cấu trúc Files

```
DACS/
├── train2Phase.ipynb              # File chính - Training 2 phase
├── quantization_script.py         # Script quantization chính
├── run_quantization.py           # Demo script
├── quantized_inference.py        # Functions để sử dụng model đã quantized
├── attack_classifier.h5          # Model gốc (float32)
├── attack_classifier_quantized.tflite  # Model đã quantized (int8)
├── quantization_info.json        # Thông tin chi tiết quantization
├── scaler_attack.pkl             # Scaler cho preprocessing
├── attack_label_encoder.pkl      # Label encoder
└── QUANTIZATION_README.md        # File này
```

## 🚀 Cách sử dụng

### Bước 1: Chuẩn bị
```bash
# Đảm bảo đã có các file cần thiết
ls -la *.h5 *.pkl
```

### Bước 2: Chạy Quantization
```bash
# Cách 1: Chạy script chính
python quantization_script.py

# Cách 2: Chạy demo (khuyến nghị)
python run_quantization.py
```

### Bước 3: Sử dụng model đã quantized
```python
from quantized_inference import load_quantized_model, predict_with_quantized_model
import pickle
import numpy as np

# Load model đã quantized
interpreter = load_quantized_model('attack_classifier_quantized.tflite')

# Load scaler và label encoder
with open('scaler_attack.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('attack_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Dữ liệu cần dự đoán (78 features)
test_data = np.random.randn(1, 78)  # Thay thế bằng dữ liệu thực tế

# Thực hiện dự đoán
predicted_labels, probabilities = predict_with_quantized_model(
    interpreter, test_data, scaler, label_encoder
)

print(f"Kết quả: {predicted_labels[0]}")
print(f"Confidence: {np.max(probabilities[0]):.3f}")
```

## 📊 Kết quả mong đợi

### So sánh hiệu suất:
| Metric | Model gốc (H5) | Model đã Quantized (TFLite) | Cải thiện |
|--------|----------------|------------------------------|-----------|
| Kích thước | ~50-100 MB | ~12-25 MB | **75%** |
| Thời gian inference | ~100ms | ~25ms | **4x** |
| Memory usage | ~200MB | ~50MB | **75%** |
| Độ chính xác | 99% | 98-99% | **<1%** |

### Output files:
- `attack_classifier_quantized.tflite`: Model đã quantized
- `quantization_info.json`: Thông tin chi tiết
- `quantized_inference.py`: Functions để sử dụng

## 🔧 Chi tiết kỹ thuật

### Quantization Process:
1. **Load model gốc** (float32)
2. **Tạo representative dataset** (calibration data)
3. **Chuyển đổi sang INT8** với TensorFlow Lite
4. **Test và validate** kết quả
5. **Lưu model đã quantized**

### Cấu hình Quantization:
```python
converter = lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

## 🎯 Ứng dụng thực tế

### 1. **Real-time DDoS Detection**
```python
def real_time_detection(network_traffic):
    """Phát hiện DDoS real-time"""
    result = predict_with_quantized_model(
        interpreter, network_traffic, scaler, label_encoder
    )
    return result
```

### 2. **Edge Computing**
- Deploy trên Raspberry Pi
- IoT devices
- Network appliances

### 3. **Mobile Applications**
- Android/iOS apps
- Offline detection
- Battery optimization

## ⚠️ Lưu ý quan trọng

### 1. **Độ chính xác**
- Quantization có thể giảm độ chính xác nhẹ (<1%)
- Test kỹ trước khi deploy production

### 2. **Input preprocessing**
- Dữ liệu input phải được preprocess giống training
- Sử dụng cùng scaler và label encoder

### 3. **Hardware compatibility**
- Đảm bảo hardware hỗ trợ INT8 operations
- Test trên target platform

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **File không tìm thấy**
```bash
# Kiểm tra các file cần thiết
ls -la attack_classifier.h5 scaler_attack.pkl attack_label_encoder.pkl
```

2. **Memory error**
```python
# Giảm batch size hoặc số lượng calibration samples
num_calibration_samples = 500  # Thay vì 1000
```

3. **Accuracy drop quá lớn**
```python
# Tăng số lượng calibration samples
# Hoặc sử dụng dynamic range quantization
```

## 📚 Tài liệu tham khảo

- [TensorFlow Lite Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [INT8 Quantization Guide](https://www.tensorflow.org/lite/performance/quantization_spec)
- [Model Optimization](https://www.tensorflow.org/lite/performance/model_optimization)

## 🤝 Đóng góp

Nếu bạn muốn cải thiện quantization process:
1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push và tạo Pull Request

## 📞 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra troubleshooting section
2. Xem quantization_info.json để debug
3. Tạo issue với log chi tiết

---

**🎉 Chỉ là test nha thấy cần gì tự bổ sung!** 