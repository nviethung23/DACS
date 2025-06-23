"""
Quantization Script for DDoS Detection Model
Sử dụng TensorFlow Lite để chuyển đổi mô hình sang dạng INT8
Tăng tốc độ xử lý và giảm kích thước mô hình
"""

import tensorflow as tf
from tensorflow import lite
from tensorflow.lite.python.interpreter import Interpreter
import numpy as np
import time
import os
import json
import pickle
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Import custom AttentionLayer
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Dropout, Bidirectional, LSTM, Dense
from tensorflow.keras.regularizers import l2

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

def load_model_and_data():
    """Load mô hình đã train và dữ liệu cần thiết"""
    print("📥 Loading mô hình và dữ liệu...")
    
    # Load model
    model = tf.keras.models.load_model('attack_classifier.h5', 
                                      custom_objects={'AttentionLayer': AttentionLayer})
    
    # Load scaler và label encoder
    with open('scaler_attack.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('attack_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load một phần dữ liệu test để làm representative dataset
    # Giả sử có file test data hoặc tạo dummy data
    print("📊 Tạo representative dataset...")
    
    # Tạo dummy data cho representative dataset (thay thế bằng dữ liệu thực tế)
    num_features = 78  # Số features trong dataset
    num_calibration_samples = 1000
    
    # Tạo dữ liệu calibration ngẫu nhiên (thay thế bằng dữ liệu thực tế)
    calibration_data = np.random.randn(num_calibration_samples, num_features)
    
    return model, scaler, label_encoder, calibration_data

def create_representative_dataset(calibration_data, scaler):
    """Tạo representative dataset cho quantization"""
    def representative_dataset():
        for data in calibration_data:
            # Reshape data để phù hợp với input shape của model
            yield [data.reshape(1, -1, 1)]
    return representative_dataset

def quantize_model(model, representative_dataset):
    """Chuyển đổi model sang TensorFlow Lite với Quantization INT8"""
    print("⚡ Bắt đầu quá trình Quantization...")
    
    # Tạo converter
    converter = lite.TFLiteConverter.from_keras_model(model)
    
    # Cấu hình quantization
    converter.optimizations = [lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Chuyển đổi model
    print("🔄 Đang chuyển đổi model...")
    tflite_model_quantized = converter.convert()
    
    return tflite_model_quantized

def save_quantized_model(tflite_model_quantized, model_path='attack_classifier_quantized.tflite'):
    """Lưu model đã quantized"""
    print(f"💾 Lưu model đã quantized: {model_path}")
    with open(model_path, 'wb') as f:
        f.write(tflite_model_quantized)

def compare_model_sizes(original_path='attack_classifier.h5', 
                       quantized_path='attack_classifier_quantized.tflite'):
    """So sánh kích thước model"""
    original_size = os.path.getsize(original_path)
    quantized_size = os.path.getsize(quantized_path)
    
    print(f"\n📏 So sánh kích thước model:")
    print(f"   Original model (H5): {original_size / (1024*1024):.2f} MB")
    print(f"   Quantized model (TFLite): {quantized_size / (1024*1024):.2f} MB")
    print(f"   Giảm kích thước: {((original_size - quantized_size) / original_size * 100):.1f}%")
    
    return original_size, quantized_size

def test_inference_performance(model, quantized_model_path, test_data, scaler):
    """Test hiệu suất inference"""
    print("\n🧪 Test hiệu suất inference...")
    
    # Load TFLite model
    interpreter = Interpreter(model_path=quantized_model_path)
    interpreter.allocate_tensors()
    
    # Lấy input và output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Test với một số mẫu
    test_samples = test_data[:10]
    
    # Test model gốc
    start_time = time.time()
    original_pred = model.predict(test_samples)
    original_time = time.time() - start_time
    
    # Test model đã quantized
    start_time = time.time()
    quantized_predictions = []
    
    for sample in test_samples:
        # Reshape và quantize input
        input_data = sample.reshape(1, -1, 1)
        
        # Quantize input data
        input_scale, input_zero_point = input_details[0]['quantization']
        input_data_quantized = np.round(input_data / input_scale + input_zero_point).astype(np.int8)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data_quantized)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data_dequantized = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        quantized_predictions.append(output_data_dequantized)
    
    quantized_time = time.time() - start_time
    
    # So sánh kết quả
    print(f"\n⏱️  Thời gian inference:")
    print(f"   Original model: {original_time:.4f} seconds")
    print(f"   Quantized model: {quantized_time:.4f} seconds")
    print(f"   Tốc độ tăng: {original_time/quantized_time:.2f}x")
    
    # So sánh độ chính xác
    original_pred_classes = np.argmax(original_pred, axis=1)
    quantized_pred_classes = np.argmax(np.array(quantized_predictions).squeeze(), axis=1)
    
    accuracy_comparison = np.mean(original_pred_classes == quantized_pred_classes)
    print(f"\n🎯 Độ chính xác so sánh: {accuracy_comparison:.4f}")
    
    return {
        'original_time': original_time,
        'quantized_time': quantized_time,
        'speedup_factor': original_time/quantized_time,
        'accuracy_comparison': accuracy_comparison,
        'input_quantization': {
            'scale': input_details[0]['quantization'][0],
            'zero_point': input_details[0]['quantization'][1]
        },
        'output_quantization': {
            'scale': output_details[0]['quantization'][0],
            'zero_point': output_details[0]['quantization'][1]
        }
    }

def save_quantization_info(quantization_info, original_size, quantized_size):
    """Lưu thông tin quantization"""
    info = {
        'original_size_mb': original_size / (1024*1024),
        'quantized_size_mb': quantized_size / (1024*1024),
        'size_reduction_percent': ((original_size - quantized_size) / original_size * 100),
        'speedup_factor': quantization_info['speedup_factor'],
        'accuracy_comparison': quantization_info['accuracy_comparison'],
        'input_quantization': quantization_info['input_quantization'],
        'output_quantization': quantization_info['output_quantization']
    }
    
    with open('quantization_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"📄 Thông tin quantization đã lưu: quantization_info.json")

def create_inference_functions():
    """Tạo các function để sử dụng model đã quantized"""
    inference_code = '''
def load_quantized_model(model_path='attack_classifier_quantized.tflite'):
    """
    Load và trả về model đã quantized
    """
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_with_quantized_model(interpreter, input_data, scaler, label_encoder):
    """
    Dự đoán sử dụng model đã quantized
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preprocess input data
    input_scaled = scaler.transform(input_data)
    input_reshaped = input_scaled.reshape(-1, input_scaled.shape[1], 1)
    
    predictions = []
    
    for sample in input_reshaped:
        # Quantize input
        input_scale, input_zero_point = input_details[0]['quantization']
        input_quantized = np.round(sample / input_scale + input_zero_point).astype(np.int8)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_quantized.reshape(1, -1, 1))
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output
        output_scale, output_zero_point = output_details[0]['quantization']
        output_dequantized = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        predictions.append(output_dequantized)
    
    predictions = np.array(predictions).squeeze()
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    
    return predicted_labels, predictions

def real_time_ddos_detection(interpreter, network_data, scaler, label_encoder):
    """
    Phát hiện DDoS real-time sử dụng model đã quantized
    """
    predicted_labels, probabilities = predict_with_quantized_model(
        interpreter, network_data, scaler, label_encoder
    )
    
    results = []
    for i, (label, prob) in enumerate(zip(predicted_labels, probabilities)):
        max_prob = np.max(prob)
        results.append({
            'sample_id': i,
            'predicted_attack': label,
            'confidence': max_prob,
            'is_ddos': label != 'Benign',
            'all_probabilities': dict(zip(label_encoder.classes_, prob))
        })
    
    return results
'''
    
    with open('quantized_inference.py', 'w') as f:
        f.write(inference_code)
    
    print(f"🔧 Function inference đã tạo: quantized_inference.py")

def main():
    """Main function để thực hiện toàn bộ quá trình quantization"""
    print("🚀 Bắt đầu quá trình Quantization cho DDoS Detection Model")
    print("=" * 60)
    
    try:
        # 1. Load model và data
        model, scaler, label_encoder, calibration_data = load_model_and_data()
        
        # 2. Tạo representative dataset
        representative_dataset = create_representative_dataset(calibration_data, scaler)
        
        # 3. Quantize model
        tflite_model_quantized = quantize_model(model, representative_dataset)
        
        # 4. Lưu model đã quantized
        save_quantized_model(tflite_model_quantized)
        
        # 5. So sánh kích thước
        original_size, quantized_size = compare_model_sizes()
        
        # 6. Test hiệu suất
        quantization_info = test_inference_performance(
            model, 'attack_classifier_quantized.tflite', calibration_data, scaler
        )
        
        # 7. Lưu thông tin
        save_quantization_info(quantization_info, original_size, quantized_size)
        
        # 8. Tạo inference functions
        create_inference_functions()
        
        print("\n" + "=" * 60)
        print("✅ Hoàn thành Quantization!")
        print(f"📁 Files đã tạo:")
        print(f"   - attack_classifier_quantized.tflite (model đã quantized)")
        print(f"   - quantization_info.json (thông tin chi tiết)")
        print(f"   - quantized_inference.py (functions để sử dụng)")
        
        print(f"\n📊 Tóm tắt kết quả:")
        print(f"   - Giảm kích thước: {((original_size - quantized_size) / original_size * 100):.1f}%")
        print(f"   - Tăng tốc độ: {quantization_info['speedup_factor']:.2f}x")
        print(f"   - Độ chính xác: {quantization_info['accuracy_comparison']:.4f}")
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình quantization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 