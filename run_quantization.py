#!/usr/bin/env python3
"""
Demo Script: Chạy Quantization cho DDoS Detection Model
Sử dụng TensorFlow Lite để chuyển đổi mô hình sang dạng INT8
"""

import os
import sys
import numpy as np
import time
from quantization_script import main as run_quantization

def check_requirements():
    """Kiểm tra các file cần thiết"""
    required_files = [
        'attack_classifier.h5',
        'scaler_attack.pkl', 
        'attack_label_encoder.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Thiếu các file cần thiết:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Hãy chạy train2Phase.ipynb trước để tạo các file này")
        return False
    
    print("✅ Tất cả file cần thiết đã có sẵn")
    return True

def demo_quantization():
    """Demo quá trình quantization"""
    print("🚀 DDoS Detection Model - Quantization Demo")
    print("=" * 50)
    
    # Kiểm tra requirements
    if not check_requirements():
        return
    
    # Chạy quantization
    print("\n🔧 Bắt đầu quá trình Quantization...")
    run_quantization()
    
    # Kiểm tra kết quả
    print("\n📋 Kiểm tra kết quả:")
    result_files = [
        'attack_classifier_quantized.tflite',
        'quantization_info.json',
        'quantized_inference.py'
    ]
    
    for file in result_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"   ✅ {file}: {size:.2f} MB")
        else:
            print(f"   ❌ {file}: Không tìm thấy")

def demo_inference():
    """Demo inference với model đã quantized"""
    print("\n🧪 Demo Inference với Model đã Quantized")
    print("=" * 50)
    
    try:
        # Import inference functions
        from quantized_inference import load_quantized_model, predict_with_quantized_model
        import pickle
        
        # Load model và data
        print("📥 Loading model đã quantized...")
        interpreter = load_quantized_model()
        
        # Load scaler và label encoder
        with open('scaler_attack.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('attack_label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Tạo dữ liệu test
        print("📊 Tạo dữ liệu test...")
        num_features = 78
        test_data = np.random.randn(5, num_features)  # 5 mẫu test
        
        # Thực hiện inference
        print("⚡ Thực hiện inference...")
        start_time = time.time()
        
        predicted_labels, probabilities = predict_with_quantized_model(
            interpreter, test_data, scaler, label_encoder
        )
        
        inference_time = time.time() - start_time
        
        # Hiển thị kết quả
        print(f"\n📈 Kết quả inference ({inference_time:.4f}s):")
        for i, (label, prob) in enumerate(zip(predicted_labels, probabilities)):
            max_prob = np.max(prob)
            print(f"   Mẫu {i+1}: {label} (confidence: {max_prob:.3f})")
        
        print(f"\n⏱️  Thời gian inference trung bình: {inference_time/5:.4f}s/mẫu")
        
    except Exception as e:
        print(f"❌ Lỗi trong demo inference: {str(e)}")

def main():
    """Main function"""
    print("🎯 DDoS Detection Model - Quantization & Inference Demo")
    print("=" * 60)
    
    # Demo quantization
    demo_quantization()
    
    # Demo inference
    demo_inference()
    
    print("\n" + "=" * 60)
    print("🎉 Demo hoàn thành!")
    print("\n📚 Hướng dẫn sử dụng:")
    print("   1. Model đã quantized: attack_classifier_quantized.tflite")
    print("   2. Functions inference: quantized_inference.py")
    print("   3. Thông tin chi tiết: quantization_info.json")
    print("\n💡 Để sử dụng trong production:")
    print("   from quantized_inference import load_quantized_model, predict_with_quantized_model")

if __name__ == "__main__":
    main() 