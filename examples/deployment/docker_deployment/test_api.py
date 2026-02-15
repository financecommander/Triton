#!/usr/bin/env python3
"""
Test script for Triton DSL Inference Server

This script tests all API endpoints to ensure the server is working correctly.
"""

import sys
import time
import requests
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image

# Configuration
BASE_URL = "http://localhost:5000"
TIMEOUT = 30


def create_test_image(size=(32, 32)):
    """Create a random test image."""
    img = Image.fromarray(
        (np.random.rand(*size, 3) * 255).astype(np.uint8)
    )
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_health():
    """Test health endpoint."""
    print("\n" + "="*70)
    print("Testing /health endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        print("✓ Health check passed")
        print(f"  Status: {data['status']}")
        print(f"  Model loaded: {data['model_loaded']}")
        print(f"  Device: {data['device']}")
        return True
        
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_info():
    """Test info endpoint."""
    print("\n" + "="*70)
    print("Testing /info endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/info", timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        
        print("✓ Info request passed")
        if 'parameters' in data:
            print(f"  Total parameters: {data['parameters']['total']:,}")
        print(f"  Device: {data['device']}")
        return True
        
    except Exception as e:
        print(f"✗ Info request failed: {e}")
        return False


def test_predict():
    """Test single prediction endpoint."""
    print("\n" + "="*70)
    print("Testing /predict endpoint...")
    
    try:
        # Create test image
        img_bytes = create_test_image()
        
        # Send request
        files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
        data = {'top_k': '5'}
        
        response = requests.post(
            f"{BASE_URL}/predict",
            files=files,
            data=data,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        
        print("✓ Prediction request passed")
        print(f"  Inference time: {result['inference_time_ms']:.2f} ms")
        print(f"  Total time: {result['total_time_ms']:.2f} ms")
        print(f"  Top prediction: {result['predictions'][0]['class_name']}")
        print(f"  Confidence: {result['predictions'][0]['confidence']:.2f}%")
        return True
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return False


def test_batch_predict():
    """Test batch prediction endpoint."""
    print("\n" + "="*70)
    print("Testing /predict_batch endpoint...")
    
    try:
        # Create multiple test images
        num_images = 3
        files = []
        for i in range(num_images):
            img_bytes = create_test_image()
            files.append(
                ('images', (f'test{i}.jpg', img_bytes, 'image/jpeg'))
            )
        
        # Send request
        response = requests.post(
            f"{BASE_URL}/predict_batch",
            files=files,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        result = response.json()
        
        print("✓ Batch prediction passed")
        print(f"  Batch size: {result['batch_size']}")
        print(f"  Total time: {result['total_time_ms']:.2f} ms")
        print(f"  Avg time per image: {result['total_time_ms'] / num_images:.2f} ms")
        return True
        
    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
        return False


def test_metrics():
    """Test metrics endpoint."""
    print("\n" + "="*70)
    print("Testing /metrics endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=TIMEOUT)
        response.raise_for_status()
        
        print("✓ Metrics request passed")
        print(f"  Response length: {len(response.text)} bytes")
        return True
        
    except Exception as e:
        print(f"✗ Metrics request failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("Triton DSL Inference Server Test Suite")
    print("="*70)
    print(f"Testing server at: {BASE_URL}")
    
    # Wait for server to start
    print("\nWaiting for server to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Server is ready!")
                break
        except:
            pass
        
        if i < max_retries - 1:
            time.sleep(1)
        else:
            print("✗ Server did not become ready in time")
            return 1
    
    # Run tests
    results = []
    results.append(("Health Check", test_health()))
    results.append(("Info", test_info()))
    results.append(("Single Prediction", test_predict()))
    results.append(("Batch Prediction", test_batch_predict()))
    results.append(("Metrics", test_metrics()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)
    
    return 0 if passed == total else 1


if __name__ == '__main__':
    sys.exit(main())
