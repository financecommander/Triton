# Tutorial 5: Model Deployment

Learn how to deploy ternary neural networks to production environments, including ONNX export, TensorRT optimization, mobile deployment, edge devices, and serving infrastructure.

## Learning Objectives

- Export ternary models to ONNX format
- Optimize models with TensorRT for GPU inference
- Deploy to mobile devices using TFLite
- Optimize for edge devices (Raspberry Pi, Jetson)
- Set up model serving with TorchServe
- Implement efficient inference pipelines
- Monitor deployed models in production

## Prerequisites

- Completed [Tutorial 1](01_basic_model.md) through [Tutorial 4](04_training.md)
- Understanding of model inference
- Basic knowledge of deployment architectures
- Familiarity with Docker (helpful)

## Deployment Overview

### Deployment Targets

```
Cloud/Server:
├── TorchServe (PyTorch native)
├── ONNX Runtime (cross-platform)
└── TensorRT (NVIDIA GPUs)

Mobile:
├── TensorFlow Lite (Android/iOS)
├── Core ML (iOS)
└── ONNX Mobile

Edge Devices:
├── Raspberry Pi (CPU/ARM)
├── NVIDIA Jetson (GPU/ARM)
└── Intel Neural Compute Stick
```

### Why Ternary Models Excel in Deployment

1. **Low Memory**: 16x smaller than FP32
2. **Fast Inference**: Sparse computation
3. **Energy Efficient**: Fewer operations
4. **Edge-Friendly**: Fits on resource-constrained devices

## ONNX Export and Optimization

### Basic ONNX Export

```python
import torch
import torch.onnx
from pathlib import Path

class ONNXExporter:
    """Export ternary models to ONNX format."""
    
    def __init__(self, model, sample_input_shape):
        self.model = model
        self.sample_input_shape = sample_input_shape
        
    def export(self, output_path, opset_version=13, 
               dynamic_axes=None, verbose=False):
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Dict of dynamic axes for inputs/outputs
            verbose: Print export details
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(self.sample_input_shape).cuda()
        
        # Default dynamic axes for batch size
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=verbose
        )
        
        print(f'Model exported to {output_path}')
        return output_path
    
    def verify_export(self, onnx_path):
        """Verify exported ONNX model."""
        import onnx
        from onnx import checker, shape_inference
        
        # Load model
        onnx_model = onnx.load(onnx_path)
        
        # Check model
        checker.check_model(onnx_model)
        
        # Infer shapes
        inferred_model = shape_inference.infer_shapes(onnx_model)
        
        print('ONNX model is valid!')
        return inferred_model

# Usage
model = load_trained_model()
exporter = ONNXExporter(model, sample_input_shape=(1, 3, 32, 32))

# Export
onnx_path = 'models/ternary_model.onnx'
exporter.export(onnx_path, opset_version=13)

# Verify
exporter.verify_export(onnx_path)
```

### ONNX Optimization

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxoptimizer

class ONNXOptimizer:
    """Optimize ONNX models for inference."""
    
    @staticmethod
    def optimize_graph(input_path, output_path):
        """Optimize ONNX graph with standard passes."""
        # Load model
        model = onnx.load(input_path)
        
        # Available optimization passes
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'eliminate_nop_pad',
            'eliminate_unused_initializer',
            'extract_constant_to_initializer',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_into_conv',
            'fuse_transpose_into_gemm',
        ]
        
        # Optimize
        optimized_model = onnxoptimizer.optimize(model, passes)
        
        # Save
        onnx.save(optimized_model, output_path)
        print(f'Optimized model saved to {output_path}')
        
        return optimized_model
    
    @staticmethod
    def quantize_model(input_path, output_path):
        """Apply dynamic quantization to ONNX model."""
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8
        )
        print(f'Quantized model saved to {output_path}')
    
    @staticmethod
    def simplify_model(input_path, output_path):
        """Simplify ONNX model using onnx-simplifier."""
        import onnxsim
        
        model = onnx.load(input_path)
        model_simp, check = onnxsim.simplify(model)
        
        if check:
            onnx.save(model_simp, output_path)
            print(f'Simplified model saved to {output_path}')
        else:
            print('Simplification failed')
        
        return model_simp

# Usage
optimizer = ONNXOptimizer()

# Optimize graph
optimizer.optimize_graph(
    'models/ternary_model.onnx',
    'models/ternary_model_optimized.onnx'
)

# Simplify
optimizer.simplify_model(
    'models/ternary_model_optimized.onnx',
    'models/ternary_model_simplified.onnx'
)
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

class ONNXInference:
    """Inference with ONNX Runtime."""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize ONNX Runtime inference.
        
        Args:
            model_path: Path to ONNX model
            device: 'cuda' or 'cpu'
        """
        # Set providers
        if device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]
        else:
            providers = ['CPUExecutionProvider']
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = \
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f'ONNX model loaded on {device}')
    
    def predict(self, input_data):
        """
        Run inference.
        
        Args:
            input_data: Input tensor (numpy array)
            
        Returns:
            Output predictions
        """
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.cpu().numpy()
        
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        
        return outputs[0]
    
    def benchmark(self, input_shape, num_iterations=1000):
        """Benchmark inference speed."""
        import time
        
        # Create random input
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.predict(input_data)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            self.predict(input_data)
        end = time.time()
        
        avg_time = (end - start) / num_iterations * 1000  # ms
        throughput = num_iterations / (end - start)
        
        print(f'Average inference time: {avg_time:.2f} ms')
        print(f'Throughput: {throughput:.2f} samples/sec')
        
        return avg_time, throughput

# Usage
inference = ONNXInference('models/ternary_model_simplified.onnx', device='cuda')

# Single prediction
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
output = inference.predict(input_data)
print(f'Output shape: {output.shape}')

# Benchmark
inference.benchmark(input_shape=(1, 3, 32, 32), num_iterations=1000)
```

## TensorRT Deployment

### Convert ONNX to TensorRT

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTConverter:
    """Convert ONNX models to TensorRT engines."""
    
    def __init__(self):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        
    def build_engine(self, onnx_path, engine_path, 
                     max_batch_size=1, fp16_mode=False,
                     int8_mode=False, max_workspace_size=1<<30):
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            engine_path: Path to save TensorRT engine
            max_batch_size: Maximum batch size
            fp16_mode: Enable FP16 precision
            int8_mode: Enable INT8 precision
            max_workspace_size: Maximum workspace size in bytes
        """
        # Create network
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = self.builder.create_network(network_flags)
        
        # Parse ONNX
        parser = trt.OnnxParser(network, self.logger)
        
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse ONNX file')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Create builder config
        config = self.builder.create_builder_config()
        config.max_workspace_size = max_workspace_size
        
        # Set precision
        if fp16_mode and self.builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print('FP16 mode enabled')
        
        if int8_mode and self.builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print('INT8 mode enabled')
        
        # Build engine
        print('Building TensorRT engine... This may take a while')
        engine = self.builder.build_engine(network, config)
        
        if engine is None:
            print('Failed to build engine')
            return None
        
        # Serialize and save
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f'TensorRT engine saved to {engine_path}')
        return engine

# Usage
converter = TensorRTConverter()

engine = converter.build_engine(
    onnx_path='models/ternary_model.onnx',
    engine_path='models/ternary_model.trt',
    max_batch_size=8,
    fp16_mode=True,
    max_workspace_size=2<<30  # 2GB
)
```

### TensorRT Inference

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class TensorRTInference:
    """Inference with TensorRT engine."""
    
    def __init__(self, engine_path):
        """Initialize TensorRT inference."""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append to lists
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def predict(self, input_data):
        """
        Run inference.
        
        Args:
            input_data: Input numpy array
            
        Returns:
            Output predictions
        """
        # Copy input to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Transfer predictions back
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        
        # Synchronize stream
        self.stream.synchronize()
        
        return self.outputs[0]['host']
    
    def benchmark(self, input_shape, num_iterations=1000):
        """Benchmark inference speed."""
        import time
        
        # Create random input
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.predict(input_data)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            self.predict(input_data)
        end = time.time()
        
        avg_time = (end - start) / num_iterations * 1000  # ms
        throughput = num_iterations / (end - start)
        
        print(f'TensorRT Average time: {avg_time:.2f} ms')
        print(f'TensorRT Throughput: {throughput:.2f} samples/sec')
        
        return avg_time, throughput

# Usage
trt_inference = TensorRTInference('models/ternary_model.trt')

# Predict
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
output = trt_inference.predict(input_data)

# Benchmark
trt_inference.benchmark(input_shape=(1, 3, 32, 32))
```

## Mobile Deployment (TensorFlow Lite)

### Convert to TFLite

```python
import tensorflow as tf
import torch
import numpy as np

class TFLiteConverter:
    """Convert PyTorch ternary models to TensorFlow Lite."""
    
    def __init__(self, model, sample_input_shape):
        self.model = model
        self.sample_input_shape = sample_input_shape
        
    def convert_via_onnx(self, onnx_path, tflite_path):
        """Convert PyTorch -> ONNX -> TFLite."""
        # First export to ONNX
        from onnx_tf.backend import prepare
        import onnx
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph('models/temp_tf_model')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model('models/temp_tf_model')
        
        # Optimization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f'TFLite model saved to {tflite_path}')
        return tflite_model
    
    @staticmethod
    def quantize_tflite(input_path, output_path, 
                        representative_dataset_gen=None):
        """Apply post-training quantization to TFLite model."""
        converter = tf.lite.TFLiteConverter.from_saved_model(input_path)
        
        # Quantization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if representative_dataset_gen:
            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # Convert
        quantized_model = converter.convert()
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(quantized_model)
        
        print(f'Quantized TFLite model saved to {output_path}')
        return quantized_model

# Representative dataset for quantization
def representative_dataset_gen():
    """Generate representative dataset for quantization."""
    for _ in range(100):
        data = np.random.randn(1, 32, 32, 3).astype(np.float32)
        yield [data]

# Usage
converter = TFLiteConverter(model, sample_input_shape=(1, 3, 32, 32))

# Convert to TFLite
converter.convert_via_onnx(
    'models/ternary_model.onnx',
    'models/ternary_model.tflite'
)

# Quantize
TFLiteConverter.quantize_tflite(
    'models/temp_tf_model',
    'models/ternary_model_quantized.tflite',
    representative_dataset_gen=representative_dataset_gen
)
```

### TFLite Inference

```python
import tensorflow as tf
import numpy as np

class TFLiteInference:
    """Inference with TensorFlow Lite."""
    
    def __init__(self, model_path, num_threads=4):
        """Initialize TFLite interpreter."""
        # Load model
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=num_threads
        )
        
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f'Input shape: {self.input_details[0]["shape"]}')
        print(f'Output shape: {self.output_details[0]["shape"]}')
    
    def predict(self, input_data):
        """Run inference."""
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            input_data
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output_data = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )
        
        return output_data
    
    def benchmark(self, input_shape, num_iterations=1000):
        """Benchmark inference speed."""
        import time
        
        # Create random input
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.predict(input_data)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            self.predict(input_data)
        end = time.time()
        
        avg_time = (end - start) / num_iterations * 1000  # ms
        throughput = num_iterations / (end - start)
        
        print(f'TFLite Average time: {avg_time:.2f} ms')
        print(f'TFLite Throughput: {throughput:.2f} samples/sec')
        
        return avg_time, throughput

# Usage
tflite_inference = TFLiteInference('models/ternary_model.tflite')

# Predict
input_data = np.random.randn(1, 32, 32, 3).astype(np.float32)
output = tflite_inference.predict(input_data)
print(f'Output: {output}')

# Benchmark
tflite_inference.benchmark(input_shape=(1, 32, 32, 3))
```

## Edge Device Optimization

### Raspberry Pi Deployment

```python
import numpy as np
import time
from PIL import Image

class RaspberryPiDeployment:
    """Optimized deployment for Raspberry Pi."""
    
    def __init__(self, model_path, use_tflite=True):
        """Initialize for Raspberry Pi."""
        self.use_tflite = use_tflite
        
        if use_tflite:
            # Use TFLite for better performance on ARM
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=4  # Use all 4 cores
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            # Use ONNX Runtime with optimizations
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 4
            sess_options.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )
    
    def preprocess_image(self, image_path, target_size=(32, 32)):
        """Preprocess image for inference."""
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize
        img = img.resize(target_size)
        
        # Convert to numpy
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Add batch dimension
        if self.use_tflite:
            img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)
        else:
            img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)
            img_array = np.expand_dims(img_array, axis=0)   # (1, C, H, W)
        
        return img_array
    
    def predict(self, input_data):
        """Run inference."""
        if self.use_tflite:
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                input_data
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
        else:
            output = self.session.run(
                None,
                {self.session.get_inputs()[0].name: input_data}
            )[0]
        
        return output
    
    def measure_power_efficiency(self, num_inferences=100):
        """Measure power efficiency (requires hardware monitoring)."""
        # This is a placeholder - actual implementation requires
        # hardware power monitoring tools
        import time
        
        dummy_input = np.random.randn(1, 32, 32, 3).astype(np.float32)
        
        start = time.time()
        for _ in range(num_inferences):
            self.predict(dummy_input)
        elapsed = time.time() - start
        
        avg_time = elapsed / num_inferences
        inferences_per_second = num_inferences / elapsed
        
        print(f'Average inference time: {avg_time*1000:.2f} ms')
        print(f'Inferences per second: {inferences_per_second:.2f}')
        
        # Estimated power (placeholder)
        estimated_power_per_inference = 0.5  # Watts (example)
        print(f'Estimated power per inference: {estimated_power_per_inference:.2f} W')

# Usage on Raspberry Pi
deployment = RaspberryPiDeployment(
    'models/ternary_model.tflite',
    use_tflite=True
)

# Process image
img_array = deployment.preprocess_image('test_image.jpg')

# Predict
output = deployment.predict(img_array)
predicted_class = np.argmax(output)
print(f'Predicted class: {predicted_class}')

# Measure efficiency
deployment.measure_power_efficiency()
```

### NVIDIA Jetson Deployment

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class JetsonDeployment:
    """Optimized deployment for NVIDIA Jetson devices."""
    
    def __init__(self, engine_path):
        """Initialize for Jetson."""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load TensorRT engine
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.allocate_buffers()
        
        # CUDA stream
        self.stream = cuda.Stream()
    
    def allocate_buffers(self):
        """Allocate CUDA buffers."""
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def predict(self, input_data):
        """Run inference on Jetson."""
        # Copy input
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        
        # Transfer to GPU
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # Execute
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Transfer back
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        
        # Synchronize
        self.stream.synchronize()
        
        return self.outputs[0]['host']
    
    def benchmark_with_power(self, input_shape, duration_seconds=60):
        """Benchmark with power monitoring on Jetson."""
        import time
        import subprocess
        
        # Start power monitoring (Jetson specific)
        # tegrastats command monitors Jetson power
        
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        start = time.time()
        count = 0
        
        while time.time() - start < duration_seconds:
            self.predict(input_data)
            count += 1
        
        elapsed = time.time() - start
        fps = count / elapsed
        avg_latency = elapsed / count * 1000
        
        print(f'FPS: {fps:.2f}')
        print(f'Average latency: {avg_latency:.2f} ms')
        print(f'Total inferences: {count}')

# Usage on Jetson
jetson = JetsonDeployment('models/ternary_model.trt')

# Predict
input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
output = jetson.predict(input_data)

# Benchmark
jetson.benchmark_with_power(input_shape=(1, 3, 32, 32))
```

## Model Serving with TorchServe

### Prepare Model for TorchServe

```python
import torch
from pathlib import Path

class TorchServePackager:
    """Package model for TorchServe deployment."""
    
    def __init__(self, model, model_name, version='1.0'):
        self.model = model
        self.model_name = model_name
        self.version = version
        
    def save_model(self, output_dir='model_store'):
        """Save model in TorchServe format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        model_file = output_dir / f'{self.model_name}.pth'
        torch.save(self.model.state_dict(), model_file)
        
        print(f'Model saved to {model_file}')
        return model_file
    
    def create_handler(self, output_dir='model_store'):
        """Create custom handler for ternary model."""
        handler_code = '''
import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
import numpy as np
from PIL import Image
import io

class TernaryModelHandler(BaseHandler):
    """Custom handler for ternary models."""
    
    def __init__(self):
        super(TernaryModelHandler, self).__init__()
        self.initialized = False
    
    def initialize(self, context):
        """Initialize model."""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load model
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = f"{model_dir}/{serialized_file}"
        
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )
        
        # Load your ternary model architecture
        self.model = load_model_architecture()
        self.model.load_state_dict(torch.load(model_pt_path))
        self.model.to(self.device)
        self.model.eval()
        
        self.initialized = True
    
    def preprocess(self, requests):
        """Preprocess input data."""
        images = []
        
        for req in requests:
            # Get image from request
            image = req.get("data") or req.get("body")
            
            if isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image))
            
            # Preprocess
            image = image.convert('RGB')
            image = image.resize((32, 32))
            image = np.array(image).astype(np.float32) / 255.0
            
            # Normalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            # To tensor
            image = torch.from_numpy(image).permute(2, 0, 1)
            images.append(image)
        
        return torch.stack(images).to(self.device)
    
    def inference(self, data):
        """Run inference."""
        with torch.no_grad():
            output = self.model(data)
        return output
    
    def postprocess(self, inference_output):
        """Postprocess output."""
        # Get probabilities
        probs = F.softmax(inference_output, dim=1)
        
        # Get predictions
        predictions = probs.argmax(dim=1)
        
        results = []
        for i in range(len(predictions)):
            results.append({
                'prediction': predictions[i].item(),
                'probabilities': probs[i].cpu().numpy().tolist()
            })
        
        return results

def load_model_architecture():
    """Load model architecture."""
    # Import your model architecture here
    from model import TernaryModel
    return TernaryModel()
'''
        
        handler_file = Path(output_dir) / 'handler.py'
        with open(handler_file, 'w') as f:
            f.write(handler_code)
        
        print(f'Handler saved to {handler_file}')
        return handler_file
    
    def create_mar_file(self, model_file, handler_file, output_dir='model_store'):
        """Create .mar file for TorchServe."""
        import subprocess
        
        cmd = [
            'torch-model-archiver',
            '--model-name', self.model_name,
            '--version', self.version,
            '--serialized-file', str(model_file),
            '--handler', str(handler_file),
            '--export-path', output_dir
        ]
        
        subprocess.run(cmd, check=True)
        print(f'MAR file created: {output_dir}/{self.model_name}.mar')

# Usage
packager = TorchServePackager(model, model_name='ternary_classifier')

# Save model
model_file = packager.save_model()

# Create handler
handler_file = packager.create_handler()

# Create MAR file
packager.create_mar_file(model_file, handler_file)
```

### Deploy with TorchServe

```bash
# Install TorchServe
pip install torchserve torch-model-archiver torch-workflow-archiver

# Start TorchServe
torchserve --start \
    --model-store model_store \
    --models ternary_classifier=ternary_classifier.mar \
    --ncs

# Check status
curl http://localhost:8080/ping

# Register model
curl -X POST "http://localhost:8081/models?url=ternary_classifier.mar&initial_workers=2"

# Inference
curl -X POST http://localhost:8080/predictions/ternary_classifier \
    -T test_image.jpg
```

### Production Serving Setup

```python
import requests
import json
from pathlib import Path

class TorchServeClient:
    """Client for TorchServe inference."""
    
    def __init__(self, base_url='http://localhost:8080'):
        self.base_url = base_url
        self.predictions_url = f'{base_url}/predictions'
        
    def predict(self, model_name, image_path):
        """Send prediction request."""
        url = f'{self.predictions_url}/{model_name}'
        
        with open(image_path, 'rb') as f:
            response = requests.post(url, files={'data': f})
        
        return response.json()
    
    def batch_predict(self, model_name, image_paths, batch_size=8):
        """Batch prediction."""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            
            for img_path in batch:
                result = self.predict(model_name, img_path)
                results.append(result)
        
        return results
    
    def health_check(self):
        """Check server health."""
        response = requests.get(f'{self.base_url}/ping')
        return response.status_code == 200
    
    def get_model_status(self, model_name):
        """Get model status."""
        response = requests.get(
            f'{self.base_url.replace("8080", "8081")}/models/{model_name}'
        )
        return response.json()

# Usage
client = TorchServeClient()

# Health check
if client.health_check():
    print('Server is healthy')

# Predict
result = client.predict('ternary_classifier', 'test_image.jpg')
print(f'Prediction: {result}')

# Batch predict
image_paths = [f'images/img_{i}.jpg' for i in range(100)]
results = client.batch_predict('ternary_classifier', image_paths)
```

## Performance Comparison

```python
import time
import numpy as np
import matplotlib.pyplot as plt

class DeploymentBenchmark:
    """Benchmark different deployment methods."""
    
    def __init__(self, models_dict):
        """
        Args:
            models_dict: Dict mapping names to model objects
        """
        self.models = models_dict
        
    def benchmark_all(self, input_shape, num_iterations=1000):
        """Benchmark all deployment methods."""
        results = {}
        
        for name, model in self.models.items():
            print(f'Benchmarking {name}...')
            latency, throughput = self.benchmark_single(
                model,
                input_shape,
                num_iterations
            )
            results[name] = {
                'latency': latency,
                'throughput': throughput
            }
        
        return results
    
    def benchmark_single(self, model, input_shape, num_iterations):
        """Benchmark single model."""
        # Create input
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            model.predict(input_data)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            model.predict(input_data)
        elapsed = time.time() - start
        
        latency = elapsed / num_iterations * 1000  # ms
        throughput = num_iterations / elapsed
        
        return latency, throughput
    
    def plot_results(self, results):
        """Plot benchmark results."""
        names = list(results.keys())
        latencies = [results[n]['latency'] for n in names]
        throughputs = [results[n]['throughput'] for n in names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Latency plot
        ax1.bar(names, latencies)
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Inference Latency')
        ax1.tick_params(axis='x', rotation=45)
        
        # Throughput plot
        ax2.bar(names, throughputs)
        ax2.set_ylabel('Throughput (samples/sec)')
        ax2.set_title('Inference Throughput')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('deployment_benchmark.png')
        print('Benchmark plot saved to deployment_benchmark.png')

# Usage
models = {
    'PyTorch': pytorch_model,
    'ONNX': onnx_inference,
    'TensorRT': tensorrt_inference,
    'TFLite': tflite_inference
}

benchmark = DeploymentBenchmark(models)
results = benchmark.benchmark_all(input_shape=(1, 3, 32, 32))
benchmark.plot_results(results)
```

## Production Monitoring

```python
import time
from collections import deque
import threading

class InferenceMonitor:
    """Monitor inference performance in production."""
    
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.errors = 0
        self.total_requests = 0
        self.lock = threading.Lock()
        
    def record_inference(self, latency_ms, error=False):
        """Record inference metrics."""
        with self.lock:
            self.latencies.append(latency_ms)
            self.total_requests += 1
            if error:
                self.errors += 1
    
    def get_stats(self):
        """Get current statistics."""
        with self.lock:
            if not self.latencies:
                return {}
            
            latencies_list = list(self.latencies)
            
            return {
                'mean_latency': np.mean(latencies_list),
                'p50_latency': np.percentile(latencies_list, 50),
                'p95_latency': np.percentile(latencies_list, 95),
                'p99_latency': np.percentile(latencies_list, 99),
                'error_rate': self.errors / self.total_requests if self.total_requests > 0 else 0,
                'total_requests': self.total_requests
            }
    
    def print_stats(self):
        """Print statistics."""
        stats = self.get_stats()
        print('=' * 50)
        print('Inference Statistics')
        print('=' * 50)
        for key, value in stats.items():
            print(f'{key}: {value:.2f}')
        print('=' * 50)

# Usage
monitor = InferenceMonitor()

# In your inference loop
start = time.time()
try:
    output = model.predict(input_data)
    latency = (time.time() - start) * 1000
    monitor.record_inference(latency, error=False)
except Exception as e:
    latency = (time.time() - start) * 1000
    monitor.record_inference(latency, error=True)

# Print stats periodically
monitor.print_stats()
```

## Best Practices

1. **Choose the Right Format**:
   - ONNX: Cross-platform, good performance
   - TensorRT: Best for NVIDIA GPUs
   - TFLite: Best for mobile/edge devices

2. **Optimize Before Deploying**:
   - Graph optimization
   - Quantization
   - Layer fusion

3. **Benchmark Thoroughly**:
   - Test on target hardware
   - Measure latency and throughput
   - Monitor resource usage

4. **Monitor in Production**:
   - Track inference times
   - Log errors and failures
   - Set up alerts

5. **Version Management**:
   - Version your models
   - Keep deployment scripts
   - Document changes

## Exercises

1. **Multi-Format Export**: Export your model to ONNX, TensorRT, and TFLite
2. **Benchmark Suite**: Create comprehensive benchmarks comparing all formats
3. **Edge Deployment**: Deploy to Raspberry Pi and measure power consumption
4. **Production Server**: Set up TorchServe with monitoring
5. **Mobile App**: Integrate TFLite model into Android/iOS app

## Next Steps

- [Tutorial 6: Advanced Features](06_advanced_features.md)
- [API Reference](../api/README.md)
- [Examples](../../examples/README.md)

## Additional Resources

- [ONNX Documentation](https://onnx.ai/)
- [TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [TorchServe Documentation](https://pytorch.org/serve/)
