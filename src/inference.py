import torch
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import onnxruntime as ort
from torch.utils.data import DataLoader


class InferenceEngine:
    def __init__(self, model_path: str, device: str = 'cuda', model_type: str = 'pytorch'):
        self.device = device
        self.model_type = model_type
        
        if model_type == 'pytorch':
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                from models import build_cswin_base
                self.model = build_cswin_base(num_classes=8, img_size=224)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                self.model = checkpoint['model']
            else:
                from models import build_cswin_base
                self.model = build_cswin_base(num_classes=8, img_size=224)
                if isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            self.model.to(device)
            self.model.eval()
        elif model_type == 'onnx':
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        elif model_type == 'tensorrt':
            import tensorrt as trt
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(model_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def predict(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        if self.model_type == 'pytorch':
            return self._predict_pytorch(image)
        elif self.model_type == 'onnx':
            return self._predict_onnx(image)
        elif self.model_type == 'tensorrt':
            return self._predict_tensorrt(image)
    
    def _predict_pytorch(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.softmax(outputs, dim=1)
            _, pred = torch.max(outputs, 1)
        
        return pred.item(), probs.cpu().numpy()[0]
    
    def _predict_onnx(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        outputs = self.session.run(None, {self.input_name: image.astype(np.float32)})
        probs = torch.softmax(torch.from_numpy(outputs[0]), dim=1).numpy()[0]
        pred = np.argmax(probs)
        
        return int(pred), probs
    
    def _predict_tensorrt(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        
        input_shape = self.engine.get_binding_shape(0)
        output_shape = self.engine.get_binding_shape(1)
        
        d_input = cuda.mem_alloc(image.nbytes)
        d_output = cuda.mem_alloc(np.prod(output_shape) * np.dtype(np.float32).itemsize)
        
        stream = cuda.Stream()
        cuda.memcpy_htod_async(d_input, image, stream)
        self.context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(np.empty(output_shape, dtype=np.float32), d_output, stream)
        stream.synchronize()
        
        outputs = np.empty(output_shape, dtype=np.float32)
        probs = torch.softmax(torch.from_numpy(outputs), dim=1).numpy()[0]
        pred = np.argmax(probs)
        
        return int(pred), probs


class FPSBenchmark:
    def __init__(self, inference_engine: InferenceEngine, img_size: Tuple[int, int] = (224, 224)):
        self.engine = inference_engine
        self.img_size = img_size
    
    def benchmark(self, num_iterations: int = 1000, warmup: int = 100) -> Dict:
        dummy_image = np.random.randn(*self.img_size, 3).astype(np.float32)
        
        for _ in range(warmup):
            _ = self.engine.predict(dummy_image)
        
        if self.engine.device == 'cuda':
            torch.cuda.synchronize()
        
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = self.engine.predict(dummy_image)
            if self.engine.device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
        
        fps = 1.0 / np.mean(times)
        std_fps = np.std([1.0 / t for t in times])
        min_fps = 1.0 / np.max(times)
        max_fps = 1.0 / np.min(times)
        
        return {
            'fps': fps,
            'std_fps': std_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
            'mean_latency_ms': np.mean(times) * 1000,
            'std_latency_ms': np.std(times) * 1000
        }
    
    def print_results(self, results: Dict):
        print(f"\n{'='*70}")
        print("FPS BENCHMARK RESULTS")
        print(f"{'='*70}")
        print(f"Mean FPS: {results['fps']:.2f} ± {results['std_fps']:.2f}")
        print(f"Min FPS: {results['min_fps']:.2f}")
        print(f"Max FPS: {results['max_fps']:.2f}")
        print(f"Mean Latency: {results['mean_latency_ms']:.2f} ms ± {results['std_latency_ms']:.2f} ms")
        print(f"{'='*70}")
