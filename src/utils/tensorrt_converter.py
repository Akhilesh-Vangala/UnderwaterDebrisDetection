import torch
from pathlib import Path
import subprocess
import sys
from typing import Tuple, Optional


class TensorRTConverter:
    def __init__(self, onnx_path: str, input_size: Tuple[int, int, int] = (3, 224, 224)):
        self.onnx_path = Path(onnx_path)
        self.input_size = input_size
        self.input_shape = f"1x{input_size[0]}x{input_size[1]}x{input_size[2]}"
    
    def convert(self, output_path: str, precision: str = 'fp16',
                max_batch_size: int = 1, max_workspace_size: int = 1 << 30) -> Optional[Path]:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import tensorrt as trt
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            with open(self.onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size
            
            if precision == 'fp16' and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == 'int8' and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
            
            profile = builder.create_optimization_profile()
            profile.set_shape('input', (1, *self.input_size), (max_batch_size, *self.input_size), (max_batch_size, *self.input_size))
            config.add_optimization_profile(profile)
            
            engine = builder.build_engine(network, config)
            
            if engine is None:
                print("✗ Failed to build TensorRT engine")
                return None
            
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            print(f"✓ TensorRT engine saved to: {output_path}")
            return output_path
        
        except ImportError:
            print("TensorRT not available. Using trtexec fallback...")
            return self._convert_with_trtexec(output_path, precision)
    
    def _convert_with_trtexec(self, output_path: str, precision: str) -> Optional[Path]:
        output_path = Path(output_path)
        cmd = [
            'trtexec',
            f'--onnx={self.onnx_path}',
            f'--saveEngine={output_path}',
            f'--shapes=input:{self.input_shape}',
            '--workspace=1024'
        ]
        
        if precision == 'fp16':
            cmd.append('--fp16')
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✓ TensorRT engine saved to: {output_path}")
            return output_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ trtexec not found. Please install TensorRT or use ONNX runtime.")
            return None
