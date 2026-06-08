import torch
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Tuple
import numpy as np


class ONNXConverter:
    def __init__(self, model: torch.nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)):
        self.model = model
        self.input_size = input_size
        self.model.eval()
    
    def convert(self, output_path: str, opset_version: int = 11):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dummy_input = torch.randn(1, *self.input_size)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        
        print(f"✓ ONNX model saved to: {output_path}")
        return output_path
    
    def validate(self, onnx_path: str, pytorch_input: torch.Tensor) -> bool:
        ort_session = ort.InferenceSession(str(onnx_path))
        
        ort_inputs = {ort_session.get_inputs()[0].name: pytorch_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        with torch.no_grad():
            pytorch_output = self.model(pytorch_input)
        
        np.testing.assert_allclose(
            ort_outputs[0],
            pytorch_output.numpy(),
            rtol=1e-03,
            atol=1e-05
        )
        
        print("✓ ONNX model validation passed")
        return True
