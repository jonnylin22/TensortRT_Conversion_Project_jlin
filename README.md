## README ##

# Summary #

1. Loaded Hugging Face image classification model
	MobileNetV2 (model loaded in PyTorch format) 
	
2. ONNX Conversion 
	PyTorch Model --> .onnx model 

3. Engine Conversion
	ONNX Model --> .engine file
	
4. Benchmarked Model with TensorRT


Successfully initalize the virtual environment and download the required packages:
```bash
python3 -m venv .venv
source .venv/bin/activate 	# Linux based machine command
pip install -r requirements.txt
```

CLI:  
After running script.py, you generate an ONNX file. Then run the followng commands

### Optionally simplify the ONNX model (recommended before TensorRT)  
```python
python3 -m onnxsim mobilenet_v2.onnx mobilenet_v2_sim.onnx
```

### Convert to TensorRT    
```python
trtexec --onnx=mobilenet_v2_sim.onnx --saveEngine=mobilenet_v2.engine
```
