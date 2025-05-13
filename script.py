from transformers import AutoImageProcessor, MobileNetV2ForImageClassification
import torch
from datasets import load_dataset

# Load image from dataset
dataset = load_dataset("huggingface/cats-image", trust_remote_code=True)
image = dataset["test"]["image"][0]

# Preprocess image
image_processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
inputs = image_processor(image, return_tensors="pt")

# Forward pass (optional: prediction)
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    print("Predicted label:", model.config.id2label[predicted_label])

# ----------------------------------
# Prepare dummy input for ONNX export
# ----------------------------------
dummy_input = {"pixel_values": torch.randn(1, 3, 224, 224)}

# Export to ONNX
torch.onnx.export(
    model,                                # Model
    (dummy_input["pixel_values"],),       # Tuple input
    "mobilenet_v2.onnx",                  # Output ONNX file
    input_names=["pixel_values"],         # Input name
    output_names=["logits"],              # Output name
    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},  # Batch dynamic
    opset_version=11,                     # ONNX opset
    do_constant_folding=True              # Optional: optimization
)

print("Model successfully exported to ONNX format.")

