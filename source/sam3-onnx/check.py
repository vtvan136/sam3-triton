import onnxruntime as ort
import os

models = [
    "/home/vanvt/triton-sam3/source/sam3-onnx/models/sam3_image_encoder.onnx",
    "/home/vanvt/triton-sam3/source/sam3-onnx/models/sam3_language_encoder.onnx",
    "/home/vanvt/triton-sam3/source/sam3-onnx/models/sam3_decoder.onnx",
]

for model_path in models:
    print("\n" + "="*60)
    print(f"MODEL: {model_path}")
    print("="*60)

    if not os.path.exists(model_path):
        print("❌ File not found")
        continue

    session = ort.InferenceSession(model_path)

    print("\n--- INPUTS ---")
    for inp in session.get_inputs():
        print(f"name  : {inp.name}")
        print(f"shape : {inp.shape}")
        print(f"type  : {inp.type}")
        print()

    print("--- OUTPUTS ---")
    for out in session.get_outputs():
        print(f"name  : {out.name}")
        print(f"shape : {out.shape}")
        print(f"type  : {out.type}")
        print()