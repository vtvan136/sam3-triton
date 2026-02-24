```bash
cd source

git clone https://github.com/wkentaro/sam3-onnx.git --recursive && cd sam3-onnx

python3 -m venv venv

pip install uv 

uvx hf download wkentaro/sam3-onnx-models --local-dir models

cd ../..

cp -r source/sam3-onnx/models/sam3_decoder.onnx repo/sam3_decoder/1/model.onnx

cp -r source/sam3-onnx/models/sam3_decoder.onnx.data  repo/sam3_decoder/1/model.onnx.data

cp -r source/sam3-onnx/models/sam3_image_encoder.onnx repo/sam3_image_encoder/1

cp -r source/sam3-onnx/models/sam3_image_encoder.onnx.data repo/sam3_image_encoder/1

cp -r source/sam3-onnx/models/sam3_language_encoder.onnx repo/sam3_language_encoder/1

cp -r source/sam3-onnx/models/sam3_language_encoder.onnx.data repo/sam3_language_encoder/1
```