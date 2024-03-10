# TrEnc-GrDec

### Requiremnets 
```
conda create -n grenc_trdec python=3.9 -y
source activate grenc_trdec
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Creating tensors and graphs from the image
```
python utils/image2graph.py
```

### Training
```
CUDA_VISIBLE_DEVICES=4,5 python run.py
```