# TrEnc-GrDec

### Requiremnets 
```
conda create -n grenc_trdec python=3.9 -y
source activate grenc_trdec
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -y
pip install -r requirements.txt
```

### Creating tensors and graphs from the image
```
python utils/image2graph.py
```

### Training
```
python run.py
```