# Cross Attentive Pooling for Speaker Verification
Pytorch code for following paper:
* **Title** : Cross Attentive Pooling for Speaker Verification. [[paper](https://arxiv.org/pdf/2008.05983.pdf)]
* **Author** : Seong Min Kye, , Yoohwan Kwon, [Joon Son Chung](https://joonson.com/)
* **Conference** : IEEE Spoken Language Technology Workshop (SLT), 2021.

#### Abstract
<img align="middle" width="1000" src="https://github.com/seongmin-kye/CAP/blob/main/overview.png">

The goal of this paper is text-independent speaker verification where utterances come from 'in the wild' videos and may contain irrelevant signal. While speaker verification is naturally a pair-wise problem, existing methods to produce the speaker embeddings are instance-wise. In this paper, we propose Cross Attentive Pooling (CAP) that utilises the context information across the reference-query pair to generate utterance-level embeddings that contain the most discriminative information for the pair-wise matching problem. Experiments are performed on the VoxCeleb dataset in which our method outperforms comparable pooling strategies.

#### Dependencies
```
pip install -r requirements.txt
```

#### Data preparation

The following script can be used to download and prepare the VoxCeleb dataset for training.

```
python ./dataprep.py --save_path ./data --download --user USERNAME --password PASSWORD 
python ./dataprep.py --save_path ./data --extract
python ./dataprep.py --save_path ./data --convert
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

### Training examples (NP+Softmax)
- TAP (Temporal average pooling):
```
CUDA_VISIBLE_DEVICES=0 python trainSpeakerNet.py --model ResNetSE34L --encoder_type TAP --trainfunc proto --global_clf --nSpeaker 3 --save_path ./data/test --batch_size 200 --max_frames 200 --eval_frames 350 --optimizer sgd --lr 0.1 --train_list ./data/train_list.txt --train_path ./data/voxceleb/voxceleb2 --test_list ./data/veri_test.txt --test_path ./data/voxceleb/voxceleb1 --test_interval 5 
```
- SAP (Self-attentive pooling):
```
CUDA_VISIBLE_DEVICES=0 python trainSpeakerNet.py --model ResNetSE34L --encoder_type SAP --trainfunc proto --global_clf --nSpeaker 3 --save_path ./data/test --batch_size 200 --max_frames 200 --eval_frames 350 --optimizer sgd --lr 0.1 --train_list ./data/train_list.txt --train_path ./data/voxceleb/voxceleb2 --test_list ./data/veri_test.txt --test_path ./data/voxceleb/voxceleb1 --test_interval 5 
```
- CAP (Cross attentive pooling):
```
CUDA_VISIBLE_DEVICES=0 python trainSpeakerNet.py --model ResNetSE34L --encoder_type CAP --trainfunc proto --global_clf --nSpeaker 3 --save_path ./data/test --batch_size 200 --max_frames 200 --eval_frames 350 --optimizer sgd --lr 0.1 --train_list ./data/train_list.txt --train_path ./data/voxceleb/voxceleb2 --test_list ./data/veri_test.txt --test_path ./data/voxceleb/voxceleb1 --test_interval 5 
```

#### Implemented models and encoders(aggregations)
```
ResNetSE34 (TAP, SAP, CAP)
ResNetSE34L (TAP, SAP, CAP)
```

#### Data

The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets are used for these experiments.

The train list should contain the identity and the file path, one line per utterance, as follows:
```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
```

The train list for VoxCeleb2 can be download from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt) and the
test list for VoxCeleb1 from [here](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt).

#### Replicating the results from the paper

1. Model definitions
  - `Thin ResNet-34` is in the paper `ResNetSE34` in the code.
  - `Fast ResNet-34` is in the paper `ResNetSE34L` in the code.

2. For metric learning objectives, the batch size in the paper is `nSpeakers` multiplied by `batch_size` in the code. For the batch size of 600 in the paper, use `--nSpeakers 3 --batch_size 200`, `--nSpeakers 2 --batch_size 300`, etc.

3. The models have been trained with `--max_frames 200` and evaluated with `--max_frames 350`.

4. You can get a good balance between speed and performance using the configuration below.
```
CUDA_VISIBLE_DEVICES=0 python trainSpeakerNet.py --model ResNetSE34L --encoder_type CAP --trainfunc proto --global_clf --nSpeaker 3 --save_path ./data/test --batch_size 200 --max_frames 200 --eval_frames 350 --optimizer sgd --lr 0.1 --train_list ./data/train_list.txt --train_path ./data/voxceleb/voxceleb2 --test_list ./data/veri_test.txt --test_path ./data/voxceleb/voxceleb1 --test_interval 5 
```

#### Citation

Please cite the following if you make use of the code.

```
@inproceedings{kye2020cross,
  title={Cross attentive pooling for speaker verification},
  author={Kye, Seong Min and Kwon, Yoohwan and Chung, Joon Son},
  booktitle={2021 IEEE Spoken Language Technology Workshop (SLT)},
  year={2021},
  organization={IEEE}
}
```

#### License
```
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
