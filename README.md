<p align="center">
    <a href="https://visurg.ai/">
    <img src="https://github.com/user-attachments/assets/92fc59cb-5392-4a72-95b5-35104df129e0">
    </a>
</p>


[📚 Paper](https://arxiv.org/abs/2503.19740) - [🤖 Code](src) - [🤗 Model](https://huggingface.co/visurg/LemonFM) - [🌐 Website](https://LEMON.visurg.ai/)

Star ⭐ us if you like it!

<div align="center">
  <img src="https://github.com/user-attachments/assets/6250cd6a-1404-4786-9c15-fe396265940d" width="70%" > </img>
</div>


## News

<!-- XX/March/2025. The [HuggingFace models and demo](TODO) are released. -->
<!--<br>-->
* 06/August/2025. Our project is now known as **LEMON**, formerly Surg-3M.
* 25/March/2025. The [arXiv](https://arxiv.org/abs/2503.19740) version of the paper is released.

<br>

This is the official repository for the paper [LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](https://arxiv.org/abs/2503.19740).

This repository provides open access to the *LEMON* dataset, *LemonFM* foundation model, and training code. 

[*LEMON*](https://surg-3m.visurg.ai/) is a dataset of 4K surgical high-resolution videos totaling 938 hours from 35 diverse surgical procedure types. Each video is annotated for multi-label classification, indicating the surgical procedures carried out in the video, and for binary classification, indicating if it is robotic or non-robotic. The dataset's annotations can be found in [labels.json](https://github.com/visurg-ai/LEMON/blob/main/labels.json).

[*LemonFM*](https://huggingface.co/visurg/LemonFM) is an image foundation model for surgery, it receives an image as input and produces a feature vector of 1536 features as output. 

<!--The website of our dataset is: [http://LEMON.org](https://LEMON.org)-->

If you use our dataset, model, or code in your research, please cite our paper:

```
@misc{che2025lemonlargeendoscopicmonocular,
      title={LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings}, 
      author={Chengan Che and Chao Wang and Tom Vercauteren and Sophia Tsoka and Luis C. Garcia-Peraza-Herrera},
      year={2025},
      eprint={2503.19740},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.19740}, 
}
```

Abstract
--------
Traditional open-access datasets focusing on surgical procedures are often limited by their small size, typically consisting of fewer than 100 videos and less than 30 hours of footage, which leads to poor model generalization. To address this constraint, a new dataset called LEMON has been compiled using a novel aggregation pipeline that collects high-resolution videos from online sources. Featuring an extensive collection of over 4K surgical videos totaling 938 hours (85 million frames) of high-quality footage across multiple procedure types, LEMON offers a comprehensive resource surpassing existing alternatives in size and scope, including two novel downstream tasks. To demonstrate the effectiveness of this diverse dataset, we introduce LemonFM, a foundation model pretrained on LEMON using a novel self-supervised augmented knowledge distillation approach. LemonFM consistently outperforms existing surgical foundation models across four downstream tasks and six datasets, achieving significant gains in surgical phase recognition (+9.5pp, +9.4pp, and +8.4pp of Jaccard in AutoLaparo, M2CAI16, and Cholec80), surgical action recognition (+4.4pp of mAP in CholecT50), surgical tool presence detection (+5.3pp and +10.2pp of mAP in Cholec80 and GraSP), and surgical semantic segmentation (+8.3pp of mDice in CholecSeg8k). LEMON and LemonFM will serve as foundational resources for the research community and industry, accelerating progress in developing autonomous robotic surgery systems and ultimately contributing to safer and more accessible surgical care worldwide.


<br>

Diversity and procedure prevalence in LEMON:

<img src="https://github.com/user-attachments/assets/67322046-5515-47e1-bb3f-621892c8608c">


Install dependencies to recreate our LEMON dataset
--------------------------------------------------
<!--
* If you want to use Docker**, follow the next steps to download our container:

   ```bash
   # Download the repo
   $ git clone git@github.com:visurg-ai/LEMON.git
   $ cd LEMON/docker

   # Build the docker image
   $ docker build --build-arg USER=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t chengan/LEMON:latest .

   # Run videosum Docker container
   $ docker run --volume $HOME:/mnt/user_home --name LEMON --runtime nvidia chengan/LEMON:latest &

   # Execute the docker container and get into the terminal
   $ docker exec --user $(whoami) --workdir $HOME -it LEMON /bin/zsh
   ```
-->

* Install the following dependencies in your local setup:

   ```bash
   $ git clone git@github.com:visurg-ai/LEMON.git
   $ cd LEMON && pip install -r requirements.txt
   ```

* **Models used in data curation.** We provide the models used in our data curation pipeline to assist with constructing the LEMON dataset, including video storyboard classification models, frame classification models, and non-surgical object detection models. The models can be downloaded from [🤗 LEMON curation models](https://huggingface.co/visurg/LEMON_curation_models).


LEMON dataset
--------------------------

> Researchers working in academic institutions can request direct access to the full LEMON dataset for non-commercial purposes by filling the request form in our [🌐 Website](https://LEMON.visurg.ai/) and [🤗 HuggingFace](https://huggingface.co/datasets/visurg/LEMON))

You can use our code of the data curation pipeline and provided annotation file (["labels.json"](https://github.com/visurg-ai/LEMON/blob/main/labels.json)) to recreate the whole LEMON dataset.

1. Get your YouTube cookie:

   You need to provide a "cookies.txt" file if you want to download videos that require Youtube login. 

   Use the [cookies](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) extension to export your Youtube cookies as "cookies.txt".


2. Download the annotation file (["labels.json"](https://github.com/visurg-ai/LEMON/blob/main/labels.json)) and use the video downloader to download the original selected Youtube videos.

   ```bash
   $ python3 src/video_downloader.py --video-path '../labels.json' --output 'your path to store the downloaded videos' --cookies 'your YouTube cookie file'
   ```

3. Curate the downloaded original videos as LEMON video dataset. In detail, use the video_processor to classify each frame as either 'surgical' or 'non-surgical', then remove the beginning and end segments of non-surgical content from the videos, and mask the non-surgical regions in 'surgical' frames and the entire 'non-surgical' frames.

   ```bash
   $ python3 src/video_processor.py --input 'your original downloaded video storage path' --input-json '../labels.json' --output 'your path to store the curated videos and their corresponding frame annotation files' --classify-models 'frame classification model' --segment-models 'non-surgical object detection models'
   ```


4. Process the LEMON video dataset as LEMON image dataset (For foundation model pre-training).

   ```bash
   $ python3 src/create_lmdb_LEMON.py --video-folder 'your directory containing the curated videos and their corresponding frame annotation files' --output-json 'your path for the json file to verify the videos and labels alignment' --lmdb-path 'your lmdb storage path'
   ```

<br>
The video processing pipeline leading to the clean videos in the LEMON dataset is as follows:

<img src="https://github.com/user-attachments/assets/cb21d841-ad49-4834-b77e-dbc24fe6699e">


LemonFM model
-------------
You can download the LemonFM full checkpoint which contains backbone and projection head weights for both student and teacher networks at [🤗 LemonFM](https://huggingface.co/visurg/LemonFM).

**LemonFM pretraining:**


```bash
$ python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=1 lemonfm/lemonfm.py --arch convnext_large --data_path 'LEMON dataset lmdb path' --output_dir 'your path to store the trained foundation model' --batch_size_per_gpu 40 --num_workers 10
```


**Fine-tuning LemonFM for surgical phase recognition:**


```bash
$ python3 downstream/train_phase_recognition_autolaparo.py --lr 1e-3 --opt adamW --nepochs 100 --bs 512 --cpdir 'path/to/store/checkpoint' --logdir 'path/to/store/log' --lmdb 'path/to/downstream_task/lmdb' --labels 'path/to/downstream_task/annotation' --seed 30 --pretrained-weights 'path/to/our/LemonFM.pth'
```

```bash
$ python3 downstream/test_phase_recognition_autolaparo.py --lmdb 'path/to/downstream_task/lmdb' --models 'path/to/your/cpdir' --labels 'path/to/downstream_task/annotation'
```



How to run our LemonFM foundation model to extract features from your video frames
----------------------------------------------------------------------------------

   ```python
   import torch
   from PIL import Image
   from model_loader import build_LemonFM

   # Load the pre-trained LemonFM model
   LemonFM = build_LemonFM(pretrained_weights = 'your path to the LemonFM')
   LemonFM.eval()

   # Load the image and convert it to a PyTorch tensor
   img_path = 'path/to/your/image.jpg'
   img = Image.open(img_path)
   img = img.resize((224, 224))
   img_tensor = torch.tensor(np.array(img)).unsqueeze(0).to('cuda')

   # Extract features from the image
   outputs = LemonFM(img_tensor)
   ```




Tasks
-------

Based on the annotation of the LEMON dataset, we propose two novel surgical downstream tasks:
1. Multi-label (35 classes) video classification of procedure types.
2. Binary video classification of surgery types.


**Leaderboard for the proposed tasks**  
To establish a baseline for the two tasks proposed in LEMON, we conducted an evaluation of SotA approaches, which serves as a benchmark for future research endeavors.


<table>
  <tr>
    <th rowspan="2" style="text-align: center;">Method</th>
    <th colspan="2" style="text-align: center;">Procedure type</th>
    <th colspan="2" style="text-align: center;">Surgery type</th>
  </tr>
  <tr>
    <th style="text-align: center;">mAP (%)</th>
    <th style="text-align: center;">F1-score (%)</th>
    <th style="text-align: center;">Accuracy (%)</th>
    <th style="text-align: center;">F1-score (%)</th>
  </tr>
  <tr>
    <td style="text-align: center;"><a href="https://arxiv.org/abs/1812.03982">SlowFast</a></td>
    <td style="text-align: center;">22.0</td>
    <td style="text-align: center;">23.9</td>
    <td style="text-align: center;">88.5</td>
    <td style="text-align: center;">87.5</td>
  </tr>
  <tr>
    <td style="text-align: center;"><a href="https://arxiv.org/abs/2102.05095">TimeSformer</a></td>
    <td style="text-align: center;">42.1</td>
    <td style="text-align: center;">37.5</td>
    <td style="text-align: center;">93.2</td>
    <td style="text-align: center;">92.7</td>
  </tr>
  <tr>
    <td style="text-align: center;"><a href="https://arxiv.org/abs/2112.01526">MViTv2</a></td>
    <td style="text-align: center;">49.5</td>
    <td style="text-align: center;">41.8</td>
    <td style="text-align: center;">95.8</td>
    <td style="text-align: center;">94.6</td>
  </tr>
  <tr>
    <td style="text-align: center;"><a href="https://arxiv.org/abs/2106.13230">Video Swin Transformer</a></td>
    <td style="text-align: center;">51.4</td>
    <td style="text-align: center;">47.9</td>
    <td style="text-align: center;">98.8</td>
    <td style="text-align: center;">98.7</td>
  </tr>
  <tr>
    <td style="text-align: center;"><a href="https://arxiv.org/abs/2503.19740">LemonFM-Vid (ours)</a></td>
    <td style="text-align: center;"><strong>57.8</strong></td>
    <td style="text-align: center;"><strong>49.3</strong></td>
    <td style="text-align: center;"><strong>98.9</strong></td>
    <td style="text-align: center;"><strong>98.9</strong></td>
  </tr>
</table>





<!--
**LemonFM performance:**

This figure shows the performance comparison between our foundation
model, LemonFM, and the state-of-the-art (SotA) models. Our
evaluation focuses on three surgical downstream tasks and six
datasets. LemonFM results are shown in bold, axis labels are presented in regular font.

<img src="https://github.com/user-attachments/assets/080ec843-fc11-4ec7-b669-0bc1de2bf16f">
-->



<!--
How to download more videos with specific procedure
---------------------------------------------------

```bash
$ cd src
$ python3 video_downloader.py --keyword 'robotic, cholecystectomy' --number 100 --cookies 'your own YouTube cookie file' --output 'your path to store the downloaded videos'
```
-->

<!--
How to classify videos as informative/uninformative after downloading more videos
---------------------------------------------------------------------------------

1. To begin with, ensure that you have installed the [videosum](https://github.com/luiscarlosgph/videosum) package correctly, including all its dependencies.

2. Run the video classifier to summarize videos into video storyboards, and then utilize our video storyboard classification models to classify each video as either 'surgical' or 'non-surgical'.

```bash
$ cd src
$ python3 video_classifier --input 'your directory containing the downloaded videos' --output 'your path to a json file which contains classification results' --models 'video storyboard classification models'
```
-->


Ethics
-------
* Sources. All the videos we collected from YouTube are from medical institutions or verified surgeons who have to obtain appropriate patient consent. 

* Curation and identifiability. During data curation we removed all out-of-body views, patient identifiers, and other nonsurgical content, and then conducted a manual review. Because the dataset contains only intra-operative endoscopic views and is anonymized, to the best of our knowledge, patients are not identifiable with current methods. 

* Regulatory. In many jurisdictions, including the EU, UK, US, and China, sharing non-identifiable data acquired for routine medical purposes does not need patient consent and is consistent with the GDPR.

* Opt out process. We will provide an online form that allows stakeholders to request the removal of their videos from our dataset. Also, if a source video is removed from YouTube, we will remove the corresponding items from our dataset.

License
-------
The LEMON dataset is provided under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/).
