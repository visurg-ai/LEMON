<p align="center">
    <a href="https://visurg.ai/">
    <img src="https://github.com/user-attachments/assets/04f6e2eb-1380-448e-a3f6-eed3e9dbf177">
    </a>
</p>

<p align="center">
  <a href="https://visurg.ai/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/04f6e2eb-1380-448e-a3f6-eed3e9dbf177">
      <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/68388c69-bfaa-499b-986c-32fed8bd1a63">
      <img src="https://github.com/user-attachments/assets/04f6e2eb-1380-448e-a3f6-eed3e9dbf177" alt="Adaptive figure">
    </picture>
  </a>
</p>


[📚 Paper](https://arxiv.org/abs/2503.19740) - [🤖 Code](src) - [🤗 Model](https://huggingface.co/visurg/SurgFM) - [🌐 Website](https://surg-3m.visurg.ai/)

Star ⭐ us if you like it!

<div align="center">
  <img src="https://github.com/user-attachments/assets/6250cd6a-1404-4786-9c15-fe396265940d" width="70%" > </img>
</div>



## News

<!-- XX/March/2025. The [HuggingFace models and demo](TODO) are released. -->
<!--<br>-->
* 25/March/2025. The [arXiv](https://arxiv.org/abs/2503.19740) version of the paper is released.

<br>

This is the official repository for the paper [Surg-3M: A Dataset and Foundation Model for Perception in Surgical Settings](https://arxiv.org/abs/2503.19740).

This repository provides open access to the *Surg-3M* dataset, *Surg-FM* foundation model, and training code. 

*Surg-3M* is a dataset of 4K surgical high-resolution videos (3M frames, when videos are sampled at 1fps) from 35 diverse surgical procedure types. Each video is annotated for multi-label classification, indicating the surgical procedures carried out in the video, and for binary classification, indicating if it is robotic or non-robotic. The dataset's annotations can be found in [labels.json](https://github.com/visurg-ai/surg-3m/blob/main/labels.json).

*Surg-FM* is an image foundation model for surgery, it receives an image as input and produces a feature vector of 1536 features as output. 

<!--The website of our dataset is: [http://surg-3m.org](https://surg-3m.org)-->

If you use our dataset, model, or code in your research, please cite our paper:

```
@misc{che2025surg3mdatasetfoundationmodel,
      title={Surg-3M: A Dataset and Foundation Model for Perception in Surgical Settings}, 
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
Advancements in computer-assisted surgical procedures heavily rely on accurate visual data interpretation from camera systems used during surgeries. Traditional open-access datasets focusing on surgical procedures are often limited by their small size, typically consisting of fewer than 100 videos with less than 100K images. To address these constraints, a new dataset called Surg-3M has been compiled using a novel aggregation pipeline that collects high-resolution videos from online sources. Featuring an extensive collection of over 4K surgical videos and more than 3 million high-quality images from multiple procedure types, Surg-3M offers a comprehensive resource surpassing existing alternatives in size and scope, including two novel tasks. To demonstrate the effectiveness of this dataset, we present SurgFM, a self-supervised foundation model pretrained on Surg-3M that achieves impressive results in downstream tasks such as surgical phase recognition, action recognition, and tool presence detection. Combining key components from ConvNeXt, DINO, and an innovative augmented distillation method, SurgFM exhibits exceptional performance compared to specialist architectures across various benchmarks. Our experimental results show that SurgFM outperforms state-of-the-art models in multiple downstream tasks, including significant gains in surgical phase recognition (+8.9pp, +4.7pp, and +3.9pp of Jaccard in AutoLaparo, M2CAI16, and Cholec80), action recognition (+3.1pp of mAP in CholecT50) and tool presence detection (+4.6pp of mAP in Cholec80). Moreover, even when using only half of the data, SurgFM outperforms state-of-the-art models in AutoLaparo and achieves state-of-the-art performance in Cholec80. Both Surg-3M and SurgFM have significant potential to accelerate progress towards developing autonomous robotic surgery systems.


<br>

Diversity and procedure prevalence in Surg-3M:

<img src="https://github.com/user-attachments/assets/67322046-5515-47e1-bb3f-621892c8608c">


Install dependencies to recreate our Surg-3M dataset
--------------------------------------------------
<!--
* If you want to use Docker**, follow the next steps to download our container:

   ```bash
   # Download the repo
   $ git clone git@github.com:visurg-ai/surg-3m.git
   $ cd surg-3m/docker

   # Build the docker image
   $ docker build --build-arg USER=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t chengan/Surg-3M:latest .

   # Run videosum Docker container
   $ docker run --volume $HOME:/mnt/user_home --name Surg-3M --runtime nvidia chengan/Surg-3M:latest &

   # Execute the docker container and get into the terminal
   $ docker exec --user $(whoami) --workdir $HOME -it Surg-3M /bin/zsh
   ```
-->

* Install the following dependencies in your local setup:

   ```bash
   $ git clone git@github.com:visurg-ai/surg-3m.git
   $ cd surg-3m && pip install -r requirements.txt
   ```

* **Models used in data curation**, We provide the models used in our data curation pipeline to assist with constructing the Surg-3M dataset, including video storyboard classification models, frame classification models, and non-surgical object detection models. The models can be downloaded from [🤗 Surg3M curation models](https://huggingface.co/visurg/Surg3M_curation_models).


Surg-3M dataset
--------------------------

> Researchers working in academic institutions can request direct access to the full Surg-3M dataset in LMDB format for non-commercial purposes by filling the request form in our [🌐 Website](https://surg-3m.visurg.ai/))

You can use our code of the data curation pipeline and provided annotation file (["labels.json"](https://github.com/visurg-ai/surg-3m/blob/main/labels.json)) to recreate the whole Surg-3M dataset.

1. Get your Youtube cookie:

   You need to provide a "cookies.txt" file if you want to download videos that require Youtube login. 

   Use the [cookies](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp) extension to export your Youtube cookies as "cookies.txt".


2. Download the annotation file (["labels.json"](https://github.com/visurg-ai/surg-3m/blob/main/labels.json)) and use the video downloader to download the original selected Youtube videos.

   ```bash
   $ python3 src/video_downloader.py --video-path '../labels.json' --output 'your path to store the downloaded videos' --cookies 'your YouTube cookie file'
   ```

3. Curate the downloaded original videos as Surg-3M video dataset. In detail, use the video_processor to classify each frame as either 'surgical' or 'non-surgical', then remove the beginning and end segments of non-surgical content from the videos, and mask the non-surgical regions in 'surgical' frames and the entire 'non-surgical' frames.

   ```bash
   $ python3 src/video_processor.py --input 'your original downloaded video storage path' --input-json '../labels.json' --output 'your path to store the curated videos and their corresponding frame annotation files' --classify-models 'frame classification model' --segment-models 'non-surgical object detection models'
   ```


4. Process the Surg-3M video dataset as Surg-3M image dataset (For foundation model pre-training).

   ```bash
   $ python3 src/create_lmdb_Surg-3M.py --video-folder 'your directory containing the curated videos and their corresponding frame annotation files' --output-json 'your path for the json file to verify the videos and labels alignment' --lmdb-path 'your lmdb storage path'
   ```

<br>
The video processing pipeline leading to the clean videos in the Surg-3M dataset is as follows:

<img src="https://github.com/user-attachments/assets/5192e42e-da73-462a-8f2e-ef422121c5cf">



SurgFM model
-------------
You can download the SurgFM full checkpoint which contains backbone and projection head weights for both student and teacher networks at [🤗 SurgFM](https://huggingface.co/visurg/SurgFM).

**SurgFM training:**

Follow the provided scripts to launch your own SurgFM training.

```bash
$ python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=1 surgfm/surgfm.py --arch convnext_large --data_path 'Surg-3M dataset lmdb path' --output_dir 'your path to store the trained foundation model' --batch_size_per_gpu 40 --num_workers 10
```


How to run our SurgFM foundation model to extract features from your video frames
----------------------------------------------------------------------------------

   ```python
   import torch
   from PIL import Image
   from model_loader import build_SurgFM

   # Load the pre-trained SurgFM model
   surgfm = build_SurgFM(pretrained_weights = 'your path to the SurgFM')
   surgfm.eval()

   # Load the image and convert it to a PyTorch tensor
   img_path = 'path/to/your/image.jpg'
   img = Image.open(img_path)
   img = img.resize((224, 224))
   img_tensor = torch.tensor(np.array(img)).unsqueeze(0).to('cuda')

   # Extract features from the image
   outputs = surgfm(img_tensor)
   ```



<!--
**surgfm performance:**

This figure shows the performance comparison between our foundation
model, SurgFM, and the state-of-the-art (SotA) models. Our
evaluation focuses on three surgical downstream tasks and six
datasets. SurgFM results are shown in bold, axis labels are presented in regular font.

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
