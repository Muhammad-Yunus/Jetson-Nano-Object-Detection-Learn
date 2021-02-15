# Pertemuan 3
- Prepare Dataset
    - Convert Video to Image dataset
    - Annotate Image Dataset using `LabelImg`
    - Split dataset
- Training Tensorflow model Faster R-CNN Inception V2 in Google Colab
- Deploy model to Jetson Nano <br>
![](resource/diag.png)
____
# 1. Prepare Dataset
- Take a video for the object to be detected. `One video` for `one data class`.<br>
![](resource/ironman.gif)
- Clone [ObjectDetection-Tensorflow](https://github.com/Muhammad-Yunus/ObjectDetection-Tensorflow) repository from Github to local computer,
- Open cloned `ObjectDetection-Tensorflow` repo folder.
- Copy the video that you have recorded to the `dataset/images/` folder.
- Convert Video to Image using `dataset_builder.ipynb`,
- Download & Install [LabelImg](https://github.com/tzutalin/labelImg) 
- Click `Open Dir` button in labelImg and choose `dataset/images/` folder.
- Annotate Dataset using LabelImg, <br>
![](resource/annotate_image.gif)
- Split dataset (20% for test dataset, 80% for training dataset) using `dataset_builder.ipynb`.
___
# 2. Training Tensorflow model Faster R-CNN Inception V2 in Google Colab
- Open [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) in local computer browser.
- Upload `Faster_R_CNN_Training_using_Custom_Dataset.ipynb` from cloned `ObjectDetection-Tensorflow` repo to Colab.
![](resource/colab-upload.png)
- Follow training step in uploaded notebook, 
- After training & testing finish, download `inference_graph.zip` from colab to local computer,
- Detection Result : <br>
    <p float="left">
    <img src="resource/batman.png" width="300" />
    <img src="resource/ironman.png" width="300" /> 
    </p>
- Evaluation report :
    - mAP : 100% (0.5IOU)
    ![](resource/eval.png)
___
# 3. Deploy model to Jetson Nano
- Extract `inference_graph.zip` in local computer
- Copy `frozen_inference_graph.pb` and `faster_rcnn_inception_v2_custom_dataset.pbtxt` from extracted `inference_graph.zip` folder to  `pertemuan_3/model/` folder in Jetson Nano.
- Copy `object-detection.json` file created in step one (prepare dataset - [ObjectDetection-Tensorflow](https://github.com/Muhammad-Yunus/ObjectDetection-Tensorflow)) to `pertemuan_3/` folder in Jetson Nano.
- Run `faster_r-cnn_flask_async.py`,
    ```
    python3 faster_r-cnn_flask_async.py
    ```
- Open result in browser (local computer) with url `http://<jetson nano IP>:5000`
- Result,<br>
![](resource/live-stream.gif)
___