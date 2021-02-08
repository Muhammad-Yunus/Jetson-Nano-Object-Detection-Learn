# Pertemuan 1
- Intro to Jetson-nano, L4T (Linux for Tegra) & JetPack,  
- GPIO Programming & Hardware interfacing (Buzzer & LED)
- Camera interfacing,

___
## Jetson-nano<br>
![](resource/jetson-nano-dev-kit-B01.png)
- NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel, easy-to-use platform that runs in as little as 5 watts.
- Feature :
    - Jetson Nano Module<br>
    ![](resource/nvidia-jetson-nano-module.jpg)
        - 128-core NVIDIA Maxwell™ GPU
        - Quad-core ARM® A57 CPU
        - 4 GB 64-bit LPDDR4
        - 10/100/1000BASE-T Ethernet 
     - Power Options
        - Micro-USB 5V 2A
        - DC power adapter 5V 4A
    -  I/O<br>
    ![](resource/jetson-nano-b01.png)   
        - USB 3.0 Type A
        - USB 2.0 Micro-B 
        - HDMI/DisplayPort
        - M.2 Key E
        - Gigabit Ethernet
        - GPIOs, I2C, I2S, SPI, UART
        - 2xMIPI-CSI camera connector
        - Fan connector
        - PoE connector
___

## L4T OS (Linux For Tegra)
- Linux for Tegra (Linux4Tegra, L4T) is a GNU/Linux based system software distribution by Nvidia for the Nvidia Tegra processor series, used in platforms like the Nvidia Jetson board series. [[wiki](https://en.wikipedia.org/wiki/Linux_for_Tegra)]
- It includes :
    - Linux Kernel 4.9, 
    - bootloader, 
    - NVIDIA drivers, 
    - flashing utilities, 
    - sample filesystem based on Ubuntu 18.04, 
    - and more for the Jetson platform
- Latest release is NVIDIA L4T 32.5, supports all Jetson modules: 
    - Jetson AGX Xavier series, 
    - Jetson Xavier NX, 
    - Jetson TX2 series, 
    - Jetson TX1, 
    - and Jetson Nano
- Check L4T version used by Jetson Nano using Terminal
    ```
    head -n 1 /etc/nv_tegra_release
    ```
- Result :
    ```
    # R32 (release), REVISION: 5.0, GCID: 25531747, BOARD: t210ref, EABI: aarch64, DATE: Fri Jan 15 22:55:35 UTC 2021
    ```
- `R32` and `REVISION: 5.0` in the log above indicates that the L4T version used is L4T 32.5.
- Check L4T Linux Kernel version,
    ```
    uname -a
    ```
- Result,
    ```
    Linux jetson 4.9.201-tegra #1 SMP PREEMPT Fri Jan 15 14:41:02 PST 2021 aarch64 aarch64 aarch64 GNU/Linux
    ```
- You can see that the kernel version used is `4.9.201-tegra` for the `aarch64` architecture.
___
## NVIDIA JetPack SDK
- NVIDIA JetPack SDK is the most comprehensive solution for building AI applications. All Jetson modules and developer kits are supported by JetPack SDK.
- JetPack SDK includes the latest Linux Driver Package (L4T) with Linux operating system and CUDA-X accelerated libraries and APIs for Deep Learning, Computer Vision, Accelerated Computing and Multimedia. <br>
![](resource/jetpack.jpg)
- Latest release JetPack is version 4.5.
- JetPack Feature :
    - **OS** : NVIDIA L4T (JetPack 4.5 includes L4T 32.5)
    - **CUDA** : is a parallel computing platform and programming model using a GPU. (JetPack 4.5 include CUDA 10.2 )
    - **cuDNN** : (CUDA Deep Neural Network) is a GPU-accelerated library of primitives for deep neural networks. Provides highly tuned implementations for standard routines such as *forward* and *backward convolution*, *pooling*, *normalization*, and *activation layers*. cuDNN accelerates widely used deep learning frameworks, including **Caffe2**, **Chainer**, **Keras**, **MATLAB**, **MxNet**, **PaddlePaddle**, **PyTorch**, and **TensorFlow** (JetPack 4.5 includes cuDNN 8.0).
    - **TensorRT** : is a high performance deep learning inference runtime, built on top of CUDA (JetPack 4.5 includes TensorRT 7.1.3)
    - **Computer Vision** : 
        - VisionWorks ver 1.6
        - **OpenCV** (default without CUDA) ver 4.1.1
        - VPI (Vision Programing Interface) ver 1.0
- Check JetPack version in L4T OS,
    ```
    sudo apt-cache show nvidia-jetpack
    ```
- Result,
    ```
    Package: nvidia-jetpack
    Version: 4.5-b129
    Architecture: arm64
    Maintainer: NVIDIA Corporation
    Installed-Size: 194
    ...
    .
    ```
- You can see that the JetPack version is `4.5-xxx` in `L4T 32.5` .
___
## Basic Linux Command
- Update & Upgrade softwre,
    ```
    sudo apt-get update
    sudo apt-get upgrade
    ```
- Check network interface,
    ```
    ifconfig
    ```
- Working with Filesystems,
    - lists the content of the current directory,
        ```
        ls
        ```
    - listing with detail,
        ```
        ls -lAh
        ```
    - change/move directory,
        ```
        cd Downloads
        ```
    - move to home dir,
        ```
        cd ~
        ```
    - check current working dir,
        ```
        pwd
        ```
    - create forlder,
        ```
        mkdir Test
        ```

    - create file,
        ```
        touch myfile.txt
        ```
    - copy file,
        ```
        cp myfile.txt Test
        ```
    - remove file,
        ```
        rm Test/myfile.txt
        ```
    - move file,
        ```
        mv myfile.txt Test
        ```
    - remove empty folder,
        ```
        rmdir Test
        ```
    - remove non empty folder,
        ```
        rm -rf Test
        ```
___
## Basic GPIO Programming
- Jetson Nano GPIO reference,<br>
    ![](resource/jetson-nano-gpio.png)
- Update & Upgrade software,
    ```
    sudo apt-get update && sudo apt-get upgrade
    ```
- Install Python Library `Jetson.GPIO` (*by default is already installed*),
    ```
    sudo pip install Jetson.GPIO
    ```
## 1. Program Blink :
- Wiring diagram, using **pin 12**.<br>
![](resource/blink-wiring.png)
- Blink program,
    ```
    import Jetson.GPIO as GPIO
    import time

    # Pin Definitions
    output_pin = 12  # BOARD pin 12

    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  # BCM pin-numbering scheme from Raspberry Pi
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

    print("Press CTRL+C to exit")

    try:
        while True:
            GPIO.output(output_pin, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(output_pin, GPIO.LOW)
            time.sleep(1)
    finally:
        GPIO.cleanup()
    ```    
- Run program,
    ```
    python '1. blink.py'
    ```
- Result,<br>
    ![](resource/blink-result.gif)
## 2. Program Buzzer :
- Wiring diagram, using **pin 7**.<br>
![](resource/buzzer-wiring.png)
- Buzzer program,
    ```
    import Jetson.GPIO as GPIO
    import time

    # Pin Definitions
    output_pin = 7  # BOARD pin 7

    # Pin Setup:
    GPIO.setmode(GPIO.BOARD)  
    GPIO.setup(output_pin, GPIO.OUT, initial=GPIO.HIGH)

    print("Press CTRL+C to exit")

    try:
        while True:
            # bep, bep, bep ...
            for _ in range(3):
                GPIO.output(output_pin, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(output_pin, GPIO.LOW)
                time.sleep(0.02)
            GPIO.output(output_pin, GPIO.LOW)
            time.sleep(0.5)

    finally:
        GPIO.output(output_pin, GPIO.LOW)
        GPIO.cleanup()
    ```
- Run program,
    ```
    python '2. buzzer.py'
    ```
___
## Camera interfacing
- List supported camera in Jetson Nano :
    - https://elinux.org/Jetson_Nano#Cameras
- Attach camera Board to Jetson Nano :
    - CSI Camera :<br>
    ![](resource/CSI-camera.jpg)
    - USB Webcam :<br>
    ![](resource/USB-camera.png)
- Check camera node,
    ```
    ls /dev/video*
    ```
- result,
    ```
    /dev/video0 /dev/video1
    ```
- You can see above, we have two attached camera into Jetson Nano,
___
## nvgstcapture
- The examples below use **nvgstcapture gstreamer application** to access the camera features via the NVIDIA API.
### 1. Start Capture and Preview display on the screen
- CSI camera
    ```
    nvgstcapture-1.0
    ```
- USB Camera, capture using node 0,
    ```
    nvgstcapture-1.0 --camsrc=0 --cap-dev-node=0
    ```
- Press `'j'` to Capture one image.
- Press `'q'` to exit
### 2. Capture a video and save to disk
- CSI camera
    ```
    nvgstcapture-1.0
    ```
- USB Camera, capture using node 0,
    ```
    nvgstcapture-1.0 --mode=2 --camsrc=0 --cap-dev-node=0
    ```
- Press `'1'` to Start recording video
- Press `'0'` to Stop recording video
- Press `'q'` to exit
___
## Access Camera using OpenCV
- Capture, preview & save to disk,
    ```
    python3 '3. OpenCV-Camera-Stream.py'
    ```
- Code,
    ```
    import cv2 

    cam = 0
    cap = cv2.VideoCapture(cam)

    while cap.isOpened():
        ret, img = cap.read()
        if not ret :
            break
        cv2.imshow("Stream", img)

        key = cv2.waitKey(10) 
        if key == ord('j') :
            cv2.imwrite("captured_photo.jpg", img)
        elif key == ord('q') :
            break

    cv2.DestroyAllWindows()
    cap.release()
    ```

## MJPEG stream Flask + OpenCV
- **MJPEG stream** working using **Server Push** mechanism.
- Server push is a simple way of **sending file updates to a browser**.
- Most major browsers, and some minor browsers support it for images, with one important exception.
- How it works ?
    - Browser requests a file (typically an image)
    - The server responds with a **multipart mixed-replace** content type header,
        ```
        Content-Type: multipart/x-mixed-replace;boundary=frame
        ```
    - Image from the server is always continue sending image data to browser (client)
    - Image is sender as byte format :
        ```
        --frame
        Content-Type: image/jpeg
        6A3D 5671 55CA 66B7
        611A A10B 1408 246A
        ....
        ....
        ```
- Install Flask,
    ```
    pip3 install flask
    pip3 install flask-socketio
    ```
### 1. Simple MJPEG Stream
- Project Structure :
    ```
    |__ 4. Flask MJPEG Stream.py
    |__ templates/
          |__ index.html
    ```
    ![](resource/flask-mjpeg.png)
- Run,
    ```
    python3 '4. Flask MJPEG Stream.py'
    ```
### 2. Facedetection MJPEG Stream
- Detector : Haar Cascade Classifier
- Run,
    ```
    python3 '5. Facedetect MJPEG Stream.py'
    ```
### 3. Facedetection Alarm
- Run,
    ```
    python3 '6. Facedetect Alarm.py'
    ```
### 4. Facedetection Alarm Async (SocketIO)
- Run,
    ```
    python3 '7. Facedetect Alarm Async.py'
    ```
# Source :
- https://developer.nvidia.com/CUDnn
- https://blogs.nvidia.com/blog/2012/09/10/what-is-cuda-2/
- https://developer.nvidia.com/EMBEDDED/Jetpack
- https://forums.developer.nvidia.com/t/how-to-check-the-jetpack-version/69549/11
- https://www.seeedstudio.com/NVIDIA-Jetson-Nano-Development-Kit-B01-p-4437.html
- https://developer.nvidia.com/embedded/jetson-nano
- https://github.com/NVIDIA/jetson-gpio
- https://developer.nvidia.com/embedded/learn/tutorials/first-picture-csi-usb-camera
- https://github.com/opencv/opencv/issues/15074
- https://github.com/madams1337/python-opencv-gstreamer-examples/blob/master/gst_device_to_rtp.py