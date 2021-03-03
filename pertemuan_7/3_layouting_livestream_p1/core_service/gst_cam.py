def camera(i, w, h):
    return "nvarguscamerasrc sensor_id=%d ! \
    video/x-raw(memory:NVMM), \
    width=%d, height=%d, \
    format=(string)NV12, \
    framerate=21/1 ! \
    nvvidconv \
    flip-method=2  ! \
    video/x-raw, \
    format=(string)BGRx ! \
    videoconvert ! \
    video/x-raw, \
    format=(string)BGR ! \
    appsink" % (i, w, h)