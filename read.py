import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tensorflow as tf
import collections
import statistics
import math
import tarfile
import os.path

from threading import Lock, Thread
from time import sleep

import cv2

# ZED imports
import pyzed.sl as sl
import time

sys.path.append('utils')

# ## Object detection imports
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_depth_into_numpy_array(depth):
    ar = depth.get_data()
    ar = ar[:, :, 0:4]
    (im_height, im_width, channels) = depth.get_data().shape
    return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)


lock = Lock()
width = 1280
height = 720
confidence = 0.35

image_np_global = np.zeros([width, height, 3], dtype=np.uint8)
depth_np_global = np.zeros([width, height, 4], dtype=np.float)

exit_signal = False
new_data = False


# ZED image capture thread function
def capture_thread_func(svo_filepath=None):
    global image_np_global, depth_np_global, exit_signal, new_data

    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.RESOLUTION_HD720
    init_params.camera_fps = 60
    init_params.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_ULTRA
    init_params.coordinate_units = sl.UNIT.UNIT_METER
    init_params.svo_real_time_mode = False
    if svo_filepath is not None:
        init_params.svo_input_filename = svo_filepath

    # Open the camera
    err = zed.open(init_params)
    print(err)
    while err != sl.ERROR_CODE.SUCCESS:
        err = zed.open(init_params)
        print(err)
        sleep(1)

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    while not exit_signal:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.VIEW_LEFT, width=width, height=height)
            zed.retrieve_measure(depth_mat, sl.MEASURE.MEASURE_XYZRGBA, width=width, height=height)
            lock.acquire()
            image_np_global = load_image_into_numpy_array(image_mat)
            depth_np_global = load_depth_into_numpy_array(depth_mat)
            new_data = True
            lock.release()

        sleep(0.01)

    zed.close()

def judge(num_detections, boxes_, classes_, scores_, category_index):
    index=[]
    num=0
    for i in range(num_detections):
        if(classes_[i]==0and scores_[i]>confidence):
            index.append[i]
            num+=1
    return index,num

#寻找质心位置，是原始display_objects_distances函数功能的一部分

def center_position(num_detections, boxes_, classes_, scores_, category_index):
    x_c=[]
    y_c=[]
    for i in range(num_detections):
        box=boxes_[i]
        ymin, xmin, ymax, xmax = box
        x_c.append( int(xmin * width + (xmax - xmin) * width * 0.5))
        y_c.append( int(ymin * height + (ymax - ymin) * height * 0.5))
    return x_c,y_c

# 对于这个display函数，image_np是用来画东西的，depth_np是用来算距离的

def display_objects_distances(image_np, depth_np, num_detections, boxes_, classes_, scores_, category_index,assigned,unassigned,trackidcount):
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)

    research_distance_box = 30

# 应当注意，这里不确定categories中人对应的是哪一个index
    for i in range(num_detections):
        #给编号为i的detection找到对应的track编号
        #分两种情况：1.对应的track编号在assigned数组里面。直接便利查询assigned数组即可
        # 2.在unassigned里面，遍历查询，用现在的总数减去对应的编号数
        idx=-1
        for i_assigned in len(assigned):
            if(assigned[i_assigned]==i):
                idx=i_assigned
                break
        #注意这里可能有计数错误
        if(idx==-1):
            for i_unassigned in len(unassigned):
                if(unassigned[i_unassigned]==i):
                    idx=trackidcount-i_unassigned-1
        if scores_[i] > confidence:
            box = tuple(boxes_[i].tolist())
            if classes_[i] in category_index.keys():
                class_name = category_index[classes_[i]]['name']
            # display_str = ''
            display_str = str(class_name)+str(idx)
            if not display_str:
                display_str = '{}%'.format(int(100 * scores_[i]))
            else:
                display_str = '{}: {}%'.format(display_str, int(100 * scores_[i]))

            # Find object distance
            ymin, xmin, ymax, xmax = box
            x_center = int(xmin * width + (xmax - xmin) * width * 0.5)
            y_center = int(ymin * height + (ymax - ymin) * height * 0.5)
            x_vect = []
            y_vect = []
            z_vect = []

            min_y_r = max(int(ymin * height), int(y_center - research_distance_box))
            min_x_r = max(int(xmin * width), int(x_center - research_distance_box))
            max_y_r = min(int(ymax * height), int(y_center + research_distance_box))
            max_x_r = min(int(xmax * width), int(x_center + research_distance_box))

            if min_y_r < 0: min_y_r = 0
            if min_x_r < 0: min_x_r = 0
            if max_y_r > height: max_y_r = height
            if max_x_r > width: max_x_r = width

            for j_ in range(min_y_r, max_y_r):
                for i_ in range(min_x_r, max_x_r):
                    z = depth_np[j_, i_, 2]
                    if not np.isnan(z) and not np.isinf(z):
                        x_vect.append(depth_np[j_, i_, 0])
                        y_vect.append(depth_np[j_, i_, 1])
                        z_vect.append(z)

            if len(x_vect) > 0:
                x = statistics.median(x_vect)
                y = statistics.median(y_vect)
                z = statistics.median(z_vect)

                distance = math.sqrt(x * x + y * y + z * z)

                display_str = display_str + " " + str('% 6.2f' % distance) + "m\n" + str('% 6.2f' % x) + "m" + str(
                    '% 6.2f' % y) + "m" + str('% 6.2f' % z) + "m"  # --------------------------------------------
                box_to_display_str_map[box].append(display_str)
                box_to_color_map[box] = vis_util.STANDARD_COLORS[classes_[i] % len(vis_util.STANDARD_COLORS)]

    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box

        vis_util.draw_bounding_box_on_image_array(
            image_np,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=50,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=True)

    return image_np


def main(args):
    svo_filepath = None
    if len(args) > 1:
        svo_filepath = args[1]

    # This main thread will run the object detection, the capture thread is loaded later

    # What model to download and load
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
    # MODEL_NAME = 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
    # MODEL_NAME = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
    # MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
    # MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28' # Accurate but heavy

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = 'data/' + MODEL_NAME + '/frozen_inference_graph.pb'

    # Check if the model is already present
    if not os.path.isfile(PATH_TO_FROZEN_GRAPH):
        print("Downloading model " + MODEL_NAME + "...")

        MODEL_FILE = MODEL_NAME + '.tar.gz'
        MODEL_PATH = 'data/' + MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_PATH)
        tar_file = tarfile.open(MODEL_PATH)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, 'data/')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    # Start the capture thread with the ZED input---------------------------------------------------------------------------
    print("Starting the ZED")
    capture_thread = Thread(target=capture_thread_func, kwargs={'svo_filepath': svo_filepath})
    capture_thread.start()
    # Shared resources
    global image_np_global, depth_np_global, new_data, exit_signal

    # Load a (frozen) Tensorflow model into memory.
    print("Loading model " + MODEL_NAME)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Limit to a maximum of 50% the GPU memory usage taken by TF https://www.tensorflow.org/guide/using_gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    print(label_map)

    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print('categories is', categories)
    print(type(categories))
    print('category_index is ', category_index)
    print(type(category_index))
    #应该在没进入循环之前加入tracker对象,将trackor类加入大文件中
    tracker = Tracker(160, 30, 5, 100)
    # Detection
    with detection_graph.as_default():
        with tf.Session(config=config, graph=detection_graph) as sess:
            while not exit_signal:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                if new_data:
                    lock.acquire()
                    image_np = np.copy(image_np_global)
                    depth_np = np.copy(depth_np_global)
                    new_data = False
                    lock.release()

                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    num_detections_ = num_detections.astype(int)[0]
                    #检测完了有用的信号，就开始进行一个tracking
                    #首先筛选置信度与标签（是不是人）
                    index,num=judge(num_detections_,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index)
                    #寻找每个检测到的人的质心坐标(inspect)
                    x_c,y_c=center_position(num,
                        np.squeeze(boxes[index]),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index)
                    centers=[]
                    for i in num:
                        tem=np.array[[x_c[i],y_c[i]]]
                        centers.append(np.round(tem))
                    # update trackers
                    # 为了能给之后的detection，对应上tracks

                    assigned,unassigned=tracker.Update(centers)

                    #说明： 每次使用tracker的Update的方法时，tracker会自动把观察到的track和之前的track自动匹配
                    #我们可以利用tracker中自带的标签来标识同一个人

                    #最终要用tracks来表示，而要输出在屏幕上的是
                    # tracks的distance，属性，标签（第几个追踪目标），和其对应的框

                    # Visualization of the results of a detection.
                    #可以说，我们只显示detection的内容，虽然一些没被检测出的track还有可能重现，
                    # 但是我们在这一次的显示中不显示没有被detect的内容

                    image_np = display_objects_distances(
                        image_np,
                        depth_np,
                        num_detections_,
                        np.squeeze(boxes[index]),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,assigned,unassigned,tracker.trackIdCount)
                    print('shape of image_np is', image_np.shape)
                    # cv2.imshow('ZED object detection', cv2.resize(image_np, (width, height)))
                    cv2.imshow('ZED object detection', image_np)
                    # timestamp = zed.get_timestamp(sl.TIME_REFERENCE.TIME_REFERENCE_CURRENT)

                    timestamp = time.time()
                    # cv2.imwrite(str(timestamp)+".png", image_np)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        exit_signal = True
                else:
                    sleep(0.01)

            sess.close()

    exit_signal = True
    capture_thread.join()  # ----------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main(sys.argv)
