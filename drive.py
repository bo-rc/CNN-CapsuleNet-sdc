import os
import os.path
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
import model
from rc_car_msgs.msg import CarController
from keras.models import load_model
import h5py
from keras import __version__ as keras_version

model_file = "keras_nvidiaNet.h5"

crop = 240
height = 160
width = 320

throttle_velocity = 0.0


class RosKeras():
    def __init__(self):
        self._cv_bridge = CvBridge()
        self.model = load_model(model_file)
        self.car = CarController()
        self._sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback, queue_size=1)
        self._pub = rospy.Publisher('/car_controller', CarController, queue_size=1)

    def callback(self, image_msg):
        cv_image = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv_image = cv_image[crop:, :, :]
        # cv2.imwrite('./tem.jpg', cv_image)
        cv_image_resize = cv2.resize(cv_image,(width,height))
        image_array = np.asarray(cv_image_resize)

        predict_angle = float(self.model.predict(image_array[None, :, :, :], batch_size=1))

        self.car.throttle = throttle_velocity
        self.car.steer = predict_angle

        self._pub.publish(self.car)

    def main(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('rostensorflow')
    tensor = RosKeras()
    tensor.main()
