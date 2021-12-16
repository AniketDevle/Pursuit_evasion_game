#!/usr/bin/env python
import cv2
import math
import rospy
import numpy as np
import tf.transformations
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from collections import namedtuple
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, Point, Quaternion

MODEL_PATH_WEIGHTS = "/home/aniketdevle/catkin_ws/src/pursuit_evasion/net/yolov3.weights"
MODEL_PATH_CFG = "/home/aniketdevle/catkin_ws/src/pursuit_evasion/net/yolov3.cfg"

Detection = namedtuple('Detection', 'center confidence')


class DetectHuman:
    def __init__(self, min_confidence=0.1):
        self.bridge = CvBridge()
        self.min_confidence = min_confidence
        self.net = cv2.dnn.readNetFromDarknet(MODEL_PATH_CFG, MODEL_PATH_WEIGHTS)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.out_layers = self.net.getLayerNames()
        self.out_layers = [self.out_layers[int(l)] for l in self.net.getUnconnectedOutLayers() - 1]

    def detect_human(self, frame):
        img_h, img_w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=False, crop=False)

        self.net.setInput(blob)
        outputs = self.net.forward(self.out_layers)

        best = None
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                (x, y, w, h) = detection[0:4] * [img_w, img_h, img_w, img_h]

                if class_id == 0 and confidence > self.min_confidence:
                    print(class_id)
                    if best:
                        if confidence > best.confidence:
                            best = Detection((int(x), int(y)), confidence)
                    else:
                        best = Detection((int(x), int(y)), confidence)

        return best

    def run(self, data):
        img = self.bridge.imgmsg_to_cv2(data)
        match = self.detect_human(img)
        return match


class Pursuer:
    def __init__(self):
        self.angle_z = None
        self.location = None
        self.frame_id = "map"

        self.detector = DetectHuman()

        rospy.init_node("Evader")
        self.frame_subscriber = rospy.Subscriber("/tb3_0/camera/rgb/image_raw", Image, callback=self.publish_goal)
        self.pose_subscriber = rospy.Subscriber("/tb3_0/amcl_pose", PoseWithCovarianceStamped, callback=self.save_pose)
        self.goal_publisher = rospy.Publisher("/tb3_0/move_base_simple/goal", PoseStamped, queue_size=10)

        rospy.spin()

    def save_pose(self, data):
        angles_q = data.pose.pose.orientation
        angles_q = [angles_q.x, angles_q.y, angles_q.z, angles_q.w]
        self.angle_z = tf.transformations.euler_from_quaternion(angles_q)[2]
        self.location = data.pose.pose.position

    def publish_goal(self, image_msg):
        detection = self.detector.run(image_msg)
        if detection is not None and self.angle_z is not None and self.location is not None:
            x = detection.center[0] - 320
            angle = math.atan(-x * 5 * math.tan(math.radians(35)) / 320) + self.angle_z
            x = self.location.x + 0.7 * math.cos(angle)
            y = self.location.y + 0.7 * math.sin(angle)
            position = Point(x, y, self.location.z)

            quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)

            stamp = rospy.Time.now()

            goal = PoseStamped()
            goal.pose.position = position
            goal.pose.orientation = Quaternion(*quaternion)
            goal.header.stamp = stamp
            goal.header.frame_id = "map"

            self.goal_publisher.publish(goal)


if __name__ == '__main__':
    pursuer = Pursuer()
