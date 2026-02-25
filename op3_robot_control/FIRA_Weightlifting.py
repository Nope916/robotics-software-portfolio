#! /usr/bin/python

import rospy
import numpy as np
import cv2
import time


from op3_ros_utils import getWalkingParams, Robot
from vision import *
from copy import copy
from sensor_msgs.msg import Imu
import sys

DEBUG_MODE = False
# DEBUG_MODE = True 
MIN_AREA = 300
CROSS_AREA = 1165
DEGREE2RADIAN = np.pi / 180

class States:
    INIT = -1
    READY = 0 # Waits for start button
    WALK_FORWARD = 1 # Moves the head, looking for the ball
    PICK_BAR = 5
    WALK_WITH_BAR = 6
    LIFT_BAR = 7
    WALK_2_FINISH = 8
    CHECK_SLOPE = 9
    CHECK_PAN = 10
    END = 99

z_gyro = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0
def imu_get_yaw_by_integral(data):
    global z_gyro
    global z_gyro_offset
    angular_velocity = data.angular_velocity
    # print("aa")
    z_gyro = z_gyro - angular_velocity.z/2 + z_gyro_offset_for_caculate
    # z_gyro = angular_velocity.z/2
    # print("z_gyro_offset_for_caculate:",z_gyro_offset_for_caculate)
    # print(z_gyro)
    z_gyro_offset = angular_velocity.z/2

rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)

global last_Px, last_Py, time_for_Pd
last_Px = 0.0
last_Py = 0.0
time_for_Pd = time.time()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Define your color range for detection

# lower_hsv = np.array([16, 77, 0])
# upper_hsv = np.array([43, 255, 198])
# (np.array([104, 86, 93]), np.array([179, 255, 255]
lower_hsv = np.array([22,  69, 128])
upper_hsv = np.array([ 38 ,255 ,255])
# lower value of yel1:[ 22  69 128]
# upper value of yel1:[ 38 255 255]

mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

# Iinitialize Node
rospy.init_node("fira_sprint")

# Create robot
robot = Robot()



rospy.sleep(3) # Make sure every publisher has registered to their topic,
               # avoiding lost messages
# rospy.sleep(2) # Make sure every publisher has registered to their topic,               # avoiding lost messages
z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0
#sure
def detect_single_color(frame, lower_hsv, upper_hsv):
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask with the given HSV range
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    
    # Convert mask to a 3-channel image to put a colored dot
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Find contours in the mask
    # Note: cv2.findContours returns 3 values in older OpenCV versions used with Python 2.7
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    # Ensure at least one contour was found
    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the center of the contour
        M = cv2.moments(largest_contour)

        selection = cv2.minAreaRect(largest_contour)
        (x, y), (width, height), slope = selection    
        angle = compute_angle(width, height, slope)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw a circle at the center on the original frame
            cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)  # Blue dot
            # Draw the largest contour on the original frame
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)  # Green contour
            return cX, cY, angle
        
        else:
            # cX, cY = 0, 0  # Default to (0,0) if contour is a line
            return None, None,None

def compute_angle(width, height, angle):
    if angle < -90 or (width > height and angle < 0):
        return 90 + angle
    if width < height and angle > 0:
        return (90 - angle) * -1 
    return angle
#sure
def camera_theta(cX, cY, frame_width, frame_height):
    global last_Px, last_Py, time_for_Pd
    
    # Convert cX and cY to normalized coordinates relative to the frame center
    Px = -1 * ((cX / float(frame_width)) - 0.5)
    Py = -1 * ((cY / float(frame_height)) - 0.5)
    
    # Assuming robot.joint_pos is accessible and contains the current angles in radians
    pan_angle_old = np.degrees(robot.joint_pos["head_pan"])
    tilt_angle_old = np.degrees(robot.joint_pos["head_tilt"])
    
    # Calculate derivative of position
    Dx = (Px - last_Px) / (time.time() - time_for_Pd)
    Dy = (Py - last_Py) / (time.time() - time_for_Pd)
    time_for_Pd = time.time()
    
    # Proportional and derivative gains (tune these according to your robot's characteristics)
    kp = 20
    kd = 0
    
    # Calculate new angles
    pan_angle = pan_angle_old + Px * kp + Dx * kd
    tilt_angle = tilt_angle_old + Py * kp + Dy * kd
    
    # Update last position for next iteration
    last_Px, last_Py = Px, Py
    
    # Apply the new angles, ensuring they're within the mechanical limits of the robot
    robot.setJointPos(["head_pan", "head_tilt"], [np.clip(np.radians(pan_angle), np.radians(-95), np.radians(95)), np.clip(np.radians(tilt_angle), np.radians(-95), np.radians(95))])
    
    print("Pan angle:", pan_angle)
    print("Tilt angle:", tilt_angle)
    return pan_angle, tilt_angle

def init():
    # Set ctrl modules of all actions to joint, so we can reset robot position 
    robot.setGeneralControlModule("action_module")

    # robot.setGrippersPos(left=0.0, right=0.0)

    # Call initial robot position
    robot.playMotion(1, wait_for_end=True)

    # Set ctrl module to walking, this actually only sets the legs
    robot.setGeneralControlModule("walking_module")
    
    # Set joint modules of head joints to none so we can control them directly
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])

    robot.setJointPos(["head_tilt" , "head_pan"] , [-0.8 , 0])

    rospy.sleep(1.0)

tickrate = 60
rate = rospy.Rate(tickrate)

# TODO remember to put to 0
STEP_LEVEL = 0

currState = States.INIT
while not rospy.is_shutdown():



    ret, frame = cap.read()
    if ret:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result = detect_single_color(frame, lower_hsv, upper_hsv)
        if result is not None:  # Ensure there's a result to unpack
            cX, cY ,ang= result
            if ang is not None:
                if ang>0:
                    ang=90.0-ang
                elif ang<=0:
                    ang=(-90.0)-ang
                    print("slope:",ang)
            if cX is not None and cY is not None:
                # Now it's safe to use cX and cY
                pan_angle, tilt_angle = camera_theta(cX, cY, frame_width, frame_height)
        cv2.imshow('MASK',mask)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
    
    if robot.buttonCheck("mode"):
        STEP_LEVEL = 0
        currState = States.INIT
    
    if DEBUG_MODE:
        lower_hsv = np.array([22,  69, 128])
        upper_hsv = np.array([ 38 ,255 ,255])
        # lower value of yellow1:[ 11  11 164]
        # upper value of yellow1:[ 68 255 255]
        # lower value of red1:[95 81  0]
# upper value of red1:[124 255 255]
        ret, frame = cap.read()
        detect_single_color(frame,lower_hsv,upper_hsv)
        
        

        if ret:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result = detect_single_color(frame, lower_hsv, upper_hsv)
    
            # cX, cY ,ang= detect_single_color(frame, lower_hsv, upper_hsv)  # This function should return the centroid coordinates\
            
            if result is not None:  # Ensure there's a result to unpack
                cX, cY , ang = result
                if ang>0:
                    ang=90-ang
                elif ang<=0:
                    ang=(-90)-ang
                if cX is not None and cY is not None:
                    # Now it's safe to use cX and cY
                    pan_angle,_= camera_theta(cX, cY, frame_width, frame_height)
                    print("slope:",ang)
        cv2.imshow('MASK',mask)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)


    if currState == States.INIT:
        print("[INIT]")
        init()

        # Transition
        tick_count = 0
        direction = False
        currState = States.READY

    elif currState == States.READY:
        print("[READY]")
        if robot.buttonCheck("start"):
            rospy.sleep(1)
            z_gyro_offset_for_caculate = z_gyro_offset
            z_gyro = 0.0
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.4])
            tick_count = 0
            robot.walkStart()
            currState = States.WALK_FORWARD
        
    elif currState == States.WALK_FORWARD:
        print("[WALK_FORWARD]")
        # robot.walkVelocities(x = -1, y = 0, th=0, z_move_amplitude=0.035, balance=True, z_offset=0)
        robot.walkVelocities(x = 3, y = 0, th=np.clip(pan_angle, -10, 10), z_move_amplitude=0.035, balance=True, z_offset=0)

        if result is not None and tilt_angle <= -72:
            robot.walkStop()
            currState = States.CHECK_PAN
            rospy.sleep(1)



    elif currState == States.CHECK_SLOPE:          #garbege
        robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"]) 
        robot.setJointPos(["head_pan", "head_tilt"], [0, -0.8])
        rospy.sleep(3)
        while abs(ang)>3:
            ret, frame = cap.read()
            if ret:
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                result = detect_single_color(frame, lower_hsv, upper_hsv)
                if result is not None:  # Ensure there's a result to unpack
                    cX, cY ,ang= result
                    if ang is not None:
                        if ang>0:
                            ang=90.0-ang
                        elif ang<=0:
                            ang=(-90.0)-ang
            if abs(ang) > 3:        #SPIN
                if ang > 3:  #original 0.02
                    print('turn left')

                    robot.onlineWalkCommand(direction="turn_left", start_leg="right", step_num=2,front_length=-0.1, step_angle=np.clip(ang,5,-5),step_time=0.5)
                    rospy.sleep(5)
                elif ang < -3 :
                    print('turn right')

                    robot.onlineWalkCommand(direction="turn_right", start_leg="right", step_num=2,front_length=-0.1, step_angle=np.clip(ang,5,-5),step_time=0.5)
                    rospy.sleep(5)
                else:
                    currState == States.CHECK_PAN
    elif currState == States.CHECK_PAN:
        print('CHECK PAN')
        while abs(np.degrees(robot.joint_pos["head_pan"]))>3:
            print('adjusting')
            ret, frame = cap.read()
            if ret:
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                result = detect_single_color(frame, lower_hsv, upper_hsv)
                if result is not None:  # Ensure there's a result to unpack
                    cX, cY ,ang= result
                    if ang is not None:
                        if ang>0:
                            ang=90.0-ang
                        elif ang<=0:
                            ang=(-90.0)-ang
                    if cX is not None and cY is not None:
                        pan_angle, tilt_angle = camera_theta(cX, cY, frame_width, frame_height)
                        robot.walkStart()
                        robot.walkVelocities(x = -2, y = 0, th=np.clip(pan_angle, -20, 20), z_move_amplitude=0.045, balance=True, z_offset=0)
    # if abs(np.degrees(robot.joint_pos["head_pan"])) < 3:
        print('bb')
        robot.walkStop()
        currState = States.PICK_BAR
 
    elif currState == States.PICK_BAR:
        rospy.loginfo("[PICK_BAR]")
        rospy.sleep(2)
        robot.setGeneralControlModule("none")
        rospy.sleep(2)
        robot.setGeneralControlModule("action_module")
        robot.playMotion(178, wait_for_end=True)     #wait pos
        time.sleep(2)
        z_gyro_offset_for_caculate = z_gyro_offset
        z_gyro = 0.0
        rospy.sleep(1)
        currState = States.WALK_WITH_BAR

    elif currState == States.WALK_WITH_BAR:
        print("[WALK_WITH_BAR]")
        robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch","l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],["walking_module"])
        robot.walkStart()
        prev_time = time.time()
        while time.time() - prev_time < 13:
            robot.walkVelocities(x = 4, y = 0, th=np.clip(z_gyro, -10, 10), z_move_amplitude=0.045, balance=True, z_offset=0,hip_pitch=-10)
            print(z_gyro)
        robot.walkStop()
        print('a')
        currState = States.LIFT_BAR
 
    elif currState == States.LIFT_BAR:
        print("[LIFT_BAR]")
        robot.setGeneralControlModule("none")
        rospy.sleep(2)
        robot.setGeneralControlModule("action_module")
        robot.playMotion(179, wait_for_end=True)
        time.sleep(2)
        rospy.sleep(1)
        currState = States.WALK_2_FINISH

    elif currState == States.WALK_2_FINISH:
        print("WALK_2_FINISH")
        robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch","l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],["walking_module"])
        prev_time2 = time.time()
        while time.time() - prev_time2 < 30:
            robot.walkStart()
            robot.walkVelocities(x = 3, y = 0, th=np.clip(z_gyro, -10, 10), z_move_amplitude=0.045, balance=True, z_offset=0,hip_pitch=2)
        robot.walkStop()
        currState = States.END
    elif currState == States.END:
        print("[END]")

        
    rate.sleep()
