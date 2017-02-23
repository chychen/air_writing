################################################################################
# Copyright (C) 2012-2016 Leap Motion, Inc. All rights reserved.               #
# Leap Motion proprietary and confidential. Not for distribution.              #
# Use subject to the terms of the Leap Motion SDK Agreement available at       #
# https://developer.leapmotion.com/sdk_agreement, or another agreement         #
# between Leap Motion and you, your company or other organization.             #
################################################################################
import os, sys, inspect, thread, time
import msvcrt
from enum import Enum
src_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
# Windows and Linux
arch_dir = '../libs/x64' if sys.maxsize > 2**32 else '../libs/x86'
collected_data_dir = 'data_collection/data/'
# Mac
#arch_dir = os.path.abspath(os.path.join(src_dir, '../lib'))
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, arch_dir)))
sys.path.insert(0, os.path.abspath(os.path.join(src_dir, "../libs")))
import Leap

Mode = Enum('Mode', 'START_RECORD END_RECORD DEFAULT_PROCESS')

class UserInterface(object):

    def __init__(self, mode):
        self.MODE = mode
    
    def getMode(self):
        return self.MODE

    def setMode(self, mode):
        self.MODE = mode

class SampleListener(Leap.Listener):
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    bone_names = ['Metacarpal', 'Proximal', 'Intermediate', 'Distal']
    fileName = []
    processMode = []

    def setFileName(self, fileName, processMode):
        self.fileName = fileName
        self.processMode = processMode
    
    @classmethod
    def on_init(cls, controller):
        print "Initialized"

    @classmethod
    def on_connect(cls, controller):
        print "Connected"

    @classmethod
    def on_disconnect(cls, controller):
        # Note: not dispatched when running in a debugger.
        print "Disconnected"

    @classmethod
    def on_exit(cls, controller):
        print "Exited"

    def on_frame(self, controller):
        # Get the most recent frame and report some basic information
        frame = controller.frame()

        # print "Frame id: %d, timestamp: %d, hands: %d, fingers: %d" % (
        #     frame.id, frame.timestamp, len(frame.hands), len(frame.fingers))

        # Get hands
        for hand in frame.hands:
            for finger in hand.fingers:
                f = open(self.fileName, 'a')
                if self.finger_names[finger.type] == "Index":
                    f.write("    %s finger, id: %d, length: %fmm, width: %fmm" % (
                        self.finger_names[finger.type],
                        finger.id,
                        finger.length,
                        finger.width))
                    print "    %s finger, id: %d, length: %fmm, width: %fmm" % (
                        self.finger_names[finger.type],
                        finger.id,
                        finger.length,
                        finger.width)

                    # Get bones
                    for b in range(0, 4):
                        bone = finger.bone(b)
                        f.write("      Bone: %s, start: %s, end: %s, direction: %s" % (
                            self.bone_names[bone.type],
                            bone.prev_joint,
                            bone.next_joint,
                            bone.direction))
                        print "      Bone: %s, start: %s, end: %s, direction: %s" % (
                            self.bone_names[bone.type],
                            bone.prev_joint,
                            bone.next_joint,
                            bone.direction)
                    
                f.close()

            # handType = "Left hand" if hand.is_left else "Right hand"

            # print "  %s, id %d, position: %s" % (
            #     handType, hand.id, hand.palm_position)

            # # Get the hand's normal vector and direction
            # normal = hand.palm_normal
            # direction = hand.direction

            # # Calculate the hand's pitch, roll, and yaw angles
            # print "  pitch: %f degrees, roll: %f degrees, yaw: %f degrees" % (
            #     direction.pitch * Leap.RAD_TO_DEG,
            #     normal.roll * Leap.RAD_TO_DEG,
            #     direction.yaw * Leap.RAD_TO_DEG)

            # # Get arm bone
            # arm = hand.arm
            # print "  Arm direction: %s, wrist position: %s, elbow position: %s" % (
            #     arm.direction,
            #     arm.wrist_position,
            #     arm.elbow_position)

            # # Get fingers
            # for finger in hand.fingers:

            #     print "    %s finger, id: %d, length: %fmm, width: %fmm" % (
            #         self.finger_names[finger.type],
            #         finger.id,
            #         finger.length,
            #         finger.width)

            #     # Get bones
            #     for b in range(0, 4):
            #         bone = finger.bone(b)
            #         print "      Bone: %s, start: %s, end: %s, direction: %s" % (
            #             self.bone_names[bone.type],
            #             bone.prev_joint,
            #             bone.next_joint,
            #             bone.direction)

        if not frame.hands.is_empty:
            print ""


def main():
    if not os.path.exists(collected_data_dir):
        os.makedirs(collected_data_dir)
    userName = raw_input("enter your name first: ")
    fileName = collected_data_dir + userName + ".txt"
    # Empty the file
    f = open(fileName, 'w')
    f.close()
    print "Hi %s!" % (userName)
    print "Instructions:"
    print "1. put your finger at starting position"
    print "2. press character 's'"
    print "3. start writing in the air and freeze at the end position"
    print "4. press character 'e'"
    print "5. loop over to step 1 (or just take a break, dont rush!)"
    print "6. press 'space' after finishing all data collection {A-Z}{a-z}{0-9}"

    # Create a sample listener and controller
    listener = SampleListener()
    processMode = UserInterface(Mode.DEFAULT_PROCESS)
    listener.setFileName(fileName, processMode)
    controller = Leap.Controller()
    
    
    while True:
        getchar = msvcrt.getch()
        if getchar == " ":
            controller.remove_listener(listener)
            break
        if getchar == "s": # start record
            processMode.setMode(Mode.START_RECORD)
            # Have the sample listener receive events from the controller
            controller.add_listener(listener)
        if getchar == "e": # end record
            processMode.setMode(Mode.END_RECORD)
            controller.remove_listener(listener)

if __name__ == "__main__":
    main()
