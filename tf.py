#!/usr/bin/env python2
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

def main():
    rospy.init_node('tf_publisher')
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    tf_pub = rospy.Publisher('/combined_tf', TFMessage, queue_size=10)
    rate = rospy.Rate(50)  # 50 Hz

    rospy.loginfo("TF publisher node started")

    while not rospy.is_shutdown():
        transforms = []
        
        try:
            # Look up the transform from base_footprint to odom (robot's position in world)
            # This gives you the robot's pose relative to the odometry frame
            robot_trans = tf_buffer.lookup_transform('odom', 'base_footprint', rospy.Time(0), rospy.Duration(0.1))
            robot_trans.child_frame_id = "frame1"  # Add identifier
            transforms.append(robot_trans)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logdebug("Robot TF lookup error: %s" % str(e))
        except Exception as e:
            rospy.logerr("Unexpected error in robot TF: %s" % str(e))
        
        try:
            # Look up the transform from odom to camera_link (camera's position in world)
            # This gives you the camera's pose relative to the odometry frame
            camera_trans = tf_buffer.lookup_transform('odom', 'base_footprint', rospy.Time(0), rospy.Duration(0.1))
            camera_trans.child_frame_id = "frame2"  # Add identifier
            transforms.append(camera_trans)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logdebug("Camera TF lookup error: %s" % str(e))
        except Exception as e:
            rospy.logerr("Unexpected error in camera TF: %s" % str(e))
        
        # Publish both transforms in a single message if we have any
        if transforms:
            tf_message = TFMessage()
            tf_message.transforms = transforms
            tf_pub.publish(tf_message)
        
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass