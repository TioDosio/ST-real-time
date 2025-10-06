#!/usr/bin/env python
import rospy
import actionlib
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler

def stop_robot():
    """
    Stop - cancels ALL goals on the move_base server.
    """
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    
    if client.wait_for_server(rospy.Duration(60.0)):
        # Cancel all goals (including from other clients)
        client.cancel_all_goals()
        rospy.loginfo("All goals cancelled - emergency stop")
        return True
    else:
        rospy.logwarn("move_base action server not available")
        return False

def mean_coordinates(coords, yaw):
    if not coords:
        print("No coordinates provided")
        return (0.0, 0.0)
    
    # Handle nested list structure - coords might be a list of trajectory lists
    # Flatten the coordinates if needed
    flattened_coords = []
    for item in coords:
        if isinstance(item, list) and len(item) > 0:
            if isinstance(item[0], list):
                # This is a trajectory (list of points)
                flattened_coords.extend(item)
            else:
                # This is a single point
                flattened_coords.append(item)
    
    if not flattened_coords:
        print("No valid coordinates found after flattening")
        return (0.0, 0.0)
    
    # Use the last 3 coordinates of the list and remove outliers
    last_coords = flattened_coords[-3:]
    
    # Ensure all coordinates have at least 2 elements (x, y)
    valid_coords = [c for c in last_coords if isinstance(c, list) and len(c) >= 2]
    
    if not valid_coords:
        print("No valid coordinates with x,y values")
        return (0.0, 0.0)
    
    x_mean = sum(c[0] for c in valid_coords) / len(valid_coords)
    y_mean = sum(c[1] for c in valid_coords) / len(valid_coords)

    return (x_mean, y_mean)

def move_robot_to_coordinate(coordinates, yaw, done_callback=None, active_callback=None, feedback_callback=None, allow_preemption=True):
    frame = "odom"
    
    # Handle single coordinate [x, y] or nested structure
    if isinstance(coordinates, list) and len(coordinates) == 2 and isinstance(coordinates[0], (int, float)):
        # Single coordinate [x, y]
        x, y = coordinates
    else:
        # Use mean_coordinates for complex structures (backward compatibility)
        x, y = mean_coordinates(coordinates, yaw)

    try:
        # Initialize node if not already done
        if not rospy.get_node_uri():
            rospy.init_node('robot_navigation', anonymous=True)
            print("Initialized ROS node 'robot_navigation'")
        
        # Create action client
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        
        # Try to connect to server with shorter timeout
        if not client.wait_for_server(rospy.Duration(60.0)):
            rospy.logerr("move_base action server not available")
            return False
        
        # Create goal
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = frame
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        
        # Convert yaw to quaternion
        q = quaternion_from_euler(0, 0, yaw)
        goal.target_pose.pose.orientation.x = q[0] + math.pi
        goal.target_pose.pose.orientation.y = q[1] + math.pi
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        
        # Send goal
        rospy.loginfo("Moving to: x={:.2f}, y={:.2f}, yaw={:.2f} rad".format(x, y, yaw))
        
        if done_callback or active_callback or feedback_callback:
            # Send goal with callbacks for async operation
            client.send_goal(goal, done_callback, active_callback, feedback_callback)
        else:
            # Send goal without callbacks
            client.send_goal(goal)
        
        # Return immediately without waiting
        rospy.loginfo("Goal sent successfully, not waiting for completion")
        return True
            
    except Exception as e:
        rospy.logerr("Error in move_robot_to_coordinate: {}".format(e))
        return False

def test_initial_goal():
    """
    Test 1: Set initial goal
    """
    rospy.loginfo("=== TEST 1: Setting initial goal ===")
    initial_coords = [0.5, 0.5]  # Single coordinate [x, y]
    initial_yaw = 0.0
    
    result = move_robot_to_coordinate(initial_coords, initial_yaw)
    if result:
        rospy.loginfo("Initial goal set successfully")
        return True
    else:
        rospy.logerr("Failed to set initial goal")
        return False

def test_update_goal():
    """
    Test 2: Update goal before reaching the first point
    """
    rospy.loginfo("=== TEST 2: Updating goal before reaching first point ===")
    updated_coords = [1.0, 1.0]  # Single coordinate [x, y]
    updated_yaw = math.pi / 4  # 45 degrees
    
    result = move_robot_to_coordinate(updated_coords, updated_yaw)
    if result:
        rospy.loginfo("Goal updated successfully")
        return True
    else:
        rospy.logerr("Failed to update goal")
        return False

def test_update_goal_again():
    """
    Test 3: Update goal again
    """
    rospy.loginfo("=== TEST 3: Updating goal again ===")
    second_update_coords = [1.5, 1.5]  # Single coordinate [x, y]
    second_update_yaw = -math.pi / 2  # -90 degrees
    
    result = move_robot_to_coordinate(second_update_coords, second_update_yaw)
    if result:
        rospy.loginfo("Goal updated again successfully")
        return True
    else:
        rospy.logerr("Failed to update goal again")
        return False

def test_goal_cancellation():
    """
    Test 4: Cancel all goals (emergency stop)
    """
    rospy.loginfo("=== TEST 4: Testing goal cancellation (emergency stop) ===")
    stop_result = stop_robot()
    if stop_result:
        rospy.loginfo("Emergency stop executed successfully")
        return True
    else:
        rospy.logerr("Failed to execute emergency stop")
        return False

def test_goal_after_cancellation():
    """
    Test 5: Set a final goal after cancellation
    """
    rospy.loginfo("=== TEST 5: Setting new goal after cancellation ===")
    final_coords = [0.0, 0.0]  # Single coordinate [x, y]
    final_yaw = 0.0
    
    result = move_robot_to_coordinate(final_coords, final_yaw)
    if result:
        rospy.loginfo("Final goal set successfully after cancellation")
        return True
    else:
        rospy.logerr("Failed to set final goal")
        return False

def test_custom_goal(x, y, yaw):
    """
    Test with custom coordinates and yaw
    """
    rospy.loginfo("=== CUSTOM TEST: Setting goal at ({}, {}) with yaw {:.2f} ===".format(x, y, yaw))
    custom_coords = [x, y]  # Single coordinate [x, y]
    
    result = move_robot_to_coordinate(custom_coords, yaw)
    if result:
        rospy.loginfo("Custom goal set successfully at ({}, {})".format(x, y))
        return True
    else:
        rospy.logerr("Failed to set custom goal at ({}, {})".format(x, y))
        return False

def wait_for_server():
    """
    Initialize connection to move_base action server
    """
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    
    if not client.wait_for_server(rospy.Duration(60.0)):
        rospy.logerr("move_base action server not available after 60 seconds")
        rospy.logwarn("You can still use the menu, but robot commands will fail")
        return False
    
    rospy.loginfo("move_base action server connected!")
    return True

def check_ros_master():
    """
    Check if ROS master is running
    """
    try:
        rospy.get_master().getSystemState()
        return True
    except Exception as e:
        rospy.logwarn("ROS master not available: {}".format(e))
        return False

def main():
    """
    Main function with interactive menu to run navigation tests on demand.
    """
    server_available = False
    
    try:
        # Initialize ROS node
        rospy.init_node('robot_navigation_test', anonymous=True)
        rospy.loginfo("Starting robot navigation test...")
        
        # Check ROS master first
        if not check_ros_master():
            print("="*60)
            print("WARNING: ROS master is not running!")
            print("To start ROS master, run: roscore")
            print("="*60)
        
        # Try to connect to server (but don't exit if it fails)
        server_available = wait_for_server()
        
        # Interactive menu
        while not rospy.is_shutdown():
            print("\n" + "="*50)
            print("ROBOT NAVIGATION TEST MENU")
            if server_available:
                print("STATUS: ROS and move_base server connected!")
            else:
                print("STATUS: WARNING - ROS/move_base not available!")
            print("="*50)
            print("1. Test initial goal")
            print("2. Test update goal")
            print("3. Test update goal again")
            print("4. Test goal cancellation (emergency stop)")
            print("5. Test goal after cancellation")
            print("6. Test custom goal (you provide coordinates)")
            print("7. Run all tests automatically")
            print("8. Stop robot (emergency stop)")
            print("9. Retry server connection")
            print("0. Exit")
            print("="*50)
            
            try:
                choice = str(input("Enter your choice (0-9): ")).strip()
                
                if choice == '0':
                    print("Exiting...")
                    break
                elif choice == '1':
                    test_initial_goal()
                    rospy.sleep(2.0)  # Wait to see robot start moving
                elif choice == '2':
                    test_update_goal()
                    rospy.sleep(2.0)
                elif choice == '3':
                    test_update_goal_again()
                    rospy.sleep(2.0)
                elif choice == '4':
                    test_goal_cancellation()
                    rospy.sleep(1.0)
                elif choice == '5':
                    test_goal_after_cancellation()
                    rospy.sleep(2.0)
                elif choice == '6':
                    try:
                        x = float(input("Enter X coordinate: "))
                        y = float(input("Enter Y coordinate: "))
                        yaw = float(input("Enter yaw (radians): "))
                        test_custom_goal(x, y, yaw)
                        rospy.sleep(2.0)
                    except ValueError:
                        print("Invalid input. Please enter numeric values.")
                elif choice == '7':
                    print("Running all tests automatically...")
                    run_all_tests()
                elif choice == '8':
                    stop_robot()
                    rospy.sleep(1.0)
                elif choice == '9':
                    print("Retrying server connection...")
                    server_available = wait_for_server()
                else:
                    print("Invalid choice. Please enter a number between 0-9.")
                    
            except (KeyboardInterrupt, EOFError):
                print("\nInterrupted by user")
                break
            except Exception as e:
                print("ERROR: {}".format(e))
                print("Returning to menu...")
                # Continue the loop instead of breaking
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted by ROS")
    except Exception as e:
        rospy.logerr("Error in main: {}".format(e))
    finally:
        # Ensure all goals are cancelled before exit
        stop_robot()
        rospy.loginfo("Test finished - all goals cancelled")

def run_all_tests():
    """
    Run all tests automatically in sequence
    """
    rospy.loginfo("Starting automatic test sequence...")
    
    # Test 1: Set initial goal
    if not test_initial_goal():
        return
    rospy.sleep(2.0)
    
    # Test 2: Update goal
    test_update_goal()
    rospy.sleep(3.0)
    
    # Test 3: Update goal again
    test_update_goal_again()
    rospy.sleep(2.0)
    
    # Test 4: Cancel goals
    test_goal_cancellation()
    rospy.sleep(1.0)
    
    # Test 5: Set goal after cancellation
    test_goal_after_cancellation()
    rospy.sleep(5.0)
    
    # Final cleanup
    rospy.loginfo("=== FINAL: Cancelling all goals before exit ===")
    stop_robot()
    
    rospy.loginfo("All automatic tests completed!")

if __name__ == "__main__":
    main()
