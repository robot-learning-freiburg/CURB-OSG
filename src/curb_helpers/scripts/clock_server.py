import rospy
import time
from rosgraph_msgs.msg import Clock


def run_clock_server():
    clock_pub = rospy.Publisher("/clock", Clock, queue_size=1)
    rate = 1000
    clock_multiplier = float(rospy.get_param("~clock_multiplier"))
    clock_msg = Clock()
    t0 = time.time()
    # rospy.loginfo(
    #     f'use_sim_time {rospy.get_param("/use_sim_time")} 2 {rospy.get_param("use_sim_time")} 3 {rospy.get_param("~use_sim_time")}'
    # )
    rospy.loginfo(f"clock server started, time multiplier: {clock_multiplier}")

    while not rospy.is_shutdown():
        clock_msg.clock = rospy.Time.from_sec(
            clock_multiplier * (time.time() - t0)
        )
        clock_pub.publish(clock_msg)
        time.sleep(1/rate)


if __name__ == "__main__":
    rospy.init_node("clock_server", anonymous=True)
    # rospy.set_param("~clock_multiplier", 0.5)
    run_clock_server()
