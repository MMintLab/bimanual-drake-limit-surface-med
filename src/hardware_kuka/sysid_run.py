
import rospy
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float32
from netft_rdt_driver.srv import Zero
import numpy as np
# have robots already preset to 0 degrees

# goal, get friction coefficient

class GammaYipppee:
    def __init__(self):
        self.thanos_ati_sub = rospy.Subscriber('/netft_thanos/netft_data', WrenchStamped, self.thanos_cb, queue_size=1)
        self.medusa_ati_sub = rospy.Subscriber('/netft_medusa/netft_data', WrenchStamped, self.medusa_cb, queue_size=1)
                
        self.thanos_ati_pub = rospy.Publisher('/netft_thanos/sysid_data', Float32, queue_size=1)
        self.medusa_ati_pub = rospy.Publisher('/netft_medusa/sysid_data', Float32, queue_size=1)
        
        self.time_start = rospy.Time.now()
        
        self.current_medusa_mu = 0.0
        self.current_thanos_mu = 0.0
    def thanos_cb(self, msg: WrenchStamped):
        force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        xy_magnitude = np.linalg.norm(force[:2])
        z = np.abs(force[2])
        
        if np.abs(z) > 1.0:
            self.current_thanos_mu = xy_magnitude / z
        self.thanos_ati_pub.publish(self.current_thanos_mu)
    
    def medusa_cb(self, msg: WrenchStamped):
        force = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
        xy_magnitude = np.linalg.norm(force[:2])
        z = np.abs(force[2])
        
        if np.abs(z) > 1.0:
            self.current_medusa_mu = xy_magnitude / z
        self.medusa_ati_pub.publish(self.current_medusa_mu)
            
if __name__ == '__main__':
    
    # start node
    rospy.init_node("gamma_yipppee")
    gamma = GammaYipppee()
    rospy.spin()
    pass