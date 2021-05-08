from scipy.spatial.transform import Rotation as R

# transform between quaternion, euler, rotation_matrix

def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler

def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat()
    return quaternion

def euler2rotation(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    rotation_matrix = r.as_dcm()
    return rotation_matrix