import pybullet as p
import pybullet_data
import time
import cv2
import numpy as np

# PyBullet başlat (GUI modunda)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Yerçekimi ve zemin
p.setGravity(0, 0, -9.81)
plane_id = p.loadURDF("plane.urdf")

# KUKA robotunu yükle
robot_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)

# Küpü yükle (başlangıç noktası)
cube_start_pos = [0.6, 0, 0.02]
cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, globalScaling=1)

joint_indices = [
    i for i in range(p.getNumJoints(robot_id))
    if p.getJointInfo(robot_id, i)[2] == p.JOINT_REVOLUTE
]

def move_to_position(position, steps=240):
    joint_angles = p.calculateInverseKinematics(robot_id, 6, position)
    for i, angle in zip(joint_indices, joint_angles):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, angle)
    for _ in range(steps):
        p.stepSimulation()
        capture_frame()  # Her adımda kare kaydet
        time.sleep(1. / 240.)

# Video için parametreler
frames = []
def capture_frame():
    width, height = 640, 480
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[1.1, 0, 0.6],
        cameraTargetPosition=[0.5, 0, 0.1],
        cameraUpVector=[0, 0, 1])
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=width/height, nearVal=0.1, farVal=2.0)
    img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix)
    frame = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
    frames.append(frame)

# 1. Küpe yaklaş (üstten)
move_to_position([cube_start_pos[0], cube_start_pos[1], 0.1])

# 2. Küpü kavra (küpü kola sabitle)
cube_cid = p.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=6,
    childBodyUniqueId=cube_id,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0.02],
    childFramePosition=[0, 0, 0]
)

# 3. Küp ile birlikte yukarı kalk
move_to_position([cube_start_pos[0], cube_start_pos[1], 0.3])

# 4. Küpü yeni hedefe taşı (örneğin [0.4, 0.2, 0.3])
cube_target_pos = [0.4, 0.2, 0.3]
move_to_position(cube_target_pos)

# 5. Küpü aşağı indir (hedef konuma)
move_to_position([cube_target_pos[0], cube_target_pos[1], 0.1])

# 6. Küpü bırak
p.removeConstraint(cube_cid)

# 7. Robot kolunu yukarı çek
move_to_position([cube_target_pos[0], cube_target_pos[1], 0.3])

# PyBullet kapat
p.disconnect()

# OpenCV ile video olarak kaydet
out = cv2.VideoWriter('pybullet_simulation.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (640, 480))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
print("Video kaydedildi: pybullet_simulation.avi")

