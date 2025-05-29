import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import json
import subprocess

# PyBullet başlat
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1/240)  # 240 Hz, stabilite için

# Düzlem ve robot kolu yükle
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

# Küp URDF
cube_urdf = """
<robot name="custom_cube">
    <link name="cube">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.1"/>
            <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="0.05 0.05 0.05"/>
            </geometry>
            <material name="red">
                <color rgba="1 0 0 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="0.05 0.05 0.05"/>
            </geometry>
        </collision>
    </link>
</robot>
"""
with open("custom_cube.urdf", "w") as f:
    f.write(cube_urdf)
cubeStartPos = [0.2, 0, 0.025]  # Küp masada (z=0.025, yüzeyde)
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
cubeId = p.loadURDF("custom_cube.urdf", cubeStartPos, cubeStartOrientation)
p.changeDynamics(cubeId, -1, mass=0.1, lateralFriction=0.5, spinningFriction=0.01, rollingFriction=0.01, restitution=0.1)
print("Küp yüklendi")

# Kamera ayarları
width, height = 1280, 720
fov = 80
aspect = width / height
near, far = 0.1, 100
cameraTargetPos = [0.2, 0, 0.3]
cameraDistance = 1.5
cameraYaw = 45
cameraPitch = -30
viewMatrix = p.computeViewMatrixFromYawPitchRoll(
    cameraTargetPos, cameraDistance, cameraYaw, cameraPitch, roll=0, upAxisIndex=2)
projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Video yazıcı
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('pybullet_magnetic_cube.mp4', fourcc, 30, (width, height))

# Robot hareketi ayarları
num_joints = p.getNumJoints(robotId)
total_steps = 600
substeps_per_action = 50
joint_limits = [
    (-2.967, 2.967), (-2.094, 2.094), (-2.967, 2.967), (-2.{Geçersiz karakter bulundu: '{'.}094, 2.094),
    (-2.967, 2.967), (-2.094, 2.094), (-3.054, 3.054)
]

def get_random_target_positions():
    return [np.random.uniform(low, high) for low, high in joint_limits]

def get_target_to_cube(cube_pos):
    """Ters kinematik ile küpe yaklaşma."""
    target_pos = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.01]  # Küpün biraz üstü
    target_orn = p.getQuaternionFromEuler([0, np.pi, 0])  # Tutucu aşağı bakar
    try:
        joint_angles = p.calculateInverseKinematics(
            robotId, num_joints - 1, target_pos, target_orn,
            lowerLimits=[low for low, _ in joint_limits],
            upperLimits=[high for _, high in joint_limits],
            jointRanges=[high - low for low, high in joint_limits],
            restPoses=[0.0] * num_joints,
            maxNumIterations=2000,  # Daha iyi yakınsama için artırıldı
            residualThreshold=1e-7  # Daha sıkı eşik
        )
        return list(joint_angles)[:num_joints]
    except Exception as e:
        print(f"IK hatası: {e}")
        return [0.0] * num_joints

# Mıknatıs kontrol
magnet_on = False
attach_constraint = None
pick_step = 100  # Küpü alma adımı
release_step = 400  # Küpü bırakma adımı
distance_threshold = 0.015  # Stabilite için

# Küpün yapıştığını kontrol için
def check_attachment():
    if magnet_on and attach_constraint is not None:
        cube_pos, _ = p.getBasePositionAndOrientation(cubeId)
        effector_state = p.getLinkState(robotId, num_joints - 1)
        eff_pos = list(effector_state[0])
        dist = np.linalg.norm(np.array(eff_pos) - np.array(cube_pos))
        return dist < 0.02  # Küpün efektöre yakın olup olmadığını kontrol et
    return False

# Veri kaydı
data = []

# Ana simülasyon döngüsü
for step in range(total_steps):
    # Küpün mevcut pozisyonunu al
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cubeId)

    # Hedef eklem pozisyonları
    if step < pick_step:
        target_positions = get_target_to_cube(cube_pos)
    else:
        target_positions = get_random_target_positions()

    # Eklem hareketlerini uygula (yumuşatma ile)
    current_positions = [p.getJointState(robotId, i)[0] for i in range(num_joints)]
    for joint_index in range(num_joints):
        interpolated_pos = current_positions[joint_index] + 0.1 * (target_positions[joint_index] - current_positions[joint_index])
        p.setJointMotorControl2(
            robotId, joint_index, p.POSITION_CONTROL,
            targetPosition=interpolated_pos,
            force=15000,  # Daha güçlü kuvvet
            maxVelocity=0.3  # Hız artırıldı
        )

    # Simülasyonu ilerlet
    for _ in range(substeps_per_action):
        p.stepSimulation()
        time.sleep(1/2400)  # 240 Hz için

    # Uç efektör ve küp durumunu al
    end_effector_link = num_joints - 1
    state = p.getLinkState(robotId, end_effector_link)
    eff_pos = list(state[0])
    eff_orn = list(p.getEulerFromQuaternion(state[1]))
    cube_pos, cube_orn = p.getBasePositionAndOrientation(cubeId)
    cube_pos = list(cube_pos)
    cube_orn = list(p.getEulerFromQuaternion(cube_orn))
    dist = np.linalg.norm(np.array(eff_pos) - np.array(cube_pos))

    # Küpü al
    if step == pick_step and dist < distance_threshold and not magnet_on:
        p.resetBaseVelocity(cubeId, [0, 0, 0], [0, 0, 0])
        try:
            # Küpü uç efektöre bağla
            attach_constraint = p.createConstraint(
                parentBodyUniqueId=robotId,
                parentLinkIndex=end_effector_link,
                childBodyUniqueId=cubeId,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, -0.01],  # Uç efektörün altında
                childFramePosition=[0, 0, 0],
                parentFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )
            p.changeConstraint(attach_constraint, maxForce=1000000)
            magnet_on = True
            print(f"Adım {step}: Küp yapıştı! Mesafe: {dist:.4f}, Efektör: {eff_pos}, Küp: {cube_pos}")
        except Exception as e:
            print(f"Kısıt oluşturma hatası: {e}")

    # Küpü bırak
    elif step == release_step and magnet_on:
        try:
            p.removeConstraint(attach_constraint)
            attach_constraint = None
            magnet_on = False
            print(f"Adım {step}: Küp bırakıldı. Küp pozisyonu: {cube_pos}")
        except Exception as e:
            print(f"Kısıt kaldırma hatası: {e}")

    # Kısıt kontrolü
    if magnet_on and step > pick_step:
        if not check_attachment():
            print(f"Adım {step}: HATA! Küp yapışık değil! Mesafe: {dist:.4f}, Efektör: {eff_pos}, Küp: {cube_pos}")
            # Kısıtı yeniden oluştur
            try:
                p.removeConstraint(attach_constraint)
                attach_constraint = p.createConstraint(
                    parentBodyUniqueId=robotId,
                    parentLinkIndex=end_effector_link,
                    childBodyUniqueId=cubeId,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, -0.01],
                    childFramePosition=[0, 0, 0],
                    parentFrameOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                    childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
                )
                p.changeConstraint(attach_constraint, maxForce=1000000)
                print(f"Adım {step}: Kısıt yeniden oluşturuldu.")
            except Exception as e:
                print(f"Yeniden kısıt oluşturma hatası: {e}")

    # Hata ayıklama
    if step % 50 == 0 or dist < distance_threshold or (magnet_on and step > pick_step):
        print(f"Adım {step}: Efektör: {eff_pos}, Küp: {cube_pos}, Mıknatıs: {magnet_on}, Mesafe: {dist:.4f}")

    # Veri kaydet
    joint_states = [p.getJointState(robotId, i)[0] for i in range(num_joints)]
    state = {
        'step': step,
        'joint_config': joint_states,
        'effector_pos': eff_pos,
        'effector_rot': eff_orn,
        'cube_pos': cube_pos,
        'cube_rot': cube_orn,
        'magnet_state': int(magnet_on),
        'cube_color': [1.0, 0.0, 0.0]  # Kırmızı, URDF'den
    }
    data.append(state)

    # Video kaydet
    img_arr = p.getCameraImage(width, height, viewMatrix, projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.array(img_arr[2], dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    video.write(frame)

    cv2.imshow('PyBullet Manyetik Küp Simülasyonu', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
video.release()
cv2.destroyAllWindows()
p.disconnect()

# Veriyi kaydet
with open('simulation_data.json', 'w') as f:
    json.dump(data, f, indent=2)

# Videoyu aç
subprocess.run(["open", "pybullet_magnetic_cube.mp4"])  # Windows için: ["start", "pybullet_magnetic_cube.mp4"]
print("Simülasyon tamamlandı!")
