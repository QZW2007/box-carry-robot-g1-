from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, choices=["unitree_g1", "booster_t1", "stanford_toddy"],
                        default="unitree_g1"
                        )
                        
    parser.add_argument("--robot_motion_path", type=str, required=True)

    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, 
                        default="videos/example.mp4")
                        
    args = parser.parse_args()
    
    robot_type = args.robot
    robot_motion_path = args.robot_motion_path
    
    if not os.path.exists(robot_motion_path):
        raise FileNotFoundError(f"Motion file {robot_motion_path} not found")
    
    motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list = load_robot_motion(robot_motion_path)
    
    env = RobotMotionViewer(robot_type=robot_type,
                            motion_fps=motion_fps,
                            record_video=args.record_video, video_path=args.video_path)
    
    for i in tqdm(range(len(motion_root_pos))):
        env.step(motion_root_pos[i], 
                motion_root_rot[i], 
                motion_dof_pos[i], 
                rate_limit=True)
    env.close()