# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation and Setup

This is a Python package for motion retargeting to humanoid robots. Install in development mode:

```bash
conda create -n gmr python=3.10 -y
conda activate gmr
pip install -e .
conda install -c conda-forge libstdcxx-ng -y
```

## Code Architecture

### Core Components

- **`GeneralMotionRetargeting`** (`general_motion_retargeting/motion_retarget.py`): Main class for motion retargeting using inverse kinematics (IK) solver built on mink/mujoco
- **`KinematicsModel`** (`general_motion_retargeting/kinematics_model.py`): Handles robot kinematics calculations
- **`RobotMotionViewer`** (`general_motion_retargeting/robot_motion_viewer.py`): MuJoCo-based visualization for robot motions
- **Configuration System** (`general_motion_retargeting/params.py`): Simplified robot definitions and IK config mappings - cleaned to focus on core supported robots

### Data Flow

1. **Human Motion Input**: SMPL-X (AMASS/OMOMO) or BVH (LAFAN1) format
2. **Motion Format**: Each frame = dict of (human_body_name, 3D translation + rotation)
3. **Robot Output**: Tuple of (base_translation, base_rotation, joint_positions)
4. **IK Configs**: JSON files in `general_motion_retargeting/ik_configs/` define human-to-robot body mappings

### Supported Robots

Core robot models in `assets/` directory:
- Unitree G1 (`unitree_g1`)
- Booster T1 (`booster_t1`)
- Stanford ToddlerBot (`stanford_toddy`) 
- Fourier N1 (`fourier_n1`)
- ENGINEAI PM01 (`engineai_pm01`)

Additional models retained in ROBOT_BASE_DICT for compatibility:
- `unitree_g1_with_hands`, `dex31_left_hand`, `dex31_right_hand`

## Common Commands

### Single Motion Retargeting
```bash
# SMPL-X to robot
python scripts/smplx_to_robot.py --smplx_file <path> --robot <robot_name> --save_path <output.pkl>

# BVH to robot  
python scripts/bvh_to_robot.py --bvh_file <path> --robot <robot_name> --save_path <output.pkl>
```

### Batch Processing
```bash
# Process datasets
python scripts/smplx_to_robot_dataset.py
python scripts/bvh_to_robot_dataset.py
```

### Visualization
```bash
# Visualize saved robot motion
python scripts/vis_robot_motion.py --robot <robot_name> --robot_motion_path <path.pkl>
```

Add `--record_video --video_path <output.mp4>` to any visualization command to record video.

## Key Technical Details

- **IK Solver**: Uses mink library with configurable solver (default: "daqp") and damping (default: 5e-1)
- **Human Height Scaling**: Automatic scaling based on `actual_human_height` parameter vs config assumptions
- **Real-time Performance**: Optimized for 60-70 FPS on high-end CPUs for teleoperation use cases
- **Body Model Dependencies**: Requires SMPL-X body models in `assets/body_models/smplx/`

## File Organization

- `scripts/`: Entry point scripts for different retargeting workflows
- `general_motion_retargeting/`: Core library code
- `assets/`: Robot models (MuJoCo XML) and body models (SMPL-X)
- `general_motion_retargeting/ik_configs/`: Cleaned JSON configuration files - simplified to core robot mappings:
  - SMPL-X configs: `smplx_to_{g1,t1,toddy,n1,pm01}.json`
  - BVH configs: `bvh_to_{g1,t1,toddy,n1,pm01}.json`