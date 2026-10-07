import pathlib

HERE = pathlib.Path(__file__).parent
IK_CONFIG_ROOT = HERE / "ik_configs"
ASSET_ROOT = HERE / ".." / "assets"

ROBOT_XML_DICT = {
    "unitree_g1": ASSET_ROOT / "unitree_g1" / "g1_mocap_29dof.xml",
    "booster_t1": ASSET_ROOT / "booster_t1" / "t1_mocap.xml",
    "stanford_toddy": ASSET_ROOT / "stanford_toddy" / "toddy_mocap.xml",
    "fourier_n1": ASSET_ROOT / "fourier_n1" / "n1_mocap.xml",
    "engineai_pm01": ASSET_ROOT / "engineai_pm01" / "pm_v2.xml",
}

IK_CONFIG_DICT = {
    # offline data
    "smplx":{
        "unitree_g1": IK_CONFIG_ROOT / "smplx_to_g1.json",
        "booster_t1": IK_CONFIG_ROOT / "smplx_to_t1.json",
        "stanford_toddy": IK_CONFIG_ROOT / "smplx_to_toddy.json",
        "fourier_n1": IK_CONFIG_ROOT / "smplx_to_n1.json",
        "engineai_pm01": IK_CONFIG_ROOT / "smplx_to_pm01.json",
    },
    "bvh":{
        "unitree_g1": IK_CONFIG_ROOT / "bvh_to_g1.json",
        "booster_t1": IK_CONFIG_ROOT / "bvh_to_t1.json",
        "fourier_n1": IK_CONFIG_ROOT / "bvh_to_n1.json",
        "stanford_toddy": IK_CONFIG_ROOT / "bvh_to_toddy.json",
        "engineai_pm01": IK_CONFIG_ROOT / "bvh_to_pm01.json",
    },
}


ROBOT_BASE_DICT = {
    "unitree_g1": "pelvis",
    "unitree_g1_with_hands": "pelvis",
    "booster_t1": "Waist",
    "stanford_toddy": "waist_link",
    "fourier_n1": "base_link",
    "engineai_pm01": "LINK_BASE",
    # Hand models use hand palm as base
    "dex31_left_hand": "left_hand_palm",
    "dex31_right_hand": "right_hand_palm",
}

VIEWER_CAM_DISTANCE_DICT = {
    "unitree_g1": 2.0,
    "unitree_g1_with_hands": 2.0,
    "booster_t1": 2.0,
    "stanford_toddy": 1.0,
    "fourier_n1": 2.0,
    "engineai_pm01": 2.0,
    # Hand models need closer camera for detailed view
    "dex31_left_hand": 0.5,
    "dex31_right_hand": 0.5,
}