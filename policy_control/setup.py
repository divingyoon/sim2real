from glob import glob

from setuptools import setup

package_name = "policy_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/config/robots", glob("config/robots/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="obs -> policy -> fabric -> pd control module for RL policies on the real robot.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "obs_node = policy_control.obs_node:main",
            "policy_node = policy_control.policy_node:main",
            "fabric_node = policy_control.fabric_node:main",
            "pd_node = policy_control.pd_node:main",
            "episode_master = policy_control.episode_master:main",
        ],
    },
)
