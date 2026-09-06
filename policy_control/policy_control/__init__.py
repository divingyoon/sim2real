"""policy_control — obs → policy → fabric → pd 4-node ROS 2 control module.

Every node is `pure core + codec + thin rclpy shell`. Task knowledge lives in
`deploy_contract.json` (built from the training run dump) and robot/sensor wiring
in `config/robots/*.yaml`; no module in this package hard-codes a task constant.
"""
