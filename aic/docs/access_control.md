# Access Control

The evaluation system uses a Zenoh access control list (ACL) to ensure that submissions do not use ROS topics or services which would allow cheating by just looking up the poses of the parts in the simulation.

To replicate the security provided by the submission environment, you can run the environment and your model submission with Zenoh security enabled. This will result in Zenoh blocking access to certain topics and services, such as those in the `gz_server` namespace, to prevent cheating.

## Test with separate terminals

The following demonstration shows the Zenoh access controls in operation. All of these steps are performed automatically by Docker in the submission portal, but they are shown here manually on the command line for interactivity and clarity. These steps can be useful for local testing, to make sure no unauthorized topics or services are being used.

### Terminal 1: Start the Zenoh router with ACL

```
. install/setup.bash
. src/aic/docker/aic_eval/zenoh_config_router.sh
ros2 run rmw_zenoh_cpp rmw_zenohd
```

### Terminal 2: Start the simulation environment

The following commands launch the simulation environment with some entities spawned in it:
```
. install/setup.bash
. src/aic/docker/aic_eval/zenoh_config_eval_session.sh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 launch aic_bringup aic_gz_bringup.launch.py nic_card_mount_0_present:=true sc_port_0_present:=true ground_truth:=false spawn_task_board:=true spawn_cable:=true attach_cable_to_gripper:=true sfp_mount_rail_0_present:=true cable_type:=sfp_sc_cable
```

### Terminal 3: Demonstrate that services are blocked
```
. install/setup.bash
. src/aic/docker/aic_model/zenoh_config_model_session.sh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 service call /gz_server/get_entities_states simulation_interfaces/srv/GetEntitiesStates
```

This call will not succeed, because it is blocked by the ACL.

If, instead, this terminal has the environment variables associated with the `eval` identity, it will return a list of the poses and velocities of all entities in the simulation:
```
. install/setup.bash
. src/aic/docker/aic_eval/zenoh_config_eval_session.sh
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 service call /gz_server/get_entities_states simulation_interfaces/srv/GetEntitiesStates
```

Note that the `eval` identity is protected by a password, which will be different when it is running in the submission portal :smile:

## Test with docker-compose

Docker-compose provides a convenient way to test the interactions between the evaluation container and the model containers:

First, build the containers:
```
docker compose -f docker/docker-compose.yaml build
```

Next, run them:
```
docker copose -f docker/docker-compose.yaml up
```
