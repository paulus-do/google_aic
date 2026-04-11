# Creating a Custom Dockerfile (Advanced)

The example dockerfile assumes that you are using `aic_model` to run your policy. If you are not using `aic_model`, you need to create a custom dockerfile.

This document assumes you have advanced knowledge of ROS 2 middleware concepts, Zenoh and Docker.

## 1. ROS 2 and rmw_zenoh_cpp

A policy node is essentially a ROS 2 node that subscribes to observations and publishes actions to be executed.
For convenience, the example policy use a Python ROS node named `aic_model` which instantiates a `Policy` class that implements the actual policy, to factor out as much boilerplate as possible.

ROS 2 is middleware agnostic, but for the AI for Industry Challenge, `rmw_zenoh_cpp` is used exclusively. Your dockerfile **must** run your policy node with `rmw_zenoh_cpp`. This is usually done by setting the `RMW_IMPLEMENTATION` environment variable to `rmw_zenoh_cpp`.

This is automatically set when running your image, you must ensure that your dockerfile do not override it and it also must have `rmw_zenoh_cpp` available.

## 2. Zenoh

During evaluation, your image will be ran with the following environment variables:

| Variable                    | Description                                                                                                     |
| :-------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| RMW_IMPLEMENTATION          | ROS 2 middleware to use. Always set to `rmw_zenoh_cpp`                                                          |
| ZENOH_ROUTER_CHECK_ATTEMPTS | Always set to `-1`, prevents your container from erroring out when it launches before the model router is ready |
| AIC_MODEL_ROUTER_ADDR       | Zenoh router address that your policy node must connect to                                                      |
| AIC_MODEL_PASSWD            | Password that you must use to identify your policy node                                                         |

Your policy node **must**:

1. Connect to the zenoh router given in `AIC_MODEL_ROUTER_ADDR` environment variable.
2. Use user-password authentication with the user `model` and password given in `AIC_MODEL_PASSWD`.

The easiest way is to set `ZENOH_CONFIG_OVERRIDE` to something like:

```bash
ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/'"$AIC_MODEL_ROUTER_ADDR"'"];transport/auth/usrpwd/user="model";transport/auth/usrpwd/password="'"$AIC_MODEL_PASSWD"'";transport/auth/usrpwd/dictionary_file="/credentials.txt"'
```

You also need to create the credentials file, you can use something like:

```bash
echo "model:$AIC_MODEL_PASSWD" >> /credentials.txt
```

See https://github.com/ros2/rmw_zenoh and https://zenoh.io/docs/manual/access-control/ for more information.

## 3. Entrypoint

Your image's entrypoint must start your policy node. No additional command line arguments will be provided.
