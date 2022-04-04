# DoubleBallBalancer

The double ball balancer is a robotic system consisting of a ball with a movable internal weight, which enables it to balance on top of another (unactuated) ball.

### Try out the interactive demo

The interactive demo allows controlling the robot through the keyboard in the [pybullet](https://pybullet.org) simulation environment.

To run the scripts in the `scripts` folder, install python3, and then install the requirements:

`python3 -m pip install -r requirements.txt`

Then, run the interactive demo:

`python3 scripts/pybullet_interactive_demo.py`

[![Watch the video](https://user-images.githubusercontent.com/4960007/161436128-bbe408ba-cfad-409d-8ecd-9b52a6a01ae4.gif)](https://youtu.be/SmjYLHc5eRc)

### Angle control demo

4 rotations of the upper ball (output of `python3 scripts/example_3d.py`)

![Angle control](https://github.com/ChristofDubs/DoubleBallBalancer/blob/master/doc/img/3d_demo.gif)

### Angular velocity control / error correction demo

Upper ball at an angular velocity of 1.5rad/s while correcting an initial lever arm angle of 90 degrees

![velocity control with initial deflection](https://user-images.githubusercontent.com/4960007/158054738-98cc35e8-96ce-41fa-ae60-0e1d14d28524.gif)