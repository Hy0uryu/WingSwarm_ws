# WingSwarm
A algorithm for drone swarm to intercept a known moving target.

test on the ubuntu 18.04, ROS melodic.

## Easy Start
```
cd WingSwarm
catkin_make
source devel/setup.bash
roslaunch planning test.launch

<!- another Terminal -/>
./sh_utils/pub_triger.sh
```
## Some important Params in launch file（test.launch）
**replan params:**
replan: wether to use replan
plan_durantion: if replan, The interval for replan
close_dist: the distance to end replan

**constraints params:**
vmax, vmin: max and min of the velocity scaled by 100 times
amax, amin: max and min of the acceleration scaled by 100 times
dSwarmMin: minimum distance between two drones
omegamax: maximum angular velocity

rhoX: weight of corresponding cost

**optimizer monitor:**
monitorUse: turning it to true will allow to show the optimization progress.
pausems: iterval between iterations.

## Others
to view the swarm data:
    subscribe to the ROS topics: /sim_node/VACX (X is 0,1,2, for example: /sim_node/VAC0)




