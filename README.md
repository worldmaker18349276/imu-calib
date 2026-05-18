# IMU-calib

fork from [imu-calib](https://github.com/mikoff/imu-calib), see [IEEE link](https://ieeexplore.ieee.org/document/9133804).

also see [imucal](https://github.com/dmckinnon/imucal), the algorithm is almost the same... [paper link](https://www.researchgate.net/publication/273383944_A_robust_and_easy_to_implement_method_for_IMU_calibration_without_external_equipments).


## Usage

place imu on a stable plane, wait few seconds, take up and put down slowly with different attitude, repeat multiple times (> 10).

prepare imu data in the format `acc_x acc_y acc_z gyr_x gyr_y gyr_z t`, each value are separated by space.

calibrate imu
```bash
python3 calibrate_imu.py --sampling-frequency=100 --imu=data/imu0.log --calib=data/calib0.json --verbose
# or use rosbag
python3 calibrate_imu_ros.py --imu=data/imu.bag --calib=data/calib.json --verbose
```

to generate simulation data
```bash
python3 run_monte_carlo.py
python3 calibrate_imu.py --imu=data/imu_sim.log --imu=data/imu_true_sim.log --calib=data/calib_sim.json --verbose
```
