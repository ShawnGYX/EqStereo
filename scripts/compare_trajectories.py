#!/usr/bin/env python3
from mypylie.pylie import SE3
from mypylie.pylie import analysis
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description="Compare estimated and groundtruth trajectories.")
parser.add_argument("est_poses", metavar='e', type=str, help="The file containing estimated poses.")
parser.add_argument("gt_poses", metavar='g', type=str, help="The file containing groundtruth poses.")
parser.add_argument("--eformat", type=str, default="xq", help="The format of estimated poses. Default xq")
parser.add_argument("--gformat", type=str, default="xw", help="The format of groundtruth poses. Default xw")
parser.add_argument("--num_frames", type=int, default=100, help="The number of frames used to match poses. Use -1 for full alignment. Default 100.")

args = parser.parse_args()


def readTimedPoses(fname : str, format_spec : str):
    with open(fname, 'r') as file:
        reader = csv.reader(file)
        # Skip header
        next(reader)
        times = []
        poses = []
        for line in reader:
            times.append(float(line[0]))
            poses.append(SE3.from_list(line[1:], format_spec))
    return times, poses

def changePoseTimes(new_times : list, times : list, poses : list):
    # Compute the poses at new_times rather than times
    new_poses = []
    for t in new_times:
        try:
            old_time_index = next(i for i, ti in enumerate(times) if ti >= t)
            new_poses.append(poses[old_time_index])
        except StopIteration:
            new_poses.append( poses[-1] )
    return new_poses

def computeStatistics(values : list) -> dict:
    # Compute the statistics (mean, std, median, min, max) from a tuple of values
    stats = {}
    stats["mean"] = np.mean(values)
    stats["std"] = np.std(values)
    stats["med"] = np.median(values)
    stats["min"] = np.min(values)
    stats["max"] = np.max(values)

    return stats

def statString(stats : dict):
    result = ""
    for key, val in stats.items():
        result += "{:>6s}: {:<.4f}\n".format(key,val)
    return result

def plotStatLines(ax, times, stats):
    tdiff = [times[0], times[-1]]
    ax.plot(tdiff, 2*[stats["mean"]], 'r--')
    ax.plot(tdiff, 2*[stats["mean"]+stats["std"]], 'g--')
    ax.plot(tdiff, 2*[stats["mean"]-stats["std"]], 'g--')
    ax.plot(tdiff, 2*[stats["med"]], 'b--')

print("Reading the estimated poses...")
etimes, eposes = readTimedPoses(args.est_poses, args.eformat)
print("Reading the ground truth poses...")
gtimes, gposes = readTimedPoses(args.gt_poses, args.gformat)
print("Matching the ground truth to the estimated...")
gposes = changePoseTimes(etimes, gtimes, gposes)

if args.num_frames >= 0:
    frame_change_n_times = min(args.num_frames, len(etimes))
    print("Using the first {} positions to align the trajectories.".format(frame_change_n_times))
else:
    frame_change_n_times = len(etimes)
    print("Using all positions to align the trajectories.")
e_positions = np.hstack([pose._x._trans for pose in eposes])
g_positions = np.hstack([pose._x._trans for pose in gposes])
frame_change = analysis.umeyama(e_positions[:,:frame_change_n_times], g_positions[:,:frame_change_n_times])
frame_change = frame_change.to_SE3().inv()
aposes = [(frame_change * pose) for pose in gposes]

n_poses = len(eposes)
rel_eposes = [eposes[i].inv() * eposes[i+1] for i in range(n_poses-1)]
rel_aposes = [aposes[i].inv() * aposes[i+1] for i in range(n_poses-1)]
rel_etrans = np.hstack([rel_pose._x._trans for rel_pose in rel_eposes])
rel_gtrans = np.hstack([rel_pose._x._trans for rel_pose in rel_aposes])

rel_erot = np.hstack([rel_pose._R.log() for rel_pose in rel_eposes])
rel_grot = np.hstack([rel_pose._R.log() for rel_pose in rel_aposes])

erot_eul = np.hstack([np.reshape(pose._R._rot.as_euler('xyz'), (3,1)) for pose in eposes])
grot_eul = np.hstack([np.reshape(pose._R._rot.as_euler('xyz'), (3,1)) for pose in aposes])
etrans = np.hstack([np.reshape(pose._x._trans, (3,1)) for pose in eposes])
gtrans = np.hstack([np.reshape(pose._x._trans, (3,1)) for pose in aposes])


# Gather statistics
# Statistics are: mean, variance, median, min, max

# Frame-to-frame errors are position change and attitude change
relative_position_error = np.linalg.norm(rel_gtrans - rel_etrans, axis=0)
relative_position_error_stats = computeStatistics(relative_position_error)
print()
print("Relative position error (m) stats:")
print(statString(relative_position_error_stats))

rel_err_rot = np.hstack([(rgpose._R.inv() * repose._R).log() for (rgpose, repose) in zip(rel_aposes, rel_eposes)])
relative_attitude_error = np.linalg.norm(rel_err_rot, axis=0) * (180.0 / np.pi)
relative_attitude_error_stats = computeStatistics(relative_attitude_error)
print("Relative attitude error (deg) stats:")
print(statString(relative_attitude_error_stats))

# Global errors are position and attitude
absolute_position_error = np.linalg.norm(gtrans - etrans, axis=0)
absolute_position_error_stats = computeStatistics(absolute_position_error)
print("Absolute position error (m) stats:")
print(statString(absolute_position_error_stats))

abs_err_rot = np.hstack([(gpose._R.inv() * epose._R).log() for (gpose, epose) in zip(aposes, eposes)])
absolute_attitude_error = np.linalg.norm(abs_err_rot, axis=0) * (180.0 / np.pi)
absolute_attitude_error_stats = computeStatistics(absolute_attitude_error)
print("Absolute attitude error (deg) stats:")
print(statString(absolute_attitude_error_stats))


# Plot the relative and absolute errors over time
fig, ax = plt.subplots(2,2)
error_line_width = 0.5
ax[0,0].plot(etimes[:-1], relative_position_error, 'm-', linewidth=error_line_width)
plotStatLines(ax[0,0], etimes, relative_position_error_stats)
ax[1,0].plot(etimes[:-1], relative_attitude_error, 'm-', linewidth=error_line_width)
plotStatLines(ax[1,0], etimes, relative_attitude_error_stats)

ax[0,1].plot(etimes, absolute_position_error, 'm-', linewidth=error_line_width)
plotStatLines(ax[0,1], etimes, absolute_position_error_stats)
ax[1,1].plot(etimes, absolute_attitude_error, 'm-', linewidth=error_line_width)
plotStatLines(ax[1,1], etimes, absolute_attitude_error_stats)

ax[0,0].set_title("Relative Position Error (m)")
ax[1,0].set_title("Relative Attitude Error (deg)")
ax[0,1].set_title("Absolute Position Error (m)")
ax[1,1].set_title("Absolute Attitude Error (deg)")
for a in ax.ravel():
    a.set_xlim([etimes[0], etimes[-1]])
    a.set_ylim([0, None])


# Plot the relative translation and rotation
fig, ax = plt.subplots(3,2)
ax[0,0].plot(etimes[:-1], rel_gtrans[0,:], 'r',
             etimes[:-1], rel_etrans[0,:], 'r--')
ax[1,0].plot(etimes[:-1], rel_gtrans[1,:], 'g',
             etimes[:-1], rel_etrans[1,:], 'g--')
ax[2,0].plot(etimes[:-1], rel_gtrans[2,:], 'b',
             etimes[:-1], rel_etrans[2,:], 'b--')
ax[0,1].plot(etimes[:-1], rel_grot[0,:], 'r',
             etimes[:-1], rel_erot[0,:], 'r--')
ax[1,1].plot(etimes[:-1], rel_grot[1,:], 'g',
             etimes[:-1], rel_erot[1,:], 'g--')
ax[2,1].plot(etimes[:-1], rel_grot[2,:], 'b',
             etimes[:-1], rel_erot[2,:], 'b--')
           
ax[0,0].set_ylabel("V_x")
ax[1,0].set_ylabel("V_y")
ax[2,0].set_ylabel("V_z")
ax[0,0].set_title("Position Change")
ax[2,0].set_xlabel('Time')
ax[0,1].set_ylabel("Omega_x")
ax[1,1].set_ylabel("Omega_y")
ax[2,1].set_ylabel("Omega_z")
ax[0,1].set_title("Attitude Change")
ax[2,1].set_xlabel('Time')
for a in ax.ravel():
    a.set_xlim((etimes[0], etimes[-2]))


# Plot the absolute translation and rotation
fig,ax = plt.subplots(3,2)
ax[0,0].plot(etimes, gtrans[0,:], 'r',
             etimes, etrans[0,:], 'r--')
ax[1,0].plot(etimes, gtrans[1,:], 'g',
             etimes, etrans[1,:], 'g--')
ax[2,0].plot(etimes, gtrans[2,:], 'b',
             etimes, etrans[2,:], 'b--')
ax[0,1].plot(etimes, grot_eul[0,:], 'r',
             etimes, erot_eul[0,:], 'r--')
ax[1,1].plot(etimes, grot_eul[1,:], 'g',
             etimes, erot_eul[1,:], 'g--')
ax[2,1].plot(etimes, grot_eul[2,:], 'b',
             etimes, erot_eul[2,:], 'b--')

ax[0,0].set_ylabel("Pos_x")
ax[1,0].set_ylabel("Pos_y")
ax[2,0].set_ylabel("Pos_z")
ax[0,0].set_title("Position")
ax[2,0].set_xlabel('Time')
ax[0,1].set_ylabel("Eul_x")
ax[1,1].set_ylabel("Eul_y")
ax[2,1].set_ylabel("Eul_z")
ax[0,1].set_title("Attitude")
ax[2,1].set_xlabel('Time')
for a in ax.ravel():
    a.set_xlim((etimes[0], etimes[-1]))
for i in range(3):
    ax[i,1].set_ylim((-np.pi, np.pi))

# Plot the full trajectories (position xy)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.hstack([pose._x._trans[0] for pose in aposes]),
        np.hstack([pose._x._trans[1] for pose in aposes]),
        np.hstack([pose._x._trans[2] for pose in aposes]))
ax.plot(np.hstack([pose._x._trans[0] for pose in eposes]),
        np.hstack([pose._x._trans[1] for pose in eposes]),
        np.hstack([pose._x._trans[2] for pose in eposes]), '--')
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("z (m)")
ax.set_title("Ground truth and Estimated Trajectories")

plt.show()