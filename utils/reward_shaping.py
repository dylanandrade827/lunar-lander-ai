import numpy as np

def shaped_reward(state, reward, done):
    # state: [x, y, vx, vy, angle, angular_velocity, leg1, leg2]
    x, y, vx, vy, theta, vtheta, leg1, leg2 = state

    angle_penalty = -abs(theta) * 0.1
    vel_penalty = -(abs(vx) + abs(vy)) * 0.05
    leg_bonus = 0.5 * (leg1 + leg2)

    shaping = angle_penalty + vel_penalty + leg_bonus
    return reward + shaping
