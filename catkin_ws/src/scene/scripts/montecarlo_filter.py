import numpy as np
import json
import os
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))  # cartella dello script
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

# TODO controllare se ho invertito w con h
X_MAX = config["table_width_m"]
Y_MAX = config["table_height_m"]

class MontecarloFilter:
    def __init__(self, N=10000, dt=0.1, f=0.05, process_noise_std=0.3, measurement_noise_std=0.05, velocity_noise_std=0.2):
        self.N = N
        self.dt = dt
        self.f = f
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.velocity_noise_std = velocity_noise_std

        self.particles = None
        self.weights = None
        self.est_positions = []
        self.real_positions = []
        self.prev_measurement = None

        self.initialize(X_MAX, Y_MAX)

    def initialize(self, X_MAX, Y_MAX):
        self.X_MAX, self.Y_MAX = X_MAX, Y_MAX
        self.particles = np.zeros((self.N, 6))
        self.particles[:, 0] = np.random.uniform(0, X_MAX, self.N) # x
        self.particles[:, 1] = np.random.uniform(0, Y_MAX, self.N) # y
        self.particles[:, 2] = np.random.normal(1.0, 0.3, self.N) # vx
        self.particles[:, 3] = np.random.normal(0.5, 0.3, self.N) # vy
        self.particles[:, 4] = np.random.normal(0.0, 0.1, self.N) # ax
        self.particles[:, 5] = np.random.normal(0.0, 0.1, self.N) # ay
        self.weights = np.ones(self.N) / self.N

    def predict(self):
        self.particles[:, 4] += np.random.normal(0, self.process_noise_std, size=self.N)
        self.particles[:, 5] += np.random.normal(0, self.process_noise_std, size=self.N)
        self.particles[:, 2] += (self.particles[:, 4] - self.f * self.particles[:, 2]) * self.dt
        self.particles[:, 3] += (self.particles[:, 5] - self.f * self.particles[:, 3]) * self.dt
        self.particles[:, 0] += self.particles[:, 2] * self.dt
        self.particles[:, 1] += self.particles[:, 3] * self.dt
        
        # Gestione rimbalzi ai bordi
        for i in range(self.N):
            if self.particles[i, 0] <= 0 or self.particles[i, 0] >= self.X_MAX:
                self.particles[i, 2] *= -1
                self.particles[i, 0] = np.clip(self.particles[i, 0], 0, self.X_MAX)
            if self.particles[i, 1] <= 0 or self.particles[i, 1] >= self.Y_MAX:
                self.particles[i, 3] *= -1
                self.particles[i, 1] = np.clip(self.particles[i, 1], 0, self.Y_MAX)

    def update(self, measurement, velocity):
        w = self.weights

        dists = np.linalg.norm(self.particles[:, 0:2] - measurement, axis=1)
        v_dists = np.linalg.norm(self.particles[:, 2:4] - velocity, axis=1)

        w *= np.exp(-0.5 * (dists / self.measurement_noise_std) ** 2)
        w *= np.exp(-0.5 * (v_dists / self.velocity_noise_std) ** 2)
        w += 1.e-300
        w /= np.sum(w)
        self.weights = w

    def resample(self):
        indices = np.random.choice(self.N, self.N, p=self.weights)
        self.particles[:] = self.particles[indices]
        self.weights.fill(1.0 / self.N)

    def estimate(self):
        est_pos = np.average(self.particles[:, 0:2], weights=self.weights, axis=0)
        est_vel = np.average(self.particles[:, 2:4], weights=self.weights, axis=0)
        est_acc = np.average(self.particles[:, 4:6], weights=self.weights, axis=0)
        return est_pos, est_vel, est_acc

    def compute_rmse(self, est_positions, real_positions):
        est_positions = np.array(est_positions)
        real_positions = np.array(real_positions)
        mse = np.mean((est_positions - real_positions) ** 2, axis=0)
        rmse = np.sqrt(mse)
        return rmse  # array [rmse_x, rmse_y]

    def run(self, cx, cy):

        #frame_h, frame_w = frame.shape[:2]
        pos = (cx, cy)
        print(cx, cy)

        plt.figure(figsize=(8, 6))

        
        if pos is None:
            prev_measurement = None
        
        measurement = np.array([pos[0], pos[1]])

        # Calcolo velocità reale dal video
        if self.prev_measurement is not None:
            velocity = (measurement - self.prev_measurement) / self.dt
        else:
            velocity = np.array([0.0, 0.0])

        self.prev_measurement = measurement

        self.predict()
        self.update(measurement, velocity)
        self.resample()
    
        est_pos, est_vel, est_acc = self.estimate()
        

        self.est_positions.append(est_pos)
        self.real_positions.append(measurement)
        #print(measurement)
        
        print(f"Est. vel: vx = {est_vel[0]:.3f}, vy = {est_vel[1]:.3f} | Est. acc: ax = {est_acc[0]:.3f}, ay = {est_acc[1]:.3f}")
        
        plt.cla()
        plt.scatter(self.particles[:, 0], self.particles[:, 1], color='gray', s=2, label='Particelle')
        plt.scatter(measurement[0], measurement[1], color='blue', s=40, label='Misura')
        plt.scatter(est_pos[0], est_pos[1], color='green', s=50, label='Stima filtro')
        plt.xlim(0, X_MAX)
        plt.ylim(0, Y_MAX)
        plt.title(f" Stima vel: ({est_vel[0]:.2f}, {est_vel[1]:.2f}) – "
                    f"Stima acc: ({est_acc[0]:.2f}, {est_acc[1]:.2f})")
        plt.legend(loc='upper right')
        plt.pause(0.1)

        plt.show()

        # Calcola e stampa la precisione finale
        rmse = self.compute_rmse(self.est_positions, self.real_positions)
        print(f"RMSE X: {rmse[0]:.4f} m, RMSE Y: {rmse[1]:.4f} m")
