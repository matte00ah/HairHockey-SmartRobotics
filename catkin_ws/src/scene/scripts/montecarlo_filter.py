import numpy as np
import yaml
import os
import time
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

X_MAX = config["table_width_m"]
Y_MAX = config["table_height_m"]

PUCK_DIAMETER = config["puck_diameter_m"]
ROBOT_REACH = config["robot_reach_m"]

GAME_POSE = config["game_pose"]
RETURN_VELOCITY = config["return_vel"]

class MontecarloFilter:
    def __init__(self,robot, N=4000, dt=0.1, f=0.01, process_noise_std=0.3, measurement_noise_std=0.05, velocity_noise_std=0.2):
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
        self.prev_robot_target = None

        self.robot = robot

        self.initialize()
        self.robot_base = [1.2 + 0.97, 0 + 0.425]  # ("0 + 0.425, 1.2 + 0.97")

    def initialize(self):
        self.particles = np.zeros((self.N, 6))
        self.particles[:, 0] = np.random.uniform(0, X_MAX, self.N) # x
        self.particles[:, 1] = np.random.uniform(0, Y_MAX, self.N) # y
        self.particles[:, 2] = np.random.normal(1.0, 0.3, self.N) # vx
        self.particles[:, 3] = np.random.normal(0.5, 0.3, self.N) # vy
        self.particles[:, 4] = np.random.normal(0.0, 0.1, self.N) # ax
        self.particles[:, 5] = np.random.normal(0.0, 0.1, self.N) # ay
        self.weights = np.ones(self.N) / self.N

    def predict(self):
        process_noise = np.random.normal(0, self.process_noise_std, size=(self.N, 2))
        self.particles[:, 4:6] += process_noise
        self.particles[:, 2:4] += (self.particles[:, 4:6] - self.f * self.particles[:, 2:4]) * self.dt
        self.particles[:, 0:2] += self.particles[:, 2:4] * self.dt
        
        # Gestione rimbalzi ai bordi
        for dim, max_val in zip([0,1], [X_MAX, Y_MAX]):
            mask_low = self.particles[:, dim] <= 0
            mask_high = self.particles[:, dim] >= max_val
            self.particles[mask_low | mask_high, dim+2] *= -1  # inverte velocità
            self.particles[:, dim] = np.clip(self.particles[:, dim], 0, max_val)
    
    def is_reachable(self, pos):
        """
        Controlla se una posizione è raggiungibile dal robot (circonferenza centrata sulla base).
    
        """
        # Calcola distanza dalla base del robot
        dist = np.linalg.norm(pos - self.robot_base)
        return dist <= ROBOT_REACH

    
    def predict_future(self, steps=10):
        future_particles = self.particles.copy()
        for step in range(steps):
            process_noise = np.random.normal(0, self.process_noise_std, size=(self.N,2))
            future_particles[:,4:6] += process_noise
            future_particles[:,2:4] += (future_particles[:,4:6] - self.f * future_particles[:,2:4]) * self.dt
            future_particles[:,0:2] += future_particles[:,2:4] * self.dt

            for dim, max_val in zip([0,1], [X_MAX, Y_MAX]):
                mask_low = future_particles[:,dim] <= 0
                mask_high = future_particles[:,dim] >= max_val
                future_particles[mask_low | mask_high, dim+2] *= -1
                future_particles[:, dim] = np.clip(future_particles[:, dim], 0, max_val)

            est_pos = np.mean(future_particles[:, 0:2], axis=0)

            if self.is_reachable(est_pos):  # assumendo che self.true_reach abbia un metodo 'contains'
                print(f"Prima posizione raggiungibile al passo {step+1}")
                return est_pos

        return None

    def update(self, measurement, velocity):
        dists = np.linalg.norm(self.particles[:,0:2] - measurement, axis=1)
        v_dists = np.linalg.norm(self.particles[:,2:4] - velocity, axis=1)
        w = self.weights * np.exp(-0.5*(dists/self.measurement_noise_std)**2) * np.exp(-0.5*(v_dists/self.velocity_noise_std)**2)
        w += 1.e-300
        self.weights = w / np.sum(w)        

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

    def run(self, wx, wy, future_steps=10):

        measurement = None if wx is None or wy is None else np.array([wx, wy])

        # Predizione step
        self.predict()

        if measurement is not None:
            velocity = (measurement - self.prev_measurement)/self.dt if self.prev_measurement is not None else np.zeros(2)

            # Aggiorniamo il filtro con misura e velocità
            self.update(measurement, velocity)
            self.resample()

            self.prev_measurement = measurement
        else:
            # Nessuna misura → solo predizione
            velocity = np.zeros(2)
            # prev_measurement non viene aggiornato

        # Se il puck sta andando verso l-avversario con una velocity alta (verso l-avversario quindi negativa)
        if velocity[0] < RETURN_VELOCITY:
            print("Torna a BASE")
            self.robot.move_to_point(GAME_POSE[0], GAME_POSE[1])
    
        est_pos, est_vel, est_acc = self.estimate()

        self.est_positions.append(est_pos)
        self.real_positions.append(measurement)

        new_target = self.predict_future(steps=future_steps)
        if new_target is not None:
            if self.prev_robot_target is None or not np.allclose(new_target, self.prev_robot_target, atol=2e-2):
                print("Chiamata panda_move", time.perf_counter())
                self.robot.move_to_point(new_target[0], new_target[1])
                self.prev_robot_target = new_target
            else:
                print("Nuova posizione simile alla precedente, nessun movimento effettuato.")
                # Strategia di attacco avanzata: colpo diretto verso la porta
                if np.linalg.norm(velocity) < 0.05:
                    # Definisci la porta come il centro del bordo opposto
                    goal = np.array([0, Y_MAX/2])
                    # Calcola la direzione dal disco verso la porta
                    direction = goal - self.prev_measurement
                    direction = direction / np.linalg.norm(direction)
                    # Posizione di partenza del robot: dietro al disco rispetto alla porta
                    hit_distance = 0.30  # distanza di sicurezza dietro il disco (modifica se necessario)
                    start_pos = self.prev_measurement - direction * hit_distance
                    print(start_pos)
                    # Verifica che la posizione di partenza sia raggiungibile
                    if self.is_reachable(start_pos):
                        print("Attacco: colpisco il disco verso la porta con movimento unico!")
                        # Muovi il robot dietro al disco
                        self.robot.move_to_point(start_pos[0], start_pos[1], wait_robot=True)
                        print("Posizione di attacco raggiunta dal robot.")
                        print(self.prev_measurement)
                        # Poi muovi il robot verso il disco (in direzione della porta)
                        self.robot.move_to_point(self.prev_measurement[0], self.prev_measurement[1], wait_robot=True)
                        print("Colpo eseguito.")
                    else:
                        #siamo nel caso in cui il disco è vicino al bordo lungo del tavolo
                        print("Posizione di attacco non raggiungibile dal robot. Provo colpo con rimbalzo!")
                        # Calcola quale bordo è più vicino al disco
                        direction_reflected = np.array([direction[0], -direction[1]])
                        # Posizione di partenza laterale rispetto al disco
                        start_pos = est_pos - direction_reflected * hit_distance
                        if self.is_reachable(start_pos):
                            print("Colpo con rimbalzo: posiziono il robot per colpire il disco verso il bordo!")
                            self.robot.move_to_point(start_pos[0], start_pos[1], wait_robot=True)
                            self.robot.move_to_point(est_pos[0], est_pos[1], wait_robot=True)
                        else:
                            #siamo nel caso in cui il disco è vicino al bordo corto del tavolo
                            if (est_pos[1] < Y_MAX / 2 ):
                                self.robot.move_to_point(X_MAX - PUCK_DIAMETER - 0.02, est_pos[1] + 0.15)
                            else:
                                self.robot.move_to_point(X_MAX - PUCK_DIAMETER - 0.02, est_pos[1] - 0.15)
        
        elif(measurement is None):
            print("Occlusione del puck")
            if (est_pos[1] < Y_MAX / 2 ):
                self.robot.move_to_point(est_pos[0], est_pos[1] + 0.15)
            else:
                self.robot.move_to_point(est_pos[0], est_pos[1] - 0.15)
        
        else:
            print("Nessuna posizione futura raggiungibile prevista.")

                       
        print(f"Est. vel: vx = {est_vel[0]:.3f}, vy = {est_vel[1]:.3f} | Est. acc: ax = {est_acc[0]:.3f}, ay = {est_acc[1]:.3f}")
        
         #Calcola e stampa la precisione finale
         #Filtra solo le posizioni reali disponibili
        #valid_real_positions = [p for p in self.real_positions if p is not None]
        #valid_est_positions = self.est_positions[-len(valid_real_positions):]  # allinea lunghezze

        #if valid_real_positions:
            #rmse = self.compute_rmse(valid_est_positions, valid_real_positions)
            #print(f"RMSE X: {rmse[0]:.4f} m, RMSE Y: {rmse[1]:.4f} m")