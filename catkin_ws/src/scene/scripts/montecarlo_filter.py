import numpy as np
import yaml
import os
import time
from stl import mesh
from shapely.geometry import Point, Polygon
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pickle

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

X_MAX = config["table_width_m"]
Y_MAX = config["table_height_m"]

PUCK_DIAMETER = config["puck_diameter_m"]
ROBOT_REACH = config["robot_reach_m"]
ROBOT_BASE = config["robot_base_tocorner"]

GAME_POSE = config["game_pose"]

RETURN_VELOCITY_X = config["return_vel_x"]
HIT_DISTANCE = config["hit_distance"]
BORDER_DISTANCE = config["border_distance"]
OCCLUSION_MOVE_Y = config["occlusion_move_y"]


class MontecarloFilter:
    def __init__(self,robot, N=2500, dt=0.05, f=0.01, process_noise_std=0.5, measurement_noise_std=0.01, velocity_noise_std=0.05):
        self.N = N
        self.dt = dt
        self.f = f
        self.process_noise_std = process_noise_std
        self.measurement_noise_std = measurement_noise_std
        self.velocity_noise_std = velocity_noise_std

        self.occlusion_state = 0

        self.particles = None
        self.weights = None
        self.est_positions = []
        self.real_positions = []
        self.prev_measurement = None
        self.prev_robot_target = None

        self.table_outline = self._load_table_outline()

        self.robot = robot

        # control measurement use / update frequency
        self.use_measurements = True       # set False to skip measurement updates entirely
        self.update_interval = 10          # call update+resample every N frames
        self._frame_count = 0

        self.prev_run_time = None
        self.current_run_time = None

    def save_filter_state(self, filename="./scene/scripts/filter_state01.pkl"): # Se lanci da cmd: "./src/scene/scripts/filter_state.pkl"
        state = {
            'particles': self.particles,
            'weights': self.weights
        }
        print("prova salva...")
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"Filter state saved to {filename}")

    def initialize_at_measurement(self, measurement):
        """Inizializza le particelle attorno alla prima misura reale del puck"""
        x0, y0 = measurement

        self.particles = np.zeros((self.N, 6))
        # Posizioni intorno al punto misurato, ±2 cm
        self.particles[:, 0] = np.random.normal(x0, 0.02, self.N)
        self.particles[:, 1] = np.random.normal(y0, 0.02, self.N)
        # Velocità iniziali molto piccole (il disco è quasi fermo appena visto)
        self.particles[:, 2:4] = np.random.normal(0.0, 0.1, (self.N, 2))
        # Accelerazioni iniziali quasi nulle
        self.particles[:, 4:6] = np.random.normal(0.0, 0.05, (self.N, 2))
        # Pesi uniformi
        self.weights = np.ones(self.N) / self.N

        self.prev_measurement = measurement

    def load_and_initialize(self, filename="./scene/scripts/filter_state.pkl", measurement=None):  # Se lanci da cmd: "./src/scene/scripts/filter_state.pkl"
        try:
            with open(filename, 'rb') as f:
                state = pickle.load(f)
                self.particles = state['particles']
                self.weights = state['weights']
                print("Filter state loaded.")
            
                # Re-weighting the loaded particles with the first measurement
                if measurement is not None:
                    velocity = self._compute_velocity(measurement)
                    self.update(measurement, velocity)
                    self.resample()
                    print("Fine load?")

        except FileNotFoundError:
            print("No saved state found. Initializing from scratch.")
            # Fall back to the original initialization if no file exists
            if measurement is not None:
                self.initialize_at_measurement(measurement)
            else:
                # Handle case where both saved state and first measurement are missing
                pass

    def predict(self):
        print(f"Nostro dt: {self.dt}")
        process_noise = np.random.normal(0, self.process_noise_std, size=(self.N, 2))
        print(f"Nostro dt: {self.dt}")
        self.particles[:, 4:6] += process_noise
        self.particles[:, 2:4] += (self.particles[:, 4:6] - self.f * self.particles[:, 2:4]) * self.dt
        self.particles[:, 0:2] += self.particles[:, 2:4] * self.dt
        
        # Gestione rimbalzi ai bordi
        for dim, max_val in zip([0,1], [X_MAX, Y_MAX]):
            mask_low = self.particles[:, dim] <= 0
            mask_high = self.particles[:, dim] >= max_val
            self.particles[mask_low | mask_high, dim+2] *= -1  # inverte velocità
            self.particles[:, dim] = np.clip(self.particles[:, dim], 0, max_val)

    def _load_table_outline(self):
        """Estrae il contorno XY della mesh STL"""
        #rospack = rospkg.RosPack()
        #pkg_path = rospack.get_path('scene')
        #stl_path = os.path.join(pkg_path, 'models', 'table_borders', 'meshes', 'airhockey-borders.stl')

        script_dir = os.path.dirname(__file__)  # src/scripts
        stl_path = os.path.join(script_dir, '..', 'models', 'table_borders', 'meshes', 'airhockey-borders.stl')
        stl_path = os.path.abspath(stl_path)

        m = mesh.Mesh.from_file(stl_path)
        points_2d = np.column_stack((m.x.flatten(), m.y.flatten()))
        hull = ConvexHull(points_2d)
        polygon = Polygon(points_2d[hull.vertices])
        return polygon
    
    def is_reachable(self, pos):
        """
        Controlla se una posizione è raggiungibile dal robot (circonferenza centrata sulla base).
        """
        # Calcola distanza dalla base del robot
        dist = np.linalg.norm(pos - ROBOT_BASE)
        #print(f"distanza: {dist}")
        reach =  dist <= ROBOT_REACH
        print(f"reachable? {reach}")
        return reach

    
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
                #print(f"Prima posizione raggiungibile al passo {step+1}")
                return est_pos

        print(f"   Posizione NON raggiungibile: {est_pos}")
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
    
    def _compute_velocity(self, measurement):
        if measurement is None or self.prev_measurement is None or self.prev_run_time is None:
            self.prev_run_time = self.current_run_time
            return np.zeros(2)

        self.dt = self.current_run_time - self.prev_run_time
        
        vel = (measurement - self.prev_measurement) / self.dt
        self.prev_run_time = self.current_run_time
        return vel
    
    def is_valid(self, pos):
        valid = (self.is_reachable(pos)
            and pos[0] <= X_MAX - BORDER_DISTANCE and pos[1] <= Y_MAX - BORDER_DISTANCE
            and pos[0] >= BORDER_DISTANCE and pos[1] >= BORDER_DISTANCE)
        print(f"Valido? {valid}")
        return valid
    
    def training(self, wx, wy):
        self.current_run_time = time.perf_counter()

        measurement = None if wx is None or wy is None else np.array([wx, wy])

        #Gestione caso in cui nel primo frame che passo ho un'occlusione del puck
        #e non ho ancora inizializzato le particelle del filtro
        if self.particles is None and measurement is None:
            print("Nessuna misura valida disponibile: attendo il primo rilevamento del puck.")
            return
        #inizializzo le particelle del filtro, si fa solo una volta
        if self.particles is None and measurement is not None:
            print(f"Inizializzo Montecarlo con prima misura {measurement}")
            self.initialize_at_measurement(measurement)

        # count frames and predict every frame
        self._frame_count += 1
        self.predict()

        # Update step (periodic)
        velocity = None
        if measurement is not None:
            velocity = self._compute_velocity(measurement)
            
            if self.use_measurements and (self._frame_count % self.update_interval) == 0:
                self.update(measurement, velocity)
                self.resample()

            # always remember last measurement for velocity computation / logic
            self.prev_measurement = measurement


    def run(self, wx, wy, future_steps=10):
        self.current_run_time = time.perf_counter()

        measurement = None if wx is None or wy is None else np.array([wx, wy])

        # increment frame counter
        self._frame_count += 1

        #Gestione caso in cui nel primo frame che passo ho un'occlusione del puck
        #e non ho ancora inizializzato le particelle del filtro
        if self.particles is None and measurement is None:
            print("Nessuna misura valida disponibile: attendo il primo rilevamento del puck.")
            return
        #inizializzo le particelle del filtro, si fa solo una volta
        if self.particles is None and measurement is not None:
        #    print(f"Inizializzo Montecarlo con prima misura {measurement}")
        #    self.initialize_at_measurement(measurement)
            self.load_and_initialize(measurement=measurement)

        print("Inizia predict?")

        # Predizione step
        self.predict()

        # Update step (periodic)
        velocity = None
        if measurement is not None:
            velocity = self._compute_velocity(measurement)
            
            if self.use_measurements and (self._frame_count % self.update_interval) == 0:
                self.update(measurement, velocity)
                self.resample()

            # always remember last measurement for velocity computation / logic
            self.prev_measurement = measurement

        # Se il puck sta andando verso l-avversario con una velocity alta (verso l-avversario quindi negativa) e in posizione oltre il reachable
        if velocity is not None and velocity[0] < RETURN_VELOCITY_X and not self.is_reachable(measurement):
            print("--- Torna a BASE. Disco va verso avversario ---")
            self.robot.move_to_point(*GAME_POSE, wait_robot=True)
    
        est_pos, est_vel, est_acc = self.estimate()
        self.est_positions.append(est_pos)
        self.real_positions.append(measurement)

        print(f"    velocity:", velocity)
        print(f"    measurement:", measurement)

        if (measurement is None):
            if self.occlusion_state > 5:
                self.occlusion_state = 0
                print("--- OCCLUSIONE del puck (fermo) ---")
                offset = OCCLUSION_MOVE_Y if self.prev_measurement[1] < Y_MAX / 2 else -OCCLUSION_MOVE_Y
                # target = [est_pos[0], est_pos[1] + offset]
                target = [self.prev_measurement[0], self.prev_measurement[1] + offset]
                print(f" --> posizione laterale: {target}")
                self.robot.move_to_point(target[0], target[1])#,wait_robot=True)
                return
            else:
                self.occlusion_state += 1
                return        
        self.occlusion_state = 0

        print(f"   vel norm: {np.linalg.norm(velocity)}")
        if (velocity is None or np.linalg.norm(velocity) < 0.025) and self.is_reachable(measurement):
            # Strategia di attacco avanzata: colpo diretto verso la porta
            print("--- Puck FERMO e raggiungibile ---")

            # Definisci la porta come il centro del bordo opposto
            goal = np.array([X_MAX, Y_MAX/2])

            # Calcola la direzione dal disco verso la porta
            direction = goal - measurement
            direction /= np.linalg.norm(direction)

            # Posizione di partenza del robot: dietro al disco rispetto alla porta
            start_pos = measurement - direction * HIT_DISTANCE
            #print(f"Start position {start_pos}")
                
            # Primo tentativo: direzione diretta verso il goal
            if self.is_valid(start_pos):
                print("___ ATTACCO: colpisco il disco verso la porta con movimento unico! ___")
                self.robot.move_to_point(*start_pos, wait_robot=True)  # Muovi il robot dietro al disco
                print(f"    1. Posizione di attacco raggiunta dal robot. measurement {measurement}")
                self.robot.move_to_point(*measurement)#, wait_robot=True)
                print("    2.Colpo eseguito.")            
            # Secondo tentativo: direzione riflessa (rimbalzo)
            else:
                # Puck vicino al bordo lungo del tavolo
                print("__ Posizione di attacco NON raggiungibile dal robot. Provo colpo con RIMBALZO! ___")      

                direction_reflected = np.array([direction[0], -direction[1]])
                start_pos_reflected = measurement - direction_reflected * HIT_DISTANCE

                if self.is_valid(start_pos_reflected):
                    print("    1. Colpo con rimbalzo: posiziono il robot per colpire il disco verso il bordo!")
                    self.robot.move_to_point(*start_pos_reflected, wait_robot=True)
                    if self.is_valid(measurement):
                        print("    2. compisco posizione vera del disco ")
                        self.robot.move_to_point(*measurement)#, wait_robot=True)
                    else: #caso in cui posizione del disco sia vicino a bordo, quindi sposto il mullet vicino al disco ma in posizione sicura
                        y_offset = BORDER_DISTANCE if measurement[1] < (Y_MAX / 2) else -BORDER_DISTANCE
                        print("    2. compisco posizione vera del disco ")
                        self.robot.move_to_point(measurement[0], measurement[1] - y_offset, wait_robot=True)
                else:
                    #siamo nel caso in cui il disco è vicino al bordo corto del 
                    print(" ___ Siamo al bordo corto! ___")                        
                    y_offset = BORDER_DISTANCE if measurement[1] < (Y_MAX / 2) else -BORDER_DISTANCE  # aggiunge o toglie un margine per prendere la rincorsa
                    safe_x = BORDER_DISTANCE  # estremo che possiamo
                    safe_y = measurement[1] + y_offset
                    print("    1. Provo rinculo!")
                    self.robot.move_to_point(safe_x, safe_y, wait_robot=True)
                    if self.is_valid(measurement):
                        print("    2. Provo colpo!")
                        self.robot.move_to_point(*measurement)#, wait_robot=True)
                    """else: #caso in cui posizione del disco sia vicino a bordo, quindi sposto il mullet vicino al disco ma in posizione sicura
                        x_offset = BORDER_DISTANCE if measurement[0] < X_MAX / 2 else -BORDER_DISTANCE
                        self.robot.move_to_point(measurement[0] - x_offset, measurement[1], wait_robot=True)"""
            return
        

        if np.linalg.norm(velocity) > 0.2: # velocity is None or np.linalg.norm(velocity) < 0.01:

            #se non è fermo calcolo il nuovo target
            new_target = self.predict_future(steps=future_steps)

            print(f"--- Target FUTURO: {new_target} ---")
            #print(f"  self.prev_robot_target {self.prev_robot_target}")
            if new_target is not None and (self.prev_robot_target is None or not np.allclose(new_target, self.prev_robot_target, atol=0.15)):
                print(f" --> Target {new_target} - CHIAMATA A MOVE FRANKA {time.perf_counter()}")
                if self.is_valid(new_target):
                    self.robot.move_to_point(*new_target)#, wait_robot=True)
                    self.prev_robot_target = new_target                  
        
        #rospy.logdebug(f"Est. vel: vx = {est_vel[0]:.3f}, vy = {est_vel[1]:.3f} | Est. acc: ax = {est_acc[0]:.3f}, ay = {est_acc[1]:.3f}")
         #Calcola e stampa la precisione finale
         #Filtra solo le posizioni reali disponibili
        #valid_real_positions = [p for p in self.real_positions if p is not None]
        #valid_est_positions = self.est_positions[-len(valid_real_positions):]  # allinea lunghezze

        #if valid_real_positions:
            #rmse = self.compute_rmse(valid_est_positions, valid_real_positions)
            #print(f"RMSE X: {rmse[0]:.4f} m, RMSE Y: {rmse[1]:.4f} m")