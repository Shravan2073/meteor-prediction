from flask import Flask, render_template, jsonify
import numpy as np
import scipy.stats as stats
import time
import threading
import math
import json

app = Flask(__name__)

# Globals to store simulation state and data
simulation_data = {
    'meteors': [],
    'region_meteor_counts': [],
    'planets': [],
    'current_state': 1,  # Start in "medium" state
    'markov_states': ["low", "medium", "high"],
    'observed_meteors': [],
    'alpha_prior': 2,
    'beta_prior': 1
}

# Simulation parameters
WIDTH, HEIGHT = 800, 600
CENTER = (WIDTH // 2, HEIGHT // 2)
MAX_METEORS = 30
REGION_X, REGION_Y = WIDTH // 2 - 100, HEIGHT // 2 - 100
REGION_WIDTH, REGION_HEIGHT = 200, 200

# Markov Chain Parameters
transition_matrix = np.array([[0.6, 0.3, 0.1], 
                              [0.2, 0.5, 0.3], 
                              [0.1, 0.3, 0.6]])
meteor_rate = {"low": 1, "medium": 2, "high": 3}

class Meteor:
    def __init__(self):
        self.id = np.random.randint(100000)
        self.x = np.random.randint(0, WIDTH)
        self.y = 0
        self.speed = np.random.uniform(0.5, 1.5)
        self.mass = np.random.uniform(0.5, 1.5)

    def fall(self):
        dx = CENTER[0] - self.x
        dy = CENTER[1] - self.y
        distance = math.sqrt(dx ** 2 + dy ** 2) + 0.1
        gravity_force = 100 / (distance ** 2)
        self.x += gravity_force * dx / distance * self.mass
        self.y += self.speed + gravity_force * dy / distance * self.mass
        
    def to_dict(self):
        return {
            'id': self.id,
            'x': self.x,
            'y': self.y
        }

class CelestialBody:
    def __init__(self, color, radius, orbit_radius, speed, orbit_color=(200, 200, 200)):
        self.color = color
        self.radius = radius
        self.orbit_radius = orbit_radius
        self.speed = speed
        self.angle = np.random.rand() * 2 * math.pi
        self.moons = []
        self.orbit_color = orbit_color
        self.x = 0
        self.y = 0

    def add_moon(self, moon):
        self.moons.append(moon)

    def update_position(self):
        self.angle += self.speed
        self.x = CENTER[0] + self.orbit_radius * math.cos(self.angle)
        self.y = CENTER[1] + self.orbit_radius * math.sin(self.angle)
        return int(self.x), int(self.y)
        
    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'radius': self.radius,
            'color': self.color,
            'orbit_radius': self.orbit_radius,
            'orbit_color': self.orbit_color,
            'moons': [moon.to_dict(self.x, self.y) for moon in self.moons]
        }

class Moon:
    def __init__(self, color, radius, orbit_radius, speed):
        self.color = color
        self.radius = radius
        self.orbit_radius = orbit_radius
        self.speed = speed
        self.angle = np.random.rand() * 2 * math.pi
        self.x = 0
        self.y = 0

    def update_position(self, planet_x, planet_y):
        self.angle += self.speed
        self.x = planet_x + self.orbit_radius * math.cos(self.angle)
        self.y = planet_y + self.orbit_radius * math.sin(self.angle)
        return int(self.x), int(self.y)
        
    def to_dict(self, planet_x, planet_y):
        self.update_position(planet_x, planet_y)
        return {
            'x': self.x,
            'y': self.y,
            'radius': self.radius,
            'color': self.color
        }

# Initialize celestial bodies
def init_celestial_bodies():
    YELLOW = (255, 255, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    ORANGE = (255, 165, 0)
    GRAY = (200, 200, 200)
    PURPLE = (128, 0, 128)
    CYAN = (0, 255, 255)
    WHITE = (255, 255, 255)

    sun = CelestialBody(YELLOW, 20, 0, 0)
    planets = [
        CelestialBody(ORANGE, 8, 60, 0.01, orbit_color=PURPLE),
        CelestialBody(BLUE, 10, 100, 0.008, orbit_color=CYAN),
        CelestialBody(RED, 6, 150, 0.006, orbit_color=(255, 182, 193)),
        CelestialBody(GREEN, 7, 200, 0.004, orbit_color=(173, 255, 47))
    ]

    # Add moons to some planets
    planets[1].add_moon(Moon(GRAY, 3, 15, 0.05))
    planets[2].add_moon(Moon(WHITE, 2, 10, 0.07))
    planets[3].add_moon(Moon(GRAY, 2, 18, 0.04))
    
    return sun, planets

def update_markov_chain():
    simulation_data['current_state'] = np.random.choice(
        [0, 1, 2], 
        p=transition_matrix[simulation_data['current_state']]
    )

def update_bayesian():
    alpha_prior = simulation_data['alpha_prior']
    beta_prior = simulation_data['beta_prior']
    observed_meteors = simulation_data['observed_meteors']
    
    alpha_post = alpha_prior + sum(observed_meteors)
    beta_post = beta_prior + len(observed_meteors)
    
    return alpha_post, beta_post

def get_bayesian_curve():
    alpha_post, beta_post = update_bayesian()
    x_vals = np.linspace(0, 10, 200)
    y_vals = stats.gamma.pdf(x_vals, alpha_post, scale=1 / beta_post).tolist()
    return x_vals.tolist(), y_vals, alpha_post, beta_post

def simulation_thread():
    sun, planets = init_celestial_bodies()
    simulation_data['planets'] = planets
    simulation_data['sun'] = sun
    frame_count = 0
    
    while True:
        frame_count += 1
        
        # Update celestial bodies
        sun.update_position()
        for planet in planets:
            planet.update_position()
            
        # Markov chain update
        if frame_count % 100 == 0:
            update_markov_chain()
            current_state = simulation_data['current_state']
            current_rate = meteor_rate[simulation_data['markov_states'][current_state]]
            simulation_data['observed_meteors'].append(current_rate)
            
        # Spawn meteors
        current_state = simulation_data['current_state']
        current_rate = meteor_rate[simulation_data['markov_states'][current_state]]
        if len(simulation_data['meteors']) < MAX_METEORS:
            for _ in range(current_rate):
                simulation_data['meteors'].append(Meteor())
        
        # Update meteors
        count_in_region = 0
        updated_meteors = []
        for meteor in simulation_data['meteors']:
            meteor.fall()
            if REGION_X <= meteor.x <= REGION_X + REGION_WIDTH and REGION_Y <= meteor.y <= REGION_Y + REGION_HEIGHT:
                count_in_region += 1
            if 0 <= meteor.y < HEIGHT and 0 <= meteor.x < WIDTH:
                updated_meteors.append(meteor)
                
        simulation_data['meteors'] = updated_meteors
        simulation_data['region_meteor_counts'].append(count_in_region)
        
        # Limit data size to prevent memory issues
        if len(simulation_data['region_meteor_counts']) > 500:
            simulation_data['region_meteor_counts'] = simulation_data['region_meteor_counts'][-500:]
        if len(simulation_data['observed_meteors']) > 500:
            simulation_data['observed_meteors'] = simulation_data['observed_meteors'][-500:]
            
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/simulation')
def simulation():
    return render_template('simulation.html')

@app.route('/api/simulation-data')
def get_simulation_data():
    meteors_data = [m.to_dict() for m in simulation_data['meteors']]
    sun_data = simulation_data['sun'].to_dict()
    planets_data = [p.to_dict() for p in simulation_data['planets']]
    region_data = {
        'x': REGION_X,
        'y': REGION_Y,
        'width': REGION_WIDTH,
        'height': REGION_HEIGHT
    }
    center_data = {
        'x': CENTER[0],
        'y': CENTER[1]
    }
    
    return jsonify({
        'meteors': meteors_data,
        'sun': sun_data,
        'planets': planets_data,
        'region': region_data,
        'center': center_data
    })

@app.route('/api/graph-data')
def get_graph_data():
    state_data = {
        'states': simulation_data['markov_states'],
        'rates': [meteor_rate[s] for s in simulation_data['markov_states']],
        'current_state': simulation_data['current_state']
    }
    
    region_counts = simulation_data['region_meteor_counts'][-50:] if simulation_data['region_meteor_counts'] else []
    
    x_vals, y_vals, alpha_post, beta_post = get_bayesian_curve()
    
    return jsonify({
        'markov': state_data,
        'region_counts': region_counts,
        'bayesian': {
            'x_vals': x_vals,
            'y_vals': y_vals,
            'alpha_post': alpha_post,
            'beta_post': beta_post
        }
    })

if __name__ == '__main__':
    simulation_thread = threading.Thread(target=simulation_thread, daemon=True)
    simulation_thread.start()
    app.run(debug=True, threaded=True)