import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Simulation Parameters ---
NUM_BUBBLES = 300
WIDTH = 100
HEIGHT = 100
MAX_RADIUS = 4.0
MIN_RADIUS = 0.5

# Initialize bubble properties
# x and y positions
x = np.random.uniform(0, WIDTH, NUM_BUBBLES)
y = np.random.uniform(0, HEIGHT, NUM_BUBBLES)

# Initial sizes and upward speeds
# Larger bubbles rise faster in reality, we simulate a slight correlation
base_radii = np.random.uniform(MIN_RADIUS, MAX_RADIUS, NUM_BUBBLES)
speeds = np.random.uniform(0.5, 2.5, NUM_BUBBLES) + (base_radii * 0.2)

# --- Setup Plot ---
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_facecolor('#1a1a24') # Dark blue/grey background resembling deep water
ax.set_xticks([]) # Hide axes for better visual
ax.set_yticks([])

# Create scatter plot. 
# 's' is the marker size in points**2. We use hollow circles to mimic bubbles.
scat = ax.scatter(x, y, s=np.pi * (base_radii**2) * 10, 
                  c='none', edgecolors='#aaccff', alpha=0.7, linewidths=1.5)

def update(frame):
    global x, y, base_radii, speeds
    
    # 1. Upward Movement
    y += speeds
    
    # 2. Horizontal Jitter (simulating turbulent water)
    x += np.random.normal(0, 0.4, NUM_BUBBLES)
    
    # Keep bubbles within horizontal bounds
    x = np.clip(x, 0, WIDTH)
    
    # 3. Simulate growth as they rise (expansion due to lower pressure)
    # The radius increases slightly as y increases
    current_radii = base_radii * (1 + (y / HEIGHT) * 0.5)
    
    # 4. Check for bursting (reaching the surface)
    popped = y > HEIGHT
    
    # Respawn popped bubbles at the bottom with new random properties
    num_popped = np.sum(popped)
    if num_popped > 0:
        y[popped] = np.random.uniform(-10, 0, num_popped) # Spawn slightly below 0
        x[popped] = np.random.uniform(0, WIDTH, num_popped)
        base_radii[popped] = np.random.uniform(MIN_RADIUS, MAX_RADIUS, num_popped)
        speeds[popped] = np.random.uniform(0.5, 2.5, num_popped) + (base_radii[popped] * 0.2)
    
    # Update the scatter plot data
    scat.set_offsets(np.c_[x, y])
    # Update sizes (converting radius to area)
    scat.set_sizes(np.pi * (current_radii**2) * 10)
    
    return scat,

# Run the animation
ani = animation.FuncAnimation(fig, update, frames=200, interval=30, blit=True)
plt.show()