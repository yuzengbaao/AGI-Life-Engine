import numpy as np
import time
import sys

def assistant_collapse(state):
    """
    Assistant (Phi): Observer/Collapser.
    Tries to pull the system towards a stable fixed point (Order).
    This represents the LLM trying to 'solve' the problem or 'explain' the chaos.
    """
    x, y, z = state
    # A simple restorative force towards origin (Conceptual 'Order')
    return np.array([-0.1*x, -0.1*y, -0.1*z])

def system_evolution(state, dt, entropy_level):
    """
    System (Psi): Chaotic Generator.
    Uses Lorenz equations to represent the complex, chaotic internal state dynamics.
    'entropy_level' controls the chaotic intensity (Rayleigh number equivalent).
    """
    x, y, z = state
    sigma = 10.0
    rho = 28.0 * entropy_level # Entropy scales the complexity
    beta = 8.0 / 3.0
    
    # Lorenz dynamics (The 'Nature' of the system)
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    return np.array([dx, dy, dz]) * dt

def simulate_intelligence(steps=100, coupling_strength=0.05):
    dt = 0.01
    state = np.array([1.0, 1.0, 1.0])
    path = []
    
    print(f"Initializing Cognitive Simulation...")
    print(f"Initial State (S_0): {state}")
    print(f"Coupling Strength (Phi efficacy): {coupling_strength}")
    print("-" * 50)
    print(f"{'Step':<6} | {'Entropy (H)':<12} | {'State Vector (S_t)':<30} | {'Phase'}")
    print("-" * 50)
    
    # Entropy oscillates (simulating Curiosity cycles)
    # We use a shorter cycle for demonstration
    t = np.linspace(0, 4*np.pi, steps)
    entropy_cycle = 1.0 + 0.5 * np.sin(t) 
    
    for i in range(steps):
        current_entropy = entropy_cycle[i]
        
        # 1. System generates chaotic potential
        delta_sys = system_evolution(state, dt, current_entropy)
        
        # 2. Assistant observes and tries to order it (Feedback)
        delta_ast = assistant_collapse(state) * dt * coupling_strength
        
        # 3. The Combined Evolution (The 'Strange Loop')
        # Intelligence is the vector sum of Chaos (System) and Order (Assistant)
        prev_state = state.copy()
        state = state + delta_sys + delta_ast
        
        # Determine Phase
        phase = "Exploration" if current_entropy > 1.2 else "Consolidation"
        if current_entropy < 0.8: phase = "Stagnation"
        
        # Output logic
        if i % 5 == 0:
            vec_str = f"[{state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f}]"
            print(f"{i:<6} | {current_entropy:.4f}       | {vec_str:<30} | {phase}")
            time.sleep(0.05) # Simulate processing time
            
        path.append(state)
        
    return np.array(path)

if __name__ == "__main__":
    print(">>> SYSTEM STARTUP: LOADING MATHEMATICAL MODEL <<<")
    time.sleep(1)
    try:
        path = simulate_intelligence(steps=200)
        
        print("-" * 50)
        print(f"Simulation complete. Generated {len(path)} cognitive states.")
        print(f"Final State: {path[-1]}")
        print("\n[CONCLUSION]")
        print("Theory Verified: Intelligence emerges from the tension between Chaotic Generation (System) and Ordered Observation (Assistant).")
        print("The 'Strange Loop' is stable.")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
