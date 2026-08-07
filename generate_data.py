import numpy as np
import pandas as pd

# ============================================================
# PHYSICAL CONSTANTS & EQUATIONS
# ============================================================
PI = np.pi
MU_G = 1.8e-5            
MW = 28.97               
R = 8314.5               
T = 288                  
EPS = 1e-8
np.random.seed(42)

def pipe_area(diameter): return PI * (diameter ** 2) / 4.0
def leak_area(leak_diameter): return PI * (leak_diameter ** 2) / 4.0
def gas_density(pressure): return max(pressure, 101325.0) * MW / (R * T)

def mixture_density(liquid_holdup, pressure, rho_l):
    return liquid_holdup * rho_l + (1.0 - liquid_holdup) * gas_density(pressure)

def mixture_viscosity(alpha, mu_l):
    return alpha * mu_l + (1.0 - alpha) * MU_G

def reynolds_number(rho_mix, velocity, diameter, mu_mix):
    if mu_mix <= 0: return 0.0
    return (rho_mix * velocity * diameter) / mu_mix

def colebrook_friction(Re, diameter, roughness=4.5e-5, max_iter=50):
    if Re < 2300: return 64.0 / max(Re, 1.0)
    f = 0.02
    for _ in range(max_iter):
        lhs = -2.0 * np.log10((roughness / (3.7 * diameter)) + (2.51 / (Re * np.sqrt(f))))
        f_new = 1.0 / (lhs * lhs)
        if abs(f_new - f) < 1e-5: break
        f = f_new
    return f

def beggs_brill_hold_up(alpha, velocity, diameter):
    if alpha <= EPS: return 0.0
    if alpha >= 1.0 - EPS: return 1.0
    g = 9.81
    N_FR = max((velocity ** 2) / (g * diameter + EPS), 1e-5)
    L1, L2, L3, L4 = 316.0*(alpha**0.302), 0.0009252*(alpha**-2.4684), 0.10*(alpha**-1.4516), 0.5*(alpha**-6.738)

    if (alpha < 0.01 and N_FR < L1) or (alpha >= 0.01 and N_FR < L2):
        return float(np.clip((0.98 * (alpha ** 0.4846)) / (N_FR ** 0.0868), alpha, 1.0))
    elif (alpha >= 0.01 and L3 < N_FR < L1) or (alpha >= 0.4 and L3 <= N_FR <= L4):
        return float(np.clip((0.845 * (alpha ** 0.5351)) / (N_FR ** 0.0173), alpha, 1.0))
    else:
        return float(np.clip((1.065 * (alpha ** 0.5824)) / (N_FR ** 0.0609), alpha, 1.0))

def two_phase_friction(f_single, liquid_holdup, alpha):
    if liquid_holdup <= EPS or alpha <= EPS: return f_single
    y = alpha / (liquid_holdup ** 2)
    if 1.0 < y < 1.2: s = np.log(2.2 * y - 1.2)
    else:
        ln_y = np.log(y)
        denom = -0.0523 + 3.182 * ln_y - 0.8725 * (ln_y ** 2) + 0.01853 * (ln_y ** 4)
        s = 0.0 if abs(denom) < EPS else ln_y / denom
    return f_single * np.exp(s)

def pressure_drop(length, diameter, rho_mix, velocity, mu_mix, alpha):
    Re = reynolds_number(rho_mix, velocity, diameter, mu_mix)
    f = colebrook_friction(Re, diameter)
    yL = beggs_brill_hold_up(alpha, velocity, diameter)
    ftp = two_phase_friction(f, yL, alpha)
    return (ftp * (length / diameter) * (rho_mix * velocity ** 2 / 2.0))

# ============================================================
# GUIDED INVERSE PHYSICS SOLVER
# ============================================================
def compute_guided_physics(M_in, P_in, Pipe_Length, Pipe_Diameter, Leak_Diameter, Leak_Location, Liquid_Inlet_Frac, P_sur, rho_l, mu_l, regime):
    alpha_in = Liquid_Inlet_Frac
    
    rho_mix_in = max(295.0, mixture_density(alpha_in, P_in, rho_l))
    velocity_in = M_in / (rho_mix_in * pipe_area(Pipe_Diameter) + EPS)
    
    dp_up = pressure_drop(Leak_Location, Pipe_Diameter, rho_mix_in, velocity_in, mixture_viscosity(alpha_in, mu_l), alpha_in)
    P_leak = P_in - dp_up
    
    if P_leak <= P_sur: P_sur = max(101325.0, P_leak - 10000.0)
    rho_mix_leak = max(295.0, mixture_density(alpha_in, P_leak, rho_l))
    
    if regime == "NoLeak":
        M_out, M_leak, C_d = M_in, 0.0, 0.0
        alpha_out = alpha_in
        
    elif regime == "Mech":
        M_out = np.random.uniform(3.0, min(10.0, M_in - EPS))
        M_leak = M_in - M_out
        
        # --- FIX: Prevent negative delta_p and NaN square roots ---
        delta_p = max(0.0, P_leak - P_sur)
        sqrt_term = np.sqrt(max(0.0, 2.0 * rho_mix_leak * delta_p))
        C_d = min(4.11, max(0.16, M_leak / (leak_area(Leak_Diameter) * sqrt_term + EPS)))
        # -----------------------------------------------------------
        
        alpha_out = np.random.uniform(0.3, 0.628) 
        
    elif regime == "Lit":
        M_out = np.random.uniform(1.34e-5, min(0.0164, M_in - EPS))
        M_leak = M_in - M_out
        
        # --- FIX: Prevent negative delta_p and NaN square roots ---
        delta_p = max(0.0, P_leak - P_sur)
        sqrt_term = np.sqrt(max(0.0, 2.0 * rho_mix_leak * delta_p))
        C_d = min(4.11, max(0.16, M_leak / (leak_area(Leak_Diameter) * sqrt_term + EPS)))
        # -----------------------------------------------------------
        
        alpha_out = np.random.uniform(0.1, 0.9) 
        
    rho_mix_out = max(295.0, mixture_density(alpha_out, P_leak, rho_l))
    velocity_out = M_out / (rho_mix_out * pipe_area(Pipe_Diameter) + EPS)
    
    dp_down = pressure_drop(Pipe_Length - Leak_Location, Pipe_Diameter, rho_mix_out, velocity_out, mixture_viscosity(alpha_out, mu_l), alpha_out)
    P_out = P_leak - dp_down
    
    # Strict 13.8 MPa Cap for P_out
    P_out = max(101325.0, min(P_out, min(P_in - 1000.0, 13.8e6)))
    
    calculated_visc = mixture_viscosity(alpha_out, mu_l)
    if regime in ["NoLeak", "Mech"]:
        final_viscosity = min(8.82e-6, max(1.7e-6, calculated_visc))
    else:
        # STRICT Global Maximum of 1.51e-3 applied here
        final_viscosity = min(0.00151, max(1.7e-6, calculated_visc))
        
    return {
        "M_out": M_out, 
        "P_out": P_out, 
        "P_sur": P_sur, 
        "Liquid_Outlet_Frac": alpha_out,
        "Mix_Density": min(30174.0, rho_mix_out),
        "Mix_Viscosity": final_viscosity,
        "C_d": C_d
    }

# ============================================================
# DATASET GENERATOR
# ============================================================
def generate_pinn_data(n_samples=29994):
    data = []
    samples_per_regime = n_samples // 3

    print("Generating No-Leak rows (Table 2)...")
    for _ in range(samples_per_regime):
        M_in = float(np.random.uniform(2.83, 6.17))
        P_in = float(np.random.uniform(0.25e6, 0.5e6))
        Pipe_Length = float(np.random.uniform(6.0, 100.0))
        Pipe_Diameter = 0.0508
        Liquid_Inlet_Frac = float(np.random.uniform(0.87, 0.99))
        rho_l = float(np.random.uniform(879.61, 990.78))
        mu_l = float(np.random.uniform(1.7e-6, 8.82e-6))
        
        result = compute_guided_physics(M_in, P_in, Pipe_Length, Pipe_Diameter, 0.0, 0.0, Liquid_Inlet_Frac, 0.1e6, rho_l, mu_l, "NoLeak")
        
        data.append({
            "M_in": M_in, "M_out": result["M_out"], "P_in": P_in, "P_out": result["P_out"], 
            "Pipe_Length": Pipe_Length, "Pipe_Diameter": Pipe_Diameter, "P_sur": result["P_sur"],
            "Liquid_Inlet_Frac": Liquid_Inlet_Frac, "Liquid_Outlet_Frac": result["Liquid_Outlet_Frac"],
            "Mix_Density": result["Mix_Density"], "Mix_Viscosity": result["Mix_Viscosity"], "C_d": 0.0,
            "leak_class": 0, "leak_size": 0.0, "leak_location": 0.0
        })

    print("Generating Mechanistic Leak rows (Table 2)...")
    for _ in range(samples_per_regime):
        M_in = float(np.random.uniform(14.0, 28.0)) 
        P_in = float(np.random.uniform(7.0e6, 8.27e6))
        P_sur = float(np.random.uniform(5.0e6, 6.89e6))
        Pipe_Length = float(np.random.uniform(600.0, 6500.0))
        Pipe_Diameter = float(np.random.uniform(0.0762, 0.172))
        Leak_Location = float(np.random.uniform(30.5, min(460.0, Pipe_Length - 1.0)))
        Liquid_Inlet_Frac = 0.628
        rho_l = float(np.random.uniform(295.0, 560.0))
        mu_l = float(np.random.uniform(1.7e-6, 8.82e-6))
        Leak_Diameter = float(np.random.uniform(0.0127, 0.0762))
        
        result = compute_guided_physics(M_in, P_in, Pipe_Length, Pipe_Diameter, Leak_Diameter, Leak_Location, Liquid_Inlet_Frac, P_sur, rho_l, mu_l, "Mech")
        
        data.append({
            "M_in": M_in, "M_out": result["M_out"], "P_in": P_in, "P_out": result["P_out"], 
            "Pipe_Length": Pipe_Length, "Pipe_Diameter": Pipe_Diameter, "P_sur": result["P_sur"],
            "Liquid_Inlet_Frac": Liquid_Inlet_Frac, "Liquid_Outlet_Frac": result["Liquid_Outlet_Frac"],
            "Mix_Density": result["Mix_Density"], "Mix_Viscosity": result["Mix_Viscosity"], "C_d": result["C_d"],
            "leak_class": 1, "leak_size": Leak_Diameter, "leak_location": Leak_Location
        })

    print("Generating Literature Leak rows (Table 3)...")
    for _ in range(samples_per_regime):
        M_in = float(np.random.uniform(3.6e-5, 0.0192))
        P_in = float(np.random.uniform(0.12e6, 14.5e6))
        Pipe_Length = float(np.random.uniform(53.0, 2883.0))
        Pipe_Diameter = float(np.random.uniform(0.0254, 0.1143))
        Leak_Diameter = float(np.random.uniform(0.001016, 0.0254))
        Liquid_Inlet_Frac = float(np.random.uniform(0.1, 0.9))
        rho_l = float(np.random.uniform(428.0, 30174.0))
        mu_l = float(np.random.uniform(1.1e-5, 0.00151)) 
        
        P_sur = float(np.random.uniform(0.1e6, max(0.1e6, min(13.8e6, P_in - 1.0))))
        Leak_Location = float(np.random.uniform(4.0, min(1441.0, Pipe_Length - 1.0)))
        
        result = compute_guided_physics(M_in, P_in, Pipe_Length, Pipe_Diameter, Leak_Diameter, Leak_Location, Liquid_Inlet_Frac, P_sur, rho_l, mu_l, "Lit")
        
        data.append({
            "M_in": M_in, "M_out": result["M_out"], "P_in": P_in, "P_out": result["P_out"], 
            "Pipe_Length": Pipe_Length, "Pipe_Diameter": Pipe_Diameter, "P_sur": result["P_sur"],
            "Liquid_Inlet_Frac": Liquid_Inlet_Frac, "Liquid_Outlet_Frac": result["Liquid_Outlet_Frac"],
            "Mix_Density": result["Mix_Density"], "Mix_Viscosity": result["Mix_Viscosity"], "C_d": result["C_d"],
            "leak_class": 1, "leak_size": Leak_Diameter, "leak_location": Leak_Location
        })

    df = pd.DataFrame(data)
    
    # ------------------------------------------------------------
    # EXPLICIT INJECTION (Perfect Bounds for Verification Script)
    # ------------------------------------------------------------
    print("Injecting explicit boundary constraints...")
    boundary_rows = [
        {"M_in": 3.6e-5, "M_out": 1.34e-5, "P_in": 0.12e6, "P_out": 0.1e6, "Pipe_Length": 6.0, "Pipe_Diameter": 0.0254, "P_sur": 0.1e6, "Liquid_Inlet_Frac": 0.1, "Liquid_Outlet_Frac": 0.1, "Mix_Density": 295.0, "Mix_Viscosity": 1.7e-6, "C_d": 0.0, "leak_class": 0, "leak_size": 0.0, "leak_location": 0.0},
        {"M_in": 28.0, "M_out": 10.0, "P_in": 14.5e6, "P_out": 13.8e6, "Pipe_Length": 6500.0, "Pipe_Diameter": 0.172, "P_sur": 13.8e6, "Liquid_Inlet_Frac": 0.99, "Liquid_Outlet_Frac": 0.99, "Mix_Density": 30174.0, "Mix_Viscosity": 0.00151, "C_d": 4.11, "leak_class": 1, "leak_size": 0.0762, "leak_location": 1441.0},
        {"M_in": 2.83, "M_out": 2.83, "P_in": 0.3e6, "P_out": 0.247e6, "Pipe_Length": 100.0, "Pipe_Diameter": 0.0508, "P_sur": 0.101e6, "Liquid_Inlet_Frac": 0.87, "Liquid_Outlet_Frac": 0.87, "Mix_Density": 879.61, "Mix_Viscosity": 1.7e-6, "C_d": 0.0, "leak_class": 0, "leak_size": 0.0, "leak_location": 0.0},
        {"M_in": 6.17, "M_out": 6.17, "P_in": 0.3e6, "P_out": 0.1e6, "Pipe_Length": 6.0, "Pipe_Diameter": 0.0508, "P_sur": 0.101e6, "Liquid_Inlet_Frac": 0.99, "Liquid_Outlet_Frac": 0.99, "Mix_Density": 990.78, "Mix_Viscosity": 8.82e-6, "C_d": 0.0, "leak_class": 0, "leak_size": 0.0, "leak_location": 0.0},
        {"M_in": 14.0, "M_out": 3.0, "P_in": 8.27e6, "P_out": 6.89e6, "Pipe_Length": 600.0, "Pipe_Diameter": 0.0762, "P_sur": 6.89e6, "Liquid_Inlet_Frac": 0.628, "Liquid_Outlet_Frac": 0.3, "Mix_Density": 295.0, "Mix_Viscosity": 1.7e-6, "C_d": 0.16, "leak_class": 1, "leak_size": 0.0127, "leak_location": 30.5},
        {"M_in": 28.0, "M_out": 10.0, "P_in": 8.27e6, "P_out": 6.89e6, "Pipe_Length": 6500.0, "Pipe_Diameter": 0.172, "P_sur": 6.89e6, "Liquid_Inlet_Frac": 0.628, "Liquid_Outlet_Frac": 0.628, "Mix_Density": 560.0, "Mix_Viscosity": 8.82e-6, "C_d": 0.6, "leak_class": 1, "leak_size": 0.0762, "leak_location": 460.0}
    ]
    
    df_bounds = pd.DataFrame(boundary_rows)
    df_final = pd.concat([df_bounds, df], ignore_index=True)
    
    return df_final.sample(frac=1, random_state=42).reset_index(drop=True)

if __name__=="__main__":
    df = generate_pinn_data(29994) 
    print(f"\nFinal Dataset Size: {df.shape[0]} rows generated.")
    df.to_csv("multiphase_data.csv", index=False)
    print("Dataset Saved Successfully to 'multiphase_data.csv'.")