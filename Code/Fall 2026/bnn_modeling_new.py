import numpy as np
import matplotlib.pyplot as plt

def derivative_values(p,q,Pi1,Pi2):

    m = np.size(Pi1,0)
    n = np.size(Pi1,1) 

    I_minus_P = np.eye(m) - np.tile(p, (m, 1))
    I_minus_Q = np.eye(n) - np.tile(q, (n, 1))

    phi_p = np.maximum(I_minus_P@Pi1@q,0)
    phi_q = np.maximum(I_minus_Q@Pi2.T@p,0)

    p_dot = I_minus_P.T@phi_p
    q_dot = I_minus_Q.T@phi_q

    return p_dot, q_dot


def bnn_ode_solver(p,q,Pi1,Pi2,T,N):
  """
    Initial Probability Arrays:
        p - Initial Player 1 Probability Array
        q - Initial Player 2 Probability Array
    
    Payoff Matrices:
        A - Player 1 Payoff Matrix
        B - Player 2 Payoff Matrix
        
    T - Time Period
    N - Number of Time Steps
    """
  
  m = np.size(Pi1,0)
  n = np.size(Pi1,1) 

 # Runge-Kutta 4 Solver
  h = T / (N-1)
  t_mesh = np.linspace(0,T,N)
  p_values = np.zeros((m,N))
  p_values[:,0] = p
  q_values = np.zeros((n,N))
  q_values[:,0] = q

  for index in range(N-1):

    p = p_values[:,index]
    q = q_values[:,index]

    p_dot, q_dot = derivative_values(p,q,Pi1,Pi2)
    k1_p = p_dot; k1_q = q_dot

    p_dot, q_dot = derivative_values(p+h*k1_p/2,q+h*k1_q/2,Pi1,Pi2)
    k2_p = p_dot;k2_q = q_dot

    p_dot, q_dot = derivative_values(p+h*k2_p/2,q+h*k2_q/2,Pi1,Pi2)
    k3_p = p_dot; k3_q = q_dot

    p_dot, q_dot = derivative_values(p+h*k3_p,q+h*k3_q,Pi1,Pi2)
    k4_p = p_dot; k4_q = q_dot
     
    p_values[:,index+1] = p_values[:,index] + h*(k1_p+2*k2_p+2*k3_p+k4_p)/6
    q_values[:,index+1] = q_values[:,index] + h*(k1_q+2*k2_q+2*k3_q+k4_q)/6

  return t_mesh,p_values,q_values
    
Pi1 = np.array([[7, 2],[3, 4]], dtype=float)
Pi2 = np.array([[1, 6],[8, 2]], dtype=float)

p_0 = [.5, .5]
q_0 = [.5, .5]

T = 50
N = 1000

t_mesh,p_values,q_values = bnn_ode_solver(p_0,q_0,Pi1,Pi2,T,N)
p1 = p_values[0,:]
p2 = p_values[1,:]
q1 = q_values[0,:]
q2 = q_values[1,:]

# Print final state
print("Final approximate strategy profile:")
print(f"Player 1: p1 = {p1[-1]:.6f}, p2 = {p2[-1]:.6f}")
print(f"Player 2: q1 = {q1[-1]:.6f}, q2 = {q2[-1]:.6f}")

# Plot trajectories
plt.figure(figsize=(10, 6))
plt.plot(t_mesh, p1, label=r"$p_1$"); plt.plot(t_mesh, p2, label=r"$p_2$")
plt.plot(t_mesh, q1, label=r"$q_1$"); plt.plot(t_mesh, q2, label=r"$q_2$")
plt.xlabel("t"); plt.ylabel("Probability")
plt.title("BNN dynamics for the 2x2 game")
plt.legend(); plt.grid(True); plt.show()

