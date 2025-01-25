import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Cobb-Douglas production function
def f_kt(alpha: float, L: float, K: float, a: float = 1) -> float:
    return a * (L**alpha) * (K**(1 - alpha))

# Calculate the next time step
def next_time(delta: float, n: float, k_t: float, s: float, alpha: float, L: float, K: float):
    Y_t = f_kt(alpha, L, K)
    k_t1 = (1 - delta - n) * k_t + s * Y_t
    L_t1 = (1 + n) * L
    C_t1 = (1 - s) * Y_t
    return k_t1, L_t1, Y_t, C_t1

# Main function to calculate the Solow growth model
def calc_vals(years, k_0, s, delta, n, L, K, alpha):
    t_points = np.arange(0, years + 1)
    k_vals, L_vals, Y_vals, C_vals = [k_0], [L], [], []

    for t in t_points:
        Y_t = f_kt(alpha, L_vals[-1], k_vals[-1])
        k_t1, L_t1, _, C_t1 = next_time(delta, n, k_vals[-1], s, alpha, L_vals[-1], k_vals[-1])
        Y_vals.append(Y_t)
        C_vals.append(C_t1)
        k_vals.append(k_t1)
        L_vals.append(L_t1)

    k_vals.pop()
    L_vals.pop()

    return {
        "capital": k_vals,
        "labor": L_vals,
        "output": Y_vals,
        "consumption": C_vals,
    }

def plot_acemoglu_diagram(alpha, s, delta):
    fig, ax = plt.subplots(figsize=(10, 6))
    k = np.linspace(0, 10, 1000)
    
    # Production function
    y = k**alpha
    ax.plot(k, y, label='f(k)', color='blue')
    
    # Savings function
    sy = s * y
    ax.plot(k, sy, label='sf(k)', color='green')
    
    # Depreciation line
    dk = delta * k
    ax.plot(k, dk, label='δk', color='red')
    
    # Find steady state
    k_star = ((s/delta)**(1/(1-alpha)))
    y_star = k_star**alpha
    
    # Highlight steady state
    ax.plot([k_star, k_star], [0, y_star], 'k--', linewidth=1)
    ax.plot([0, k_star], [y_star, y_star], 'k--', linewidth=1)
    ax.scatter([k_star], [y_star], color='purple', s=100, zorder=5, label='Steady State')
    
    ax.set_xlabel('Capital per worker (k)')
    ax.set_ylabel('Output per worker (y)')
    ax.set_title('Acemoglu Diagram for Solow Model')
    ax.legend()
    ax.grid(True)
    
    # Annotate steady state
    ax.annotate(f'k* = {k_star:.2f}\ny* = {y_star:.2f}', 
                xy=(k_star, y_star), xytext=(k_star+0.5, y_star-0.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    return fig



# Streamlit app
st.title("Interactive Solow Growth Model")

# Initialize session state to store previous results
if 'prev_result' not in st.session_state:
    st.session_state.prev_result = None
    st.session_state.prev_params = None

# Sidebar for input parameters
st.sidebar.header("Model Parameters")
years = st.sidebar.slider("Years", 1, 100, 10)
k_0 = st.sidebar.number_input("Initial Capital (k_0)", 1.0, 1000.0, 1.0)
s = st.sidebar.slider("Savings Rate (s)", 0.0, 1.0, 0.7)
delta = st.sidebar.slider("Depreciation Rate (delta)", 0.0, 1.0, 0.3)
n = st.sidebar.slider("Population Growth Rate (n)", 0.0, 1.0, 0.3)
L = st.sidebar.number_input("Initial Labor (L)", 1.0, 1000.0, 10.0)
K = st.sidebar.number_input("Initial Capital (K)", 1.0, 1000.0, 10.0)
alpha = st.sidebar.slider("Labor Share (alpha)", 0.0, 1.0, 0.5)

# Calculate model results
current_params = (years, k_0, s, delta, n, L, K, alpha)
result = calc_vals(years, k_0, s, delta, n, L, K, alpha)

# Function to plot results
def plot_results(result, years, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    time = np.arange(0, years + 1)
    
    ax.plot(time, result['capital'], label="Capital (K_t)", linestyle='-', marker='o')
    ax.plot(time, result['labor'], label="Labor (L_t)", linestyle='--', marker='x')
    ax.plot(time, result['output'], label="Output (Y_t)", linestyle='-.', marker='s')
    ax.plot(time, result['consumption'], label="Consumption (C_t)", linestyle=':', marker='d')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Time (t)", fontsize=14)
    ax.set_ylabel("Values", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    return fig

# Plot current results
st.subheader("Current Model Results")
st.pyplot(plot_results(result, years, "Solow Growth Model Over Time (Current)"))



# Display parameter changes
if st.session_state.prev_params is not None:
    st.subheader("Parameter Changes")
    changes = []
    for (prev, curr, name) in zip(st.session_state.prev_params, current_params, 
                                  ["Years", "k_0", "s", "delta", "n", "L", "K", "alpha"]):
        if prev != curr:
            changes.append(f"{name}: {prev} → {curr}")
    if changes:
        st.write("\n".join(changes))
    else:
        st.write("No changes in parameters.")

# After displaying parameter changes
# Current Acemoglu Diagram
st.subheader("Current Acemoglu Diagram")
current_acemoglu_fig = plot_acemoglu_diagram(alpha, s, delta)
st.pyplot(current_acemoglu_fig)

# Update previous results button
if st.button("See Previous Results"):
    st.session_state.prev_result = result
    st.session_state.prev_params = current_params
    st.session_state.prev_acemoglu = current_acemoglu_fig

# Plot previous results if available
if st.session_state.prev_result is not None:
    st.subheader("Previous Model Results")
    prev_years = st.session_state.prev_params[0]
    st.pyplot(plot_results(st.session_state.prev_result, prev_years, "Solow Growth Model Over Time (Previous)"))

    # Previous Acemoglu Diagram
    if 'prev_acemoglu' in st.session_state:
        st.subheader("Previous Acemoglu Diagram")
        st.pyplot(st.session_state.prev_acemoglu)

# Display raw data
if st.checkbox("Show raw data"):
    st.write(result)