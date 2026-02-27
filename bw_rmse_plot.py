import numpy as np
import matplotlib.pyplot as plt

# Explicitly define the arrays from your data
bw_values = [0.04, 0.03, 0.02, 0.01, 0.05]
rmse_values = [0.045729, 0.074818, 0.075293, 0.073217, 0.075887]

# Convert to numpy arrays
bw_array = np.array(bw_values)
rmse_array = np.array(rmse_values)

# Sort by bandwidth for better plotting
sort_idx = np.argsort(bw_array)
bw_array = bw_array[sort_idx]
rmse_array = rmse_array[sort_idx]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(bw_array, rmse_array, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Bandwidth', fontsize=12)
plt.ylabel('RMSE Translation (m)', fontsize=12)
plt.title('RMSE Translation Error vs Bandwidth', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xscale('log')  # Log scale for bandwidth

# Add value labels on points
for i, (bw, rmse) in enumerate(zip(bw_array, rmse_array)):
    plt.annotate(f'{rmse:.3f}', (bw, rmse), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.show()

# Print the parsed data
print("Parsed data:")
print(f"Bandwidths: {bw_array}")
print(f"RMSE values: {rmse_array}")