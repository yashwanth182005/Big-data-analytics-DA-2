# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from pylab import colorbar, plot

# Load the dataset
data = pd.read_csv('Carbon_(CO2)_Emissions_by_Country.csv')

# Step 1: Data Preprocessing
# Select relevant columns (Kilotons of CO2, Metric Tons Per Capita) for analysis
data = data[['Kilotons of Co2', 'Metric Tons Per Capita']]
data.dropna(inplace=True)  # Drop missing values

# Step 2: Normalizing the features
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Step 3: Initialize and Train SOM
som_grid_rows = 10  # SOM grid size
som_grid_cols = 10
som = MiniSom(x=som_grid_rows, y=som_grid_cols, input_len=data_scaled.shape[1], sigma=1.0, learning_rate=0.5)

# Initialize weights and train the SOM
som.random_weights_init(data_scaled)
som.train_random(data_scaled, 5000)  # Train with 5000 iterations

# Step 4: U-Matrix Visualization (Distance Map)
plt.figure(figsize=(12, 10))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Distance map
plt.colorbar(label='Distance (U-Matrix)', orientation='vertical', pad=0.02)
plt.title('SOM U-Matrix for Carbon Emissions Data', fontsize=16)
plt.xlabel('SOM X-axis', fontsize=14)
plt.ylabel('SOM Y-axis', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.savefig('som_umatrix.png', dpi=300)  # Save the plot as a PNG file
plt.show()

# Step 5: Hit Map Visualization (Data Density)
plt.figure(figsize=(12, 10))
plt.pcolor(som.activation_response(data_scaled).T, cmap='Blues')  # Activation map
plt.colorbar(label='Activation Count', orientation='vertical', pad=0.02)
plt.title('SOM Hit Map: Emissions Data Density on SOM Grid', fontsize=16)
plt.xlabel('SOM X-axis', fontsize=14)
plt.ylabel('SOM Y-axis', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.savefig('som_hitmap.png', dpi=300)  # Save the hit map as a PNG file
plt.show()

# Step 6: Visualize SOM with Markers using different symbols
plt.figure(figsize=(12, 10))
plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Distance map as background (colorful grid)
colorbar()  # Show color legend

# Define markers and colors for the visualization
markers = ['^', 'o', 'D']  # Triangle, Circle, Diamond
colors = ['g', 'y', 'w']  # Green, Yellow, White

# Assign markers and colors based on some criteria (e.g., cluster position or random assignment)
for i, x in enumerate(data_scaled):
    win_position = som.winner(x)  # Get winning node for each data point
    marker_index = i % 3  # Cycle through the markers
    plt.plot(win_position[0] + 0.5, win_position[1] + 0.5, markers[marker_index], 
             markerfacecolor=colors[marker_index], markeredgecolor='k', 
             markeredgewidth=2, markersize=12)

# Set white background for the plot
plt.gca().set_facecolor('white')  # Ensure the background is white
plt.title('Carbon Emissions Segmentation using SOM with Different Symbols', fontsize=16)
plt.xlabel('SOM X-axis', fontsize=14)
plt.ylabel('SOM Y-axis', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.tight_layout()
plt.savefig('som_emissions_segmentation_symbols.png', dpi=300)  # Save the final plot as a PNG file
plt.show()

# Step 7: Cluster Assignment
clustered_data = data.copy()
clustered_data['Cluster'] = [som.winner(x)[0] * som_grid_cols + som.winner(x)[1] for x in data_scaled]

# Save clustered data
clustered_data.to_csv('emissions_segmentation_som.csv', index=False)
