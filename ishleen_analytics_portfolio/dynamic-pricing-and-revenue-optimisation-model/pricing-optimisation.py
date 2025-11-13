#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:47:37 2025

@author: Group 8
"""

import numpy as np
import pandas as pd
from pyomo.environ import *

# Load dataset
file_path = "BuildMax_Rentals_Dataset_Updated.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Define equipment types and rental durations
equipment_types = ["Excavators", "Cranes", "Bulldozers"]
durations = [1, 4, 8, 16]  # Rental durations in weeks
num_weeks = 52

# Extract initial inventory
initial_inventory = {eq: df[f"{eq} - Start of Week Inventory"].iloc[0] for eq in equipment_types}

# Extract prices, demand, and returns with (i, t, j) indexing
prices = {(eq, t, j+1): df[f"{eq} - {t}-week Price per day (£)"].iloc[j] 
          if t == 1 else df[f"{eq} - {t}-weeks Price per day (£)"].iloc[j]
          for eq in equipment_types for t in durations for j in range(num_weeks)}

demand = {(eq, t, j+1): df[f"{eq} - {t}-week Demand (units)"].iloc[j] 
          if t == 1 else df[f"{eq} - {t}-weeks Demand (units)"].iloc[j]
          for eq in equipment_types for t in durations for j in range(num_weeks)}

# Define model
model = ConcreteModel()

# Decision Variables
model.X = Var(equipment_types, durations, range(1, num_weeks+1), domain=NonNegativeIntegers) #the units accepted to optimize revenue
model.I = Var(equipment_types, range(1, num_weeks+1), domain=NonNegativeIntegers)  # Inventory tracking

# Objective Function: Maximize Revenue
def revenue_rule(model):
    return sum(
        t * 7 * prices[i, t, j] * model.X[i, t, j]  
        for i in equipment_types for t in durations for j in range(1, num_weeks+1)
    )

model.obj = Objective(rule=revenue_rule, sense=maximize)

# Inventory Balance Constraint (Returns = Rentals from t Weeks Ago)
def inventory_rule(model, i, j):
    if j == 1:
        return model.I[i, j] == initial_inventory[i]  # Week 1: Start with initial inventory
    else:
        # Rentals accepted in the previous week (j-1) are subtracted
        # Returns occur for rentals accepted t weeks ago
        returns_total = sum(model.X[i, t, j-t] for t in durations if (j-t) >= 1)  

        return model.I[i, j] == model.I[i, j-1] - sum(model.X[i, t, j-1] for t in durations) + returns_total

model.inventory_constraint = Constraint(equipment_types, range(1, num_weeks+1), rule=inventory_rule)

# Demand Constraint
def demand_rule(model, i, t, j):
    return model.X[i, t, j] <= demand[i, t, j]

model.demand_constraint = Constraint(equipment_types, durations, range(1, num_weeks+1), rule=demand_rule)

# Capacity Constraint
def capacity_rule(model, i, j):
    return sum(model.X[i, t, j] for t in durations) <= model.I[i, j]

model.capacity_constraint = Constraint(equipment_types, range(1, num_weeks+1), rule=capacity_rule)

# Solve model
solver = SolverFactory("glpk")
solver.solve(model)

# Print results in the required format
print(f"Total optimized revenue: {model.obj():,.2f}")

for i in equipment_types:
    for t in durations:
        total_rentals = sum(model.X[i, t, j]() for j in range(1, num_weeks+1))
        print(f"{i} - {t}-week rentals: {total_rentals:.1f}")
        
# ---------------- ANALYSIS SECTION ---------------- #

# Revenue per Equipment Type
revenue_equipment_type = {eq: sum(
    t * 7 * prices[eq, t, j] * model.X[eq, t, j]()  
    for t in durations for j in range(1, num_weeks+1)) 
    for eq in equipment_types}

for eq in equipment_types:
    print(f"Total Revenue {eq}: £{revenue_equipment_type[eq]:,.2f}")

# Revenue per Duration and Equipment Type
revenue_per_duration_eq = {
    (eq, t): sum(
        t * 7 * prices[eq, t, j] * model.X[eq, t, j]()  
        for j in range(1, num_weeks+1)) 
    for eq in equipment_types for t in durations}

# Print Revenue per Duration and Equipment Type
print("\nRevenue per Duration and Equipment Type:")
for (eq, t), revenue in revenue_per_duration_eq.items():
    print(f"{eq} - {t}-week rentals: £{revenue:,.2f}")
    
# Compute Weekly Fleet Utilization Rate
weekly_utilization = {
    eq: [
        (sum(model.X[eq, t, j]() for t in durations) / model.I[eq, j]()) * 100
        if model.I[eq, j]() > 0 else 0  # Avoid division by zero
        for j in range(1, num_weeks+1)
    ]
    for eq in equipment_types
}

# Compute Average Fleet Utilization Rate across all weeks
avg_fleet_utilization = {
    eq: sum(weekly_utilization[eq]) / num_weeks
    for eq in equipment_types
}

# Print Results
print("\n--- Fleet Utilization Rate ---")
for eq in equipment_types:
    print(f"{eq}: {avg_fleet_utilization[eq]:.2f}%")
    
# Compute Total Rentals and Total Inventory for each equipment type
total_rentals = {
    eq: sum(sum(model.X[eq, t, j]() for t in durations) for j in range(1, num_weeks+1))
    for eq in equipment_types
}

total_inventory = {
    eq: sum(model.I[eq, j]() for j in range(1, num_weeks+1))
    for eq in equipment_types
}

# Compute Overall Fleet Utilization Rate
total_rentals_all_eq = sum(total_rentals[eq] for eq in equipment_types)
total_inventory_all_eq = sum(total_inventory[eq] for eq in equipment_types)

overall_utilization_rate = (total_rentals_all_eq / total_inventory_all_eq) * 100

# Print Results
print(f"Overall Fleet Utilization Rate: {overall_utilization_rate:.2f}%")

# Revenue per Unit (RPU)
rpu = {
    eq: revenue_equipment_type[eq] / initial_inventory[eq]
    for eq in equipment_types}

for eq in equipment_types:
    print(f"Revenue per {eq}: £{rpu[eq]:,.2f}")
    
# Define equipment purchase costs for each equipment type (in £)
equipment_costs = {
    "Excavators": 12000,  
    "Cranes": 15000,      
    "Bulldozers": 25000    
}

original_total_revenue = {
    "Excavators": 45274005,   
    "Cranes": 61785689,       
    "Bulldozers": 53413920    
}
    
# Calculate Original RPU for each equipment type
original_rpu = {
    eq: original_total_revenue[eq] / initial_inventory[eq]
    for eq in equipment_types
}

# Print the Original RPU for each equipment type
for eq in equipment_types:
    print(f"Original RPU for {eq}: £{original_rpu[eq]:,.2f}")
    
# Calculate Total Equipment Cost for each equipment type (Inventory * Cost)
total_equipment_cost = {
    eq: initial_inventory[eq] * equipment_costs[eq]
    for eq in equipment_types
}

# Calculate Original Total Revenue (Sum of original revenues)
original_total_revenue_all = sum(original_total_revenue.values())

# Calculate Optimized Total Revenue (Revenue from model)
optimized_total_revenue_all = sum(revenue_equipment_type.values())

# Calculate Original ROI using the formula: (Total Revenue - Total Equipment Cost) / Total Equipment Cost * 100
original_roi_all = original_total_revenue_all - sum(total_equipment_cost.values())
original_roi_percentage = (original_roi_all / sum(total_equipment_cost.values())) * 100

# Calculate Optimized ROI using the formula: (Optimized Revenue - Total Equipment Cost) / Total Equipment Cost * 100
optimized_roi_all = optimized_total_revenue_all - sum(total_equipment_cost.values())
optimized_roi_percentage = (optimized_roi_all / sum(total_equipment_cost.values())) * 100

# Calculate ROI Improvement
roi_improvement_percentage = optimized_roi_percentage - original_roi_percentage

# Print Results
print(f"\nOriginal Total Revenue: £{original_total_revenue_all:,.2f}")
print(f"Optimized Total Revenue: £{optimized_total_revenue_all:,.2f}")
print(f"Total Equipment Cost: £{sum(total_equipment_cost.values()):,.2f}")
print(f"Original ROI: {original_roi_percentage:.2f}%")
print(f"Optimized ROI: {optimized_roi_percentage:.2f}%")
print(f"ROI Improvement: {roi_improvement_percentage:.2f}%")


