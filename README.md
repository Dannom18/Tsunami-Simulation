﻿# Tsunami-Simulation

This is the system of semi-discrete equations:

### First Equation:
$$
\dot{a}_i(t) = -\frac{1}{2 \Delta x} \left[ a_i(t) \cdot \left(b_{i+1}(t) - b_{i-1}(t)\right) + b_i(t) \cdot \left(a_{i+1}(t) - a_{i-1}(t)\right) \right]
$$

### Second Equation:
$$
\dot{b}_i(t) = -\frac{1}{2 \Delta x} \left[ g \cdot \left(a_{i+1}(t) - a_{i-1}(t)\right) + b_i(t) \cdot \left(b_{i+1}(t) - b_{i-1}(t)\right) \right]
$$
