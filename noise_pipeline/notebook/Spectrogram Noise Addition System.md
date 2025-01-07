# Spectrogram Noise Addition System

## Overview

This system facilitates adding noise to spectrograms using a combination of predefined shapes and customizable patterns. It supports both simple noise additions at specific locations and complex patterned noise additions that mimic real-world noise distributions.

---

## Noise Addition Methods

### 1. Single Shape Noise Addition

Adds noise in specific shapes (e.g., circle, horizontal stripe, etc.) to the spectrogram.

#### **Example: Circle Shape**

```yaml
shape: "circle"
parameters:
    power_distribution:
        type: "normal"         # Distribution type ("normal", "uniform", "none")
        mean: -40              # Mean value (dB power)
        stddev: 5              # Standard deviation (for "normal")
        min: -80               # Minimum value (for "uniform")
        max: -10               # Maximum value (for "uniform")
    position: (time, freq)      # Center position (time, frequency)
    size: (time, freq)          # Size (time-axis, frequency-axis)
    rotation: 0                 # Rotation angle
    gradient_time: 0.5          # Gradient increase along the time-axis
    gradient_freq: -0.5         # Gradient decrease along the frequency-axis


shape: "horizontal_stripe" # Horizontal stripe
parameters:
    power_distribution:
        type: "normal"         # Distribution type ("normal", "uniform", "none")
        mean: -40              # Mean value (dB power)
        stddev: 5              # Standard deviation (for "normal")
        min: -80               # Minimum value (for "uniform")
        max: -10               # Maximum value (for "uniform")
    position: (time, freq)      # Center position (time, frequency)
    size: (abs(gaussian(0,1)), abs(gaussian(0,1))) # Scale (time-axis, frequency-axis)
    rotation: 0                 # Rotation angle
    gradient_time: gaussian(0,1)         # Gradient function for time-axis
    gradient_freq: gaussian(0,1)         # Gradient function for frequency-axis
    is_dot_line: false         # Is it a dotted line?


shape: "wav_file"
parameters:
    power_distribution:
        type: "normal"         # Distribution type ("normal", "uniform", "none")
        mean: -40              # Mean value (dB power)
        stddev: 5              # Standard deviation (for "normal")
        min: -80               # Minimum value (for "uniform")
        max: -10               # Maximum value (for "uniform")
    position: (time, freq)      # Center position (time, frequency)
    size: (2, 2)               # Scale (time-axis, frequency-axis)
    rotation: (90, 0)          # Rotation angle (time-axis: 90°, frequency-axis: 0°)
    gradient_time: 0.5          # Gradient increase along the time-axis
    gradient_freq: -0.5         # Gradient decrease along the frequency-axis
    audio_path: ./audio.wav     # Path to the noise file
    audio_name: "clap"          # Noise file name
```




### 2. Patterned Noise Addition
Adds noise patterns such as random distributions, linear repetitions, or more complex functions to the spectrogram.


**Example: Random Shape on Range**

```yaml
pattern: "random_shape_on_range"
parameters:
    n: 50                      # Number of shapes to generate
    time_start: 0              # Start time
    time_end: 6.0              # End time
    freq_start: 0              # Start frequency
    freq_end: 800              # End frequency
    position_distribution: "uniform" # Position distribution type
    gradient_time: 0.5         # Gradient increase along the time-axis
    gradient_freq: -0.5        # Gradient decrease along the frequency-axis
    shape: "horizontal_stripe" # Shape type to generate
    parameters:
        power_distribution:
            type: "normal"     # Distribution type ("normal", "uniform", "none")
            mean: -40          # Mean value (dB power)
            stddev: 5          # Standard deviation (for "normal")
            min: -80           # Minimum value (for "uniform")
            max: -10           # Maximum value (for "uniform")
        position: (time, freq)  # Center position (time, frequency)
        size: (1, 1)            # Scale (time-axis, frequency-axis)
        rotation: 0             # Rotation angle
        gradient_time: gaussian(0,1)     # Gradient function for time-axis
        gradient_freq: gaussian(0,1)     # Gradient function for frequency-axis
        is_dot_line: false     # Is it a dotted line?
```


**Example: Linear Repeat with Sleep**

```yaml
pattern: "n_linear_repeat_t_time_sleep" # Linear repeat with sleep
parameters:
    repeat: 3                  # Number of repetitions
    repeat_time: 0.5           # Repeat every 0.5 seconds
    sleep_time: 5              # Pause for 5 seconds after repetitions
    time_start: 2.0            # Start time
    time_end: 6.0              # End time
    freq_start: 0              # Start frequency
    freq_end: 800              # End frequency
    position_distribution: "uniform" # Position distribution type
    gradient_time: 0.5         # Gradient increase along the time-axis
    gradient_freq: -0.5        # Gradient decrease along the frequency-axis
    shape: "spike"             # Shape type to generate
    parameters:
        power_distribution:
            type: "normal"     # Distribution type ("normal", "uniform", "none")
            mean: -40          # Mean value (dB power)
            stddev: 5          # Standard deviation (for "normal")
            min: -80           # Minimum value (for "uniform")
            max: -10           # Maximum value (for "uniform")
        position: (time, freq)  # Center position (time, frequency)
        size: (abs(gaussian(0,1)), abs(gaussian(0,1))) # Scale (time-axis, frequency-axis)
        rotation: 0             # Rotation angle
        gradient_time: gaussian(0,1)     # Gradient function for time-axis
        gradient_freq: gaussian(0,1)     # Gradient function for frequency-axis
```


### Python Usage

```python
import numpy as np

# Define the SpectrogramModifier and NoisePipeline classes as above

# Usage Example
sample_rate = 16000
duration = 12
n_samples = sample_rate * duration
np.random.seed(42)
signal = np.random.normal(-80, 1, n_samples)  # Generate base Gaussian signal

# Initialize SpectrogramModifier
spectro_mod = SpectrogramModifier(
    sample_rate=sample_rate,
    n_fft=1024,
    hop_length=512,
    noise_strength=600,
    noise_type='perlin',
    noise_params={'seed': 42, 'scale': 100.0}
)

# Initialize NoisePipeline
pipeline = NoisePipeline(spectro_mod)

# Add Circle Shape
pipeline.addShape(
    shape_name="circle",
    distribution_name="normal",
    shape_params={
        "position": (512, 200),
        "size": (10, 10),
        "rotation": 0,
        "gradient_time": 0.5,
        "gradient_freq": -0.5
    },
    dist_params={
        "mean": 0,
        "stddev": 1
    }
)

# Add Horizontal Stripe Shape
pipeline.addShape(
    shape_name="horizontal_stripe",
    distribution_name="uniform",
    shape_params={
        "position": (100, 300),
        "size": (abs(np.random.gaussian(0,1)), abs(np.random.gaussian(0,1))),
        "rotation": 0,
        "gradient_time": np.random.gaussian(0,1),
        "gradient_freq": np.random.gaussian(0,1),
        "is_dot_line": False
    },
    dist_params={
        "min": -1,
        "max": 1
    }
)

# Generate Spectrogram with Added Shapes
result_spectrogram = pipeline.generate(signal)

# Plot the Spectrogram
fig, ax = spectro_mod.plot_spectrogram(
    show_labels=True,
    colormap='magma',
    title='Spectrogram with Added Shapes'
)
plt.show()
```


```python
# Add Shape from JSON Example
import json

json_data = '''
{
    "shape_name": "circle",
    "distribution_name": "normal",
    "shape_params": {
        "position": [512, 200],
        "size": [10, 10],
        "rotation": 0,
        "gradient_time": 0.5,
        "gradient_freq": -0.5
    },
    "dist_params": {
        "mean": 0,
        "stddev": 1
    }
}
'''

pipeline.addShapeFromJSON(json_data)

# Add Random Shape on Range Pattern
pipeline.addPattern(
    "random_shape_on_range",
    {
        "n": 10,
        "time_start": 0,
        "time_end": 6.0,
        "freq_start": 500,
        "freq_end": 1000,
        "position_distribution": "uniform",
        "gradient_time": 0.5,
        "gradient_freq": -0.5,
        "shape": "horizontal_stripe",
        "parameters": {
            "power_distribution": {
                "type": "normal",
                "mean": -40,
                "stddev": 5
            },
            "size": (1, 1),
            "rotation": 0,
            "gradient_time": 0.5,
            "gradient_freq": -0.5
        }
    }
)

# Generate Spectrogram with Added Patterns
result_spectrogram = pipeline.generate(signal)

# Plot the Spectrogram
fig, ax = spectro_mod.plot_spectrogram(
    show_labels=True,
    colormap='magma',
    title='Spectrogram with Patterned Noise'
)
plt.show()
```


---



# Flexible and Extensible Spectrogram Noise Addition Pipeline

## Overview

This document outlines the implementation plan for a **Flexible** and **Extensible** spectrogram noise addition pipeline. The goal is to create a system that programmatically and automatically adds various noise patterns to spectrograms without relying on external configuration files like YAML or JSON. The design draws inspiration from Keras' pipeline-building approach, enabling users to build complex noise addition workflows through intuitive and modular code.

---

## **1. Architectural Overview**

### **1.1. Core Components**

1. **SpectrogramModifier**: 
   - Handles spectrogram generation from audio signals.
   - Applies noise masks based on added shapes and patterns.
   - Provides visualization capabilities.

2. **Shape Classes**:
   - Represent different noise shapes (e.g., Circle, HorizontalStripe).
   - Each shape defines how it modifies the spectrogram.

3. **Pattern Classes**:
   - Define patterns for noise addition (e.g., RandomShapePattern, LinearRepeatPattern).
   - Manage the distribution and repetition of shapes.

4. **NoisePipeline**:
   - Orchestrates the addition of shapes and patterns.
   - Manages the sequence and integration of noise elements.
   - Facilitates method chaining for a Keras-like experience.

5. **ShapeFactory**:
   - Dynamically creates instances of Shape classes.
   - Allows for easy registration of new shapes.

6. **PatternFactory**:
   - Dynamically creates instances of Pattern classes.
   - Allows for easy registration of new patterns.

7. **DistributionEngine**:
   - Manages different power distribution strategies.
   - Supports dynamic addition of new distributions.

8. **ParameterValidator**:
   - Ensures consistency and validity of parameters.
   - Prevents runtime errors due to missing or incorrect parameters.

### **1.2. Design Patterns Utilized**

- **Factory Pattern**: For dynamic creation of Shape and Pattern instances.
- **Builder Pattern**: For constructing complex NoisePipeline objects step-by-step.
- **Strategy Pattern**: For interchangeable power distribution strategies.
- **Composite Pattern**: To handle complex patterns composed of multiple shapes/patterns.

---

## **2. Implementation Strategy**

### **2.1. Transitioning to Code-Driven Automation**

To shift from configuration-driven to code-driven noise addition, the pipeline will incorporate methods that automatically generate and add random noise shapes and patterns. This approach enhances flexibility and enables dynamic noise generation without manual configurations.

### **2.2. Enhancing NoisePipeline for Automation**

- **Method Chaining**: Allow methods like `add_random_shape` and `add_random_pattern` to return `self`, enabling seamless chaining.
- **Random Parameter Generation**: Integrate randomness within these methods to vary shape types, positions, sizes, and other attributes.
- **Extensibility**: Ensure new shapes and patterns can be easily integrated without modifying existing code.

---

## **3. Detailed Component Design**

### **3.1. SpectrogramModifier**

**Responsibilities:**

- Generate spectrogram from audio signals.
- Apply noise masks to spectrogram based on shapes and patterns.
- Provide visualization capabilities.

**Key Methods:**

- `compute_spectrogram(signal)`: Computes the dB-scaled spectrogram.
- `apply_mask(mask)`: Applies a noise mask to the spectrogram.
- `plot_spectrogram()`: Visualizes the spectrogram.



...



### Final: Building the Pipeline with Automated Random Noise

```python
import numpy as np
import matplotlib.pyplot as plt

# Assuming all classes (SpectrogramModifier, ShapeFactory, Shape classes, Pattern classes, DistributionEngine, ParameterValidator, NoisePipeline) are defined as above

# Initialize SpectrogramModifier
spectro_mod = SpectrogramModifier(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    noise_strength=600,
    noise_type='gaussian',
    noise_params={'mean': 0, 'std': 1}
)

# Initialize NoisePipeline
pipeline = NoisePipeline(spectro_mod)

# Define the number of random shapes and patterns
num_random_shapes = 20
num_random_patterns = 5

# Automatically add random shapes
for _ in range(num_random_shapes):
    shape_type = np.random.choice(['circle', 'horizontal_stripe'])
    if shape_type == 'circle':
        shape = pipeline.shape_factory.create(
            "circle",
            position=(np.random.uniform(0, spectro_mod.n_fft),
                      np.random.uniform(0, spectro_mod.sample_rate / 2)),
            size=(np.random.uniform(5, 15), np.random.uniform(5, 15)),
            rotation=np.random.uniform(0, 360),
            gradient_time=np.random.uniform(-1, 1),
            gradient_freq=np.random.uniform(-1, 1),
            power_distribution=pipeline.distribution_engine.create(
                "normal",
                mean=np.random.uniform(-50, -30),
                stddev=np.random.uniform(1, 5)
            )
        )
    elif shape_type == 'horizontal_stripe':
        shape = pipeline.shape_factory.create(
            "horizontal_stripe",
            position=(np.random.uniform(0, spectro_mod.n_fft),
                      np.random.uniform(0, spectro_mod.sample_rate / 2)),
            size=(np.random.uniform(10, 50), np.random.uniform(5, 15)),
            rotation=0,  # Horizontal stripes typically have 0 rotation
            gradient_time=np.random.uniform(-1, 1),
            gradient_freq=np.random.uniform(-1, 1),
            is_dot_line=np.random.choice([True, False]),
            power_distribution=pipeline.distribution_engine.create(
                "uniform",
                min=np.random.uniform(-60, -40),
                max=np.random.uniform(-20, 0)
            )
        )
    pipeline.add_shape(shape)

# Automatically add random patterns
for _ in range(num_random_patterns):
    pattern_type = np.random.choice(['random_shape_on_range'])
    if pattern_type == 'random_shape_on_range':
        pattern = pipeline.pattern_factory.create(
            "random_shape_on_range",
            {
                "n": np.random.randint(5, 15),
                "time_start": np.random.uniform(0, 5),
                "time_end": np.random.uniform(5, 10),
                "freq_start": np.random.uniform(0, 400),
                "freq_end": np.random.uniform(400, 800),
                "position_distribution": np.random.choice(['uniform', 'normal']),
                "gradient_time": np.random.uniform(-1, 1),
                "gradient_freq": np.random.uniform(-1, 1),
                "shape": np.random.choice(['circle', 'horizontal_stripe']),
                "parameters": {
                    "power_distribution": {
                        "type": "normal",
                        "mean": np.random.uniform(-50, -30),
                        "stddev": np.random.uniform(1, 5)
                    },
                    "size": (np.random.uniform(5, 15), np.random.uniform(5, 15)),
                    "rotation": np.random.uniform(0, 360),
                    "gradient_time": np.random.uniform(-1, 1),
                    "gradient_freq": np.random.uniform(-1, 1),
                    "is_dot_line": np.random.choice([True, False])
                }
            }
        )
    pipeline.add_pattern(pattern)

# Generate a random signal
duration = 12  # seconds
sample_rate = spectro_mod.sample_rate
n_samples = sample_rate * duration
np.random.seed(42)
signal = np.random.normal(-80, 1, n_samples)  # Base Gaussian signal

# Generate spectrogram with all added shapes and patterns
result_spectrogram = pipeline.generate(signal)

# Plot the spectrogram
fig, ax = spectro_mod.plot_spectrogram(
    show_labels=True,
    colormap='magma',
    title='Spectrogram with Automated Random Noise Shapes and Patterns'
)
plt.show()
```