# Linear Regression Implementation from Scratch

## Overview
This project implements Linear Regression from scratch using NumPy. The implementation includes gradient descent optimization, MSE loss calculation, and R-squared score computation. The model supports both simple and multiple linear regression tasks.

## Features
- Custom implementation of Linear Regression algorithm
- Gradient Descent optimization
- Mean Squared Error (MSE) loss tracking
- R-squared score calculation
- Support for multiple features
- Custom exception handling
- Logging functionality

## Requirements
```
numpy
matplotlib
scikit-learn (for data preprocessing and comparison)
```

## Project Structure
```
Linear_Regression/
│
├── src/
│   ├── __init__.py
│   ├── exception.py
│   └── logger.py
│
├── lin_reg.py
├── test.py
└── README.md
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/Linear_Regression.git
cd Linear_Regression
```

2. Install required packages:
```bash
pip install numpy matplotlib scikit-learn
```

## Usage
### Basic Example
```python
from lin_reg import LinearRegression
import numpy as np

# Create sample data
X = np.random.randn(100, 3)  # 100 samples, 3 features
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(100)

# Create and train model
model = LinearRegression(learning_rate=0.01, no_of_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Get model performance
r2_score = model.score(X, y)
print(f"R-squared Score: {r2_score}")
```

### Complete Testing Script
Check `test.py` for a complete example including:
- Dataset generation
- Train-test splitting
- Feature scaling
- Model training and evaluation
- Performance visualization

## Implementation Details
The `LinearRegression` class implements:

1. **Initialization**:
   - Learning rate and number of iterations as hyperparameters
   - Weights and bias initialization

2. **Training** (`fit` method):
   - Gradient descent optimization
   - Loss history tracking
   - Parameter updates

3. **Prediction** (`predict` method):
   - Forward pass implementation
   - Input validation

4. **Evaluation** (`score` method):
   - R-squared score calculation
   - Error handling

5. **Error Handling**:
   - Custom exception class
   - Detailed error logging

## Error Handling
The implementation includes comprehensive error handling using custom exceptions:
```python
try:
    # Your code
except Exception as e:
    raise CustomException(e, sys)
```

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments
- Implementation based on gradient descent optimization
- Inspired by scikit-learn's LinearRegression implementation
- Thanks to the NumPy community for the excellent array operations library

## Contact
Your Name - niharsanoria80@gmail.com
Project Link: https://github.com/NiharSanoria1/linear_regression