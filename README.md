# Signal Type Analyzer

### Advanced Signal Processing and Analysis Tool with Interactive Streamlit Interface

A comprehensive web-based application for signal processing education and analysis, featuring real-time signal generation, mathematical expression parsing, and detailed signal characterization.

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

### Core Capabilities
- **Interactive Signal Generation**: Create signals using mathematical expressions with automatic type detection
- **Real-Time Voice Recording**: Capture and analyze live audio signals with professional-grade processing  
- **Comprehensive Signal Analysis**: Energy, power, periodicity, and causality analysis with mathematical rigor
- **Educational Interface**: Detailed explanations of signal theory with visual demonstrations
- **Professional Visualizations**: Publication-quality plots with customizable parameters

### Signal Types Supported
- **Continuous-Time Signals**: Use variable 't' for functions like `sin(2*pi*t)`, `exp(-t)*Heaviside(t)`
- **Discrete-Time Signals**: Use variable 'n' for sequences like `0.8**n * Heaviside(n)`, `sin(2*pi*0.1*n)`
- **Predefined Signals**: 20+ built-in signals including unit step, impulse, ramp, sinusoids, and noise
- **Signal Operations**: Modulus, derivative, integral, and amplitude scaling transformations

### Advanced Analysis Features
- **Mathematical Causality Detection**: Automatic analysis of time shifts and future dependencies
- **Energy vs Power Classification**: Rigorous mathematical classification with detailed explanations
- **Periodicity Detection**: Automatic period identification with tolerance-based algorithms
- **Expression Parsing**: Support for complex mathematical expressions with special functions

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Audio input device (for voice recording features)

### Quick Installation
Clone the repository
git clone https://github.com/yourusername/signal-type-analyzer.git
cd signal-type-analyzer

Install dependencies
pip install -r requirements.txt

Run the application
streamlit run app.py


### Dependencies
streamlit>=1.28.0 # Web interface framework
numpy>=1.21.0 # Numerical computations
matplotlib>=3.5.0 # Visualization
pandas>=1.3.0 # Data handling
sympy>=1.9.0 # Symbolic mathematics
sounddevice>=0.4.0 # Audio recording
soundfile>=0.10.0 # Audio file processing



## Quick Start

1. **Launch the Application**:
streamlit run app.py



2. **Select Signal Input Method**:
- Choose from predefined signals
- Input custom mathematical expressions
- Record real-time voice signals
- Upload CSV files

3. **Analyze Your Signal**:
- View professional visualizations
- Get comprehensive analysis results
- Understand signal theory with detailed explanations

## Usage Examples

### Custom Signal Definition
Continuous-time signals (use 't')
"sin(2pit)" # Sine wave
"exp(-t)Heaviside(t)" # Exponential decay (causal)
"exp(-(t/2)**2)" # Gaussian pulse
"tHeaviside(t)" # Unit ramp function

Discrete-time signals (use 'n')
"0.8**n * Heaviside(n)" # Geometric sequence (causal)
"sin(2pi0.1*n)" # Discrete sine sequence
"(-1)*n" # Alternating sequence
"n(n+1)/2" # Triangular numbers



### Signal Operations
- **Modulus**: Takes absolute value |x(t)| or |x[n]|
- **Derivative**: Computes dx(t)/dt or difference x[n] - x[n-1]
- **Integral**: Computes ∫x(t)dt or cumulative sum Σx[n]
- **Amplitude Scaling**: Multiplies signal by constant factor a·x(t)

### Voice Signal Analysis
1. Select "Real-Time Voice Signal"
2. Choose recording duration (1-10 seconds)  
3. Click "Start Recording"
4. Analyze energy, power, and frequency characteristics

## File Structure

signal-type-analyzer/
├── app.py # Main Streamlit application
├── signal_utils.py # Core signal processing functions
├── sample_signals.py # Predefined signal library
├── requirements.txt # Python dependencies
└── README.md # This documentation



### Core Files Description

**app.py** - Main Streamlit Application:
- Professional tabbed interface with session state management
- Interactive parameter configuration panels
- Mathematical expression parsing with variable detection
- Real-time voice recording capabilities
- Comprehensive error handling and input validation

**signal_utils.py** - Signal Processing Engine:
- `calculate_energy()`: Computes total signal energy using E = Σ|x[n]|²
- `calculate_power()`: Calculates average power P = E/N
- `is_periodic()`: Detects signal periodicity with configurable tolerance
- `analyze_expression_causality()`: Advanced mathematical causality analysis
- `get_causality_explanation()`: Generates detailed theoretical explanations

**sample_signals.py** - Predefined Signal Library:
- **Basic Signals**: Unit step, impulse, ramp functions
- **Sinusoidal Signals**: Sine, cosine, multi-frequency combinations
- **Exponential Signals**: Growth, decay, and damped oscillations
- **Complex Signals**: Chirp, Gaussian pulses, pulse trains
- **Random Signals**: White noise, random walk sequences

## Mathematical Background

### Signal Classification Theory

**Energy Signals** (Finite Energy, Zero Average Power):
- Continuous: E = ∫₋∞^∞ |x(t)|² dt < ∞
- Discrete: E = Σₙ₌₋∞^∞ |x[n]|² < ∞
- Examples: Exponential decay, rectangular pulses

**Power Signals** (Finite Average Power, Infinite Energy):
- Continuous: P = lim(T→∞) (1/T) ∫₋T/₂^T/₂ |x(t)|² dt
- Discrete: P = lim(N→∞) (1/2N+1) Σₙ₌₋ₙ^N |x[n]|²
- Examples: Sinusoidal signals, periodic waveforms

### Causality Analysis

**Causal Systems**: Output depends only on present and past inputs
- Mathematical condition: h(t) = 0 for t < 0
- Physical realizability: Can be implemented in real-time

**Non-Causal Systems**: Require future information
- Contains terms like x(t+τ) where τ > 0
- Not physically realizable but useful for analysis

### Advanced Features

**Automatic Variable Detection**: Uses SymPy to parse expressions and detect:
- Continuous-time variables ('t') vs discrete-time variables ('n')
- Time shifts and delays in mathematical expressions
- Special functions (Heaviside, DiracDelta) for causality enforcement

**Expression Parsing Support**:
- Trigonometric functions: sin, cos, tan, sinh, cosh, tanh
- Exponential/logarithmic: exp, log, log10, sqrt
- Special functions: Heaviside (unit step), DiracDelta (unit impulse)
- Mathematical constants: pi, e (natural logarithm base)

## Key Functions Reference

### Core Processing Functions

def calculate_energy(signal):
"""Computes total energy: E = Σ|x[n]|²"""

def calculate_power(signal):
"""Computes average power: P = E/N"""

def is_periodic(signal, tolerance=1e-10):
"""Detects periodicity with numerical tolerance"""

def analyze_expression_causality(expression, variable='t'):
"""Mathematical causality analysis of expressions"""



### Interactive Features

- **Session State Management**: Preserves voice recordings and user inputs
- **Dynamic Parameter Configuration**: Adjustable time ranges and sampling rates
- **Professional Error Handling**: Comprehensive validation with helpful messages
- **Educational Content**: Detailed theory explanations with mathematical formulations

## Educational Value

Perfect for:
- **Signal Processing Courses**: Hands-on learning with immediate visual feedback
- **Engineering Students**: Practical implementation of theoretical concepts
- **Research Applications**: Professional-grade analysis tools
- **Self-Learning**: Comprehensive explanations with mathematical rigor

## Contributing

We welcome contributions! Areas for improvement:
- Additional signal transformations (FFT, filtering, convolution)
- Enhanced visualization options (3D plots, spectrograms)
- Performance optimizations for large signals
- Mobile-responsive interface improvements
- Advanced machine learning-based signal classification

### Development Setup
git clone https://github.com/yourusername/signal-type-analyzer.git
cd signal-type-analyzer
pip install -r requirements.txt



## License

MIT License - Feel free to use, modify, and distribute.

## Acknowledgments

- **Signal Processing Theory**: Oppenheim & Willsky's "Signals and Systems"
- **Streamlit Framework**: Excellent web application development platform
- **Scientific Python Stack**: NumPy, SciPy, SymPy for robust numerical computing
- **Open Source Community**: Collaborative development and knowledge sharing

## Support & Contact

- **Issues**: [GitHub Issues Page](https://github.com/yourusername/signal-type-analyzer/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/yourusername/signal-type-analyzer/discussions)
- **Academic Collaboration**: [your.email@domain.com]

---

**Made with care for signal processing education and research**

*This project demonstrates the power of combining theoretical signal processing knowledge with modern interactive web technologies for enhanced learning and analysis.*
