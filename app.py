import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from signal_utils import calculate_energy, calculate_power, is_periodic, analyze_expression_causality, get_causality_explanation
from sample_signals import get_sample_signals
import sympy as sp

# Audio imports with error handling for deployment
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False
    st.sidebar.warning("⚠️ Audio recording disabled for cloud deployment")

st.title("Signal Type Analyzer")

# --- Initialize session state for voice recording ---
if "voice_recorded" not in st.session_state:
    st.session_state.voice_recorded = False
    st.session_state.signal = None
    st.session_state.time_axis = None

# --- Load predefined signals ---
signals = get_sample_signals()
options = list(signals.keys()) + [
    "Modulus of Signal",
    "Derivative of Signal", 
    "Integral of Signal",
    "Amplitude Scaling",
    "Custom Input"
]

# Only add voice recording option if audio is enabled
if AUDIO_ENABLED:
    options.append("Real-Time Voice Signal")

# --- Dropdown selection ---
option = st.selectbox("Select a sample signal or input your own:", options)

# --- Initialize variables ---
signal = None
time_axis = None
signal_type = 'Discrete'  # Default signal type
signal_input = None  # Store mathematical expression if applicable

# --- Helper function for proper variable detection ---
def detect_signal_variables(expression):
    """
    Properly detect variables in mathematical expressions using SymPy
    Returns tuple: (has_t, has_n, variable_names)
    """
    try:
        # Remove equation format if present
        if '=' in expression:
            expression = expression.split('=', 1)[1].strip()
        
        # Parse expression and get variables
        expr = sp.sympify(expression, evaluate=False)
        variables = expr.free_symbols
        variable_names = {str(var) for var in variables}
        
        has_t = 't' in variable_names
        has_n = 'n' in variable_names
        
        return has_t, has_n, variable_names
    except:
        return False, False, set()

# --- Predefined signals ---
if option in signals:
    signal = signals[option]
    time_axis = np.arange(len(signal))
    signal_type = 'Discrete'

# --- Signal Operations ---
elif option in ["Modulus of Signal", "Derivative of Signal", "Integral of Signal", "Amplitude Scaling"]:
    base_signal_option = st.selectbox(
        "Choose base signal:", 
        list(signals.keys()) + ["Custom Input"]
    )
    base_signal = None
    base_signal_type = 'Discrete'
    base_input_expr = None
    
    if base_signal_option == "Custom Input":
        st.subheader(f"Custom Base Signal for {option}")
        
        # Initialize session state for base signal text area if not exists
        base_signal_key = f'base_signal_input_{option.replace(" ", "_")}'
        if base_signal_key not in st.session_state:
            st.session_state[base_signal_key] = ""
        
        # Professional input configuration panel for base signal
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Base Signal Expression Input
            Define the base signal using mathematical expressions. The system will automatically detect:
            - **Continuous-time signals**: Use variable **'t'** in your expression
            - **Discrete-time signals**: Use variable **'n'** in your expression
            """)
            
            # Tabbed interface for better organization
            tab1, tab2, tab3 = st.tabs(["Input", "Examples", "Advanced"])
            
            with tab1:
                # Input area with better formatting - now connected to session state
                base_input = st.text_area(
                    "Mathematical Expression for Base Signal:",
                    height=100,
                    placeholder="Enter expression using 't' for continuous (e.g., sin(2*pi*t)) or 'n' for discrete (e.g., 2*n + 1)",
                    help="Use 't' for continuous-time signals or 'n' for discrete-time signals",
                    value=st.session_state[base_signal_key],
                    key=f"base_signal_input_area_{option.replace(' ', '_')}"
                )
                
                # Update session state when text area changes
                if base_input != st.session_state[base_signal_key]:
                    st.session_state[base_signal_key] = base_input
                
                # Optional: Support for equation format
                st.info("**Tip**: You can use equation format like `y(t) = sin(2*pi*t)` or just the expression `sin(2*pi*t)`")
            
            with tab2:
                st.markdown("""
                ### Common Signal Examples
                
                **Continuous-Time Signals (use 't'):**
                - **Unit Step**: `Heaviside(t)`
                - **Unit Impulse**: `DiracDelta(t)`
                - **Unit Ramp**: `t*Heaviside(t)`
                - **Sine Wave**: `sin(2*pi*f*t)` where f is frequency
                - **Cosine Wave**: `cos(2*pi*f*t + phi)` where phi is phase
                - **Exponential Decay**: `exp(-a*t)*Heaviside(t)` where a > 0
                - **Exponential Growth**: `exp(a*t)*Heaviside(t)` where a > 0
                - **Two-sided Exponential**: `exp(-abs(t))`
                - **Damped Sinusoid**: `exp(-a*t)*sin(2*pi*f*t)*Heaviside(t)`
                - **Chirp Signal**: `sin(2*pi*(f0 + k*t/2)*t)`
                - **Gaussian Pulse**: `exp(-(t/sigma)**2)`
                
                **Discrete-Time Signals (use 'n'):**
                - **Unit Step**: `Heaviside(n)`
                - **Unit Impulse**: `DiracDelta(n)`
                - **Unit Ramp**: `n*Heaviside(n)`
                - **Geometric Sequence**: `a**n * Heaviside(n)` where 0 < a < 1
                - **Fibonacci-like**: `n*(n+1)/2` (triangular numbers)
                - **Alternating**: `(-1)**n`
                - **Sine Sequence**: `sin(2*pi*f*n)`
                - **Exponential Sequence**: `a**n` where a is a constant
                
                **Causality Examples:**
                - **Causal**: `sin(t-5)*Heaviside(t)`, `exp(-(t-2))*Heaviside(t-2)`
                - **Non-Causal**: `sin(t+3)`, `exp(-(t+1))*Heaviside(t)`
                """)
                
                # Quick insert buttons for common signals
                st.markdown("**Quick Insert:**")
                col_ex1, col_ex2, col_ex3 = st.columns(3)
                with col_ex1:
                    if st.button("Sine Wave (t)", key=f"base_sine_t_{option}", help="Insert sin(2*pi*t)"):
                        st.session_state[base_signal_key] = "sin(2*pi*t)"
                        st.rerun()
                with col_ex2:
                    if st.button("Step Function (t)", key=f"base_step_t_{option}", help="Insert Heaviside(t)"):
                        st.session_state[base_signal_key] = "Heaviside(t)"
                        st.rerun()
                with col_ex3:
                    if st.button("Exponential (t)", key=f"base_exp_t_{option}", help="Insert exp(-t)*Heaviside(t)"):
                        st.session_state[base_signal_key] = "exp(-t)*Heaviside(t)"
                        st.rerun()
                
                col_ex4, col_ex5, col_ex6 = st.columns(3)
                with col_ex4:
                    if st.button("Sine Sequence (n)", key=f"base_sine_n_{option}", help="Insert sin(2*pi*0.1*n)"):
                        st.session_state[base_signal_key] = "sin(2*pi*0.1*n)"
                        st.rerun()
                with col_ex5:
                    if st.button("Step Sequence (n)", key=f"base_step_n_{option}", help="Insert Heaviside(n)"):
                        st.session_state[base_signal_key] = "Heaviside(n)"
                        st.rerun()
                with col_ex6:
                    if st.button("Geometric (n)", key=f"base_geom_n_{option}", help="Insert 0.8**n * Heaviside(n)"):
                        st.session_state[base_signal_key] = "0.8**n * Heaviside(n)"
                        st.rerun()
            
            with tab3:
                st.markdown("""
                ### Advanced Features
                
                **Mathematical Functions Available:**
                - **Trigonometric**: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
                - **Exponential/Log**: exp, log, log10, sqrt, abs
                - **Special**: Heaviside (unit step), DiracDelta (unit impulse)
                - **Constants**: pi, e (use as `pi` and `E` in expressions)
                
                **Operators:**
                - **Arithmetic**: +, -, *, /, ** (power)
                - **Comparison**: Use Heaviside for conditional logic
                
                **Signal Type Detection:**
                - **Continuous**: Expression contains variable **'t'** → Continuous-time signal
                - **Discrete**: Expression contains variable **'n'** → Discrete-time signal
                - **Auto-detection**: No manual selection needed!
                
                **Signal Operations:**
                - **Modulus**: Takes absolute value of base signal
                - **Derivative**: Computes difference (discrete) or derivative (continuous)
                - **Integral**: Computes cumulative sum (discrete) or integral (continuous)
                - **Amplitude Scaling**: Multiplies signal by constant factor
                
                **Tips for Professional Usage:**
                - Use parentheses for clear operator precedence
                - For causal signals, multiply by `Heaviside(t)` or `Heaviside(n)`
                - For periodic analysis, ensure integer periods in your time range
                - For energy/power analysis, consider signal duration and amplitude
                """)
        
        with col2:
            st.markdown("### Signal Parameters")
            
            # Show current detection status
            if base_input:
                has_t, has_n, var_names = detect_signal_variables(base_input)
                
                if has_t and not has_n:
                    detected_type = "Continuous-Time"
                    st.success("**Detected**: Continuous-Time Signal (contains 't')")
                elif has_n and not has_t:
                    detected_type = "Discrete-Time"
                    st.success("**Detected**: Discrete-Time Signal (contains 'n')")
                elif has_t and has_n:
                    detected_type = "Mixed Variables"
                    st.warning("**Warning**: Expression contains both 't' and 'n'")
                else:
                    detected_type = "No Variables"
                    st.error("**Error**: No 't' or 'n' variable found")
                    
                # Show detected variables
                if var_names:
                    st.info(f"**Variables detected**: {', '.join(sorted(var_names))}")
            else:
                detected_type = "None"
                st.info("Enter an expression to detect signal type")
            
            st.markdown("---")
            
            # Time/Index range configuration based on detected type
            if base_input:
                has_t, has_n, var_names = detect_signal_variables(base_input)
                
                if has_t and not has_n:
                    st.markdown("**Time Domain Settings:**")
                    t_min = st.number_input(
                        "Start Time (t₀)", 
                        value=-5.0, 
                        step=0.1,
                        format="%.2f",
                        help="Starting time for signal evaluation",
                        key=f"base_t_min_{option}"
                    )
                    t_max = st.number_input(
                        "End Time (t₁)", 
                        value=5.0, 
                        step=0.1,
                        format="%.2f",
                        help="Ending time for signal evaluation",
                        key=f"base_t_max_{option}"
                    )
                    
                    num_samples = st.number_input(
                        "Number of Samples", 
                        value=1000, 
                        min_value=100, 
                        max_value=10000,
                        step=100,
                        help="Resolution of the continuous signal",
                        key=f"base_num_samples_{option}"
                    )
                    
                    # Display computed parameters
                    st.info(f"**Δt**: {(t_max-t_min)/num_samples:.4f}s\n**Duration**: {t_max-t_min:.2f}s")
                    
                elif has_n and not has_t:
                    st.markdown("**Discrete Domain Settings:**")
                    n_start = st.number_input(
                        "Start Index (n₀)", 
                        value=0, 
                        step=1,
                        help="Starting index for discrete signal",
                        key=f"base_n_start_{option}"
                    )
                    n_end = st.number_input(
                        "End Index (n₁)", 
                        value=20, 
                        step=1,
                        help="Ending index for discrete signal",
                        key=f"base_n_end_{option}"
                    )
                    
                    st.info(f"**Samples**: {n_end-n_start+1}\n**Range**: [{n_start}, {n_end}]")
            
            # Validation and preview
            st.markdown("### Input Validation")
            
            if base_input:
                # Basic syntax validation
                try:
                    has_t, has_n, var_names = detect_signal_variables(base_input)
                    st.success("Expression syntax is valid")
                        
                except Exception as e:
                    st.error(f"Syntax Error: {str(e)[:100]}")
            else:
                st.info("Enter an expression to validate")
        
        # Process the base signal input
        if base_input:
            try:
                has_t, has_n, var_names = detect_signal_variables(base_input)
                
                if has_t and not has_n:
                    base_signal_type = 'Continuous'
                    t = np.linspace(t_min, t_max, num_samples)
                    expr = sp.sympify(base_input.split('=', 1)[1].strip() if '=' in base_input else base_input, evaluate=False)
                    f = sp.lambdify(sp.Symbol('t'), expr, modules=['numpy', {
                        'Heaviside': lambda x: np.heaviside(x, 1), 
                        'DiracDelta': lambda x: np.isclose(x, 0, atol=1e-3).astype(float)
                    }])
                    base_signal = f(t)
                    time_axis = t
                elif has_n and not has_t:
                    base_signal_type = 'Discrete'
                    n = np.arange(n_start, n_end + 1)
                    expr = sp.sympify(base_input.split('=', 1)[1].strip() if '=' in base_input else base_input, evaluate=False)
                    f = sp.lambdify(sp.Symbol('n'), expr, modules=['numpy', {
                        'Heaviside': lambda x: np.heaviside(x, 1), 
                        'DiracDelta': lambda x: (x==0).astype(float)
                    }])
                    base_signal = f(n)
                    time_axis = n
                elif has_t and has_n:
                    st.error("Expression contains both 't' and 'n' as variables. Please use only one variable.")
                    base_signal = None
                else:
                    st.error("Expression must contain either 't' (continuous) or 'n' (discrete) as a variable.")
                    base_signal = None
                    
                base_input_expr = base_input
            except Exception as e:
                st.error(f"Invalid input: {e}")
                base_signal = None
    else:
        base_signal = signals[base_signal_option]
        time_axis = np.arange(len(base_signal))
        base_signal_type = 'Discrete'
    
    # Apply operation
    if base_signal is not None and len(base_signal) > 0:
        try:
            if option == "Modulus of Signal":
                signal = np.abs(base_signal)
            elif option == "Derivative of Signal":
                if len(base_signal) > 1:
                    signal = np.diff(base_signal, prepend=base_signal[0] if len(base_signal) > 0 else 0)
                else:
                    signal = np.array([0])
            elif option == "Integral of Signal":
                signal = np.cumsum(base_signal)
            elif option == "Amplitude Scaling":
                factor = st.number_input("Enter scaling factor:", value=1.0)
                signal = base_signal * factor
            
            signal_type = base_signal_type
            signal_input = base_input_expr
        except Exception as e:
            st.error(f"Error in signal operation: {e}")
            signal = None

# --- Custom Input ---
elif option == "Custom Input":
    st.subheader("Custom Signal Definition")
    
    # Initialize session state for text area if not exists
    if 'custom_input_text' not in st.session_state:
        st.session_state.custom_input_text = ""
    
    # Professional input configuration panel
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Signal Expression Input
        Define your signal using mathematical expressions. The system will automatically detect:
        - **Continuous-time signals**: Use variable **'t'** in your expression
        - **Discrete-time signals**: Use variable **'n'** in your expression
        """)
        
        # Tabbed interface for better organization
        tab1, tab2, tab3 = st.tabs(["Input", "Examples", "Advanced"])
        
        with tab1:
            # Input area with better formatting - now connected to session state
            signal_input = st.text_area(
                "Mathematical Expression:",
                height=100,
                placeholder="Enter expression using 't' for continuous (e.g., sin(2*pi*t)) or 'n' for discrete (e.g., 2*n + 1)",
                help="Use 't' for continuous-time signals or 'n' for discrete-time signals",
                value=st.session_state.custom_input_text,
                key="signal_input_area"
            )
            
            # Update session state when text area changes
            if signal_input != st.session_state.custom_input_text:
                st.session_state.custom_input_text = signal_input
            
            # Optional: Support for equation format
            st.info("**Tip**: You can use equation format like `y(t) = sin(2*pi*t)` or just the expression `sin(2*pi*t)`")
        
        with tab2:
            st.markdown("""
            ### Common Signal Examples
            
            **Continuous-Time Signals (use 't'):**
            - **Unit Step**: `Heaviside(t)`
            - **Unit Impulse**: `DiracDelta(t)`
            - **Unit Ramp**: `t*Heaviside(t)`
            - **Sine Wave**: `sin(2*pi*f*t)` where f is frequency
            - **Cosine Wave**: `cos(2*pi*f*t + phi)` where phi is phase
            - **Exponential Decay**: `exp(-a*t)*Heaviside(t)` where a > 0
            - **Exponential Growth**: `exp(a*t)*Heaviside(t)` where a > 0
            - **Two-sided Exponential**: `exp(-abs(t))`
            - **Damped Sinusoid**: `exp(-a*t)*sin(2*pi*f*t)*Heaviside(t)`
            - **Chirp Signal**: `sin(2*pi*(f0 + k*t/2)*t)`
            - **Gaussian Pulse**: `exp(-(t/sigma)**2)`
            
            **Discrete-Time Signals (use 'n'):**
            - **Unit Step**: `Heaviside(n)`
            - **Unit Impulse**: `DiracDelta(n)`
            - **Unit Ramp**: `n*Heaviside(n)`
            - **Geometric Sequence**: `a**n * Heaviside(n)` where 0 < a < 1
            - **Fibonacci-like**: `n*(n+1)/2` (triangular numbers)
            - **Alternating**: `(-1)**n`
            - **Sine Sequence**: `sin(2*pi*f*n)`
            - **Exponential Sequence**: `a**n` where a is a constant
            
            **Causality Examples:**
            - **Causal**: `sin(t-5)*Heaviside(t)`, `exp(-(t-2))*Heaviside(t-2)`
            - **Non-Causal**: `sin(t+3)`, `exp(-(t+1))*Heaviside(t)`
            """)
            
            # Quick insert buttons for common signals
            st.markdown("**Quick Insert:**")
            col_ex1, col_ex2, col_ex3 = st.columns(3)
            with col_ex1:
                if st.button("Sine Wave (t)", help="Insert sin(2*pi*t)"):
                    st.session_state.custom_input_text = "sin(2*pi*t)"
                    st.rerun()
            with col_ex2:
                if st.button("Step Function (t)", help="Insert Heaviside(t)"):
                    st.session_state.custom_input_text = "Heaviside(t)"
                    st.rerun()
            with col_ex3:
                if st.button("Exponential (t)", help="Insert exp(-t)*Heaviside(t)"):
                    st.session_state.custom_input_text = "exp(-t)*Heaviside(t)"
                    st.rerun()
            
            col_ex4, col_ex5, col_ex6 = st.columns(3)
            with col_ex4:
                if st.button("Sine Sequence (n)", help="Insert sin(2*pi*0.1*n)"):
                    st.session_state.custom_input_text = "sin(2*pi*0.1*n)"
                    st.rerun()
            with col_ex5:
                if st.button("Step Sequence (n)", help="Insert Heaviside(n)"):
                    st.session_state.custom_input_text = "Heaviside(n)"
                    st.rerun()
            with col_ex6:
                if st.button("Geometric (n)", help="Insert 0.8**n * Heaviside(n)"):
                    st.session_state.custom_input_text = "0.8**n * Heaviside(n)"
                    st.rerun()
        
        with tab3:
            st.markdown("""
            ### Advanced Features
            
            **Mathematical Functions Available:**
            - **Trigonometric**: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
            - **Exponential/Log**: exp, log, log10, sqrt, abs
            - **Special**: Heaviside (unit step), DiracDelta (unit impulse)
            - **Constants**: pi, e (use as `pi` and `E` in expressions)
            
            **Operators:**
            - **Arithmetic**: +, -, *, /, ** (power)
            - **Comparison**: Use Heaviside for conditional logic
            
            **Signal Type Detection:**
            - **Continuous**: Expression contains variable **'t'** → Continuous-time signal
            - **Discrete**: Expression contains variable **'n'** → Discrete-time signal
            - **Auto-detection**: No manual selection needed!
            
            **Causality Analysis:**
            - **Time Shifts**: The system analyzes expressions like `f(t+5)` vs `f(t-5)`
            - **Causal**: `f(t-delay)` uses past values → Physically realizable
            - **Non-Causal**: `f(t+advance)` requires future values → Not physically realizable
            
            **Tips for Professional Usage:**
            - Use parentheses for clear operator precedence
            - For causal signals, multiply by `Heaviside(t)` or `Heaviside(n)`
            - For periodic analysis, ensure integer periods in your time range
            - For energy/power analysis, consider signal duration and amplitude
            """)
    
    with col2:
        st.markdown("### Signal Parameters")
        
        # Show current detection status
        if signal_input:
            has_t, has_n, var_names = detect_signal_variables(signal_input)
            
            if has_t and not has_n:
                detected_type = "Continuous-Time"
                st.success("**Detected**: Continuous-Time Signal (contains 't')")
            elif has_n and not has_t:
                detected_type = "Discrete-Time"
                st.success("**Detected**: Discrete-Time Signal (contains 'n')")
            elif has_t and has_n:
                detected_type = "Mixed Variables"
                st.warning("**Warning**: Expression contains both 't' and 'n'")
            else:
                detected_type = "No Variables"
                st.error("**Error**: No 't' or 'n' variable found")
                
            # Show detected variables
            if var_names:
                st.info(f"**Variables detected**: {', '.join(sorted(var_names))}")
        else:
            detected_type = "None"
            st.info("Enter an expression to detect signal type")
        
        st.markdown("---")
        
        # Time/Index range configuration based on detected type
        if signal_input:
            has_t, has_n, var_names = detect_signal_variables(signal_input)
            
            if has_t and not has_n:
                st.markdown("**Time Domain Settings:**")
                t_min = st.number_input(
                    "Start Time (t₀)", 
                    value=-5.0, 
                    step=0.1,
                    format="%.2f",
                    help="Starting time for signal evaluation"
                )
                t_max = st.number_input(
                    "End Time (t₁)", 
                    value=5.0, 
                    step=0.1,
                    format="%.2f",
                    help="Ending time for signal evaluation"
                )
                
                num_samples = st.number_input(
                    "Number of Samples", 
                    value=1000, 
                    min_value=100, 
                    max_value=10000,
                    step=100,
                    help="Resolution of the continuous signal"
                )
                
                # Display computed parameters
                st.info(f"**Δt**: {(t_max-t_min)/num_samples:.4f}s\n**Duration**: {t_max-t_min:.2f}s")
                
            elif has_n and not has_t:
                st.markdown("**Discrete Domain Settings:**")
                n_start = st.number_input(
                    "Start Index (n₀)", 
                    value=0, 
                    step=1,
                    help="Starting index for discrete signal"
                )
                n_end = st.number_input(
                    "End Index (n₁)", 
                    value=50, 
                    step=1,
                    help="Ending index for discrete signal"
                )
                
                st.info(f"**Samples**: {n_end-n_start+1}\n**Range**: [{n_start}, {n_end}]")
        
        # Validation and preview
        st.markdown("### Input Validation")
        
        if signal_input:
            # Basic syntax validation
            try:
                has_t, has_n, var_names = detect_signal_variables(signal_input)
                st.success("Expression syntax is valid")
                    
            except Exception as e:
                st.error(f"Syntax Error: {str(e)[:100]}")
        else:
            st.info("Enter an expression to validate")
    
    # Process the input
    if signal_input:
        try:
            has_t, has_n, var_names = detect_signal_variables(signal_input)
            
            # Extract right-hand side if input is in form y(t)=x(t)
            if '=' in signal_input:
                signal_input = signal_input.split('=', 1)[1].strip()
                
            # Auto-detect signal type based on actual variables (not substrings)
            if has_t and not has_n:
                signal_type = 'Continuous'
                t = np.linspace(t_min, t_max, num_samples)
                expr = sp.sympify(signal_input, evaluate=False)
                f = sp.lambdify(sp.Symbol('t'), expr, modules=['numpy', {
                    'Heaviside': lambda x: np.heaviside(x, 1), 
                    'DiracDelta': lambda x: np.isclose(x, 0, atol=1e-3).astype(float)
                }])
                signal = f(t)
                time_axis = t
                
            elif has_n and not has_t:
                signal_type = 'Discrete'
                n = np.arange(n_start, n_end + 1)
                expr = sp.sympify(signal_input, evaluate=False)
                f = sp.lambdify(sp.Symbol('n'), expr, modules=['numpy', {
                    'Heaviside': lambda x: np.heaviside(x, 1), 
                    'DiracDelta': lambda x: (x==0).astype(float)
                }])
                signal = f(n)
                time_axis = n
            
            elif has_t and has_n:
                st.error("Expression contains both 't' and 'n' as variables. Please use only one variable.")
                signal = None
                
            else:
                st.error("Expression must contain either 't' (continuous) or 'n' (discrete) as a variable.")
                signal = None
                
        except Exception as e:
            st.error(f"Error evaluating expression: {e}")
            signal = None
            signal_type = None
            time_axis = None

# --- Real-Time Voice Signal (only if audio is enabled) ---
elif option == "Real-Time Voice Signal" and AUDIO_ENABLED:
    st.subheader("Input Voice Signal")
    input_mode = st.radio("Choose Input Mode:", ["Record Real-Time Voice", "Upload WAV File"])
    
    if input_mode == "Record Real-Time Voice":
        duration = st.slider("Select recording duration (seconds):", 1, 10, 3)
        sample_rate = 44100
        
        if not st.session_state.voice_recorded:
            if st.button("Start Recording"):
                try:
                    st.info("Recording in progress...")
                    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                    sd.wait()
                    signal = recording.flatten()
                    time_axis = np.linspace(0, duration, len(signal))
                    st.session_state.signal = signal
                    st.session_state.time_axis = time_axis
                    st.session_state.voice_recorded = True
                    st.success("Recording completed!")
                except Exception as e:
                    st.error(f"Recording failed: {e}")
        else:
            st.success("Voice already recorded.")
            if st.button("Record Again"):
                try:
                    st.info("Re-recording in progress...")
                    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                    sd.wait()
                    signal = recording.flatten()
                    time_axis = np.linspace(0, duration, len(signal))
                    st.session_state.signal = signal
                    st.session_state.time_axis = time_axis
                    st.success("Re-recording completed!")
                except Exception as e:
                    st.error(f"Re-recording failed: {e}")
                    
        if st.session_state.voice_recorded and st.session_state.signal is not None:
            signal = st.session_state.signal
            time_axis = st.session_state.time_axis
            signal_type = 'Continuous'
            
    else:
        uploaded_audio = st.file_uploader("Upload your .wav file", type=['wav'])
        if uploaded_audio is not None:
            try:
                data, sample_rate = sf.read(uploaded_audio)
                if data.ndim > 1:  # Handle multi-channel audio
                    signal = data[:, 0]  # Take first channel
                else:
                    signal = data
                duration = len(signal) / sample_rate
                time_axis = np.linspace(0, duration, len(signal))
                signal_type = 'Continuous'
                st.success("Audio file loaded successfully.")
            except Exception as e:
                st.error(f"Error loading audio file: {e}")

# --- CSV Upload Support ---
uploaded_file = st.file_uploader("Or upload a CSV file (with columns 'time' and 'amplitude')", type=['csv'])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if 'time' in data.columns and 'amplitude' in data.columns:
            time_axis = data['time'].values
            signal = data['amplitude'].values
            signal_type = 'Discrete'
            st.success("CSV file loaded successfully.")
        else:
            st.error("CSV must contain 'time' and 'amplitude' columns.")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# --- Visualization & Analysis ---
if signal is not None and time_axis is not None and len(signal) > 0:
    # Ensure signal and time_axis have same length
    min_length = min(len(signal), len(time_axis))
    signal = signal[:min_length]
    time_axis = time_axis[:min_length]
    
    st.subheader("Signal Visualization")
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Make the zero of both axes common and remove space between axes
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    # Remove space between axes by setting limits to include zero
    x_min, x_max = np.min(time_axis), np.max(time_axis)
    y_min, y_max = np.min(signal), np.max(signal)
    
    # Handle case where min and max are the same
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    if y_min == y_max:
        y_min, y_max = y_min - 1, y_max + 1
        
    ax.set_xlim(left=min(0, x_min), right=max(0, x_max))
    ax.set_ylim(bottom=min(0, y_min), top=max(0, y_max))
    
    if option == "Custom Input":
        if signal_type == "Discrete":
            ax.stem(time_axis, signal, linefmt='b-', markerfmt='bo', basefmt='r-')
            ax.set_xlabel('n (samples)')
            ax.set_title("Discrete-Time Signal")
        else:
            ax.plot(time_axis, signal)
            ax.set_xlabel('t (seconds)')
            ax.set_title("Continuous-Time Signal")
    elif option == "Real-Time Voice Signal" and AUDIO_ENABLED:
        display_type = st.radio("Choose waveform type to display:", ["Continuous-Time", "Discrete-Time"])
        if display_type == "Discrete-Time":
            # Downsample for discrete display if signal is too long
            if len(signal) > 1000:
                step = len(signal) // 1000
                display_signal = signal[::step]
                display_indices = np.arange(0, len(signal), step)
            else:
                display_signal = signal
                display_indices = np.arange(len(signal))
            ax.stem(display_indices, display_signal, linefmt='b-', markerfmt='bo', basefmt='r-')
            ax.set_xlabel('n (samples)')
            ax.set_title("Discrete-Time Representation")
        else:
            ax.plot(time_axis, signal)
            ax.set_xlabel('t (seconds)')
            ax.set_title("Continuous-Time Representation")
    else:
        signal_type_option = st.radio("Select Signal Type:", ["Discrete-Time", "Continuous-Time"])
        if signal_type_option == "Discrete-Time":
            ax.stem(time_axis, signal, linefmt='b-', markerfmt='bo', basefmt='r-')
            ax.set_xlabel('n (samples)')
        else:
            ax.plot(time_axis, signal)
            ax.set_xlabel('t (seconds)')
        ax.set_title("Signal Visualization")
    
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # --- Signal Analysis ---
    try:
        energy = calculate_energy(signal)
        power = calculate_power(signal)
        periodic, period = is_periodic(signal)
        
        # --- Advanced Mathematical Causality Detection ---
        causal = True
        causality_analysis = None
        
        if option == "Custom Input" and signal_input:
            # Use the new mathematical causality analysis
            has_t, has_n, var_names = detect_signal_variables(signal_input)
            variable = 't' if has_t else 'n' if has_n else 't'
            causal, causality_analysis = analyze_expression_causality(signal_input, variable)
        elif option == "Real-Time Voice Signal":
            # Voice signals recorded from t=0 are always causal
            causal = True
        elif option in signals.keys():
            # For predefined signals, assume causal
            causal = True
        elif option in ["Modulus of Signal", "Derivative of Signal", "Integral of Signal", "Amplitude Scaling"]:
            # For signal operations, inherit causality from base signal
            if 'base_input_expr' in locals() and base_input_expr:
                has_t, has_n, var_names = detect_signal_variables(base_input_expr)
                variable = 't' if has_t else 'n' if has_n else 't'
                causal, causality_analysis = analyze_expression_causality(base_input_expr, variable)
            else:
                causal = True
        else:
            # Default case
            causal = True
        
        # --- Display Analysis ---
        st.subheader("Analysis Results")
        st.write(f"**Energy**: {energy:.4f}")
        st.write(f"**Power**: {power:.4f}")
        
        # Improved classification logic
        if energy < np.inf and power == 0:
            st.write("Classified as **Energy Signal**")
        elif power > 0 and power < np.inf:
            st.write("Classified as **Power Signal**")
        else:
            st.write("Neither Energy nor Power Signal")
            
        st.write(f"**Periodic**: {'Yes' if periodic else 'No'}" + (f" (Period = {period})" if periodic else ""))
        st.write(f"**Causal**: {'Yes' if causal else 'No'}")
        
        # --- Educational Section: Energy and Power Signals ---
        st.subheader("Understanding Energy and Power Signals")
        
        with st.expander("**Signal Theory Explanation**", expanded=False):
            st.markdown("""
            ### **Energy Signals vs Power Signals**
            
            Signals can be classified into two fundamental categories based on their energy and power characteristics:
            
            #### **Energy Signals**
            - **Definition**: Signals with **finite energy** and **zero average power**
            - **Mathematical Condition**: 
              - Energy: $E = \\int_{-\\infty}^{\\infty} |x(t)|^2 dt < \\infty$ (continuous)
              - Energy: $E = \\sum_{n=-\\infty}^{\\infty} |x[n]|^2 < \\infty$ (discrete)
              - Average Power: $P = 0$
            - **Examples**: Exponentially decaying signals, rectangular pulses, Gaussian pulses
            
            #### **Power Signals**  
            - **Definition**: Signals with **finite average power** and **infinite energy**
            - **Mathematical Condition**:
              - Average Power: $P = \\lim_{T \\to \\infty} \\frac{1}{T} \\int_{-T/2}^{T/2} |x(t)|^2 dt$ (continuous)
              - Average Power: $P = \\lim_{N \\to \\infty} \\frac{1}{2N+1} \\sum_{n=-N}^{N} |x[n]|^2$ (discrete)
              - Energy: $E = \\infty$
            - **Examples**: Sinusoidal signals, periodic signals, random noise
            
            #### **Neither Energy nor Power**
            - Signals like $x(t) = t$ (ramp) or $x(t) = e^t$ have both infinite energy and infinite power
            """)
        
        with st.expander("**Energy Calculation for Current Signal**", expanded=True):
            st.markdown("### **Step-by-Step Energy Calculation**")
            
            # Show the mathematical formula being used
            if signal_type == 'Discrete' or len(time_axis) < 2000:
                st.latex(r"E = \sum_{n} |x[n]|^2")
                st.write("**For discrete signals, energy is calculated as the sum of squared magnitudes:**")
            else:
                st.latex(r"E = \int |x(t)|^2 dt \approx \sum_{i} |x[i]|^2 \cdot \Delta t")
                st.write("**For continuous signals, energy is approximated using numerical integration:**")
            
            # Calculate and display step-by-step
            signal_squared = np.abs(signal)**2
            
            # Show sample calculations (first 10 points)
            st.write("**Sample calculation (first 10 points):**")
            calc_df = pd.DataFrame({
                'Index/Time': time_axis[:min(10, len(time_axis))],
                'x[n] or x(t)': signal[:min(10, len(signal))],
                '|x|²': signal_squared[:min(10, len(signal_squared))]
            })
            st.dataframe(calc_df, use_container_width=True)
            
            if len(signal) > 10:
                st.write("... (calculation continues for all signal points)")
            
            # Show the final calculation
            if signal_type == 'Continuous' and len(time_axis) > 1:
                dt = time_axis[1] - time_axis[0] if len(time_axis) > 1 else 1
                st.write(f"**Sampling interval (Δt)**: {dt:.6f}")
                st.write(f"**Sum of |x|²**: {np.sum(signal_squared):.6f}")
                st.write(f"**Final Energy**: {energy:.6f} = {np.sum(signal_squared):.6f} × {dt:.6f}")
            else:
                st.write(f"**Sum of |x[n]|²**: {np.sum(signal_squared):.6f}")
                st.write(f"**Final Energy**: {energy:.6f}")
            
            # Power calculation explanation
            st.markdown("### **Average Power Calculation**")
            if signal_type == 'Discrete':
                st.latex(r"P = \frac{1}{N} \sum_{n=0}^{N-1} |x[n]|^2")
            else:
                st.latex(r"P = \frac{1}{T} \int_0^T |x(t)|^2 dt")
            
            N = len(signal)
            if signal_type == 'Continuous' and len(time_axis) > 1:
                T = time_axis[-1] - time_axis[0]
                st.write(f"**Signal Duration (T)**: {T:.6f}")
                st.write(f"**Average Power**: {power:.6f} = {energy:.6f} / {T:.6f}")
            else:
                st.write(f"**Number of samples (N)**: {N}")
                st.write(f"**Average Power**: {power:.6f} = {energy:.6f} / {N}")
        
        with st.expander("**Signal Classification Logic**", expanded=False):
            st.markdown("""
            ### **Classification Decision Tree**
            
            The classification follows this logic:
            
            1. **Calculate Energy (E)** and **Average Power (P)**
            2. **Apply Classification Rules**:
               - If `E < ∞` and `P ≈ 0` → **Energy Signal**
               - If `P > 0` and `P < ∞` → **Power Signal** 
               - If `E = ∞` and `P = ∞` → **Neither**
            
            ### **For Your Current Signal**:
            """)
            
            # Current signal analysis
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Energy", f"{energy:.4f}", 
                         "Finite" if energy < 1e10 else "Large/Infinite")
            with col2:
                st.metric("Power", f"{power:.4f}",
                         "Non-zero" if power > 1e-10 else "~Zero")
            
            # Classification explanation
            if energy < 1e10 and power < 1e-6:
                st.success("**Classification**: Energy Signal - Energy is finite and power is approximately zero")
            elif power > 1e-6 and power < 1e10:
                st.success("**Classification**: Power Signal - Power is finite and non-zero") 
            else:
                st.warning("**Classification**: Neither - Both energy and power are very large")
            
            st.markdown("""
            ### **Physical Interpretation**
            - **Energy signals** represent transient phenomena (like a lightning strike)
            - **Power signals** represent continuous phenomena (like AC voltage)
            - **Neither category** often represents growing/unbounded signals
            """)
        
        # --- Educational Section: Periodicity Analysis ---
        st.subheader("Understanding Signal Periodicity")
        
        with st.expander("**Periodicity Theory Explanation**", expanded=False):
            st.markdown("""
            ### **What Makes a Signal Periodic?**
            
            A signal is **periodic** if it repeats its pattern at regular intervals.
            
            #### **Mathematical Definition**
            - **Continuous-time**: $x(t) = x(t + T)$ for all $t$, where $T > 0$ is the **fundamental period**
            - **Discrete-time**: $x[n] = x[n + N]$ for all $n$, where $N > 0$ is the **fundamental period**
            
            #### **Key Properties**
            - **Fundamental Period**: The **smallest positive value** of $T$ or $N$ for which the condition holds
            - **Frequency**: $f = \\frac{1}{T}$ (continuous) or $f = \\frac{1}{N}$ (discrete)
            - **Angular Frequency**: $\\omega = 2\\pi f = \\frac{2\\pi}{T}$
            
            #### **Common Periodic Signals**
            - **Sinusoidal**: $\\sin(\\omega t)$, $\\cos(\\omega t)$ with period $T = \\frac{2\\pi}{\\omega}$
            - **Square Wave**: Alternating high/low values with fixed period
            - **Sawtooth Wave**: Linear ramp that resets periodically
            - **Complex Exponential**: $e^{j\\omega t}$ with period $T = \\frac{2\\pi}{\\omega}$
            
            #### **Non-Periodic (Aperiodic) Signals**
            - **Exponential Decay**: $e^{-at}$ (never repeats exactly)
            - **Chirp Signals**: Frequency changes over time
            - **Random Noise**: No predictable pattern
            - **Impulse**: $\\delta(t)$ occurs only once
            """)
        
        with st.expander("**Periodicity Detection for Current Signal**", expanded=True):
            st.markdown("### **Periodicity Detection Algorithm**")
            
            if periodic:
                st.success(f"**Signal is PERIODIC** with period = {period}")
            else:
                st.info("**Signal is APERIODIC** (non-periodic)")
            
            st.markdown("""
            #### **Detection Method:**
            The algorithm checks if the signal repeats by:
            1. **Testing different period lengths** from 2 to N/2 samples
            2. **Comparing signal segments** for exact matches
            3. **Finding the smallest period** that satisfies the repetition condition
            """)
            
            # Show periodicity analysis details
            if len(signal) >= 4:  # Need minimum length for analysis
                st.write("**Detailed Analysis:**")
                
                max_period_to_check = min(len(signal) // 2, 20)  # Limit display
                test_periods = range(2, max(3, max_period_to_check + 1))
                
                period_results = []
                for test_p in test_periods:
                    if test_p < len(signal):
                        # Check if signal repeats with this period
                        is_periodic_with_p = True
                        for i in range(len(signal) - test_p):
                            if abs(signal[i] - signal[i + test_p]) > 1e-10:
                                is_periodic_with_p = False
                                break
                        
                        status = "Match" if is_periodic_with_p else "No match"
                        period_results.append({
                            "Test Period": test_p,
                            "Status": status,
                            "Sample Comparison": f"{signal[0]:.3f} vs {signal[test_p % len(signal)]:.3f}"
                        })
                        
                        if len(period_results) >= 10:  # Limit display
                            break
                
                if period_results:
                    results_df = pd.DataFrame(period_results)
                    st.dataframe(results_df, use_container_width=True)
                
                if periodic:
                    st.write(f"**Found repetition at period {period}:**")
                    # Show a few cycles if period is reasonable
                    if period <= 10 and period * 3 <= len(signal):
                        cycles_to_show = min(3, len(signal) // period)
                        for cycle in range(cycles_to_show):
                            start_idx = cycle * period
                            end_idx = start_idx + period
                            cycle_values = signal[start_idx:end_idx]
                            st.write(f"   Cycle {cycle + 1}: {[f'{val:.3f}' for val in cycle_values]}")
                else:
                    st.write("**No repetition found** - Signal is aperiodic")
            else:
                st.write("Signal too short for meaningful periodicity analysis")
        
        # --- Educational Section: Advanced Causality Analysis ---
        st.subheader("Understanding Signal Causality")
        
        with st.expander("**Causality Theory Explanation**", expanded=False):
            st.markdown("""
            ### **What Makes a Signal Causal?**
            
            A signal is **causal** if it is zero for all negative time (or negative indices).
            
            #### **Mathematical Definition**
            - **Continuous-time**: $x(t) = 0$ for all $t < 0$
            - **Discrete-time**: $x[n] = 0$ for all $n < 0$
            
            #### **Physical Significance**
            - **Real-world systems** are causal - they cannot respond before an input is applied
            - **Causal signals** represent **physically realizable** phenomena
            - **Non-causal signals** may exist mathematically but not physically
            
            #### **Time Shift Analysis**
            
            **Causal Time Shifts:**
            - **Past Reference**: $x(t-5)$ at $t=1$ → $x(-4)$ (uses past value)
            - **Present Reference**: $x(t)$ at $t=1$ → $x(1)$ (uses present value)
            - **Delayed Response**: System output depends only on current and past inputs
            
            **Non-Causal Time Shifts:**
            - **Future Reference**: $x(t+5)$ at $t=1$ → $x(6)$ (requires future value)
            - **Predictive Response**: System would need to "see the future"
            - **Not physically realizable** in real-time systems
            
            #### **Engineering Implications**
            - **Filter Design**: Causal filters have practical delay
            - **Control Systems**: Causal controllers ensure stability
            - **Signal Processing**: Real-time systems require causality
            """)
        
        with st.expander("**Advanced Causality Analysis**", expanded=True):
            st.markdown("### **Mathematical Expression Analysis**")
            
            if causality_analysis and (option == "Custom Input" or option in ["Modulus of Signal", "Derivative of Signal", "Integral of Signal", "Amplitude Scaling"]):
                # Show detailed mathematical analysis
                explanation = get_causality_explanation(causal, causality_analysis)
                st.markdown(explanation)
                
                # Show time shift details if any were found
                if causality_analysis.get('shifts_found'):
                    st.markdown("### **Time Shift Examples**")
                    
                    example_time = 1.0 if 't' in (signal_input or locals().get('base_input_expr', '') or "") else 1
                    var_name = 't' if 't' in (signal_input or locals().get('base_input_expr', '') or "") else 'n'
                    
                    for shift in causality_analysis['shifts_found']:
                        shift_expr = shift['expression']
                        shift_val = shift['shift']
                        
                        # Calculate what value this would reference
                        reference_time = example_time + shift_val
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.code(f"At {var_name}={example_time}:")
                            st.code(f"{shift_expr} → references {var_name}={reference_time}")
                        
                        with col2:
                            if shift_val > 0:
                                st.error(f"**Future reference** (non-causal)")
                                st.write(f"Requires future value at {var_name}={reference_time}")
                            elif shift_val < 0:
                                st.success(f"**Past reference** (causal)")
                                st.write(f"Uses past value at {var_name}={reference_time}")
                            else:
                                st.info(f"**Present reference** (causal)")
                                st.write(f"Uses current value at {var_name}={reference_time}")
                
            elif option == "Custom Input" and signal_input:
                # Basic analysis for expressions without detected shifts
                st.success("**Signal is CAUSAL**") if causal else st.error("**Signal is NON-CAUSAL**")
                
                st.markdown("### **Basic Analysis**")
                st.write("**Expression Analysis:**")
                
                # Show what type of signal this appears to be
                expr_lower = signal_input.lower().replace(' ', '')
                
                if 'heaviside' in expr_lower:
                    st.write("• Contains Heaviside step function → Likely causal")
                elif any(func in expr_lower for func in ['sin', 'cos', 'tan']):
                    if 'heaviside' not in expr_lower:
                        st.write("• Trigonometric function without step → Non-causal (extends to -∞)")
                    else:
                        st.write("• Trigonometric function with step → Made causal")
                elif 'exp' in expr_lower:
                    st.write("• Exponential function → Analysis depends on form")
                
                st.info("**Tip**: Use expressions like `f(t-delay)*Heaviside(t)` to ensure causality")
                
            elif option == "Real-Time Voice Signal":
                st.success("**Voice recordings are inherently causal** (start from t=0)")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Recording Start", "t = 0")
                    st.metric("Duration", f"{np.max(time_axis):.3f}s")
                with col2:
                    st.metric("Total Samples", len(signal))
                    st.metric("Sample Rate", "44.1 kHz")
                    
            else:
                st.success("**Signal is assumed CAUSAL**")
                st.write("**Analysis:**")
                if option in signals.keys():
                    st.write("• Predefined signals are designed to be causal")
                elif option in ["Modulus of Signal", "Derivative of Signal", "Integral of Signal", "Amplitude Scaling"]:
                    st.write("• Signal operations inherit causality from base signal")
                else:
                    st.write("• Default assumption for this signal type")
            
            st.markdown("""
            ### **Engineering Context:**
            - **Causal signals** can be implemented in real-time systems
            - **Non-causal signals** require future information (not physically realizable)
            - **Most practical signals** are designed to be causal for realizability
            
            ### **Making Non-Causal Signals Causal:**
            - Multiply by `Heaviside(t)` or `Heaviside(n)` to enforce causality
            - Use time delays: `f(t-delay)` instead of `f(t+advance)`
            - Apply windowing to limit the signal to causal time region
            """)
        
    except Exception as e:
        st.error(f"Error in signal analysis: {e}")
else:
    if signal is None and option not in ["Real-Time Voice Signal"] and uploaded_file is None:
        st.info("Please select or input a signal to analyze.")
    elif signal is not None and len(signal) == 0:
        st.warning("Signal is empty. Please check your input.")
