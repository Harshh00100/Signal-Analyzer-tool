import numpy as np
import sympy as sp
import re

def calculate_energy(signal):
    """Calculate the energy of a signal."""
    return np.sum(np.abs(signal)**2)

def calculate_power(signal):
    """Calculate the average power of a signal."""
    return np.mean(np.abs(signal)**2)

def is_periodic(signal, tolerance=1e-10):
    """
    Check if a signal is periodic and return the period.
    Uses the mathematical definition: f(n) = f(n+N) for discrete or f(t) = f(t+T) for continuous
    """
    n = len(signal)
    if n < 2:
        return False, 0
    
    # Check for periods from 1 to n//2
    for period in range(1, n//2 + 1):
        is_periodic_with_period = True
        
        # Check if signal[i] == signal[i + period] for all valid i
        # We can only check up to (n - period) to avoid index out of bounds
        for i in range(n - period):
            if abs(signal[i] - signal[i + period]) > tolerance:
                is_periodic_with_period = False
                break
                
        # Additional check: ensure the remaining part also matches the pattern
        # For a true period, we need the signal to repeat completely
        if is_periodic_with_period:
            # Check if the signal pattern repeats for the entire length
            complete_periods = n // period
            remainder = n % period
            
            # Check all complete periods
            valid_period = True
            for cycle in range(1, complete_periods):
                for i in range(period):
                    if abs(signal[i] - signal[cycle * period + i]) > tolerance:
                        valid_period = False
                        break
                if not valid_period:
                    break
            
            # Check the remainder part (if any)
            if valid_period and remainder > 0:
                for i in range(remainder):
                    if abs(signal[i] - signal[complete_periods * period + i]) > tolerance:
                        valid_period = False
                        break
            
            if valid_period:
                return True, period
    
    return False, 0

def is_causal(signal):
    """Basic causality check - this is kept for compatibility but not used for expressions."""
    # This is a simple check - for expressions, we use analyze_expression_causality
    return True

def analyze_expression_causality(expression, variable='t'):
    """
    Analyze mathematical expression for causality based on time shifts.
    
    Args:
        expression (str): Mathematical expression to analyze
        variable (str): Time variable ('t' for continuous, 'n' for discrete)
    
    Returns:
        tuple: (is_causal, analysis_details)
    """
    try:
        # Remove equation format if present
        if '=' in expression:
            expression = expression.split('=', 1)[1].strip()
        
        # Convert expression to lowercase for analysis
        expr_lower = expression.lower().replace(' ', '')
        
        # Parse the expression using SymPy
        expr = sp.sympify(expression, evaluate=False)
        
        # Get all sub-expressions involving the time variable
        time_shifts = find_time_shifts(expression, variable)
        
        # Analyze causality based on time shifts
        is_causal = True
        non_causal_terms = []
        causal_terms = []
        analysis_details = {
            'shifts_found': time_shifts,
            'non_causal_terms': [],
            'causal_terms': [],
            'reasoning': []
        }
        
        if time_shifts:
            for shift in time_shifts:
                shift_value = shift['shift']
                shift_expr = shift['expression']
                
                if shift_value > 0:
                    # Positive shift means looking into the future - NON-CAUSAL
                    is_causal = False
                    non_causal_terms.append(shift_expr)
                    analysis_details['non_causal_terms'].append(shift_expr)
                    analysis_details['reasoning'].append(f"'{shift_expr}' requires future values (shift: +{shift_value})")
                else:
                    # Zero or negative shift means present/past values - CAUSAL
                    causal_terms.append(shift_expr)
                    analysis_details['causal_terms'].append(shift_expr)
                    if shift_value < 0:
                        analysis_details['reasoning'].append(f"'{shift_expr}' uses past values (shift: {shift_value})")
                    else:
                        analysis_details['reasoning'].append(f"'{shift_expr}' uses present value (no shift)")
        else:
            # No explicit time shifts found - analyze the base function
            base_analysis = analyze_base_function_causality(expression, variable)
            is_causal = base_analysis['is_causal']
            analysis_details.update(base_analysis)
        
        return is_causal, analysis_details
        
    except Exception as e:
        # If parsing fails, return default analysis
        return True, {
            'shifts_found': [],
            'non_causal_terms': [],
            'causal_terms': [],
            'reasoning': [f"Could not parse expression: {str(e)}"],
            'error': str(e)
        }

def find_time_shifts(expression, variable='t'):
    """
    Find time shifts in mathematical expressions like t+5, t-3, n+2, n-1, etc.
    
    Args:
        expression (str): Mathematical expression
        variable (str): Time variable to search for
    
    Returns:
        list: List of dictionaries containing shift information
    """
    shifts = []
    
    # Patterns to match time shifts
    # Matches: t+5, t-3, (t+2), t + 5, t - 3, etc.
    patterns = [
        rf'{variable}\s*\+\s*(\d+(?:\.\d+)?)',  # t+number
        rf'{variable}\s*-\s*(\d+(?:\.\d+)?)',   # t-number
        rf'\(\s*{variable}\s*\+\s*(\d+(?:\.\d+)?)\s*\)',  # (t+number)
        rf'\(\s*{variable}\s*-\s*(\d+(?:\.\d+)?)\s*\)',   # (t-number)
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, expression, re.IGNORECASE)
        for match in matches:
            shift_value = float(match.group(1))
            full_match = match.group(0)
            
            # Determine if it's positive or negative shift
            if '+' in full_match:
                actual_shift = shift_value
            else:  # '-' in full_match
                actual_shift = -shift_value
            
            shifts.append({
                'expression': full_match.strip(),
                'shift': actual_shift,
                'position': match.span()
            })
    
    return shifts

def analyze_base_function_causality(expression, variable='t'):
    """
    Analyze causality of base functions without explicit time shifts.
    
    Args:
        expression (str): Mathematical expression
        variable (str): Time variable
    
    Returns:
        dict: Analysis results
    """
    expr_lower = expression.lower().replace(' ', '')
    
    analysis = {
        'is_causal': True,
        'shifts_found': [],
        'non_causal_terms': [],
        'causal_terms': [],
        'reasoning': []
    }
    
    # Check for functions that are inherently causal or non-causal
    if 'heaviside' in expr_lower:
        if f'heaviside({variable})' in expr_lower:
            analysis['is_causal'] = True
            analysis['reasoning'].append(f"Heaviside({variable}) is causal (zero for {variable}<0)")
        else:
            analysis['is_causal'] = True
            analysis['reasoning'].append("Contains Heaviside function - likely causal")
    
    elif 'dirac' in expr_lower or 'delta' in expr_lower:
        analysis['is_causal'] = True
        analysis['reasoning'].append("Impulse function is causal when applied at t=0 or later")
    
    elif any(func in expr_lower for func in ['sin', 'cos', 'tan']):
        # Trigonometric functions without Heaviside are typically non-causal
        # unless explicitly made causal
        if 'heaviside' not in expr_lower:
            analysis['is_causal'] = False
            analysis['reasoning'].append("Trigonometric functions without Heaviside are non-causal (extend to -∞)")
        else:
            analysis['is_causal'] = True
            analysis['reasoning'].append("Trigonometric function made causal with Heaviside")
    
    elif 'exp' in expr_lower:
        if f'exp(-{variable})' in expr_lower or f'exp(-abs({variable}))' in expr_lower:
            if 'heaviside' not in expr_lower:
                analysis['is_causal'] = False
                analysis['reasoning'].append("Exponential function without Heaviside extends to negative time")
            else:
                analysis['is_causal'] = True
                analysis['reasoning'].append("Exponential function made causal with Heaviside")
        else:
            analysis['is_causal'] = True
            analysis['reasoning'].append("Exponential function appears to be causal")
    
    else:
        # For other functions, assume causal unless evidence suggests otherwise
        analysis['is_causal'] = True
        analysis['reasoning'].append("Function appears to be causal based on structure")
    
    return analysis

def get_causality_explanation(is_causal, analysis_details):
    """
    Generate a human-readable explanation of causality analysis.
    
    Args:
        is_causal (bool): Whether the signal is causal
        analysis_details (dict): Detailed analysis results
    
    Returns:
        str: Human-readable explanation
    """
    if 'error' in analysis_details:
        return f"Could not analyze expression: {analysis_details['error']}"
    
    explanation = []
    
    if is_causal:
        explanation.append("✅ **Signal is CAUSAL**")
    else:
        explanation.append("❌ **Signal is NON-CAUSAL**")
    
    # Add reasoning
    if analysis_details.get('reasoning'):
        explanation.append("\n**Analysis:**")
        for reason in analysis_details['reasoning']:
            explanation.append(f"• {reason}")
    
    # Add shift details if found
    if analysis_details.get('shifts_found'):
        explanation.append(f"\n**Time shifts detected:** {len(analysis_details['shifts_found'])}")
        for shift in analysis_details['shifts_found']:
            shift_type = "future" if shift['shift'] > 0 else "past" if shift['shift'] < 0 else "present"
            explanation.append(f"• {shift['expression']} → {shift_type} value (shift: {shift['shift']:+g})")
    
    return "\n".join(explanation)
