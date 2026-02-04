import time
from typing import List, Dict, Union

def fibonacci_recursive(n: int) -> int:
    """
    Calculate the nth Fibonacci number iteratively for O(n) performance.
    
    Args:
        n: The position in the Fibonacci sequence (must be non-negative)
    
    Returns:
        The nth Fibonacci number
    
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    
    # Iterative implementation to avoid O(2^n) complexity
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

def calculate_stats(numbers: List[Union[int, float]]) -> Dict[str, Union[float, int]]:
    """
    Calculate basic statistics on a list of numbers.
    
    Args:
        numbers: List of numeric values
    
    Returns:
        Dictionary containing sum and average
    
    Raises:
        ValueError: If the input list is empty
        ZeroDivisionError: If attempting to divide by zero when calculating average
    """
    if not numbers:
        raise ValueError("Cannot calculate stats on empty list")
        
    total = sum(numbers)
    try:
        avg = total / len(numbers)
    except ZeroDivisionError:
        raise ZeroDivisionError("Cannot calculate average: division by zero")
    
    return {
        "sum": total,
        "avg": avg
    }