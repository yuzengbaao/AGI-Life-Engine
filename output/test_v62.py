class Calculator:
    """A simple calculator class with basic arithmetic operations."""
    
    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """
        Subtract b from a.
        
        Args:
            a: First number
            b: Second number to subtract from a
            
        Returns:
            Result of a - b
        """
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """
        Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b
        """
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """
        Divide a by b.
        
        Args:
            a: Numerator
            b: Denominator
            
        Returns:
            Result of a / b
            
        Raises:
            ZeroDivisionError: If b is 0
        """
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b


# Example usage
if __name__ == "__main__":
    calc = Calculator()
    
    # Test the calculator methods
    print(f"Addition: 5 + 3 = {calc.add(5, 3)}")
    print(f"Subtraction: 10 - 4 = {calc.subtract(10, 4)}")
    print(f"Multiplication: 6 * 7 = {calc.multiply(6, 7)}")
    print(f"Division: 15 / 3 = {calc.divide(15, 3)}")
    
    # Test with floating point numbers
    print(f"Addition (float): 2.5 + 3.7 = {calc.add(2.5, 3.7)}")
    print(f"Subtraction (float): 8.9 - 2.3 = {calc.subtract(8.9, 2.3)}")
    print(f"Multiplication (float): 1.5 * 4.2 = {calc.multiply(1.5, 4.2)}")
    print(f"Division (float): 10.5 / 2.5 = {calc.divide(10.5, 2.5)}")
    
    # Test division by zero (with error handling)
    try:
        print(f"Division by zero: 5 / 0 = {calc.divide(5, 0)}")
    except ZeroDivisionError as e:
        print(f"Division by zero error: {e}")