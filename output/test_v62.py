class Calculator:
    """A simple calculator class with basic arithmetic operations."""
    
    def add(self, a: float, b: float) -> float:
        """
        Add two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            The sum of a and b
        """
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """
        Subtract b from a.
        
        Args:
            a: First number
            b: Second number to subtract from a
            
        Returns:
            The result of a - b
        """
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """
        Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            The product of a and b
        """
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """
        Divide a by b.
        
        Args:
            a: The numerator (dividend)
            b: The denominator (divisor)
            
        Returns:
            The result of a divided by b
            
        Raises:
            ZeroDivisionError: If b is zero
        """
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b


# Example usage
if __name__ == "__main__":
    calc = Calculator()
    
    # Test the methods
    print(f"Addition: 5.5 + 3.2 = {calc.add(5.5, 3.2)}")
    print(f"Subtraction: 10.0 - 4.5 = {calc.subtract(10.0, 4.5)}")
    print(f"Multiplication: 2.5 * 4.0 = {calc.multiply(2.5, 4.0)}")
    print(f"Division: 10.0 / 2.5 = {calc.divide(10.0, 2.5)}")
    
    # Type hints demonstration
    result: float = calc.add(3.14, 2.86)
    print(f"\nType hinted result: {result}")
    
    # Test division by zero (uncomment to see the error)
    # print(f"Division by zero: 5.0 / 0.0 = {calc.divide(5.0, 0.0)}")