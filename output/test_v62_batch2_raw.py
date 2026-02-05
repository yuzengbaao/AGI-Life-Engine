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
            a: Numerator (dividend)
            b: Denominator (divisor)
            
        Returns:
            The result of a / b
            
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
    print("Calculator Demo:")
    print(f"Addition: 5.5 + 3.2 = {calc.add(5.5, 3.2)}")
    print(f"Subtraction: 10.0 - 4.5 = {calc.subtract(10.0, 4.5)}")
    print(f"Multiplication: 2.5 * 4.0 = {calc.multiply(2.5, 4.0)}")
    print(f"Division: 10.0 / 2.5 = {calc.divide(10.0, 2.5)}")
    
    # Test with integers (they'll be converted to float)
    print(f"\nWith integers:")
    print(f"7 + 3 = {calc.add(7, 3)}")
    print(f"15 / 3 = {calc.divide(15, 3)}")
    
    # Test division by zero handling
    print(f"\nTesting division by zero:")
    try:
        result = calc.divide(10.0, 0.0)
        print(f"10.0 / 0.0 = {result}")
    except ZeroDivisionError as e:
        print(f"Error: {e}")
    
    # Additional test cases
    print(f"\nAdditional tests:")
    print(f"Division with negative numbers: -8.0 / 2.0 = {calc.divide(-8.0, 2.0)}")
    print(f"Division resulting in decimal: 5.0 / 2.0 = {calc.divide(5.0, 2.0)}")