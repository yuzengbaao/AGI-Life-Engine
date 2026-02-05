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


# Example usage
if __name__ == "__main__":
    calc = Calculator()
    
    # Test the calculator methods
    print("Calculator Demo:")
    print(f"Addition: 5.5 + 3.2 = {calc.add(5.5, 3.2)}")
    print(f"Subtraction: 10.0 - 4.5 = {calc.subtract(10.0, 4.5)}")
    print(f"Multiplication: 2.5 * 4.0 = {calc.multiply(2.5, 4.0)}")
    
    # Test with integers (they'll be converted to float)
    print(f"\nWith integers: 7 + 3 = {calc.add(7, 3)}")