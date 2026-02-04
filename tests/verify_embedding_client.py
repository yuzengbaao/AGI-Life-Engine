
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm_client import LLMService

class TestLLMClientEmbedding(unittest.TestCase):
    def setUp(self):
        # Create a mock logger
        self.logger = MagicMock()
        
    def test_get_embedding_real_client(self):
        """Test that get_embedding uses the real client when available."""
        # Setup mock client
        mock_openai_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_openai_client.embeddings.create.return_value = mock_response
        
        # Initialize LLMService with the mock
        client = LLMService()
        client.client = mock_openai_client # Inject mock
        client.embedding_client = None # Ensure it uses self.client
        client.logger = self.logger
        client.mock_mode = False # Ensure not in mock mode
        
        # Call get_embedding
        embedding = client.get_embedding("test text")
        
        # Verify
        mock_openai_client.embeddings.create.assert_called_once()
        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        print("✅ Real client embedding call verified.")

    def test_get_embedding_fallback(self):
        """Test that get_embedding falls back to random when client fails or is missing."""
        client = LLMService()
        client.client = None # No client
        client.embedding_client = None # Ensure no embedding client
        client.logger = self.logger
        client.mock_mode = False
        
        # Call get_embedding
        embedding = client.get_embedding("test text")
        
        # Verify it returns a random vector (length 1536)
        self.assertEqual(len(embedding), 1536)
        print("✅ Fallback to random vector verified.")

if __name__ == '__main__':
    unittest.main()
