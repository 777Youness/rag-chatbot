import unittest
from src.main import initialize_rag_system
from src.chatbot import generate_response

class TestRAGChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the RAG system once for all tests
        cls.rag_chain = initialize_rag_system()
        
    def test_basic_question(self):
        """Test a basic question about LangChain"""
        query = "What is LangChain?"
        response = generate_response(self.rag_chain, query)
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")
        
    def test_technical_question(self):
        """Test a more technical question about LangChain"""
        query = "How do I use LangChain with OpenAI models?"
        response = generate_response(self.rag_chain, query)
        self.assertIsNotNone(response)
        self.assertTrue("OpenAI" in response or "model" in response.lower())
        
    def test_conceptual_question(self):
        """Test a conceptual question about LangChain"""
        query = "Explain the concept of chains in LangChain"
        response = generate_response(self.rag_chain, query)
        self.assertIsNotNone(response)
        self.assertTrue("chain" in response.lower())
        
    def test_usage_question(self):
        """Test a usage question about LangChain"""
        query = "How do I implement a simple chatbot with LangChain?"
        response = generate_response(self.rag_chain, query)
        self.assertIsNotNone(response)
        self.assertTrue("chatbot" in response.lower() or "implement" in response.lower())
        
    def test_comparison_question(self):
        """Test a comparison question about LangChain"""
        query = "What is the difference between a Chain and an Agent in LangChain?"
        response = generate_response(self.rag_chain, query)
        self.assertIsNotNone(response)
        self.assertTrue("chain" in response.lower() and "agent" in response.lower())
        
if __name__ == "__main__":
    unittest.main()