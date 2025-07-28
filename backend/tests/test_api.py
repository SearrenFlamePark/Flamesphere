import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server import app, llm_service, obsidian_service

client = TestClient(app)

class TestHealthAPI:
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        with patch('server.client.admin.command', new_callable=AsyncMock) as mock_db:
            mock_db.return_value = {"ok": 1}
            with patch.object(llm_service, 'health_check', new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = True
                with patch.object(obsidian_service.vault_path, 'exists') as mock_vault:
                    mock_vault.return_value = True
                    
                    response = client.get("/api/health")
                    assert response.status_code == 200
                    data = response.json()
                    assert "status" in data
                    assert "database" in data
                    assert "llm_service" in data
                    assert "obsidian_vault" in data

class TestConnectionAPI:
    @patch('server.db.sync_configs.find_one', new_callable=AsyncMock)
    @patch.object(llm_service, 'health_check', new_callable=AsyncMock)
    def test_connection_test(self, mock_health, mock_config):
        """Test the connection test endpoint"""
        mock_config.return_value = {
            "llm_provider": "openai",
            "openai_model": "gpt-4"
        }
        mock_health.return_value = True
        
        response = client.post("/api/test/connection")
        assert response.status_code == 200
        data = response.json()
        assert "provider" in data
        assert "model" in data
        assert "status" in data

class TestImportAPI:
    @patch('server.db.processed_conversations.insert_one', new_callable=AsyncMock)
    def test_import_chatgpt_conversation(self, mock_insert):
        """Test importing a ChatGPT conversation"""
        mock_insert.return_value = MagicMock()
        
        test_data = {
            "content": "User: Hello\nAssistant: Hi there!",
            "import_type": "text",
            "title": "Test Conversation",
            "tags": ["test"]
        }
        
        response = client.post("/api/import/chatgpt", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["original_title"] == "Test Conversation"
        assert "test" in data["tags"]

class TestSyncAPI:
    @patch('server.db.sync_jobs.insert_one', new_callable=AsyncMock)
    def test_manual_sync(self, mock_insert):
        """Test manual sync trigger"""
        mock_insert.return_value = MagicMock()
        
        response = client.post("/api/sync/manual")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "job_id" in data

    @patch('server.db.sync_configs.find_one', new_callable=AsyncMock)
    def test_get_sync_config(self, mock_find):
        """Test getting sync configuration"""
        mock_find.return_value = None
        
        response = client.get("/api/sync/config")
        assert response.status_code == 200
        data = response.json()
        assert "llm_provider" in data
        assert "obsidian_vault_path" in data

class TestLLMService:
    def test_get_openai_llm(self):
        """Test creating OpenAI LLM instance"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'mock_api_key_for_testing'}):
            llm = llm_service.get_llm("openai", "gpt-4")
            assert llm is not None

    def test_get_ollama_llm(self):
        """Test creating Ollama LLM instance"""
        llm = llm_service.get_llm("ollama", "llama2")
        assert llm is not None

    def test_invalid_provider(self):
        """Test error handling for invalid provider"""
        with pytest.raises(Exception):
            llm_service.get_llm("invalid", "model")

class TestObsidianService:
    def test_generate_filename(self):
        """Test filename generation for Obsidian notes"""
        from server import ProcessedConversation
        from datetime import datetime
        
        conv = ProcessedConversation(
            original_title="Test Conversation",
            structured_title="🧠 Test Conversation",
            raw_content="test content",
            created_at=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        filename = obsidian_service.generate_filename(conv)
        assert filename.endswith('.md')
        assert '20240101_120000' in filename

    def test_format_conversation_as_markdown(self):
        """Test markdown formatting for Obsidian"""
        from server import ProcessedConversation
        from datetime import datetime
        
        conv = ProcessedConversation(
            original_title="Test Conversation",
            structured_title="🧠 Test Conversation",
            raw_content="test content",
            structured_content="# Test Content",
            summary="Test summary",
            tags=["test", "example"],
            key_concepts=["concept1", "concept2"]
        )
        
        markdown = obsidian_service.format_processed_conversation_as_markdown(conv)
        assert "---" in markdown  # YAML frontmatter
        assert "# 🧠 Test Conversation" in markdown
        assert "Test summary" in markdown
        assert "#test" in markdown

class TestChatGPTParser:
    def test_parse_text_format(self):
        """Test parsing plain text conversation"""
        from server import ChatGPTParser
        
        content = "User: Hello\nAssistant: Hi there!"
        result = ChatGPTParser.parse_text_format(content, "Test Title")
        
        assert result["title"] == "Test Title"
        assert result["raw_content"] == content

    def test_parse_json_export(self):
        """Test parsing JSON export format"""
        from server import ChatGPTParser
        
        json_data = '{"title": "Test Conversation", "messages": [{"role": "user", "content": "Hello"}]}'
        result = ChatGPTParser.parse_json_export(json_data)
        
        assert len(result) > 0
        assert result[0]["title"] == "Test Conversation"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()