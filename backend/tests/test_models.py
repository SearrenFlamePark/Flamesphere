import pytest
from datetime import datetime
from pydantic import ValidationError
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from server import (
    ProcessedConversation, 
    SyncConfiguration, 
    ConversationMessage,
    SyncJob,
    ChatGPTImport
)

class TestProcessedConversation:
    def test_create_processed_conversation(self):
        """Test creating a ProcessedConversation instance"""
        conv = ProcessedConversation(
            original_title="Test Title",
            structured_title="🧠 Test Title",
            raw_content="Raw content here",
            structured_content="Structured content",
            summary="Test summary"
        )
        
        assert conv.original_title == "Test Title"
        assert conv.structured_title == "🧠 Test Title"
        assert conv.synced_to_obsidian == False
        assert isinstance(conv.created_at, datetime)
        assert len(conv.id) > 0

    def test_processed_conversation_with_tags(self):
        """Test ProcessedConversation with tags and concepts"""
        conv = ProcessedConversation(
            original_title="Test Title",
            structured_title="🧠 Test Title", 
            raw_content="Raw content",
            tags=["test", "example"],
            key_concepts=["concept1", "concept2"],
            frameworks=["framework1"]
        )
        
        assert "test" in conv.tags
        assert "concept1" in conv.key_concepts
        assert "framework1" in conv.frameworks

class TestSyncConfiguration:
    def test_create_sync_config(self):
        """Test creating a SyncConfiguration instance"""
        config = SyncConfiguration()
        
        assert config.llm_provider == "openai"
        assert config.openai_model == "gpt-4"
        assert config.sync_interval_minutes == 60
        assert config.auto_sync_enabled == True
        assert config.obsidian_vault_path == "/app/obsidian_vault"

    def test_sync_config_with_custom_values(self):
        """Test SyncConfiguration with custom values"""
        config = SyncConfiguration(
            llm_provider="ollama",
            ollama_model="llama3",
            sync_interval_minutes=30,
            auto_sync_enabled=False
        )
        
        assert config.llm_provider == "ollama"
        assert config.ollama_model == "llama3"
        assert config.sync_interval_minutes == 30
        assert config.auto_sync_enabled == False

class TestConversationMessage:
    def test_create_conversation_message(self):
        """Test creating a ConversationMessage instance"""
        msg = ConversationMessage(
            role="user",
            content="Hello, world!"
        )
        
        assert msg.role == "user"
        assert msg.content == "Hello, world!"
        assert isinstance(msg.timestamp, datetime)
        assert len(msg.id) > 0

    def test_message_roles(self):
        """Test different message roles"""
        roles = ["user", "assistant", "system"]
        
        for role in roles:
            msg = ConversationMessage(role=role, content="Test content")
            assert msg.role == role

class TestSyncJob:
    def test_create_sync_job(self):
        """Test creating a SyncJob instance"""
        job = SyncJob(
            status="pending",
            job_type="manual_sync"
        )
        
        assert job.status == "pending"
        assert job.job_type == "manual_sync"
        assert job.items_processed == 0
        assert job.errors == []
        assert isinstance(job.created_at, datetime)

    def test_sync_job_completion(self):
        """Test SyncJob with completion data"""
        job = SyncJob(
            status="completed",
            job_type="auto_sync",
            started_at=datetime.now(),
            completed_at=datetime.now(),
            items_processed=5,
            errors=[]
        )
        
        assert job.status == "completed"
        assert job.items_processed == 5
        assert job.errors == []

class TestChatGPTImport:
    def test_create_chatgpt_import(self):
        """Test creating a ChatGPTImport instance"""
        import_data = ChatGPTImport(
            content="User: Hello\nAssistant: Hi!",
            import_type="text",
            title="Test Conversation",
            tags=["test"]
        )
        
        assert import_data.import_type == "text"
        assert import_data.title == "Test Conversation"
        assert "test" in import_data.tags

    def test_chatgpt_import_defaults(self):
        """Test ChatGPTImport with default values"""
        import_data = ChatGPTImport(
            content="Test content"
        )
        
        assert import_data.import_type == "text"
        assert import_data.title is None
        assert import_data.tags == []