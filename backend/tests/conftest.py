import pytest
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_vault():
    """Create a temporary Obsidian vault directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        vault_path = Path(temp_dir) / "test_vault"
        vault_path.mkdir()
        yield vault_path

@pytest.fixture
def mock_db():
    """Mock database connection"""
    db = MagicMock()
    db.processed_conversations = MagicMock()
    db.sync_configs = MagicMock()
    db.sync_jobs = MagicMock()
    
    # Setup async methods
    db.processed_conversations.insert_one = AsyncMock()
    db.processed_conversations.find = MagicMock()
    db.processed_conversations.find_one = AsyncMock()
    db.processed_conversations.update_one = AsyncMock()
    
    db.sync_configs.find_one = AsyncMock()
    db.sync_configs.update_one = AsyncMock()
    db.sync_configs.insert_one = AsyncMock()
    
    db.sync_jobs.insert_one = AsyncMock()
    db.sync_jobs.find = MagicMock()
    db.sync_jobs.find_one = AsyncMock()
    
    return db

@pytest.fixture 
def sample_conversation_data():
    """Sample conversation data for testing"""
    return {
        "content": "User: What are spaced repetition benefits?\nAssistant: Spaced repetition helps with long-term retention.",
        "title": "Test Conversation",
        "tags": ["learning", "memory"]
    }