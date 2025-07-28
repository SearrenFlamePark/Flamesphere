from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import yaml

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="ChatGPT to Obsidian Memory Sync", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize scheduler
scheduler = AsyncIOScheduler()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic Models
class ConversationMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: str  # "user" or "assistant" or "system"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    messages: List[ConversationMessage]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    synced_to_obsidian: bool = False
    obsidian_file_path: Optional[str] = None
    tags: List[str] = []
    summary: Optional[str] = None

class SyncConfiguration(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    llm_provider: str = "openai"  # "openai" or "ollama"
    openai_model: str = "gpt-4"
    ollama_model: str = "llama2"
    obsidian_vault_path: str
    sync_interval_minutes: int = 60
    auto_sync_enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class SyncJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str  # "pending", "running", "completed", "failed"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    conversations_synced: int = 0
    errors: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ConversationCreate(BaseModel):
    title: str
    messages: List[ConversationMessage]
    tags: Optional[List[str]] = []

class MessageCreate(BaseModel):
    content: str
    role: str = "user"

class SyncConfigUpdate(BaseModel):
    llm_provider: Optional[str] = None
    openai_model: Optional[str] = None
    ollama_model: Optional[str] = None
    sync_interval_minutes: Optional[int] = None
    auto_sync_enabled: Optional[bool] = None

# Services
class LLMService:
    def __init__(self):
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        
    def get_llm(self, provider: str, model: str):
        if provider == "openai":
            if not self.openai_api_key:
                raise HTTPException(status_code=400, detail="OpenAI API key not configured")
            return ChatOpenAI(
                model=model,
                api_key=self.openai_api_key,
                temperature=0.7
            )
        elif provider == "ollama":
            return ChatOllama(
                model=model,
                base_url=self.ollama_base_url,
                temperature=0.7
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")
    
    async def generate_conversation_summary(self, messages: List[ConversationMessage], provider: str, model: str) -> str:
        try:
            llm = self.get_llm(provider, model)
            
            # Prepare conversation text
            conversation_text = "\n".join([
                f"{msg.role.upper()}: {msg.content}" for msg in messages
            ])
            
            summary_prompt = f"""Please provide a concise summary of the following conversation in 2-3 sentences:

{conversation_text}

Summary:"""
            
            response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Summary generation failed"

class ObsidianService:
    def __init__(self):
        self.vault_path = Path(os.environ.get('OBSIDIAN_VAULT_PATH', '/app/obsidian_vault'))
        self.vault_path.mkdir(exist_ok=True)
        
    def generate_filename(self, conversation: Conversation) -> str:
        """Generate a safe filename for the conversation"""
        # Clean title for filename
        safe_title = "".join(c for c in conversation.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
        
        # Add timestamp to avoid conflicts
        timestamp = conversation.created_at.strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{safe_title}.md"
    
    def format_conversation_as_markdown(self, conversation: Conversation) -> str:
        """Convert conversation to Obsidian markdown format with YAML frontmatter"""
        
        # YAML frontmatter
        frontmatter = {
            'id': conversation.id,
            'title': conversation.title,
            'created': conversation.created_at.isoformat(),
            'updated': conversation.updated_at.isoformat(),
            'tags': conversation.tags or [],
            'type': 'chatgpt_conversation',
            'summary': conversation.summary or ''
        }
        
        yaml_content = yaml.dump(frontmatter, default_flow_style=False)
        
        # Markdown content
        markdown_content = f"""---
{yaml_content}---

# {conversation.title}

"""
        
        if conversation.summary:
            markdown_content += f"## Summary\n{conversation.summary}\n\n"
        
        markdown_content += "## Conversation\n\n"
        
        for msg in conversation.messages:
            role_emoji = "🧑" if msg.role == "user" else "🤖" if msg.role == "assistant" else "⚙️"
            role_name = msg.role.title()
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            markdown_content += f"### {role_emoji} {role_name} - {timestamp}\n\n"
            markdown_content += f"{msg.content}\n\n"
        
        # Add tags as Obsidian tags
        if conversation.tags:
            markdown_content += "---\n\n"
            markdown_content += "**Tags:** " + " ".join([f"#{tag}" for tag in conversation.tags])
        
        return markdown_content
    
    async def sync_conversation_to_obsidian(self, conversation: Conversation) -> str:
        """Sync a conversation to Obsidian vault"""
        try:
            filename = self.generate_filename(conversation)
            file_path = self.vault_path / filename
            
            markdown_content = self.format_conversation_as_markdown(conversation)
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Synced conversation to Obsidian: {file_path}")
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Error syncing to Obsidian: {str(e)}")
            raise

# Initialize services
llm_service = LLMService()
obsidian_service = ObsidianService()

# API Routes
@api_router.get("/")
async def root():
    return {"message": "ChatGPT to Obsidian Memory Sync API", "version": "1.0.0"}

@api_router.post("/conversations", response_model=Conversation)
async def create_conversation(conversation_data: ConversationCreate):
    """Create a new conversation"""
    conversation = Conversation(
        title=conversation_data.title,
        messages=conversation_data.messages,
        tags=conversation_data.tags or []
    )
    
    # Save to database
    await db.conversations.insert_one(conversation.dict())
    return conversation

@api_router.get("/conversations", response_model=List[Conversation])
async def get_conversations(limit: int = 50, skip: int = 0):
    """Get all conversations"""
    conversations = await db.conversations.find().skip(skip).limit(limit).to_list(limit)
    return [Conversation(**conv) for conv in conversations]

@api_router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """Get a specific conversation"""
    conversation = await db.conversations.find_one({"id": conversation_id})
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Conversation(**conversation)

@api_router.post("/conversations/{conversation_id}/messages", response_model=ConversationMessage)
async def add_message_to_conversation(conversation_id: str, message_data: MessageCreate):
    """Add a message to an existing conversation"""
    conversation = await db.conversations.find_one({"id": conversation_id})
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    new_message = ConversationMessage(
        content=message_data.content,
        role=message_data.role
    )
    
    # Update conversation
    await db.conversations.update_one(
        {"id": conversation_id},
        {
            "$push": {"messages": new_message.dict()},
            "$set": {"updated_at": datetime.utcnow(), "synced_to_obsidian": False}
        }
    )
    
    return new_message

@api_router.post("/conversations/{conversation_id}/chat", response_model=ConversationMessage)
async def chat_with_llm(conversation_id: str, message_data: MessageCreate):
    """Add a user message and get LLM response"""
    conversation = await db.conversations.find_one({"id": conversation_id})
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get current sync config
    config = await db.sync_configs.find_one({}) or SyncConfiguration().dict()
    
    # Add user message
    user_message = ConversationMessage(
        content=message_data.content,
        role="user"
    )
    
    try:
        # Get LLM response
        llm = llm_service.get_llm(config["llm_provider"], 
                                  config.get("openai_model" if config["llm_provider"] == "openai" else "ollama_model", "gpt-4"))
        
        # Prepare conversation history
        messages = []
        for msg in conversation["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # Add new user message
        messages.append(HumanMessage(content=message_data.content))
        
        # Get response
        response = await llm.ainvoke(messages)
        
        ai_message = ConversationMessage(
            content=response.content,
            role="assistant"
        )
        
        # Update conversation with both messages
        await db.conversations.update_one(
            {"id": conversation_id},
            {
                "$push": {"messages": {"$each": [user_message.dict(), ai_message.dict()]}},
                "$set": {"updated_at": datetime.utcnow(), "synced_to_obsidian": False}
            }
        )
        
        return ai_message
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@api_router.post("/sync/manual")
async def manual_sync(background_tasks: BackgroundTasks):
    """Trigger manual sync to Obsidian"""
    sync_job = SyncJob(status="pending")
    await db.sync_jobs.insert_one(sync_job.dict())
    
    background_tasks.add_task(perform_sync, sync_job.id)
    return {"message": "Sync started", "job_id": sync_job.id}

@api_router.get("/sync/jobs", response_model=List[SyncJob])
async def get_sync_jobs(limit: int = 20):
    """Get sync job history"""
    jobs = await db.sync_jobs.find().sort("created_at", -1).limit(limit).to_list(limit)
    return [SyncJob(**job) for job in jobs]

@api_router.get("/sync/config", response_model=SyncConfiguration)
async def get_sync_config():
    """Get current sync configuration"""
    config = await db.sync_configs.find_one({})
    if not config:
        # Create default config
        default_config = SyncConfiguration(
            obsidian_vault_path=os.environ.get('OBSIDIAN_VAULT_PATH', '/app/obsidian_vault')
        )
        await db.sync_configs.insert_one(default_config.dict())
        return default_config
    return SyncConfiguration(**config)

@api_router.put("/sync/config", response_model=SyncConfiguration)
async def update_sync_config(config_update: SyncConfigUpdate):
    """Update sync configuration"""
    current_config = await db.sync_configs.find_one({})
    if not current_config:
        current_config = SyncConfiguration().dict()
    
    # Update fields
    update_data = config_update.dict(exclude_unset=True)
    update_data["updated_at"] = datetime.utcnow()
    
    await db.sync_configs.update_one(
        {"id": current_config.get("id", str(uuid.uuid4()))},
        {"$set": update_data},
        upsert=True
    )
    
    updated_config = await db.sync_configs.find_one({})
    return SyncConfiguration(**updated_config)

# Background sync function
async def perform_sync(job_id: str):
    """Perform the actual sync operation"""
    try:
        # Update job status
        await db.sync_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "running", "started_at": datetime.utcnow()}}
        )
        
        # Get unsynced conversations
        conversations = await db.conversations.find({"synced_to_obsidian": False}).to_list(None)
        logger.info(f"Found {len(conversations)} conversations to sync")
        
        synced_count = 0
        errors = []
        
        # Get sync config
        config = await db.sync_configs.find_one({})
        if config:
            provider = config.get("llm_provider", "openai")
            model = config.get(f"{provider}_model", "gpt-4" if provider == "openai" else "llama2")
        else:
            provider = "openai"
            model = "gpt-4"
        
        for conv_data in conversations:
            try:
                conversation = Conversation(**conv_data)
                
                # Generate summary if not exists
                if not conversation.summary and len(conversation.messages) > 0:
                    conversation.summary = await llm_service.generate_conversation_summary(
                        conversation.messages, provider, model
                    )
                
                # Sync to Obsidian
                file_path = await obsidian_service.sync_conversation_to_obsidian(conversation)
                
                # Update conversation in database
                await db.conversations.update_one(
                    {"id": conversation.id},
                    {
                        "$set": {
                            "synced_to_obsidian": True,
                            "obsidian_file_path": file_path,
                            "summary": conversation.summary,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
                synced_count += 1
                logger.info(f"Synced conversation: {conversation.title}")
                
            except Exception as e:
                error_msg = f"Error syncing conversation {conv_data.get('id', 'unknown')}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Update job completion
        await db.sync_jobs.update_one(
            {"id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "conversations_synced": synced_count,
                    "errors": errors
                }
            }
        )
        
        logger.info(f"Sync completed: {synced_count} conversations synced")
        
    except Exception as e:
        logger.error(f"Sync job {job_id} failed: {str(e)}")
        await db.sync_jobs.update_one(
            {"id": job_id},
            {
                "$set": {
                    "status": "failed",
                    "completed_at": datetime.utcnow(),
                    "errors": [str(e)]
                }
            }
        )

# Auto-sync scheduler job
async def scheduled_sync():
    """Scheduled sync job"""
    config = await db.sync_configs.find_one({})
    if config and config.get("auto_sync_enabled", True):
        sync_job = SyncJob(status="pending")
        await db.sync_jobs.insert_one(sync_job.dict())
        await perform_sync(sync_job.id)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize scheduler on startup"""
    # Get sync config
    config = await db.sync_configs.find_one({})
    if not config:
        config = SyncConfiguration().dict()
    
    interval_minutes = config.get("sync_interval_minutes", 60)
    
    # Schedule auto-sync
    scheduler.add_job(
        scheduled_sync,
        IntervalTrigger(minutes=interval_minutes),
        id="auto_sync",
        replace_existing=True
    )
    
    scheduler.start()
    logger.info(f"Scheduler started with {interval_minutes} minute intervals")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    scheduler.shutdown()
    client.close()

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)