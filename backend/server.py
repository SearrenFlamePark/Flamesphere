from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, UploadFile, File
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import uuid
from datetime import datetime
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import yaml
import json
import re
import time
from functools import wraps
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

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

class ProcessedConversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_title: str
    structured_title: str
    raw_content: str
    structured_content: str
    key_concepts: List[str] = []
    frameworks: List[str] = []
    action_items: List[str] = []
    tags: List[str] = []
    summary: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    synced_to_obsidian: bool = False
    obsidian_file_path: Optional[str] = None

class SyncConfiguration(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    llm_provider: str = "openai"  # "openai" or "ollama"
    openai_model: str = "gpt-4"
    ollama_model: str = "llama2"
    obsidian_vault_path: str = Field(default="/app/obsidian_vault")
    sync_interval_minutes: int = 60
    auto_sync_enabled: bool = True
    processing_template: str = "advanced_structured"  # "basic", "advanced_structured", "custom"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class SyncJob(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str  # "pending", "running", "completed", "failed"
    job_type: str  # "manual_sync", "auto_sync", "import_processing"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    items_processed: int = 0
    errors: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatGPTImport(BaseModel):
    content: str
    import_type: str = "text"  # "text", "json", "markdown"
    title: Optional[str] = None
    tags: List[str] = []

class SyncConfigUpdate(BaseModel):
    llm_provider: Optional[str] = None
    openai_model: Optional[str] = None 
    ollama_model: Optional[str] = None
    sync_interval_minutes: Optional[int] = None
    auto_sync_enabled: Optional[bool] = None
    processing_template: Optional[str] = None

# Services
class ConnectionManager:
    """Manage API connections with retry logic and rate limiting"""
    
    def __init__(self):
        self.last_request_time = {}
        self.request_counts = {}
        self.rate_limit_window = 60  # 1 minute window
        self.max_requests_per_window = 50  # Conservative limit
    
    def should_rate_limit(self, service: str) -> bool:
        """Check if we should rate limit requests to a service"""
        now = time.time()
        
        # Clean old entries
        if service in self.request_counts:
            self.request_counts[service] = [
                req_time for req_time in self.request_counts[service]
                if now - req_time < self.rate_limit_window
            ]
        
        # Check current request count
        current_count = len(self.request_counts.get(service, []))
        return current_count >= self.max_requests_per_window
    
    def record_request(self, service: str):
        """Record a request to a service"""
        now = time.time()
        if service not in self.request_counts:
            self.request_counts[service] = []
        self.request_counts[service].append(now)
        self.last_request_time[service] = now
    
    async def wait_if_needed(self, service: str):
        """Wait if rate limiting is needed"""
        if self.should_rate_limit(service):
            wait_time = 60  # Wait 1 minute if rate limited
            logger.warning(f"Rate limiting {service}, waiting {wait_time} seconds")
            await asyncio.sleep(wait_time)

class LLMService:
    def __init__(self):
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.ollama_base_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.connection_manager = ConnectionManager()
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception))
    )
    def get_llm(self, provider: str, model: str):
        """Get LLM instance with retry logic"""
        try:
            if provider == "openai":
                if not self.openai_api_key:
                    raise HTTPException(status_code=400, detail="OpenAI API key not configured")
                return ChatOpenAI(
                    model=model,
                    api_key=self.openai_api_key,
                    temperature=0.7,
                    timeout=30,
                    max_retries=3
                )
            elif provider == "ollama":
                return ChatOllama(
                    model=model,
                    base_url=self.ollama_base_url,
                    temperature=0.7,
                    timeout=30
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported LLM provider: {provider}")
        except Exception as e:
            logger.error(f"Error creating LLM instance: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, Exception))
    )
    async def process_chatgpt_conversation(self, raw_content: str, provider: str, model: str, template: str = "advanced_structured") -> Dict[str, Any]:
        """Process raw ChatGPT conversation into structured Obsidian format with robust error handling"""
        try:
            # Rate limiting check
            await self.connection_manager.wait_if_needed(f"{provider}_{model}")
            
            llm = self.get_llm(provider, model)
            self.connection_manager.record_request(f"{provider}_{model}")
            
            if template == "advanced_structured":
                prompt = ChatPromptTemplate.from_template("""
You are an expert at analyzing ChatGPT conversations and extracting structured knowledge for Obsidian note-taking.

Analyze the following ChatGPT conversation and create a structured breakdown:

CONVERSATION:
{raw_content}

Please provide a JSON response with the following structure:
{{
    "structured_title": "A clear, concept-focused title with emoji if appropriate",
    "summary": "2-3 sentence summary of key insights",
    "key_concepts": ["list", "of", "main", "concepts"],
    "frameworks": ["any", "frameworks", "or", "methodologies", "discussed"],
    "action_items": ["specific", "actionable", "items", "or", "instructions"],
    "tags": ["relevant", "obsidian", "tags"],
    "structured_content": "Well-formatted markdown content with clear sections, bullet points, and proper Obsidian formatting. Include > Usage notes where appropriate."
}}

Focus on creating content similar to these example formats:
- Use emojis for section headers where appropriate
- Create clear bullet points for key information
- Include usage notes in blockquotes
- Structure information hierarchically
- Maintain any specialized terminology

IMPORTANT: Return only valid JSON. No additional text or explanations.
""")
                
            elif template == "basic":
                prompt = ChatPromptTemplate.from_template("""
Convert this ChatGPT conversation to a clean Obsidian note:

{raw_content}

Provide JSON with: structured_title, summary, tags, and structured_content (markdown).
Return only valid JSON.
""")
            
            messages = prompt.format_messages(raw_content=raw_content)
            
            # Add timeout handling
            try:
                response = await asyncio.wait_for(llm.ainvoke(messages), timeout=60.0)
            except asyncio.TimeoutError:
                logger.error("LLM request timed out")
                raise
            
            # Try to parse JSON response with better error handling
            response_text = response.content.strip()
            
            # Clean up response if it has markdown code blocks
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            try:
                result = json.loads(response_text)
                
                # Validate required fields
                required_fields = ["structured_title", "summary", "structured_content"]
                for field in required_fields:
                    if field not in result:
                        result[field] = f"Generated {field}"
                
                # Ensure lists exist
                for field in ["key_concepts", "frameworks", "action_items", "tags"]:
                    if field not in result or not isinstance(result[field], list):
                        result[field] = []
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {str(e)}, Response: {response_text[:500]}")
                # Create a fallback structure
                return {
                    "structured_title": "Processed ChatGPT Conversation",
                    "summary": "Content processing completed with parsing issues",
                    "key_concepts": [],
                    "frameworks": [],
                    "action_items": [],
                    "tags": ["chatgpt", "processed", "parsing-issue"],
                    "structured_content": f"# Processed Content\n\n{response_text}\n\n---\n\n*Note: Original processing had formatting issues*"
                }
                
        except Exception as e:
            logger.error(f"Error processing conversation: {str(e)}")
            # Create error fallback
            return {
                "structured_title": "Error Processing Conversation",
                "summary": f"Processing failed: {str(e)[:100]}",
                "key_concepts": [],
                "frameworks": [],
                "action_items": [],
                "tags": ["chatgpt", "error", "needs-reprocessing"],
                "structured_content": f"# Processing Error\n\nOriginal content:\n\n{raw_content[:1000]}...\n\n**Error:** {str(e)}"
            }

    async def health_check(self, provider: str, model: str) -> bool:
        """Check if the LLM service is healthy"""
        try:
            llm = self.get_llm(provider, model)
            test_response = await asyncio.wait_for(
                llm.ainvoke([HumanMessage(content="Say 'OK'")]), 
                timeout=10.0
            )
            return "ok" in test_response.content.lower()
        except Exception as e:
            logger.error(f"Health check failed for {provider}:{model} - {str(e)}")
            return False

class ChatGPTParser:
    """Parse different ChatGPT export formats"""
    
    @staticmethod
    def parse_json_export(content: str) -> List[Dict[str, Any]]:
        """Parse ChatGPT JSON export format"""
        try:
            data = json.loads(content)
            conversations = []
            
            if isinstance(data, list):
                # List of conversations
                for item in data:
                    conversations.append(ChatGPTParser._extract_conversation_from_json(item))
            elif isinstance(data, dict):
                # Single conversation or root object
                if 'title' in data or 'messages' in data:
                    conversations.append(ChatGPTParser._extract_conversation_from_json(data))
                elif 'conversations' in data:
                    for conv in data['conversations']:
                        conversations.append(ChatGPTParser._extract_conversation_from_json(conv))
            
            return conversations
        except Exception as e:
            logger.error(f"Error parsing JSON export: {str(e)}")
            return []
    
    @staticmethod
    def _extract_conversation_from_json(conv_data: dict) -> Dict[str, Any]:
        """Extract conversation data from JSON structure"""
        title = conv_data.get('title', 'Untitled Conversation')
        
        # Handle different message structures
        messages = []
        if 'mapping' in conv_data:
            # OpenAI export format with mapping
            for msg_id, msg_data in conv_data['mapping'].items():
                if msg_data.get('message') and msg_data['message'].get('content'):
                    content = msg_data['message']['content']
                    role = msg_data['message'].get('author', {}).get('role', 'user')
                    
                    if isinstance(content, dict) and 'parts' in content:
                        text = ' '.join(content['parts'])
                    elif isinstance(content, list):
                        text = ' '.join(str(part) for part in content)
                    else:
                        text = str(content)
                    
                    if text.strip():
                        messages.append({
                            'role': role,
                            'content': text,
                            'timestamp': datetime.utcnow()
                        })
        
        elif 'messages' in conv_data:
            # Simple message array format
            for msg in conv_data['messages']:
                messages.append({
                    'role': msg.get('role', 'user'),
                    'content': msg.get('content', ''),
                    'timestamp': datetime.utcnow()
                })
        
        # Combine all message content
        raw_content = f"# {title}\n\n"
        for msg in messages:
            role_name = "User" if msg['role'] == 'user' else "Assistant"
            raw_content += f"**{role_name}:** {msg['content']}\n\n"
        
        return {
            'title': title,
            'raw_content': raw_content,
            'messages': messages
        }
    
    @staticmethod
    def parse_text_format(content: str, title: str = None) -> Dict[str, Any]:
        """Parse plain text conversation format"""
        if not title:
            # Try to extract title from first line or create one
            lines = content.strip().split('\n')
            first_line = lines[0].strip() if lines else ""
            if first_line.startswith('#'):
                title = first_line.replace('#', '').strip()
            else:
                title = "Imported ChatGPT Conversation"
        
        return {
            'title': title,
            'raw_content': content,
            'messages': []  # Will be parsed if needed
        }

class ObsidianService:
    def __init__(self):
        self.vault_path = Path(os.environ.get('OBSIDIAN_VAULT_PATH', '/app/obsidian_vault'))
        self.vault_path.mkdir(exist_ok=True)
        
    def generate_filename(self, processed_conv: ProcessedConversation) -> str:
        """Generate a safe filename for the processed conversation"""
        # Clean title for filename
        safe_title = "".join(c for c in processed_conv.structured_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]  # Limit length
        
        # Remove emojis for filename
        safe_title = re.sub(r'[^\w\s-]', '', safe_title).strip()
        
        # Add timestamp to avoid conflicts
        timestamp = processed_conv.created_at.strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{safe_title}.md"
    
    def format_processed_conversation_as_markdown(self, processed_conv: ProcessedConversation) -> str:
        """Convert processed conversation to Obsidian markdown format with YAML frontmatter"""
        
        # YAML frontmatter
        frontmatter = {
            'id': processed_conv.id,
            'title': processed_conv.structured_title,
            'original_title': processed_conv.original_title,
            'created': processed_conv.created_at.isoformat(),
            'processed': processed_conv.processed_at.isoformat(),
            'tags': processed_conv.tags,
            'type': 'chatgpt_knowledge',
            'summary': processed_conv.summary,
            'key_concepts': processed_conv.key_concepts,
            'frameworks': processed_conv.frameworks,
            'action_items': processed_conv.action_items
        }
        
        yaml_content = yaml.dump(frontmatter, default_flow_style=False, allow_unicode=True)
        
        # Markdown content
        markdown_content = f"""---
{yaml_content}---

# {processed_conv.structured_title}

## Summary
{processed_conv.summary}

{processed_conv.structured_content}

"""
        
        # Add concept links if any
        if processed_conv.key_concepts:
            markdown_content += "\n---\n\n**Key Concepts:** " + " | ".join([f"[[{concept}]]" for concept in processed_conv.key_concepts])
        
        # Add tags
        if processed_conv.tags:
            markdown_content += "\n\n**Tags:** " + " ".join([f"#{tag}" for tag in processed_conv.tags])
        
        return markdown_content
    
    async def sync_processed_conversation_to_obsidian(self, processed_conv: ProcessedConversation) -> str:
        """Sync a processed conversation to Obsidian vault"""
        try:
            filename = self.generate_filename(processed_conv)
            file_path = self.vault_path / filename
            
            markdown_content = self.format_processed_conversation_as_markdown(processed_conv)
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Synced processed conversation to Obsidian: {file_path}")
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Error syncing to Obsidian: {str(e)}")
            raise

# Initialize services
llm_service = LLMService()
obsidian_service = ObsidianService()
chatgpt_parser = ChatGPTParser()

# API Routes
@api_router.get("/")
async def root():
    return {"message": "ChatGPT to Obsidian Memory Sync API", "version": "1.0.0"}

@api_router.post("/test/connection")
async def test_llm_connection():
    """Test LLM connection"""
    try:
        config = await db.sync_configs.find_one({})
        if config:
            provider = config.get("llm_provider", "openai")
            model = config.get(f"{provider}_model", "gpt-4" if provider == "openai" else "llama2")
        else:
            provider = "openai"
            model = "gpt-4"
        
        # Test connection
        start_time = time.time()
        is_healthy = await llm_service.health_check(provider, model)
        response_time = time.time() - start_time
        
        return {
            "provider": provider,
            "model": model,
            "status": "connected" if is_healthy else "failed",
            "response_time_seconds": round(response_time, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@api_router.get("/health")
async def health_check():
    """System health check endpoint"""
    try:
        # Check database connection
        db_status = "healthy"
        try:
            await db.admin.command('ping')
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        # Check LLM services
        config = await db.sync_configs.find_one({})
        if config:
            provider = config.get("llm_provider", "openai")
            model = config.get(f"{provider}_model", "gpt-4" if provider == "openai" else "llama2")
        else:
            provider = "openai"
            model = "gpt-4"
        
        llm_status = "healthy" if await llm_service.health_check(provider, model) else "unhealthy"
        
        # Check Obsidian vault
        vault_status = "healthy" if obsidian_service.vault_path.exists() else "vault_missing"
        
        # Get recent sync status
        recent_job = await db.sync_jobs.find_one({}, sort=[("created_at", -1)])
        last_sync_status = recent_job.get("status", "no_jobs") if recent_job else "no_jobs"
        
        return {
            "status": "healthy" if all(s == "healthy" for s in [db_status, llm_status, vault_status]) else "degraded",
            "database": db_status,
            "llm_service": llm_status,
            "obsidian_vault": vault_status,
            "last_sync": last_sync_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@api_router.post("/import/chatgpt", response_model=ProcessedConversation)
async def import_chatgpt_conversation(import_data: ChatGPTImport, background_tasks: BackgroundTasks):
    """Import and process a ChatGPT conversation"""
    try:
        # Parse the input based on type
        if import_data.import_type == "json":
            parsed_conversations = chatgpt_parser.parse_json_export(import_data.content)
            if not parsed_conversations:
                raise HTTPException(status_code=400, detail="Failed to parse JSON export")
            # For now, take the first conversation
            parsed_data = parsed_conversations[0]
        else:
            # Text or markdown format
            parsed_data = chatgpt_parser.parse_text_format(import_data.content, import_data.title)
        
        # Create processed conversation record
        processed_conv = ProcessedConversation(
            original_title=parsed_data['title'],
            structured_title=parsed_data['title'],  # Will be updated after processing
            raw_content=parsed_data['raw_content'],
            structured_content="Processing...",
            tags=import_data.tags
        )
        
        # Save to database
        await db.processed_conversations.insert_one(processed_conv.dict())
        
        # Process asynchronously
        background_tasks.add_task(process_conversation_async, processed_conv.id)
        
        return processed_conv
        
    except Exception as e:
        logger.error(f"Error importing conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Import error: {str(e)}")

@api_router.post("/import/file")
async def import_chatgpt_file(file: UploadFile = File(...), tags: str = ""):
    """Import ChatGPT conversation from uploaded file"""
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Determine file type
        file_ext = Path(file.filename).suffix.lower()
        import_type = "json" if file_ext == ".json" else "text"
        
        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()] if tags else []
        
        import_data = ChatGPTImport(
            content=content_str,
            import_type=import_type,
            title=Path(file.filename).stem,
            tags=tag_list
        )
        
        return await import_chatgpt_conversation(import_data, BackgroundTasks())
        
    except Exception as e:
        logger.error(f"Error importing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File import error: {str(e)}")

@api_router.get("/conversations/processed", response_model=List[ProcessedConversation])
async def get_processed_conversations(limit: int = 50, skip: int = 0):
    """Get all processed conversations"""
    conversations = await db.processed_conversations.find().skip(skip).limit(limit).sort("created_at", -1).to_list(limit)
    return [ProcessedConversation(**conv) for conv in conversations]

@api_router.get("/conversations/processed/{conversation_id}", response_model=ProcessedConversation)
async def get_processed_conversation(conversation_id: str):
    """Get a specific processed conversation"""
    conversation = await db.processed_conversations.find_one({"id": conversation_id})
    if not conversation:
        raise HTTPException(status_code=404, detail="Processed conversation not found")
    return ProcessedConversation(**conversation)

@api_router.post("/sync/manual")
async def manual_sync(background_tasks: BackgroundTasks):
    """Trigger manual sync to Obsidian"""
    sync_job = SyncJob(status="pending", job_type="manual_sync")
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

# Background processing functions
async def process_conversation_async(conversation_id: str):
    """Process a conversation asynchronously"""
    try:
        # Get the conversation
        conv_data = await db.processed_conversations.find_one({"id": conversation_id})
        if not conv_data:
            logger.error(f"Conversation {conversation_id} not found")
            return
        
        processed_conv = ProcessedConversation(**conv_data)
        
        # Get sync config
        config = await db.sync_configs.find_one({})
        if config:
            provider = config.get("llm_provider", "openai")
            model = config.get(f"{provider}_model", "gpt-4" if provider == "openai" else "llama2")
            template = config.get("processing_template", "advanced_structured")
        else:
            provider = "openai"
            model = "gpt-4"
            template = "advanced_structured"
        
        # Process the conversation
        processed_data = await llm_service.process_chatgpt_conversation(
            processed_conv.raw_content, provider, model, template
        )
        
        # Update the conversation with processed data
        update_data = {
            "structured_title": processed_data.get("structured_title", processed_conv.original_title),
            "structured_content": processed_data.get("structured_content", processed_conv.raw_content),
            "summary": processed_data.get("summary", ""),
            "key_concepts": processed_data.get("key_concepts", []),
            "frameworks": processed_data.get("frameworks", []),
            "action_items": processed_data.get("action_items", []),
            "tags": processed_conv.tags + processed_data.get("tags", []),
            "processed_at": datetime.utcnow()
        }
        
        await db.processed_conversations.update_one(
            {"id": conversation_id},
            {"$set": update_data}
        )
        
        logger.info(f"Successfully processed conversation: {conversation_id}")
        
    except Exception as e:
        logger.error(f"Error processing conversation {conversation_id}: {str(e)}")

async def perform_sync(job_id: str):
    """Perform the actual sync operation"""
    try:
        # Update job status
        await db.sync_jobs.update_one(
            {"id": job_id},
            {"$set": {"status": "running", "started_at": datetime.utcnow()}}
        )
        
        # Get unsynced processed conversations
        conversations = await db.processed_conversations.find({
            "synced_to_obsidian": False,
            "structured_content": {"$ne": "Processing..."}
        }).to_list(None)
        
        logger.info(f"Found {len(conversations)} processed conversations to sync")
        
        synced_count = 0
        errors = []
        
        for conv_data in conversations:
            try:
                processed_conv = ProcessedConversation(**conv_data)
                
                # Sync to Obsidian
                file_path = await obsidian_service.sync_processed_conversation_to_obsidian(processed_conv)
                
                # Update conversation in database
                await db.processed_conversations.update_one(
                    {"id": processed_conv.id},
                    {
                        "$set": {
                            "synced_to_obsidian": True,
                            "obsidian_file_path": file_path
                        }
                    }
                )
                
                synced_count += 1
                logger.info(f"Synced processed conversation: {processed_conv.structured_title}")
                
            except Exception as e:
                error_msg = f"Error syncing processed conversation {conv_data.get('id', 'unknown')}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Update job completion
        await db.sync_jobs.update_one(
            {"id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "items_processed": synced_count,
                    "errors": errors
                }
            }
        )
        
        logger.info(f"Sync completed: {synced_count} processed conversations synced")
        
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
        sync_job = SyncJob(status="pending", job_type="auto_sync")
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