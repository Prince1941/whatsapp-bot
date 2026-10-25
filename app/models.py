from datetime import datetime, timezone
from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents a single parsed message from a WhatsApp export."""
    sender: str
    timestamp_str: Optional[str] = None
    timestamp: Optional[datetime] = None
    content: str
    is_system: bool = False
    is_media: bool = False
    raw_line_num: int = 0


class ParseResult(BaseModel):
    """Result of parsing a WhatsApp export file."""
    messages: List[ChatMessage]
    senders: Dict[str, int]  # sender_name -> count
    total_lines: int
    system_message_count: int
    media_count: int
    user_message_count: int


class PrivacyFilterConfig(BaseModel):
    """Rules for filtering sensitive information before learning."""
    redact_phone_numbers: bool = True
    redact_emails: bool = True
    redact_urls: bool = True
    redact_financial: bool = True
    custom_banned_terms: List[str] = Field(default_factory=list)
    action: Literal["redact", "drop"] = "redact"


class LengthStats(BaseModel):
    avg_words_per_message: float = 0.0
    avg_chars_per_message: float = 0.0
    short_ratio: float = 0.0  # < 8 words
    medium_ratio: float = 0.0  # 8 - 25 words
    long_ratio: float = 0.0  # > 25 words


class CapitalizationStyle(BaseModel):
    primary_case: Literal["all_lowercase", "sentence_case", "mixed", "all_caps"] = "sentence_case"
    lowercase_rate: float = 0.0


class PunctuationStats(BaseModel):
    exclamation_freq: float = 0.0  # per message
    question_freq: float = 0.0
    ellipsis_freq: float = 0.0
    period_ending_freq: float = 0.0  # frequency of ending messages with a dot
    comma_freq: float = 0.0


class EmojiStats(BaseModel):
    emoji_frequency: float = 0.0  # emojis per message
    top_emojis: List[str] = Field(default_factory=list)


class StyleProfile(BaseModel):
    """Lightweight, inspectable style profile representing user writing habits."""
    id: str = "default"
    profile_name: str = "Default Profile"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sender_name: str = ""
    total_training_messages: int = 0
    confidence_level: Literal["high", "medium", "low", "none"] = "none"
    confidence_description: str = ""
    length_stats: LengthStats = Field(default_factory=LengthStats)
    capitalization: CapitalizationStyle = Field(default_factory=CapitalizationStyle)
    punctuation: PunctuationStats = Field(default_factory=PunctuationStats)
    emojis: EmojiStats = Field(default_factory=EmojiStats)
    spelling_preferences: Dict[str, str] = Field(default_factory=dict)  # standard/alt -> user_preferred
    frequent_short_phrases: List[str] = Field(default_factory=list)  # recurring real short phrases
    intent_phrases: Dict[str, List[str]] = Field(default_factory=dict)  # intent_category -> phrases
    slang_and_phrases: List[str] = Field(default_factory=list)
    common_openers: List[str] = Field(default_factory=list)
    common_closers: List[str] = Field(default_factory=list)
    sample_snippets: List[str] = Field(default_factory=list)
    custom_notes: str = ""


class ChatPair(BaseModel):
    """Represents an authentic (incoming_message -> user_reply) exchange pair."""
    id: Optional[int] = None
    profile_id: str = "default"
    incoming: str
    reply: str
    other_sender: Optional[str] = None
    timestamp_str: Optional[str] = None
    similarity_score: Optional[float] = None


class DraftRequest(BaseModel):
    """Input payload to draft replies."""
    context: str = Field(..., description="Received message or conversation thread")
    intent: Optional[str] = Field("", description="What you want to convey or accomplish")
    length_preference: Literal["short", "medium", "long"] = "medium"
    style_intensity: Literal["subtle", "moderate", "strong"] = "moderate"
    num_drafts: int = Field(default=2, ge=1, le=5)
    model_name: Optional[str] = None
    provider: Optional[str] = "ollama"  # "ollama", "gemini", "groq", "openai"
    language_script: Literal["auto", "hinglish", "devanagari", "english"] = "auto"


class DraftResponse(BaseModel):
    """Response containing generated reply options."""
    drafts: List[str]
    model_used: str
    confidence_level: str = "none"
    learned_phrases_applied: List[str] = Field(default_factory=list)
    detected_intent: Optional[str] = None
    retrieved_examples: List[str] = Field(default_factory=list)
    needs_user_input: bool = False
    user_input_prompt: Optional[str] = None
    is_cloud: bool = False
    language_script_used: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    disclaimer: str = "AI-generated draft — review and approve before sending"


class ProviderConfig(BaseModel):
    """Configuration for LLM provider."""
    provider: Literal["ollama", "gemini", "groq", "openai"] = "ollama"
    model: str = "qwen2.5:7b"
    api_key_env_var: Optional[str] = None
    is_cloud: bool = False


class OllamaConfig(BaseModel):
    """Ollama connection configuration."""
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    timeout_seconds: float = 120.0


class OllamaStatus(BaseModel):
    """Health check status for Ollama server."""
    is_connected: bool
    available_models: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None

