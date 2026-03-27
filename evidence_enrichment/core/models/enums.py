"""Shared enums for the evidence-first pipeline."""

from enum import Enum


class ProviderType(str, Enum):
    REPLAY = "replay"
    SERPER = "serper"
    TAVILY = "tavily"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ReviewDecision(str, Enum):
    AUTO_APPROVE = "auto_approve"
    NEEDS_REVIEW = "needs_review"
    AUTO_REJECT = "auto_reject"


class SourceType(str, Enum):
    SEARCH = "search"
    DOCUMENT = "document"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"


class DocumentType(str, Enum):
    COMPANY_WEBSITE = "company_website"
    REGULATORY_FILING = "regulatory_filing"
    NEWS_ARTICLE = "news_article"
    UNKNOWN = "unknown"

