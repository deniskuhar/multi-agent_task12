from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import get_settings
from langfuse_utils import get_prompt_text
from schemas import ResearchPlan
from tools import knowledge_search, web_search

settings = get_settings()
model = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.openai_api_key.get_secret_value(),
    temperature=0.1,
    timeout=settings.request_timeout_seconds,
)

planner_web_search = tool(
    "web_search",
    description="Search the public web for relevant pages and return concise results with URLs.",
)(web_search)

planner_knowledge_search = tool(
    "knowledge_search",
    description="Search the local knowledge base using hybrid retrieval and reranking.",
)(knowledge_search)

planner_agent = create_agent(
    model=model,
    tools=[planner_web_search, planner_knowledge_search],
    system_prompt=get_prompt_text(settings.planner_prompt_name),
    response_format=ResearchPlan,
)
