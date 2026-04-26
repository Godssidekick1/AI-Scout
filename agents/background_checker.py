"""
Agent 3: Background Checker ("The Investigator")
=================================================
Searches for character red flags, injury history, and sentiment analysis.
Uses Serper.dev for news and Transfermarkt for injury data.
"""

import asyncio
import logging
from typing import Dict, List

from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from data_sources.news_search import NewsSearchSource
from data_sources.transfermarkt_source import TransfermarktSource
from data_sources.data_cache import DataCache
from config import settings

logger = logging.getLogger(__name__)

SENTIMENT_PROMPT = """You are a football character analyst. Analyze these news articles about {player_name} and provide:

1. attitude_score (0-10): Overall character assessment
2. discipline_issues: List any red cards, suspensions, controversies
3. maturity_assessment: One sentence on their professional maturity
4. locker_room_presence: Positive/Neutral/Negative — based on teammate reports
5. key_concerns: Any red flags that a Premier League club should know about

Articles:
{articles}

Respond in JSON format ONLY (no markdown):
{{
    "attitude_score": 7.5,
    "discipline_issues": [],
    "maturity_assessment": "...",
    "locker_room_presence": "...",
    "key_concerns": []
}}"""


def check_backgrounds(state: Dict) -> Dict:
    """
    Check character and injury background for all candidates.
    Runs news search + Transfermarkt injury lookup.
    """
    candidates = state.get("candidate_pool", [])

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                reports = pool.submit(asyncio.run, _async_check_backgrounds(candidates)).result()
        else:
            reports = asyncio.run(_async_check_backgrounds(candidates))
    except RuntimeError:
        reports = asyncio.run(_async_check_backgrounds(candidates))

    state["character_reports"] = state.get("character_reports", []) + reports
    state["messages"] = [
        AIMessage(content=f"🔎 Completed background checks for {len(reports)} candidates. "
                          f"Checked news, injury history, and character reports.")
    ]

    return state


async def _async_check_backgrounds(candidates: List[Dict]) -> List[Dict]:
    """Run background checks on all candidates."""
    cache = DataCache(
        cache_dir=settings.CACHE_DIR,
        ttl_hours=settings.CACHE_TTL_HOURS,
    )
    news_source = NewsSearchSource(cache=cache)
    tmk = TransfermarktSource(cache=cache)

    reports = []

    for candidate in candidates[:10]:  # Limit to top 10
        name = candidate.get("name", "Unknown")
        report = {
            "player_name": name,
            "attitude_score": 7.5,
            "discipline_issues": [],
            "injury_history": [],
            "injury_risk_score": 2.0,
            "total_days_injured_24mo": 0,
            "maturity_assessment": "No data available",
            "locker_room_presence": "Unknown",
            "sources": [],
        }

        # --- Injury data from Transfermarkt ---
        if settings.ENABLE_TRANSFERMARKT:
            try:
                tm_data = await tmk.get_player_stats(name)
                if tm_data and "injury_history" in tm_data:
                    injuries = tm_data["injury_history"]
                    report["injury_history"] = [
                        f"{inj.get('injury', 'Unknown')} — {inj.get('days_missed', 0)} days ({inj.get('from_date', '')})"
                        for inj in injuries[:5]
                    ]
                    total_days = sum(inj.get("days_missed", 0) for inj in injuries)
                    report["total_days_injured_24mo"] = total_days

                    # Calculate injury risk score (0-10)
                    if total_days > 180:
                        report["injury_risk_score"] = 9.0
                    elif total_days > 90:
                        report["injury_risk_score"] = 7.0
                    elif total_days > 45:
                        report["injury_risk_score"] = 5.0
                    elif total_days > 14:
                        report["injury_risk_score"] = 3.0
                    else:
                        report["injury_risk_score"] = 1.0

                    # Check for recurring soft-tissue injuries
                    soft_tissue_keywords = ["hamstring", "calf", "muscle", "groin", "thigh"]
                    soft_tissue_count = sum(
                        1 for inj in injuries
                        if any(kw in inj.get("injury", "").lower() for kw in soft_tissue_keywords)
                    )
                    if soft_tissue_count >= 3:
                        report["injury_risk_score"] = min(report["injury_risk_score"] + 2, 10)

                    report["sources"].append("transfermarkt")
            except Exception as e:
                logger.debug(f"Transfermarkt injury check failed for {name}: {e}")

        # --- News sentiment ---
        if settings.has_serper():
            try:
                sentiment = await news_source.get_sentiment_summary(name)
                all_articles = (
                    sentiment.get("character_articles", []) +
                    sentiment.get("injury_articles", []) +
                    sentiment.get("transfer_articles", [])
                )

                if all_articles:
                    report["sources"].append("serper_news")

                    # Use LLM to analyze sentiment if available
                    if settings.has_gemini() and all_articles:
                        llm_analysis = _analyze_sentiment_with_llm(name, all_articles)
                        if llm_analysis:
                            report.update(llm_analysis)

            except Exception as e:
                logger.debug(f"News search failed for {name}: {e}")

        # Discipline from yellow/red cards
        yellows = candidate.get("yellow_cards", 0)
        reds = candidate.get("red_cards", 0)
        if reds > 0:
            report["discipline_issues"].append(f"{reds} red card(s) this season")
            report["attitude_score"] = max(report["attitude_score"] - reds * 1.5, 0)
        if yellows > 8:
            report["discipline_issues"].append(f"{yellows} yellow cards this season")
            report["attitude_score"] = max(report["attitude_score"] - 1, 0)

        reports.append(report)

    return reports


def _analyze_sentiment_with_llm(player_name: str, articles: List[Dict]) -> Dict:
    """Use Gemini to analyze news sentiment."""
    try:
        llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2,
        )

        articles_text = "\n".join(
            f"- [{a.get('source', 'Unknown')}] {a.get('title', '')}: {a.get('snippet', '')}"
            for a in articles[:8]
        )

        response = llm.invoke([
            SystemMessage(content=SENTIMENT_PROMPT.format(
                player_name=player_name,
                articles=articles_text,
            )),
        ])

        import json
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content.strip())

    except Exception as e:
        logger.debug(f"LLM sentiment analysis failed: {e}")
        return {}
