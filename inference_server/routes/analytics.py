"""
Analytics Routes
================
Endpoints for connection quality reporting and retrain status.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from inference_server import config
from inference_server.schemas import (
    ConnectionReportRequest,
    ConnectionReportResponse,
    RetrainStatusResponse,
)

logger = logging.getLogger("inference_server.routes.analytics")
router = APIRouter(tags=["analytics"])

# Analytics storage
ANALYTICS_DIR = Path(config.BASE_DIR) / "analytics_logs"


def _ensure_analytics_dir() -> Path:
    """Ensure analytics directory exists."""
    ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    return ANALYTICS_DIR


@router.post("/connection/report", response_model=ConnectionReportResponse)
async def report_connection(request: ConnectionReportRequest) -> ConnectionReportResponse:
    """
    Report client connection quality metrics.

    Used for monitoring network conditions from mobile/edge clients.

    Args:
        request: Connection metrics (latency, upload speed, connection type)

    Returns:
        Confirmation of metric recording
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    connection_record = {
        "client_id": request.client_id,
        "latency_ms": request.latency_ms,
        "upload_speed_mbps": request.upload_speed_mbps,
        "connection_type": request.connection_type,
        "timestamp": timestamp,
    }

    try:
        analytics_dir = _ensure_analytics_dir()
        log_file = analytics_dir / "connection_log.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(connection_record) + "\n")

        logger.debug("Connection report from %s: latency=%.1fms",
                     request.client_id, request.latency_ms)

        return ConnectionReportResponse(status="recorded")
    except Exception as e:
        logger.error("Failed to record connection report: %s", e)
        return ConnectionReportResponse(status=f"error: {e}")


@router.get("/connection/stats")
async def get_connection_stats() -> dict:
    """
    Get connection quality statistics.

    Returns aggregate metrics from reported connections.
    """
    try:
        analytics_dir = _ensure_analytics_dir()
        log_file = analytics_dir / "connection_log.jsonl"

        if not log_file.exists():
            return {
                "total_reports": 0,
                "unique_clients": 0,
                "avg_latency_ms": None,
                "connection_types": {},
            }

        total = 0
        clients: set[str] = set()
        latencies: list[float] = []
        conn_types: dict[str, int] = {}

        with open(log_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    total += 1
                    clients.add(record.get("client_id", ""))
                    if record.get("latency_ms") is not None:
                        latencies.append(record["latency_ms"])
                    conn_type = record.get("connection_type", "unknown")
                    conn_types[conn_type] = conn_types.get(conn_type, 0) + 1

        return {
            "total_reports": total,
            "unique_clients": len(clients),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "connection_types": conn_types,
        }
    except Exception as e:
        logger.error("Failed to get connection stats: %s", e)
        return {"error": str(e)}


@router.get("/retrain/status", response_model=RetrainStatusResponse)
async def get_retrain_status() -> RetrainStatusResponse:
    """
    Get auto-retraining pipeline status.

    Returns information about scheduled/pending model retraining.
    """
    # Count pending feedback
    feedback_count = 0
    try:
        feedback_file = Path(config.BASE_DIR) / "feedback_logs" / "feedback_log.jsonl"
        if feedback_file.exists():
            with open(feedback_file, encoding="utf-8") as f:
                feedback_count = sum(1 for line in f if line.strip())
    except Exception:
        pass

    # Note: Auto-retraining is not implemented yet
    # This is a placeholder that returns current status
    return RetrainStatusResponse(
        enabled=False,
        last_retrain=None,
        next_scheduled=None,
        pending_feedback_count=feedback_count,
        status="disabled",
    )
