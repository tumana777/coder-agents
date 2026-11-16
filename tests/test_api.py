import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Mock external modules before importing the app
# --------------------------------------------------

database_mock = ModuleType("backend.database")

@asynccontextmanager
async def fake_lifespan(app):
    yield


database_mock.lifespan = fake_lifespan

agents_mock = ModuleType("backend.agents")
agents_mock.build_graph = MagicMock()

sys.modules["backend.database"] = database_mock
sys.modules["backend.agents"] = agents_mock


from backend.main import app


# --------------------------------------------------
# Fake Graph
# --------------------------------------------------

class FakeGraph:

    def __init__(self):
        self.invoke = MagicMock()
        self.get_state = MagicMock()


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def graph():
    return FakeGraph()


@pytest.fixture
def client(graph):
    app.state.graph = graph

    return TestClient(app)


# --------------------------------------------------
# GET /
# --------------------------------------------------

def test_read_root(client):

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "service": "AI Code Review Service",
        "version": "0.1.0",
        "status": "running"
    }


# --------------------------------------------------
# POST /api/code-review
# --------------------------------------------------

def test_create_code_review(client, graph):

    graph.get_state.return_value = SimpleNamespace(
        values={
            "generated_code": "print('Hello')",
            "review_status": "approved",
            "review_feedback": "Code looks good",
            "attempts": 1,
            "human_decision": ""
        },
        next=("human_approval",)
    )

    response = client.post(
        "/api/code-review",
        json={
            "request": "Write a Hello World program"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "waiting_human_approval"
    assert data["code"] == "print('Hello')"
    assert data["review_status"] == "approved"
    assert data["attempts"] == 1

    graph.invoke.assert_called_once()

    initial_state = graph.invoke.call_args.args[0]

    assert initial_state["user_request"] == "Write a Hello World program"


# --------------------------------------------------
# GET /api/code-review/{thread_id}
# --------------------------------------------------

def test_get_code_review(client, graph):

    graph.get_state.return_value = SimpleNamespace(
        values={
            "generated_code": "print('Hello')",
            "review_status": "approved",
            "review_feedback": "Looks good",
            "attempts": 1,
            "human_decision": ""
        },
        next=("human_approval",)
    )

    response = client.get("/api/code-review/test-thread-id")

    assert response.status_code == 200

    data = response.json()

    assert data["thread_id"] == "test-thread-id"
    assert data["status"] == "waiting_human_approval"
    assert data["review_status"] == "approved"


def test_get_code_review_not_found(client, graph):

    graph.get_state.return_value = SimpleNamespace(
        values={},
        next=()
    )

    response = client.get("/api/code-review/unknown-thread")

    assert response.status_code == 404


# --------------------------------------------------
# POST /api/code-review/{thread_id}/resume
# --------------------------------------------------

def test_resume_code_review(client, graph):

    graph.get_state.side_effect = [
        SimpleNamespace(
            values={
                "generated_code": "print('Hello')",
                "review_status": "approved",
                "review_feedback": "Looks good",
                "attempts": 1,
                "human_decision": ""
            },
            next=("human_approval",)
        ),
        SimpleNamespace(
            values={
                "generated_code": "print('Hello')",
                "review_status": "approved",
                "review_feedback": "Looks good",
                "attempts": 1,
                "human_decision": "approved",
                "status": "completed"
            },
            next=()
        )
    ]

    response = client.post(
        "/api/code-review/test-thread-id/resume",
        json={
            "decision": "approved"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "completed"
    assert data["human_decision"] == "approved"

    assert graph.invoke.call_count == 1


def test_resume_code_review_not_waiting(client, graph):

    graph.get_state.return_value = SimpleNamespace(
        values={
            "generated_code": "print('Hello')"
        },
        next=()
    )

    response = client.post(
        "/api/code-review/test-thread-id/resume",
        json={
            "decision": "approved"
        }
    )

    assert response.status_code == 400

    graph.invoke.assert_not_called()


def test_resume_code_review_not_found(client, graph):

    graph.get_state.return_value = SimpleNamespace(
        values={},
        next=()
    )

    response = client.post(
        "/api/code-review/unknown-thread/resume",
        json={
            "decision": "approved"
        }
    )

    assert response.status_code == 404

    graph.invoke.assert_not_called()


# --------------------------------------------------
# Request validation
# --------------------------------------------------

def test_create_code_review_invalid_request(client):

    response = client.post(
        "/api/code-review",
        json={}
    )

    assert response.status_code == 422


def test_resume_code_review_invalid_decision(client):

    response = client.post(
        "/api/code-review/test-thread-id/resume",
        json={
            "decision": "maybe"
        }
    )

    assert response.status_code == 422
