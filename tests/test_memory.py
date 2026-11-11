import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.memory import ConversationBuffer, VectorMemory


def test_conversation_buffer_trims_old_messages():
    buf = ConversationBuffer(system_prompt="sys", max_messages=4)
    for i in range(10):
        buf.add("user", f"message {i}")
    messages = buf.as_list()
    assert messages[0] == {"role": "system", "content": "sys"}
    assert len(messages) == 1 + 4  # system + max_messages
    assert messages[-1]["content"] == "message 9"  # most recent kept


def fake_embed(text: str) -> list[float]:
    """A deterministic 'embedding': word-length bucket counts. Good
    enough to prove retrieval ranks relevant text higher without a real
    model -- this is the test-double equivalent of OllamaClient.embed."""
    buckets = [0.0] * 8
    for word in text.lower().split():
        buckets[len(word) % 8] += 1.0
    return buckets


def test_vector_memory_add_and_search_ranks_relevant_chunk_higher(tmp_path):
    mem = VectorMemory(tmp_path / "test.sqlite3", embed_fn=fake_embed)
    mem.add_text("doc1", "cats cats cats cats cats cats cats cats")
    mem.add_text("doc2", "a a a a a a a a a a a a a a a a a a a a")

    results = mem.search("cats cats cats cats cats", top_k=2)

    assert mem.count() == 2
    assert results[0]["source"] == "doc1"
    assert results[0]["score"] > results[1]["score"]


def test_vector_memory_persists_across_instances(tmp_path):
    db_path = tmp_path / "persist.sqlite3"
    mem1 = VectorMemory(db_path, embed_fn=fake_embed)
    mem1.add_text("doc1", "hello world")
    mem1.close()

    mem2 = VectorMemory(db_path, embed_fn=fake_embed)
    assert mem2.count() == 1
