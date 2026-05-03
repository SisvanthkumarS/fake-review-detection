"""End-to-end Kafka smoke test.

Produces 5 messages, then consumes them back. Confirms the broker is up and
reachable. Requires Kafka running via docker compose.
"""
import json
import time

from kafka import KafkaProducer, KafkaConsumer

TOPIC = "test-topic"
BOOTSTRAP = "localhost:9092"


def produce():
    print("Producing 5 messages...")
    p = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    for i in range(5):
        msg = {"id": i, "text": f"hello {i}", "ts": time.time()}
        p.send(TOPIC, msg)
        print(f"  sent: {msg}")
    p.flush()
    p.close()


def consume():
    print("\nConsuming...")
    c = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        auto_offset_reset="earliest",
        consumer_timeout_ms=5000,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="test-group",
    )
    received = 0
    for msg in c:
        print(f"  got: {msg.value}")
        received += 1
    c.close()

    print(f"\nReceived {received} messages")
    if received == 0:
        raise SystemExit("ERROR: 0 messages received. Is Kafka running?")
    print("Kafka is working.")


if __name__ == "__main__":
    produce()
    time.sleep(1)
    consume()