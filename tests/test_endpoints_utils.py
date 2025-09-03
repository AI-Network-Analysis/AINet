import math


def test_parse_timestamp_handles_epoch_and_iso():
    from server.api.endpoints import _parse_timestamp

    # Epoch int
    dt1 = _parse_timestamp(0)
    assert dt1.year == 1970

    # Epoch float string
    dt2 = _parse_timestamp("1712345678.5")
    assert dt2.year >= 2024

    # ISO with Z
    dt3 = _parse_timestamp("2025-01-02T03:04:05Z")
    assert dt3.year == 2025 and dt3.month == 1

    # ISO with offset
    dt4 = _parse_timestamp("2025-01-02T03:04:05+00:00")
    assert dt4.year == 2025 and dt4.minute == 4


def test_normalize_connections_various_shapes():
    from server.api.endpoints import _normalize_connections

    # New format
    conn1 = _normalize_connections({
        'total_connections': 10,
        'tcp_established': 3,
        'tcp_listen': 2,
        'udp_connections': 5
    })
    assert conn1['total_connections'] == 10
    assert conn1['tcp_established'] == 3
    assert conn1['tcp_listen'] == 2
    assert conn1['udp_connections'] == 5

    # Old aggregate format
    conn2 = _normalize_connections({
        'total': 7,
        'tcp': 4,
        'udp': 3,
        'established': 2
    })
    assert conn2['total_connections'] == 7
    assert conn2['tcp_established'] == 2
    assert conn2['udp_connections'] == 3
