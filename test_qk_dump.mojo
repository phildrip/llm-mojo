from std.memory import alloc
from std.testing import assert_equal, assert_true
from qk_dump import (
    QKDumpWriter, QKDumpReader,
    global_layer_to_index, index_to_global_layer,
    QK_MAGIC, QK_VERSION, NUM_GLOBAL_LAYERS,
    Q_FLOATS, K_FLOATS, V_FLOATS, RECORD_BYTES
)
import std.os

def test_layer_index_conversions() raises:
    assert_equal(global_layer_to_index(5), 0)
    assert_equal(global_layer_to_index(11), 1)
    assert_equal(global_layer_to_index(17), 2)
    assert_equal(global_layer_to_index(23), 3)
    assert_equal(global_layer_to_index(29), 4)
    assert_equal(global_layer_to_index(0), -1)
    assert_equal(global_layer_to_index(4), -1)
    assert_equal(global_layer_to_index(30), -1)

    assert_equal(index_to_global_layer(0), 5)
    assert_equal(index_to_global_layer(1), 11)
    assert_equal(index_to_global_layer(2), 17)
    assert_equal(index_to_global_layer(3), 23)
    assert_equal(index_to_global_layer(4), 29)
    print("test_layer_index_conversions passed")

def test_write_and_read_roundtrip() raises:
    var test_path = String("/tmp/test_dump_roundtrip.bin")
    var num_test_tokens = 4

    var q_buf = alloc[Float32](Q_FLOATS)
    var k_buf = alloc[Float32](K_FLOATS)
    var v_buf = alloc[Float32](V_FLOATS)

    # 1. Write records
    var writer = QKDumpWriter(test_path)
    for t in range(num_test_tokens):
        for l_idx in range(5):
            var l_num = index_to_global_layer(l_idx)
            # Fill with unique deterministic pattern
            for i in range(Q_FLOATS):
                q_buf[i] = Float32(t * 100000 + l_num * 1000 + i)
            for i in range(K_FLOATS):
                k_buf[i] = Float32(t * 100000 + l_num * 1000 + 20000 + i)
            for i in range(V_FLOATS):
                v_buf[i] = Float32(t * 100000 + l_num * 1000 + 30000 + i)
            writer.write_record(t, l_num, q_buf, k_buf, v_buf)
    writer.close()

    # 2. Read back and verify
    var reader = QKDumpReader(test_path)
    assert_equal(reader.num_tokens, num_test_tokens)

    var q_read = alloc[Float32](Q_FLOATS)
    var k_read = alloc[Float32](K_FLOATS)
    var v_read = alloc[Float32](V_FLOATS)

    for t in range(num_test_tokens):
        for l_idx in range(5):
            var l_num = index_to_global_layer(l_idx)
            reader.read_layer(t, l_idx, q_read, k_read, v_read)
            for i in range(Q_FLOATS):
                var expected = Float32(t * 100000 + l_num * 1000 + i)
                assert_equal(q_read[i], expected)
            for i in range(K_FLOATS):
                var expected = Float32(t * 100000 + l_num * 1000 + 20000 + i)
                assert_equal(k_read[i], expected)
            for i in range(V_FLOATS):
                var expected = Float32(t * 100000 + l_num * 1000 + 30000 + i)
                assert_equal(v_read[i], expected)

    # Test read_k_only and read_q_only
    for t in range(num_test_tokens):
        var l_idx = 2 # layer 17
        var l_num = 17
        reader.read_k_only(t, l_idx, k_read)
        for i in range(K_FLOATS):
            var expected = Float32(t * 100000 + l_num * 1000 + 20000 + i)
            assert_equal(k_read[i], expected)

        reader.read_q_only(t, l_idx, q_read)
        for i in range(Q_FLOATS):
            var expected = Float32(t * 100000 + l_num * 1000 + i)
            assert_equal(q_read[i], expected)

    reader.close()

    # Clean up
    q_buf.free()
    k_buf.free()
    v_buf.free()
    q_read.free()
    k_read.free()
    v_read.free()
    print("test_write_and_read_roundtrip passed")

def test_default_writer_inactive() raises:
    var writer = QKDumpWriter()
    assert_true(not writer.is_active())
    writer.close()
    print("test_default_writer_inactive passed")

def main() raises:
    test_layer_index_conversions()
    test_default_writer_inactive()
    test_write_and_read_roundtrip()
    print("ALL QK DUMP TESTS PASSED")
