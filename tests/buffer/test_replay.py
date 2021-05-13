from rllr.buffer import ReplayBuffer
import numpy as np


def test_get():
    buffer = ReplayBuffer(buffer_size=3, batch_size=3, device="cpu")
    buffer.add(1, 5)
    buffer.add(2, 6)
    buffer.add(3, 7)
    assert buffer.is_enough()
    res = list(buffer.get())
    assert all(res[0].numpy() == np.array([[1], [2], [3]]))
    assert all(res[1].numpy() == np.array([[5], [6], [7]]))
    buffer.add(4, 8)
    res = list(buffer.get())
    assert all(res[0].numpy() == np.array([[2], [3], [4]]))
    assert all(res[1].numpy() == np.array([[6], [7], [8]]))


def test_clear():
    buffer = ReplayBuffer(buffer_size=3, batch_size=3, device="cpu")
    buffer.add(1)
    buffer.add(2)
    buffer.add(3)
    assert buffer.is_enough()
    assert len(buffer.buffer) == 3
    buffer.add(4)
    buffer.add(5)
    assert buffer.is_enough()
    assert len(buffer.buffer) == 3
    buffer.clear()
    assert len(buffer.buffer) == 0
    assert not buffer.is_enough()


def test_is_enough():
    buffer = ReplayBuffer(buffer_size=3, batch_size=3, device="cpu")
    buffer.add(1)
    buffer.add(2)
    assert not buffer.is_enough()
    buffer.add(3)
    assert buffer.is_enough()
    buffer.add(4)
    buffer.add(5)
    assert buffer.is_enough()