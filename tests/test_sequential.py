from __future__ import annotations

from src.inference import sequential_map


def test_sequential_map_never_overlaps_items() -> None:
    active = 0
    maximum_active = 0
    order: list[int] = []

    def processor(item: int) -> int:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        order.append(item)
        active -= 1
        return item * 2

    assert sequential_map(range(5), processor) == [0, 2, 4, 6, 8]
    assert maximum_active == 1
    assert order == [0, 1, 2, 3, 4]

