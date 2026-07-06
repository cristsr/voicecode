from domain.models import CleanText
from pipeline.writer import SequenceBuffer


def test_emits_immediately_when_seq_matches_expected() -> None:
    buffer = SequenceBuffer()
    item = CleanText(seq=0, text='hola')

    result = buffer.process(item)

    assert result == [item]
    assert buffer.expected_seq == 1


def test_buffers_item_that_arrives_ahead_of_order() -> None:
    buffer = SequenceBuffer()
    item = CleanText(seq=2, text='tercero')

    result = buffer.process(item)

    assert result == []
    assert buffer.pending == {2: item}
    assert buffer.expected_seq == 0


def test_drains_pending_in_ascending_order_after_gap_fills() -> None:
    buffer = SequenceBuffer()
    item1 = CleanText(seq=1, text='segundo')
    item2 = CleanText(seq=2, text='tercero')
    item0 = CleanText(seq=0, text='primero')

    assert buffer.process(item1) == []
    assert buffer.process(item2) == []
    result = buffer.process(item0)

    assert result == [item0, item1, item2]
    assert buffer.expected_seq == 3
    assert buffer.pending == {}


def test_stops_draining_at_first_gap() -> None:
    buffer = SequenceBuffer()
    item0 = CleanText(seq=0, text='primero')
    item1 = CleanText(seq=1, text='segundo')
    item3 = CleanText(seq=3, text='cuarto')

    buffer.process(item3)
    result = buffer.process(item0)

    assert result == [item0]
    assert buffer.expected_seq == 1
    assert buffer.pending == {3: item3}

    result = buffer.process(item1)
    assert result == [item1]
    assert buffer.expected_seq == 2
    assert buffer.pending == {3: item3}


def test_overlapping_recordings_out_of_order_end_to_end() -> None:
    buffer = SequenceBuffer()
    items = [
        CleanText(seq=1, text='b'),
        CleanText(seq=0, text='a'),
        CleanText(seq=3, text='d'),
        CleanText(seq=2, text='c'),
    ]

    emitted: list[CleanText] = []
    for item in items:
        emitted.extend(buffer.process(item))

    assert [i.text for i in emitted] == ['a', 'b', 'c', 'd']
