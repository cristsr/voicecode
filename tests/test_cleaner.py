import asyncio

import pytest

from config import VoiceCodeConfig
from domain.models import CleanText, TranscribedText
from pipeline.cleaner import RegexCleaner, clean_text


@pytest.fixture
def patterns() -> list[str]:
    return VoiceCodeConfig().filler_patterns


def test_removes_eh_variations(patterns: list[str]) -> None:
    assert clean_text('eh quiero eh un café', patterns) == 'Quiero un café'


def test_removes_mmm_variations(patterns: list[str]) -> None:
    assert clean_text('mmm vamos mmmm a comer', patterns) == 'Vamos a comer'


def test_removes_o_sea(patterns: list[str]) -> None:
    assert clean_text('o sea que no', patterns) == 'Que no'


def test_removes_digamos(patterns: list[str]) -> None:
    assert clean_text('digamos que sí', patterns) == 'Que sí'


def test_removes_basicamente(patterns: list[str]) -> None:
    assert clean_text('básicamente es esto', patterns) == 'Es esto'


def test_removes_pues_aislado(patterns: list[str]) -> None:
    assert clean_text('pues vamos', patterns) == 'Vamos'


def test_removes_entonces_variants(patterns: list[str]) -> None:
    assert clean_text('entonces vamos', patterns) == 'Vamos'
    assert clean_text('entoces vamos', patterns) == 'Vamos'


def test_removes_la_verdad(patterns: list[str]) -> None:
    assert clean_text('la verdad no sé', patterns) == 'No sé'


def test_case_insensitive(patterns: list[str]) -> None:
    assert clean_text('Eh Pues vamos', patterns) == 'Vamos'


def test_collapses_multiple_spaces(patterns: list[str]) -> None:
    assert clean_text('hola    mundo', patterns) == 'Hola mundo'


def test_empty_text_does_not_break(patterns: list[str]) -> None:
    assert clean_text('', patterns) == ''


def test_only_fillers_results_in_empty(patterns: list[str]) -> None:
    assert clean_text('eh mmm pues', patterns) == ''


def test_capitalizes_first_letter(patterns: list[str]) -> None:
    assert clean_text('hola mundo', patterns) == 'Hola mundo'


def test_strips_dangling_comma_left_by_filler_removal(patterns: list[str]) -> None:
    assert clean_text('eh, quiero un cafe', patterns) == 'Quiero un cafe'


def test_strips_dangling_punctuation_from_leading_filler(patterns: list[str]) -> None:
    assert clean_text('pues, la verdad no se', patterns) == 'No se'


@pytest.mark.asyncio
async def test_regex_cleaner_publishes_clean_text() -> None:
    config = VoiceCodeConfig()
    cleaner = RegexCleaner(config)
    text_queue: asyncio.Queue[TranscribedText] = asyncio.Queue()
    clean_queue: asyncio.Queue[CleanText] = asyncio.Queue()

    await text_queue.put(TranscribedText(seq=3, raw='eh hola mundo'))
    task = asyncio.create_task(cleaner.clean(text_queue, clean_queue))

    result = await asyncio.wait_for(clean_queue.get(), timeout=1)
    task.cancel()

    assert result == CleanText(seq=3, text='Hola mundo')
