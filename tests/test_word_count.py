"""Unit tests for the deterministic Unicode-aware word counter."""

from __future__ import annotations

from src.pipeline.word_count import count_translation_words


def test_empty_and_whitespace():
    assert count_translation_words("") == 0
    assert count_translation_words("   \n\t  ") == 0


def test_french_accented_text():
    text = "L'entreprise a été créée à Genève en février."
    # "L'entreprise", "a", "été", "créée", "à", "Genève", "en", "février"
    assert count_translation_words(text) == 8


def test_french_apostrophe_counts_as_one_word():
    assert count_translation_words("l'entreprise") == 1
    assert count_translation_words("aujourd'hui") == 1


def test_german_compound_words():
    text = "Donaudampfschifffahrt ist ein langes Wort."
    assert count_translation_words(text) == 5
    assert count_translation_words("Donaudampfschifffahrt") == 1


def test_turkish_dotted_dotless_characters():
    text = "İstanbul'da güzel bir gün, çiçekler açtı ve İşığı gördük."
    words = ["İstanbul'da", "güzel", "bir", "gün,", "çiçekler", "açtı", "ve", "İşığı", "gördük."]
    # Punctuation like trailing commas/periods is not part of \w, so it's
    # stripped automatically; verify the word count matches the number of
    # actual words regardless of attached punctuation.
    assert count_translation_words(text) == len(words)
    assert count_translation_words("ığİÖŞÇÜ") == 1


def test_serbo_croatian_diacritics():
    text = "Čovjek je otišao u šumu s pet pčela i žutom kućom."
    assert count_translation_words(text) == 11
    assert count_translation_words("čćšžđ") == 1


def test_hyphenated_word_not_split():
    assert count_translation_words("state-of-the-art") == 1
    assert count_translation_words("This is state-of-the-art technology.") == 4


def test_markdown_formatting_stripped():
    text = "# Heading\n**bold text** and *italic* and _underline_ and `code`."
    # words: Heading, bold, text, and, italic, and, underline, and, code
    assert count_translation_words(text) == 9


def test_html_page_break_ignored():
    text = (
        "First page content.\n"
        '<div style="page-break-after: always;"></div>\n'
        "Second page content."
    )
    assert count_translation_words(text) == 6


def test_html_page_break_variants_ignored():
    variants = [
        '<div style="page-break-after: always;"></div>',
        "<div style='page-break-after: always;'></div>",
        "<DIV STYLE=\"page-break-after: always;\"></DIV>",
    ]
    for divider in variants:
        text = f"Alpha {divider} Beta"
        assert count_translation_words(text) == 2


def test_mixed_markdown_and_html_and_unicode():
    text = (
        "# Başlık\n"
        '<div style="page-break-after: always;"></div>\n'
        "**Čovjek** je kupio *state-of-the-art* uređaj u Zürichu."
    )
    # Başlık, Čovjek, je, kupio, state-of-the-art, uređaj, u, Zürichu
    assert count_translation_words(text) == 8
