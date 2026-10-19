from skysun.tool.tools import count_words


def test_count_words_uses_whitespace() -> None:
    assert count_words("one  two\nthree") == 3
    assert count_words("  ") == 0
