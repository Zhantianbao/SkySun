# 本文件负责测试用例中的 test、tools 相关流程。
from skysun.tool.tools import count_words


# 调用位置：pytest 执行该测试用例时调用。
def test_count_words_uses_whitespace() -> None:
    """验证计数词数使用空白字符。"""
    assert count_words("one  two\nthree") == 3
    assert count_words("  ") == 0
