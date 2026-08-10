# Contributing

1. 从主分支创建短生命周期分支。
2. 一次提交聚焦一个行为，并添加回归测试。
3. 运行完整测试和包检查。
4. 公共契约、数据 schema 或会计行为变化需补充 ADR。
5. 策略插件必须通过合同测试后才能发布。

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src tests
mypy src/zyquant
python -m pip check
python -m pip wheel . --no-deps -w dist
```
