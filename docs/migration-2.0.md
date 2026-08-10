# ZyQuant 1.2 到 2.0 迁移

ZyQuant 2.0 只将 `src/zyquant` 打包进 wheel，具体策略保留在用户项目。

- 策略配置从安装包中的短 entry-point 名改为
  `module.path:factory`。
- `zyquant.data` 不再导出 Hermes、JQData 和 SQL 类；如需 Python API，
  从 `zyquant.connectors` 对应模块导入。
- `zyq data publish` 必须显式传入 `--source` 和连接器请求文件。
- Hermes 采集统一为
  `zyq data acquire --source hermes --action run|resume|status --request ...`。
- 自定义执行、费用和报告实现也使用 `module.path:factory`。
