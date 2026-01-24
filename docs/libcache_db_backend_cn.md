# LibCache 数据库后端配置

`LibCache` 为用于存储基准测试结果的数据库提供了一套调度机制。该功能依赖 `sqlalchemy` 库，请确保运行环境中已安装此依赖。

通过设置环境变量 `FLAGGEMS_DB_URL` 可以选择相应的数据库后端。

## SQLite3

默认支持的后端是 `SQLite3`。请确保环境中已安装相关的 `sqlite3` 库。如果您希望将数据库文件存储在指定路径，请设置环境变量：
```bash
export FLAGGEMS_DB_URL=sqlite:///${db_path}
```
如果您想将其作为内存数据库使用，或者不希望在当前环境中保留或复用缓存，可以将环境变量设置为：
```bash
export FLAGGEMS_DB_URL=sqlite:///:memory:
```
此时缓存将仅存储在内存中，连接断开后数据库内容将会丢失。

## PostgreSQL

作为一种嵌入式数据库，`SQLite3` 不支持并发写入（multi-writers），这在某些并发场景下可能会受到限制。因此，我们也支持使用 `PostgreSQL` 作为后端。与嵌入式数据库不同，PostgreSQL 需要在使用前进行安装和配置。您可以参考[官方文档](https://documentation.ubuntu.com/server/how-to/databases/install-postgresql/)了解安装方法。此外，您也可以使用远程数据库，从而允许多个 `FlagGems` 实例同时连接并共享基准测试结果。

在创建好数据库后，可以使用以下 URL 格式设置环境变量：
```bash
export FLAGGEMS_DB_URL=postgresql+psycopg:///${user}:${password}@${host}:${port}/${db}
```
如果数据库位于本地且当前系统用户拥有直接访问权限，可以使用更简洁的配置：
```bash
export FLAGGEMS_DB_URL=postgresql+psycopg:///${db}
```

在使用 PostgreSQL 之前，请确保您的机器上已安装 `psycopg` 驱动。
