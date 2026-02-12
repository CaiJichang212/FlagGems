## 项目环境说明
运行、测试项目代码，需要激活使用flagos开发环境。
规则要求：
1. 在Terminal执行命令等操作时，需要进入到lmft_liyc容器：`docker exec lmft_liyc '命令'`。
2. 在Terminal运行代码时，需要先进入容器，再激活conda环境：`conda activate flagos`。
3. 项目所使用的python3路径（lmft_liyc容器里）是：`/opt/conda/envs/flagos/bin/python3`。

## 编写代码说明
1. 尽量新创建独立的代码文件，避免影响原始代码。
2. 如果确实需要修改原始代码，遵循最小改动原则。
3. 多多学习项目的代码，查看项目代码，编写新代码。