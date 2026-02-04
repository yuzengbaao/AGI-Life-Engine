# Bytebot 集成配置计划

## 版本
- v1.0

## 目标
- 将 Bytebot 作为桌面执行器融入现有 AGI 编排，实现 GUI 多步任务与文档处理，提供可观测与可复现的执行环境。

## 范围与前提
- 宿主：Windows 10/11，安装 Docker Desktop（Linux 容器）
- 虚拟桌面：Ubuntu 22.04 + XFCE，预装浏览器、VS Code 等
- 模型：OpenAI/Anthropic/Gemini 至少其一

## 环境矩阵
- 端口：`9990`（桌面与 computer-use）、`9991`（任务 API）、`9992`（Web UI）
- 数据库：Postgres 16（容器）
- 网络：Docker bridge，服务间内网互通

## 安全策略
- 仅内网开放端口，必要时置于反向代理并加鉴权
- 密钥入 `docker/.env`，最小权限，禁止日志输出密钥
- 文件卷按任务分域挂载：只读/读写分离

## 网络与拓扑
- `bytebot-desktop` 暴露 `9990`（noVNC 与 computer-use）
- `bytebot-agent` 暴露 `9991`（任务管理与编排）
- `bytebot-ui` 暴露 `9992`（任务创建与可视化桌面）
- `postgres` 仅容器内网访问

## 环境变量与密钥
- `ANTHROPIC_API_KEY` 可选
- `OPENAI_API_KEY` 可选
- `GEMINI_API_KEY` 可选
- `DATABASE_URL` 默认 `postgresql://postgres:postgres@postgres:5432/bytebotdb`
- `BYTEBOT_DESKTOP_BASE_URL` 默认 `http://bytebot-desktop:9990`
- `BYTEBOT_AGENT_BASE_URL` 默认 `http://bytebot-agent:9991`
- `BYTEBOT_DESKTOP_VNC_URL` 默认 `http://bytebot-desktop:9990/websockify`

## 卷映射设计
- 输入卷：`D:/AGI/input` → `/workspace/input`（只读）
- 输出卷：`D:/AGI/output` → `/workspace/output`（读写）
- 日志卷：`D:/AGI/logs` → `/workspace/logs`（读写）

## 观测与存证
- 截图与录屏存入 `/workspace/logs/screenshots`、`/workspace/logs/recordings`
- 任务日志归档 `/workspace/logs/tasks`
- 指标：成功率、耗时、准确率、人工接管率

## 集成与编排
- 在 AGI 中新增适配器：
  - 任务创建：`POST http://localhost:9991/tasks`（文本描述 + 附件）
  - 桌面动作：`POST http://localhost:9990/computer-use`（截图、点击、键盘输入等）
- 统一链路 ID 与重试策略，接管模式支持人工介入

## 里程碑
- P0：本机启动 + 示例任务（网页抓取、PDF提取、批量下载）
- P1：适配器上线 + 指标采集 + 文件卷映射
- P2：安全加固 + 多实例并发 + Prompt 模板库
- P3：远控宿主机方案与合规审计（如需）

## 风险与回退
- 风险：界面变化、登录失败、长流程中断、敏感数据泄露
- 回退：人工接管、任务取消、恢复至快照环境、撤销卷写入

## 验收标准
- 端口与服务联通检查通过
- 示例任务 3 项成功率 ≥ 90%
- 日志与截图完整入库
- 安全清单全项合格