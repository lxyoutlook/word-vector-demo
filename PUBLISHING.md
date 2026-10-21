# 教师发布步骤

1. 把本目录中的源码文件上传到 GitHub 仓库根目录。保留 `.streamlit/config.toml` 和 `.gitignore`。
2. 不要把 `.venv`、缓存或两个大模型文件提交到普通 Git 仓库；`.gitignore` 已排除这些内容。
3. 创建 `v1.0.0` Release，把 `vector-demo-full-v1.0.0.zip` 和 `SHA256SUMS.txt` 作为附件上传。
4. 给学员发送 Release 页或完整版附件的链接。请学员下载附件中的完整版 ZIP。
5. GitHub 自动生成的 `Source code (zip)` 和绿色按钮 `Code → Download ZIP` 都只有源码，不含模型；不作为学员的完整版下载入口。

完整版包含源码和两个模型，但不包含Python解释器或依赖包。学员需要先安装64位Python 3.12，首次启动需要联网安装依赖。启动器不依赖教师电脑路径、Codex或Git。

后续更新源码时，重新生成完整版 ZIP 并发布新版本，保持仓库源码与 Release 内容一致。模型来源和第三方说明请保留。
