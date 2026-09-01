# -*- coding: utf-8 -*-
"""
summarizer.py

调用 DeepSeek（OpenAI 兼容接口）为 arXiv 论文批量生成：
1. 中文标题 title_zh
2. 一句话中文总结 summary
3. 中文摘要 abstract_zh
4. 3-5 句 AI 总结 ai_summary

特点：
- DeepSeek API 失败自动重试
- HTTP 错误打印完整响应
- JSON 解析失败自动重试
- 支持清理 ```json ... ``` 包裹
- 批量失败后自动拆成单篇继续尝试
- 严格校验返回的论文 ID
"""

import json
import os
import re
import time
from typing import Dict, List, Optional

import requests


DEFAULT_BASE_URL = "https://api.deepseek.com"

SYSTEM_PROMPT = """
你是一名精通微分几何、一般拓扑学、几何拓扑学、偏微分方程、
泛函分析、复变函数与度量几何的科研助理。

用户会给你一批 arXiv 论文，每篇包含：
- arXiv 编号
- 英文标题
- 英文摘要

请严格为每篇论文生成以下四个字段：

1. title_zh
将英文论文标题准确翻译成中文。
要求：
- 数学专业术语准确
- 不要遗漏公式、专有名词和限定词

2. summary
一句话中文总结，不超过 60 个汉字。
说明：
- 论文研究什么问题
- 核心贡献是什么

3. abstract_zh
将英文摘要忠实翻译成中文。
要求：
- 尽量完整
- 不超过 250 个汉字
- 保留数学专业术语
- 不要自行添加论文没有表达的结论

4. ai_summary
3-5 句中文总结，总长度约 60-150 个汉字。
分别概括：
- 研究问题
- 使用的方法
- 主要结果
- 理论意义或应用意义

每句话之间使用换行符分隔。

非常重要：
- 必须处理输入中的每一篇论文
- id 必须原样返回，不能修改
- 不要遗漏论文
- 不要输出 Markdown
- 不要输出解释
- 不要输出 ```json
- 只返回一个合法 JSON 对象

JSON 格式必须严格为：

{
  "papers": [
    {
      "id": "论文编号",
      "title_zh": "中文标题",
      "summary": "一句话总结",
      "abstract_zh": "中文摘要",
      "ai_summary": "第一句\n第二句\n第三句"
    }
  ]
}
""".strip()


USER_TEMPLATE = """
以下是本批 {count} 篇 arXiv 论文。

请逐篇处理，必须保证每一个输入 id 都出现在输出 JSON 的 papers 数组中。

论文列表：

{body}
""".strip()


class DeepSeekError(RuntimeError):
    """DeepSeek API 或返回内容异常。"""


def _extract_json(content: str) -> dict:
    """
    从 DeepSeek 返回内容中提取 JSON。

    正常情况下 content 本身就是 JSON。
    如果模型错误地返回：
        ```json
        {...}
        ```
    也可以自动清理。
    """
    if not isinstance(content, str):
        raise DeepSeekError(
            f"DeepSeek 返回 content 不是字符串，而是 {type(content).__name__}"
        )

    content = content.strip()

    if not content:
        raise DeepSeekError("DeepSeek 返回了空 content")

    # 直接解析
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 清理 Markdown JSON 代码块
    fenced = re.search(
        r"```(?:json)?\s*(.*?)\s*```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:
        cleaned = fenced.group(1).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # 尝试寻找第一个 { 到最后一个 }
    start = content.find("{")
    end = content.rfind("}")

    if start >= 0 and end > start:
        candidate = content[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    preview = content[:1000]
    raise DeepSeekError(
        "无法解析 DeepSeek 返回的 JSON。\n"
        f"实际返回内容前 1000 字符：\n{preview}"
    )


def _call_chat(
    api_key: str,
    base_url: str,
    model: str,
    user_content: str,
    timeout: int = 180,
) -> dict:
    """
    调用一次 DeepSeek chat/completions。

    失败时抛出 DeepSeekError，并尽可能保留服务器返回信息。
    """
    if not api_key:
        raise DeepSeekError("DEEPSEEK_API_KEY 为空")

    url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "temperature": 0.2,
        "response_format": {
            "type": "json_object"
        },
        # 批量 20 篇时输出可能比较长，避免被默认输出长度截断。
        "max_tokens": 12000,
    }

    print(f"      [DeepSeek] POST {url}")
    print(f"      [DeepSeek] model={model}")

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise DeepSeekError(
            f"网络请求失败：{type(exc).__name__}: {exc}"
        ) from exc

    # HTTP 非 2xx
    if not response.ok:
        try:
            error_body = response.json()
            error_text = json.dumps(
                error_body,
                ensure_ascii=False,
            )
        except ValueError:
            error_text = response.text[:3000]

        raise DeepSeekError(
            f"DeepSeek HTTP {response.status_code}\n"
            f"响应内容：{error_text}"
        )

    # JSON response
    try:
        data = response.json()
    except ValueError as exc:
        raise DeepSeekError(
            "DeepSeek HTTP 请求成功，但响应不是合法 JSON。\n"
            f"响应前 3000 字符：{response.text[:3000]}"
        ) from exc

    # 检查 OpenAI-compatible response
    try:
        choices = data["choices"]
        if not choices:
            raise KeyError("choices 为空")

        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise DeepSeekError(
            "DeepSeek 返回结构异常。\n"
            f"响应内容：{json.dumps(data, ensure_ascii=False)[:5000]}"
        ) from exc

    # 检查是否因为长度达到上限而截断
    finish_reason = choices[0].get("finish_reason")

    if finish_reason == "length":
        raise DeepSeekError(
            "DeepSeek 输出被 max_tokens 截断，"
            "当前批次太大，应该缩小 batch_size。"
        )

    return _extract_json(content)


def _validate_parsed(
    parsed: dict,
    expected_ids: List[str],
) -> Dict[str, dict]:
    """
    验证 DeepSeek 返回的数据。

    要求：
    - 必须有 papers
    - 每个返回项必须有合法 id
    - id 必须属于本批输入
    - title_zh 必须存在
    - 至少有 summary / abstract_zh / ai_summary
    """
    if not isinstance(parsed, dict):
        raise DeepSeekError("DeepSeek JSON 顶层不是对象")

    papers_data = parsed.get("papers")

    if not isinstance(papers_data, list):
        raise DeepSeekError(
            "DeepSeek 返回 JSON 中没有合法的 papers 数组。\n"
            f"实际 JSON：{json.dumps(parsed, ensure_ascii=False)[:5000]}"
        )

    expected_set = set(expected_ids)
    results = {}

    for item in papers_data:
        if not isinstance(item, dict):
            continue

        pid = str(item.get("id", "")).strip()

        if not pid:
            continue

        if pid not in expected_set:
            print(
                f"      [警告] DeepSeek 返回了本批不存在的论文 ID：{pid}"
            )
            continue

        title_zh = str(item.get("title_zh", "")).strip()
        summary = str(item.get("summary", "")).strip()
        abstract_zh = str(item.get("abstract_zh", "")).strip()
        ai_summary = str(item.get("ai_summary", "")).strip()

        if not title_zh:
            print(f"      [警告] {pid} 缺少 title_zh，跳过")
            continue

        # AI 总结是本项目网页的核心字段。
        if not ai_summary:
            print(f"      [警告] {pid} 缺少 ai_summary，跳过")
            continue

        results[pid] = {
            "title_zh": title_zh,
            "summary": summary,
            "abstract_zh": abstract_zh,
            "ai_summary": ai_summary,
        }

    return results


def _make_user_content(batch: List[dict]) -> str:
    """
    构造本批输入。
    """
    lines = []

    for p in batch:
        pid = str(p["id"]).strip()
        title = str(p.get("title", "")).strip()

        # 摘要截断到 2500 字符，控制 token 成本。
        abstract = str(p.get("summary", ""))[:2500].strip()

        lines.append(
            f"[{pid}]\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}"
        )

    return USER_TEMPLATE.format(
        count=len(batch),
        body="\n\n".join(lines),
    )


def _call_with_retry(
    api_key: str,
    base_url: str,
    model: str,
    user_content: str,
    expected_ids: List[str],
    batch_label: str,
    max_attempts: int = 3,
) -> Dict[str, dict]:
    """
    调用 DeepSeek，并进行多次重试。
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            print(
                f"      [{batch_label}] "
                f"DeepSeek 第 {attempt}/{max_attempts} 次请求..."
            )

            parsed = _call_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                user_content=user_content,
            )

            results = _validate_parsed(
                parsed,
                expected_ids,
            )

            if not results:
                raise DeepSeekError(
                    "DeepSeek 请求成功，但没有返回任何有效论文总结。"
                )

            missing = set(expected_ids) - set(results.keys())

            if missing:
                print(
                    f"      [{batch_label}] "
                    f"成功 {len(results)}/{len(expected_ids)} 篇，"
                    f"缺失 {len(missing)} 篇。"
                )
            else:
                print(
                    f"      [{batch_label}] "
                    f"成功 {len(results)}/{len(expected_ids)} 篇。"
                )

            return results

        except Exception as exc:
            last_error = exc

            print(
                f"      [错误] {batch_label} 第 {attempt} 次失败："
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < max_attempts:
                # 指数退避：5s → 10s
                sleep_seconds = 5 * attempt
                print(
                    f"      [重试] {sleep_seconds} 秒后重试..."
                )
                time.sleep(sleep_seconds)

    raise DeepSeekError(
        f"{batch_label} 连续 {max_attempts} 次失败。"
        f"最后错误：{last_error}"
    )


def summarize_papers(
    papers,
    api_key,
    base_url=None,
    model="deepseek-chat",
    batch_size=20,
):
    """
    对论文列表批量生成中文总结。

    返回：
        {
            "论文id": {
                "title_zh": "...",
                "summary": "...",
                "abstract_zh": "...",
                "ai_summary": "..."
            }
        }

    失败策略：
    1. 先按 batch_size 批量请求
    2. 某批失败后，将该批拆成单篇请求
    3. 单篇仍失败，则记录失败，但继续处理其他论文

    这样即使某一批因为 token、JSON 或特殊论文导致失败，
    也不会让整个 253 篇全部失败。
    """
    base_url = base_url or DEFAULT_BASE_URL

    if not papers:
        return {}

    if batch_size < 1:
        raise ValueError("batch_size 必须 >= 1")

    results: Dict[str, dict] = {}

    total = len(papers)
    total_batches = (total + batch_size - 1) // batch_size

    print(
        f"      [DeepSeek] 共 {total} 篇论文，"
        f"批大小={batch_size}，预计 {total_batches} 批。"
    )

    failed_batches = []

    # ============================================================
    # 第一阶段：批量请求
    # ============================================================
    for batch_index, start in enumerate(
        range(0, total, batch_size),
        start=1,
    ):
        batch = papers[start : start + batch_size]

        expected_ids = [
            str(p["id"]).strip()
            for p in batch
        ]

        print(
            f"\n      ===== 批次 {batch_index}/{total_batches} "
            f"（{len(batch)} 篇）====="
        )

        user_content = _make_user_content(batch)

        try:
            batch_results = _call_with_retry(
                api_key=api_key,
                base_url=base_url,
                model=model,
                user_content=user_content,
                expected_ids=expected_ids,
                batch_label=f"批次 {batch_index}/{total_batches}",
                max_attempts=3,
            )

            results.update(batch_results)

            # 如果批量请求没有返回全部论文，
            # 后面再对缺失论文进行单篇补救。
            missing = [
                p for p in batch
                if str(p["id"]).strip() not in batch_results
            ]

            if missing:
                print(
                    f"      [补救] 本批有 {len(missing)} 篇未成功，"
                    f"稍后进行单篇补救。"
                )
                failed_batches.extend(missing)

        except Exception as exc:
            print(
                f"      [批量失败] 批次 {batch_index}/{total_batches}："
                f"{type(exc).__name__}: {exc}"
            )

            # 整批失败，加入单篇补救队列
            failed_batches.extend(batch)

    # ============================================================
    # 第二阶段：对失败/缺失论文逐篇补救
    # ============================================================
    if failed_batches:
        # 去重
        unique_failed = {}
        for p in failed_batches:
            unique_failed[str(p["id"]).strip()] = p

        failed_batches = list(unique_failed.values())

        print(
            f"\n      [补救阶段] "
            f"共有 {len(failed_batches)} 篇需要单篇重试。"
        )

        for index, paper in enumerate(
            failed_batches,
            start=1,
        ):
            pid = str(paper["id"]).strip()

            # 如果前面已经成功，就不再重复请求
            if pid in results:
                continue

            print(
                f"\n      ----- 单篇补救 "
                f"{index}/{len(failed_batches)}：{pid} -----"
            )

            user_content = _make_user_content([paper])

            try:
                single_result = _call_with_retry(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    user_content=user_content,
                    expected_ids=[pid],
                    batch_label=f"单篇 {pid}",
                    max_attempts=2,
                )

                results.update(single_result)

            except Exception as exc:
                print(
                    f"      [最终失败] {pid}："
                    f"{type(exc).__name__}: {exc}"
                )

    # ============================================================
    # 最终统计
    # ============================================================
    success_count = len(results)
    failed_count = total - success_count

    print("\n      ===============================")
    print(
        f"      DeepSeek 总结结果："
        f"{success_count}/{total} 成功"
    )
    print(
        f"      失败：{failed_count} 篇"
    )
    print(
        f"      成功率："
        f"{success_count / total * 100:.1f}%"
    )
    print("      ===============================")

    if failed_count:
        failed_ids = [
            str(p["id"]).strip()
            for p in papers
            if str(p["id"]).strip() not in results
        ]

        print(
            "      [警告] 以下论文最终没有获得 AI 总结："
        )

        # 最多打印 50 个，避免 Actions 日志过长
        for pid in failed_ids[:50]:
            print(f"        - {pid}")

        if len(failed_ids) > 50:
            print(
                f"        ... 还有 {len(failed_ids) - 50} 篇"
            )

    return results


if __name__ == "__main__":
    # 本地自测：
    #
    # Windows:
    #   set DEEPSEEK_API_KEY=你的Key
    #   python summarizer.py
    #
    # Linux/macOS:
    #   export DEEPSEEK_API_KEY=你的Key
    #   python summarizer.py

    if os.name == "nt":
        pass

    key = os.environ.get("DEEPSEEK_API_KEY")

    if not key:
        print(
            "请先设置环境变量 DEEPSEEK_API_KEY "
            "再运行自测。"
        )
        raise SystemExit(1)

    try:
        from arxiv_fetcher import fetch_new_papers

        demo = fetch_new_papers(
            ["math.DG"],
            hours_back=24,
            max_results=3,
        )

        print(f"测试论文数：{len(demo)}")

        if not demo:
            print("没有抓到测试论文。")
            raise SystemExit(0)

        out = summarize_papers(
            demo,
            api_key=key,
            batch_size=3,
        )

        print("\n========== 测试结果 ==========")

        for pid, info in out.items():
            print(f"\n[{pid}]")
            print(f"中文标题：{info['title_zh']}")
            print(f"一句话总结：{info['summary']}")
            print(f"AI 总结：\n{info['ai_summary']}")
            print(f"中文摘要：{info['abstract_zh']}")

    except Exception as exc:
        print(
            f"\n[测试失败] {type(exc).__name__}: {exc}"
        )
        raise SystemExit(1)