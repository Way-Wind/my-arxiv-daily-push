# -*- coding: utf-8 -*-
"""
main.py —— arXiv 每日论文速览

流程：
    arXiv 抓取
        ↓
    去重
        ↓
    DeepSeek 中文总结
        ↓
    生成 GitHub Pages
        ↓
    微信模板消息

用法：
    本地预览：
        python main.py --dry-run

    本地测试少量论文：
        python main.py --dry-run --max-papers 3

    正式运行：
        python main.py

配置优先级：
    环境变量 > config.json
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

from arxiv_fetcher import fetch_new_papers
from summarizer import summarize_papers
from wechat_push import WeChatPusher


# ============================================================
# 默认配置
# ============================================================

DEFAULT_TEMPLATE_FIELDS = {
    "first": "first",
    "keyword1": "keyword1",
    "keyword2": "keyword2",
    "keyword3": "keyword3",
    "remark": "remark",
}

DEFAULT_MESSAGE_MODE = "digest"

DETAILED_PAPERS_PER_MESSAGE = 1

CATEGORY_NAMES = {
    "math.AP": "偏微分方程",
    "math.AT": "代数拓扑",
    "math.CV": "复变函数",
    "math.DG": "微分几何",
    "math.FA": "泛函分析",
    "math.GN": "一般拓扑",
    "math.GT": "几何拓扑",
    "math.HO": "数学史与综述",
    "math.MG": "度量几何",
    "math.SG": "辛几何",
}

DEFAULT_CATEGORIES = [
    "math.AP",
    "math.DG",
    "math.FA",
    "math.GT",
    "math.CV",
    "math.MG",
]

STATE_FILE = "last_pushed.json"

# 北京时间 UTC+8
BEIJING_TZ = timezone(timedelta(hours=8))


# ============================================================
# 基础工具
# ============================================================

def _load_pushed_ids():
    """
    读取已经推送过的论文 ID。
    """
    try:
        with open(
            STATE_FILE,
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return set(
            str(x)
            for x in data.get("ids", [])
        )

    except (OSError, ValueError, TypeError):
        return set()


def _save_pushed_ids(ids):
    """
    保存已经推送过的论文 ID。
    """
    with open(
        STATE_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "ids": sorted(ids)
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _utf8():
    """
    确保终端输出 UTF-8。
    """
    if (
        sys.stdout.encoding
        and sys.stdout.encoding.lower()
        not in ("utf-8", "utf8")
    ):
        try:
            sys.stdout.reconfigure(
                encoding="utf-8"
            )
        except AttributeError:
            pass


def _today_str():
    """
    返回北京时间日期。
    """
    return datetime.now(
        BEIJING_TZ
    ).strftime("%Y-%m-%d")


def _category_label(primary: str) -> str:
    return CATEGORY_NAMES.get(
        primary,
        primary,
    )


def _assign_section(p, categories):
    """
    确定论文所属目标分区。
    """
    if p["primary"] in categories:
        return p["primary"]

    for c in p["categories"]:
        if c in categories:
            return c

    return p["primary"]


# ============================================================
# 配置读取
# ============================================================

def load_config():
    """
    配置优先级：

        环境变量
            ↓
        config.json
    """

    cfg = {}

    # --------------------------------------------------------
    # config.json
    # --------------------------------------------------------

    if os.path.exists("config.json"):
        try:
            with open(
                "config.json",
                encoding="utf-8",
            ) as f:
                cfg = json.load(f)
        except (OSError, ValueError) as exc:
            print(
                f"[警告] config.json 读取失败：{exc}"
            )

    # --------------------------------------------------------
    # arXiv
    # --------------------------------------------------------

    categories_env = os.environ.get(
        "ARXIV_CATEGORIES"
    )

    arxiv_cfg = cfg.get(
        "arxiv",
        {},
    )

    categories = (
        [
            c.strip()
            for c in categories_env.split(",")
            if c.strip()
        ]
        if categories_env
        else arxiv_cfg.get(
            "categories",
            DEFAULT_CATEGORIES,
        )
    )

    hours_back = int(
        os.environ.get(
            "ARXIV_HOURS_BACK"
        )
        or arxiv_cfg.get(
            "hours_back",
            30,
        )
    )

    max_papers_per_run = int(
        os.environ.get(
            "ARXIV_MAX_PAPERS"
        )
        or arxiv_cfg.get(
            "max_papers_per_run",
            200,
        )
    )

    # --------------------------------------------------------
    # DeepSeek
    # --------------------------------------------------------

    ds_cfg = cfg.get(
        "deepseek",
        {},
    )

    deepseek = {
        "api_key": (
            os.environ.get(
                "DEEPSEEK_API_KEY"
            )
            or ds_cfg.get(
                "api_key",
                "",
            )
        ),
        "base_url": (
            os.environ.get(
                "DEEPSEEK_BASE_URL"
            )
            or ds_cfg.get(
                "base_url",
                "https://api.deepseek.com",
            )
        ),
        "model": (
            os.environ.get(
                "DEEPSEEK_MODEL"
            )
            or ds_cfg.get(
                "model",
                "deepseek-chat",
            )
        ),
    }

    # --------------------------------------------------------
    # 微信
    # --------------------------------------------------------

    wx_cfg = cfg.get(
        "wechat",
        {},
    )

    fields = dict(
        DEFAULT_TEMPLATE_FIELDS
    )

    fields.update(
        wx_cfg.get(
            "template_fields",
            {},
        )
    )

    mode = (
        os.environ.get(
            "WECHAT_MODE"
        )
        or wx_cfg.get(
            "mode"
        )
        or DEFAULT_MESSAGE_MODE
    ).strip().lower()

    if mode not in (
        "digest",
        "detailed",
    ):
        mode = DEFAULT_MESSAGE_MODE

    wechat = {
        "app_id": (
            os.environ.get(
                "WECHAT_APP_ID"
            )
            or wx_cfg.get(
                "app_id",
                "",
            )
        ),
        "app_secret": (
            os.environ.get(
                "WECHAT_APP_SECRET"
            )
            or wx_cfg.get(
                "app_secret",
                "",
            )
        ),
        "template_id": (
            os.environ.get(
                "WECHAT_TEMPLATE_ID"
            )
            or wx_cfg.get(
                "template_id",
                "",
            )
        ),
        "user_openid": (
            os.environ.get(
                "WECHAT_OPENID"
            )
            or wx_cfg.get(
                "user_openid",
                "",
            )
        ),
        "template_fields": fields,
        "mode": mode,
    }

    return {
        "arxiv": {
            "categories": categories,
            "hours_back": hours_back,
            "max_papers_per_run": max_papers_per_run,
        },
        "deepseek": deepseek,
        "wechat": wechat,
    }


# ============================================================
# 微信消息构建
# ============================================================

def build_messages(
    papers,
    summaries,
    template_fields,
    categories,
    date_str,
    mode=DEFAULT_MESSAGE_MODE,
    page_url=None,
):
    """
    组织微信模板消息。

    digest：
        一条汇总消息，点击进入 GitHub Pages。

    detailed：
        每篇论文一条消息。
    """

    category_text = "、".join(
        _category_label(c)
        for c in categories
    )

    keyword3_text = date_str

    f = template_fields["first"]
    k1 = template_fields["keyword1"]
    k2 = template_fields["keyword2"]
    k3 = template_fields["keyword3"]
    r = template_fields["remark"]

    def _base_data(
        remark_text,
        first_text,
        keyword1_text,
        keyword2_text,
    ):
        return {
            f: {
                "value": first_text
            },
            k1: {
                "value": keyword1_text
            },
            k2: {
                "value": keyword2_text
            },
            k3: {
                "value": keyword3_text
            },
            r: {
                "value": remark_text
            },
        }

    def _cut(text, limit):
        text = str(text or "")
        return (
            text[:limit]
            + ("…" if len(text) > limit else "")
        )

    messages = []

    # ========================================================
    # detailed 模式
    # ========================================================

    if mode == "detailed":

        first_text = "📚 arXiv 每日论文速览"
        keyword1_text = category_text
        keyword2_text = (
            f"今日新增 {len(papers)} 篇"
        )

        for p in papers:

            info = summaries.get(
                p["id"],
                {},
            )

            title_zh = (
                info.get("title_zh")
                or p["title"]
            )

            one_line = (
                info.get("summary")
                or "（未生成总结）"
            )

            abstract_en = _cut(
                p["summary"],
                220,
            )

            abstract_zh = (
                info.get("abstract_zh")
                or "（未生成翻译）"
            )

            ai_summary = (
                info.get("ai_summary")
                or "（未生成 AI 总结）"
            )

            label = _category_label(
                p["primary"]
            )

            remark_text = (
                f"[{p['id']}] "
                f"{title_zh}（{label}）\n\n"
                f"💡 {one_line}\n\n"
                f"🤖 AI 总结\n"
                f"{ai_summary}\n\n"
                f"\n"
                f"{abstract_en}\n\n"
                f"\n"
                f"{abstract_zh}"
            )

            if len(remark_text) > 600:
                remark_text = _cut(
                    remark_text,
                    590,
                )

            messages.append(
                {
                    "data": _base_data(
                        remark_text,
                        first_text,
                        keyword1_text,
                        keyword2_text,
                    ),
                    "url": p["url"],
                }
            )

        return messages

    # ========================================================
    # digest 模式
    # ========================================================

    sections = {
        c: []
        for c in categories
    }

    for p in papers:
        sec = _assign_section(
            p,
            categories,
        )

        sections.setdefault(
            sec,
            []
        ).append(p)

    lines = []

    for sec in categories:
        count = len(
            sections.get(
                sec,
                [],
            )
        )

        if count:
            lines.append(
                f"{_category_label(sec)}："
                f"{count} 篇"
            )

    lines.append("")

    lines.append(
        "📌 点击消息查看全部论文的 "
        "AI 总结、中英文摘要与 arXiv 链接"
    )

    remark_text = "\n".join(lines)

    if len(remark_text) > 600:
        remark_text = _cut(
            remark_text,
            590,
        )

    first_text = "📚 arXiv 每日论文速览"
    keyword1_text = (
        f"共 {len(papers)} 篇"
    )
    keyword2_text = date_str
    keyword3_text = category_text

    messages.append(
        {
            "data": _base_data(
                remark_text,
                first_text,
                keyword1_text,
                keyword2_text,
            ),
            "url": page_url,
        }
    )

    return messages


# ============================================================
# DeepSeek 总结结果检查
# ============================================================

def _validate_summary_results(
    papers,
    summaries,
):
    """
    检查总结结果是否足够生成网页。

    核心原则：
        不允许 0 篇总结时继续推送。

    对于少量失败论文：
        允许继续，因为 summarizer 会尽量抢救。

    但是如果成功率低于 50%，
    认为 DeepSeek 服务存在明显异常，
    直接终止。
    """

    total = len(papers)
    success = len(summaries)

    if total == 0:
        return

    success_rate = success / total

    print(
        f"      [检查] AI 总结成功率："
        f"{success}/{total} "
        f"({success_rate * 100:.1f}%)"
    )

    # 一个都没有成功：绝对不能继续
    if success == 0:
        raise RuntimeError(
            "DeepSeek 未成功生成任何论文总结。"
            "为了避免发送空白 AI 页面，"
            "程序已停止，不会生成页面，也不会发送微信。"
        )

    # 成功率过低：大概率 API / 配置存在异常
    if success_rate < 0.5:
        raise RuntimeError(
            f"DeepSeek 总结成功率只有 "
            f"{success_rate * 100:.1f}%，"
            f"低于 50%。"
            f"为了避免生成大量缺失 AI 总结的网页，"
            f"程序已停止。"
        )

    # 有少量失败时允许继续
    if success < total:
        print(
            f"      [警告] "
            f"{total - success} 篇论文没有获得 AI 总结。"
        )
        print(
            "      [提示] "
            "这些论文将在网页中显示为未生成总结。"
        )


# ============================================================
# 主程序
# ============================================================

def main():

    _utf8()

    parser = argparse.ArgumentParser(
        description=(
            "arXiv 每日论文速览："
            "抓取 + DeepSeek 总结 + 微信推送"
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只抓取和总结并打印结果，不发送微信",
    )

    parser.add_argument(
        "--max-papers",
        type=int,
        default=0,
        help=(
            "最多处理多少篇论文；"
            "0 表示不限制"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # 配置
    # ========================================================

    cfg = load_config()

    arxiv_cfg = cfg["arxiv"]
    ds_cfg = cfg["deepseek"]
    wx_cfg = cfg["wechat"]

    # ========================================================
    # 检查 DeepSeek
    # ========================================================

    if not ds_cfg["api_key"]:

        print(
            "[错误] 缺少 DeepSeek API Key。\n"
            "请设置：DEEPSEEK_API_KEY"
        )

        sys.exit(1)

    print(
        "[配置] DeepSeek Base URL："
        f"{ds_cfg['base_url']}"
    )

    print(
        "[配置] DeepSeek Model："
        f"{ds_cfg['model']}"
    )

    # ========================================================
    # 检查微信
    # ========================================================

    if not args.dry_run:

        missing_wechat = []

        if not wx_cfg["app_id"]:
            missing_wechat.append(
                "WECHAT_APP_ID"
            )

        if not wx_cfg["app_secret"]:
            missing_wechat.append(
                "WECHAT_APP_SECRET"
            )

        if not wx_cfg["template_id"]:
            missing_wechat.append(
                "WECHAT_TEMPLATE_ID"
            )

        if not wx_cfg["user_openid"]:
            missing_wechat.append(
                "WECHAT_OPENID"
            )

        if missing_wechat:

            print(
                "[错误] 微信配置不完整，缺少："
                + ", ".join(missing_wechat)
            )

            sys.exit(1)

    # ========================================================
    # 1. 抓取 arXiv
    # ========================================================

    print(
        f"\n[1/3] 正在从 arXiv 抓取 "
        f"{', '.join(arxiv_cfg['categories'])} "
        f"最近 {arxiv_cfg['hours_back']} 小时的新论文 ..."
    )

    papers = fetch_new_papers(
        arxiv_cfg["categories"],
        hours_back=arxiv_cfg["hours_back"],
    )

    if args.max_papers > 0:
        papers = papers[
            :args.max_papers
        ]

    # 如果当前窗口无结果，扩大到 72 小时
    if not papers:

        print(
            "      [提示] 当前窗口无结果。"
            "可能是 arXiv 索引延迟，"
            "扩大到 72 小时重新查询..."
        )

        papers = fetch_new_papers(
            arxiv_cfg["categories"],
            hours_back=72,
        )

        if args.max_papers > 0:
            papers = papers[
                :args.max_papers
            ]

    print(
        f"      共抓到 {len(papers)} 篇"
    )

    if not papers:

        print(
            "[完成] 今天没有新论文，不推送。"
        )

        return

    # ========================================================
    # 2. 去重
    # ========================================================

    pushed_ids = _load_pushed_ids()

    new_papers = [
        p
        for p in papers
        if p["id"] not in pushed_ids
    ]

    if not new_papers:

        print(
            f"[完成] 没有新论文。"
            f"窗口内 {len(papers)} 篇均已推送过，"
            f"不重复推送。"
        )

        return

    if len(new_papers) < len(papers):

        print(
            f"      [去重] 已推送过 "
            f"{len(papers) - len(new_papers)} 篇，"
            f"本次实际新增 "
            f"{len(new_papers)} 篇"
        )

    papers = new_papers

    # ========================================================
    # 3. DeepSeek 总结
    # ========================================================

    print(
        f"\n[2/3] 调用 DeepSeek "
        f"（{ds_cfg['model']}）"
        f"生成中文总结..."
    )

    try:

        summaries = summarize_papers(
            papers,
            api_key=ds_cfg["api_key"],
            base_url=ds_cfg["base_url"],
            model=ds_cfg["model"],
            batch_size=20,
        )

    except Exception as exc:

        print(
            "\n================================================"
        )
        print(
            "[致命错误] DeepSeek 总结阶段失败"
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print(
            "================================================"
        )

        # 不生成页面，不发送微信
        sys.exit(1)

    # ========================================================
    # 检查总结质量
    # ========================================================

    try:

        _validate_summary_results(
            papers,
            summaries,
        )

    except RuntimeError as exc:

        print(
            "\n================================================"
        )
        print(
            "[致命错误] AI 总结结果不合格"
        )
        print(
            str(exc)
        )
        print(
            "为了避免发送错误页面，程序已停止。"
        )
        print(
            "================================================"
        )

        sys.exit(1)

    # ========================================================
    # 北京时间日期
    # ========================================================

    date_str = _today_str()

    print(
        f"      [日期] 北京时间：{date_str}"
    )

    # ========================================================
    # 微信模式
    # ========================================================

    mode = wx_cfg.get(
        "mode",
        DEFAULT_MESSAGE_MODE,
    )

    # digest 模式：
    # 网页显示全部论文
    #
    # detailed 模式：
    # 微信只推送 max_papers_per_run 篇

    all_papers = papers

    if (
        mode == "detailed"
        and len(papers)
        > arxiv_cfg["max_papers_per_run"]
    ):

        print(
            f"      [提示] 论文较多，"
            f"详细模式本次仅推送最新的 "
            f"{arxiv_cfg['max_papers_per_run']} 篇"
        )

        papers = papers[
            :arxiv_cfg["max_papers_per_run"]
        ]

    # ========================================================
    # GitHub Pages
    # ========================================================

    page_base_url = os.environ.get(
        "PAGE_BASE_URL",
        "",
    ).strip()

    page_url = None

    if page_base_url:

        try:

            from page_builder import (
                write_daily_page
            )

            daily_path, index_path = (
                write_daily_page(
                    all_papers,
                    summaries,
                    arxiv_cfg["categories"],
                    date_str,
                    output_dir="pages",
                )
            )

            page_url = (
                page_base_url.rstrip("/")
                + "/"
                + os.path.basename(
                    daily_path
                )
            )

            print(
                f"      [Pages] 已生成："
                f"{daily_path}"
            )

            print(
                f"      [Pages] 首页："
                f"{index_path}"
            )

            print(
                f"      [Pages] URL："
                f"{page_url}"
            )

        except Exception as exc:

            # 页面是微信跳转的核心。
            # 页面生成失败时不应该继续发送一个没有链接的 digest。
            print(
                "\n================================================"
            )

            print(
                "[致命错误] GitHub Pages 页面生成失败"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "================================================"
            )

            sys.exit(1)

    else:

        print(
            "      [警告] 未设置 PAGE_BASE_URL。"
        )

        if mode == "digest":

            print(
                "      digest 模式将无法提供 GitHub Pages 跳转链接。"
            )

    # ========================================================
    # 组织微信消息
    # ========================================================

    messages = build_messages(
        papers,
        summaries,
        wx_cfg["template_fields"],
        arxiv_cfg["categories"],
        date_str,
        mode=mode,
        page_url=page_url,
    )

    if mode == "digest":

        print(
            f"\n[3/3] 模式=速览汇总"
        )

        print(
            f"      共组织 {len(messages)} 条微信消息"
        )

        if page_url:

            print(
                f"      点击消息将跳转："
                f"{page_url}"
            )

    else:

        print(
            f"\n[3/3] 模式=详细"
        )

        print(
            f"      共组织 {len(messages)} 条微信消息"
        )

    # ========================================================
    # dry-run
    # ========================================================

    if args.dry_run:

        print(
            "\n=============================================="
        )

        print(
            "              DRY RUN 预览"
        )

        print(
            "=============================================="
        )

        for idx, msg in enumerate(
            messages,
            1,
        ):

            jump = ""

            if msg.get("url"):
                jump = (
                    f"\n点击跳转："
                    f"{msg['url']}"
                )

            print(
                f"\n----- 消息 {idx} -----"
            )

            for k, v in msg["data"].items():

                print(
                    f"{k}: "
                    f"{v['value']}"
                )

            if jump:
                print(jump)

        print(
            "\n[dry-run] "
            "仅预览，没有发送微信。"
        )

        return

    # ========================================================
    # 微信推送
    # ========================================================

    try:

        pusher = WeChatPusher(
            wx_cfg["app_id"],
            wx_cfg["app_secret"],
            wx_cfg["template_id"],
            wx_cfg["user_openid"],
        )

        sent = pusher.send_batch(
            messages
        )

    except Exception as exc:

        print(
            "\n================================================"
        )

        print(
            "[致命错误] 微信推送失败"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "================================================"
        )

        # 注意：
        # 微信发送失败时不要记录 last_pushed.json，
        # 否则下一次运行会被错误地认为已经推送。
        sys.exit(1)

    print(
        f"\n[完成] 已发送 {sent} 条消息到微信。"
    )

    # ========================================================
    # 记录已经推送的论文
    # ========================================================

    # detailed 模式如果只推送了一部分：
    # 这里只记录实际发送的 papers。
    #
    # digest 模式：
    # papers 是全部论文，所以全部记录。
    newly_pushed_ids = {
        p["id"]
        for p in papers
    }

    updated_ids = (
        pushed_ids
        | newly_pushed_ids
    )

    try:

        _save_pushed_ids(
            updated_ids
        )

    except OSError as exc:

        print(
            f"[警告] last_pushed.json 保存失败："
            f"{exc}"
        )

    print(
        f"      [去重] 本次记录 "
        f"{len(newly_pushed_ids)} 篇"
    )

    print(
        f"      [去重] 累计 "
        f"{len(updated_ids)} 篇"
    )


if __name__ == "__main__":
    main()