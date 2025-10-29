#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户代理工具模块
提供动态随机用户代理生成功能
"""

import random
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

from fake_useragent import UserAgent

# 全局 UserAgent 实例（单例模式）
_user_agent_instance: Optional[UserAgent] = None

# 语言与编码候选，避免每次完全相同
_ACCEPT_LANGUAGE_CANDIDATES = [
    "zh-CN,zh;q=0.9,en;q=0.8",
    "zh-CN,zh;q=0.8,en-US;q=0.6,en;q=0.5",
    "en-US,en;q=0.9,zh-CN;q=0.7",
]
_ACCEPT_ENCODING_CANDIDATES = [
    "gzip, deflate, br",
    "gzip, deflate",
]


def get_user_agent_instance() -> UserAgent:
    """获取或创建 UserAgent 实例（单例模式）"""
    global _user_agent_instance
    if _user_agent_instance is None:
        try:
            _user_agent_instance = UserAgent()
        except Exception:
            # 在无法连接到服务器时提供一个备用方案
            _user_agent_instance = None
    return _user_agent_instance


def get_random_user_agent(browser_type: Optional[str] = None) -> str:
    """
    获取随机用户代理字符串

    Args:
        browser_type: 指定浏览器类型 ('chrome', 'firefox', 'safari', 'edge')
                     如果为 None，则随机选择

    Returns:
        str: 用户代理字符串
    """
    ua = get_user_agent_instance()

    if ua is None:
        # 如果 UserAgent 初始化失败，返回一个硬编码的默认值
        return "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

    # 如果没有指定浏览器类型，随机选择一个（偏向 Chrome 和 Edge）
    if browser_type is None:
        browser_choices = ["chrome", "chrome", "chrome", "edge", "edge", "firefox", "safari"]
        browser_type = random.choice(browser_choices)

    # 根据浏览器类型获取用户代理
    try:
        if browser_type == "chrome":
            user_agent = ua.chrome
        elif browser_type == "edge":
            user_agent = ua.edge
        elif browser_type == "firefox":
            user_agent = ua.firefox
        elif browser_type == "safari":
            user_agent = ua.safari
        else:
            user_agent = ua.random
    except Exception:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

    return user_agent


def _detect_browser_type(user_agent: str) -> str:
    ua_lower = user_agent.lower()
    if "edg/" in ua_lower:
        return "edge"
    if "chrome/" in ua_lower:
        return "chrome"
    if "firefox/" in ua_lower:
        return "firefox"
    if "safari" in ua_lower:
        return "safari"
    return "chrome"


def _detect_platform(user_agent: str) -> str:
    if "Windows" in user_agent:
        return '"Windows"'
    if "Mac OS X" in user_agent or "Macintosh" in user_agent:
        return '"macOS"'
    if "Android" in user_agent:
        return '"Android"'
    if "iPhone" in user_agent or "iPad" in user_agent:
        return '"iOS"'
    if "Linux" in user_agent:
        return '"Linux"'
    return '"Windows"'


def _infer_fetch_site(origin: Optional[str], referer: Optional[str], target_url: Optional[str]) -> str:
    def _normalize_host(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        parsed = urlparse(value) if "://" in value else urlparse(f"https://{value}")
        return (parsed.netloc or parsed.path or "").lower() or None

    def _site(host: Optional[str]) -> Optional[str]:
        if not host:
            return None
        parts = host.split(".")
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return host

    origin_host = _normalize_host(origin)
    referer_host = _normalize_host(referer)
    target_host = _normalize_host(target_url)

    if target_host and origin_host:
        if origin_host == target_host:
            return "same-origin"
        if _site(origin_host) and _site(origin_host) == _site(target_host):
            return "same-site"
        return "cross-site"

    if target_host and referer_host:
        if referer_host == target_host:
            return "same-origin"
        if _site(referer_host) and _site(referer_host) == _site(target_host):
            return "same-site"
        return "cross-site"

    if origin_host or referer_host:
        return "same-origin" if origin_host == referer_host else "cross-site"

    return "none"


@dataclass(frozen=True)
class BrowserFingerprint:
    """用于在整个会话周期保持一致的浏览器指纹"""

    user_agent: str
    browser_type: str
    accept_language: str
    accept_encoding: str

    @classmethod
    def create(cls, browser_type: Optional[str] = None) -> "BrowserFingerprint":
        if browser_type is None:
            browser_type = random.choice(["chrome", "chrome", "edge", "firefox", "safari"])
        user_agent = get_random_user_agent(browser_type)
        resolved_browser_type = _detect_browser_type(user_agent)
        accept_language = random.choice(_ACCEPT_LANGUAGE_CANDIDATES)
        accept_encoding = random.choice(_ACCEPT_ENCODING_CANDIDATES)
        return cls(
            user_agent=user_agent,
            browser_type=resolved_browser_type,
            accept_language=accept_language,
            accept_encoding=accept_encoding,
        )

    def build_headers(
        self,
        target_url: Optional[str] = None,
        referer: Optional[str] = None,
        origin: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        return get_dynamic_headers(
            target_url=target_url,
            referer=referer,
            origin=origin,
            browser_type=self.browser_type,
            additional_headers=additional_headers,
            user_agent=self.user_agent,
            accept_language=self.accept_language,
            accept_encoding=self.accept_encoding,
        )

    def build_ws_headers(
        self,
        referer: Optional[str] = None,
        origin: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        headers = {
            "User-Agent": self.user_agent,
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
        if referer:
            headers["Referer"] = referer
        if origin:
            headers["Origin"] = origin
        if additional_headers:
            headers.update(additional_headers)
        return headers


# 通用 UserAgent headers 生成函数
def get_dynamic_headers(
    target_url: Optional[str] = None,
    referer: Optional[str] = None,
    origin: Optional[str] = None,
    browser_type: Optional[str] = None,
    additional_headers: Optional[Dict[str, str]] = None,
    user_agent: Optional[str] = None,
    accept_language: Optional[str] = None,
    accept_encoding: Optional[str] = None,
) -> Dict[str, str]:
    """
    生成动态浏览器 headers，包含随机 User-Agent

    Args:
        referer: 引用页面 URL
        origin: 源站 URL
        browser_type: 指定浏览器类型
        additional_headers: 额外的 headers

    Returns:
        Dict[str, str]: 包含动态 User-Agent 的 headers
    """
    if user_agent is None:
        user_agent = get_random_user_agent(browser_type)
    if browser_type is None:
        browser_type = _detect_browser_type(user_agent)

    if accept_language is None:
        accept_language = random.choice(_ACCEPT_LANGUAGE_CANDIDATES)
    if accept_encoding is None:
        accept_encoding = random.choice(_ACCEPT_ENCODING_CANDIDATES)

    # 基础 headers
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json, text/event-stream",
        "Accept-Language": accept_language,
        "Accept-Encoding": accept_encoding,
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
    }

    # 添加可选的 headers
    if referer:
        headers["Referer"] = referer

    if origin:
        headers["Origin"] = origin

    # 根据用户代理添加浏览器特定的 headers
    if browser_type in {"chrome", "edge"}:
        # Chrome/Edge 特定的 headers
        chrome_version = "126"  # 默认值

        try:
            if "Chrome/" in user_agent:
                chrome_version = user_agent.split("Chrome/")[1].split(".")[0]
        except IndexError:
            pass

        sec_ch_ua_parts = []
        if "Edg/" in user_agent:
            try:
                edge_version = user_agent.split("Edg/")[1].split(".")[0]
                sec_ch_ua_parts.append(f'"Microsoft Edge";v="{edge_version}"')
            except IndexError:
                pass

        sec_ch_ua_parts.append(f'"Chromium";v="{chrome_version}"')

        if "Google Chrome" in user_agent:
            sec_ch_ua_parts.append(f'"Google Chrome";v="{chrome_version}"')

        sec_ch_ua_parts.append('"Not_A Brand";v="8"')

        headers.update(
            {
                "sec-ch-ua": ", ".join(sec_ch_ua_parts),
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": _detect_platform(user_agent),
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": _infer_fetch_site(origin, referer, target_url),
            }
        )

    # 添加额外的 headers
    if additional_headers:
        headers.update(additional_headers)

    return headers
