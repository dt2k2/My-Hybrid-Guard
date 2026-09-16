import math
from collections import Counter
from ipaddress import ip_address
from typing import Dict
from urllib.parse import ParseResult, urlparse


def safe_urlparse(raw_url: str) -> ParseResult:
    try:
        return urlparse(raw_url)
    except ValueError:
        sanitized = (raw_url or "").replace("[", "%5B").replace("]", "%5D")
        try:
            return urlparse(sanitized)
        except ValueError:
            return ParseResult(scheme="http", netloc="", path=raw_url, params="", query="", fragment="")


NUMERIC_FEATURES = [
    "url_len",
    "dom_len",
    "is_ip",
    "tld_len",
    "subdom_cnt",
    "letter_cnt",
    "digit_cnt",
    "special_cnt",
    "eq_cnt",
    "qm_cnt",
    "amp_cnt",
    "dot_cnt",
    "dash_cnt",
    "under_cnt",
    "letter_ratio",
    "digit_ratio",
    "spec_ratio",
    "is_https",
    "slash_cnt",
    "entropy",
    "path_len",
    "query_len",
]


def _safe_ratio(part: int, total: int) -> float:
    return float(part) / float(total) if total else 0.0


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _extract_hostname(parsed_url) -> str:
    hostname = parsed_url.netloc or parsed_url.path
    hostname = hostname.split("@")[(-1)]
    hostname = hostname.split(":")[0]
    return hostname.lower().strip(".")


def _extract_tld(hostname: str) -> str:
    if not hostname:
        return ""
    parts = [part for part in hostname.split(".") if part]
    if len(parts) < 2:
        return parts[-1] if parts else ""
    if len(parts[-1]) == 2 and len(parts) >= 3 and len(parts[-2]) <= 3:
        return ".".join(parts[-2:])
    return parts[-1]


def _count_subdomains(hostname: str) -> int:
    parts = [part for part in hostname.split(".") if part]
    if len(parts) <= 2:
        return 0
    return len(parts) - 2


def _is_ip_host(hostname: str) -> int:
    try:
        ip_address(hostname)
        return 1
    except ValueError:
        return 0


def normalize_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        url = f"http://{url}"
    return url


def extract_url_features(raw_url: str) -> Dict[str, float]:
    url = normalize_url(raw_url)
    parsed = safe_urlparse(url)
    hostname = _extract_hostname(parsed)
    tld = _extract_tld(hostname)

    letter_cnt = sum(character.isalpha() for character in url)
    digit_cnt = sum(character.isdigit() for character in url)
    special_cnt = sum(not character.isalnum() for character in url)

    features = {
        "url": raw_url,
        "url_len": len(url),
        "dom": hostname,
        "dom_len": len(hostname),
        "is_ip": _is_ip_host(hostname),
        "tld": tld,
        "tld_len": len(tld),
        "subdom_cnt": _count_subdomains(hostname),
        "letter_cnt": letter_cnt,
        "digit_cnt": digit_cnt,
        "special_cnt": special_cnt,
        "eq_cnt": url.count("="),
        "qm_cnt": url.count("?"),
        "amp_cnt": url.count("&"),
        "dot_cnt": url.count("."),
        "dash_cnt": url.count("-"),
        "under_cnt": url.count("_"),
        "letter_ratio": _safe_ratio(letter_cnt, len(url)),
        "digit_ratio": _safe_ratio(digit_cnt, len(url)),
        "spec_ratio": _safe_ratio(special_cnt, len(url)),
        "is_https": int(parsed.scheme.lower() == "https"),
        "slash_cnt": url.count("/"),
        "entropy": _shannon_entropy(url),
        "path_len": len(parsed.path or ""),
        "query_len": len(parsed.query or ""),
    }
    return features