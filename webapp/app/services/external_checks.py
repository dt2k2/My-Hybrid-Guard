import json
import re
import shutil
import socket
import ssl
import subprocess
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

try:
    import httpx
except ImportError:
    httpx = None

from webapp.app.config import SUSPICIOUS_KEYWORDS


SOCIAL_KEYWORDS = ("facebook", "instagram", "tiktok", "twitter", "x.com", "youtube", "telegram", "linkedin")
CONTACT_KEYWORDS = (
    "contact",
    "address",
    "phone",
    "hotline",
    "support",
    "customer service",
    "zalo",
    "whatsapp",
    "email",
)


def _run_command(command: list[str], timeout: int) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return completed.returncode == 0, output.strip()


def _extract_ips(text: str) -> list[str]:
    return sorted(set(re.findall(r"(?:\d{1,3}\.){3}\d{1,3}", text)))


def _parse_headers(raw_headers: str) -> dict:
    blocks = [block.strip() for block in re.split(r"\r?\n\r?\n", raw_headers) if block.strip()]
    if not blocks:
        return {"status_code": None, "headers": {}, "redirect_hops": 0}
    final_block = blocks[-1]
    lines = [line.strip() for line in final_block.splitlines() if line.strip()]
    status_code = None
    headers = {}
    if lines:
        status_match = re.search(r"\b(\d{3})\b", lines[0])
        if status_match:
            status_code = int(status_match.group(1))
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return {"status_code": status_code, "headers": headers, "redirect_hops": max(0, len(blocks) - 1)}


def _flatten_name(name_tuple) -> str:
    pairs = []
    for item in name_tuple or []:
        for key, value in item:
            pairs.append(f"{key}={value}")
    return ", ".join(pairs)


def _format_network_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = repr(exc)
    lowered = message.lower()
    if "getaddrinfo failed" in lowered or "name or service not known" in lowered:
        return f"DNS resolution failed: {message}"
    return message


def collect_dns(hostname: str, timeout: int, cache, ttl_hours: int, no_cache: bool = False) -> dict:
    cache_key = f"dns::{hostname}"
    cached = None if no_cache else cache.get(cache_key, ttl_hours)
    if cached:
        cached["from_cache"] = True
        return cached

    result = {
        "resolved": False,
        "ips": [],
        "tool": None,
        "error": None,
        "from_cache": False,
    }

    if shutil.which("nslookup"):
        ok, output = _run_command(["nslookup", hostname], timeout)
        result["tool"] = "nslookup"
        ips = _extract_ips(output)
        result["ips"] = ips
        result["resolved"] = ok and bool(ips)

    if not result["resolved"]:
        result["tool"] = "socket.getaddrinfo" if result["tool"] is None else f"{result['tool']} -> socket.getaddrinfo"
        try:
            addrinfo = socket.getaddrinfo(hostname, None)
            result["ips"] = sorted({item[4][0] for item in addrinfo})
            result["resolved"] = bool(result["ips"])
        except OSError as exc:
            result["error"] = str(exc)

    cache.set(cache_key, dict(result))
    return result


def collect_ssl(scheme: str, hostname: str, timeout: int, cache, ttl_hours: int, no_cache: bool = False) -> dict:
    normalized_scheme = (scheme or "").lower().strip()
    cache_key = f"ssl::{normalized_scheme}::{hostname}"
    cached = None if no_cache else cache.get(cache_key, ttl_hours)
    if cached:
        cached["from_cache"] = True
        return cached

    result = {
        "available": False,
        "issuer": None,
        "subject": None,
        "not_before": None,
        "not_after": None,
        "days_until_expiration": None,
        "expired": None,
        "error": None,
        "from_cache": False,
    }

    if normalized_scheme != "https":
        result["error"] = "URL không dùng HTTPS."
        cache.set(cache_key, dict(result))
        return result

    try:
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as secure_sock:
                cert = secure_sock.getpeercert()

        not_before = datetime.strptime(cert["notBefore"], "%b %d %H:%M:%S %Y %Z").replace(tzinfo=UTC)
        not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z").replace(tzinfo=UTC)
        result.update(
            {
                "available": True,
                "issuer": _flatten_name(cert.get("issuer")),
                "subject": _flatten_name(cert.get("subject")),
                "not_before": not_before.isoformat(),
                "not_after": not_after.isoformat(),
                "days_until_expiration": int((not_after - datetime.now(UTC)).total_seconds() // 86400),
                "expired": datetime.now(UTC) > not_after,
            }
        )
    except (ssl.SSLError, OSError, KeyError, ValueError) as exc:
        result["error"] = str(exc)

    cache.set(cache_key, dict(result))
    return result


def collect_domain_age(hostname: str, timeout: int, cache, no_cache: bool = False) -> dict:
    cache_key = f"rdap::{hostname}"
    cached = None if no_cache else cache.get(cache_key, 24 * 7)
    if cached:
        cached["from_cache"] = True
        return cached

    result = {
        "registered_at": None,
        "expires_at": None,
        "days_since_registration": None,
        "days_until_expiration": None,
        "error": None,
        "source": "rdap.org",
        "from_cache": False,
    }

    request = Request(
        f"https://rdap.org/domain/{quote(hostname)}",
        headers={"User-Agent": "Mozilla/5.0 URL-Phishing-WebApp"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))

        now = datetime.now(UTC)
        for event in payload.get("events", []):
            action = (event.get("eventAction") or "").lower()
            event_date = event.get("eventDate")
            if not event_date:
                continue
            try:
                event_time = datetime.fromisoformat(event_date.replace("Z", "+00:00")).astimezone(UTC)
            except ValueError:
                continue

            if action in {"registration", "registered"} and result["registered_at"] is None:
                result["registered_at"] = event_time.isoformat()
                result["days_since_registration"] = int((now - event_time).total_seconds() // 86400)
            if action in {"expiration", "expired", "expiration date"} and result["expires_at"] is None:
                result["expires_at"] = event_time.isoformat()
                result["days_until_expiration"] = int((event_time - now).total_seconds() // 86400)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError, Exception) as exc:
        result["error"] = str(exc)

    cache.set(cache_key, dict(result))
    return result


def collect_http(url: str, timeout: int, max_content_bytes: int, cache, ttl_hours: int, no_cache: bool = False) -> dict:
    cache_key = f"http::{url}"
    cached = None if no_cache else cache.get(cache_key, ttl_hours)
    if cached:
        cached["from_cache"] = True
        return cached

    result = {
        "tool": None,
        "status_code": None,
        "headers": {},
        "redirect_hops": 0,
        "keyword_hits": [],
        "password_form_detected": False,
        "social_icon_mentions": 0,
        "social_link_count": 0,
        "empty_social_anchor_count": 0,
        "contact_keyword_hits": [],
        "content_preview": None,
        "error": None,
        "from_cache": False,
    }

    def _extract_content_signals(text: str) -> dict:
        lowered = text.lower()
        content_preview = re.sub(r"\s+", " ", text).strip()[:max_content_bytes]
        keyword_hits = sorted(keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in lowered)
        password_form_detected = "password" in lowered and "<input" in lowered
        contact_keyword_hits = sorted(keyword for keyword in CONTACT_KEYWORDS if keyword in lowered)

        social_mentions = 0
        social_links = 0
        empty_social_links = 0
        for social in SOCIAL_KEYWORDS:
            mention_count = lowered.count(social)
            social_mentions += mention_count
            social_links += len(
                re.findall(
                    rf"(?:href|src|data-href|data-url)\s*=\s*[\"'][^\"']*{re.escape(social)}[^\"']*[\"']",
                    lowered,
                )
            )
            empty_social_links += len(
                re.findall(
                    rf"<a[^>]*href\s*=\s*[\"'](?:#|javascript:void\(0\)|)\s*[\"'][^>]*>[^<]*?(?:{re.escape(social)})",
                    lowered,
                )
            )

        # Common phishing pattern: icon classes are present but anchor href is empty/#
        empty_social_links += len(
            re.findall(
                r"<a[^>]*href\s*=\s*[\"'](?:#|javascript:void\(0\)|)\s*[\"'][^>]*class\s*=\s*[\"'][^\"']*(?:facebook|instagram|tiktok|twitter|x-twitter|youtube|telegram|linkedin|fa-)\S*[^\"']*[\"']",
                lowered,
            )
        )

        return {
            "content_preview": content_preview,
            "keyword_hits": keyword_hits,
            "password_form_detected": password_form_detected,
            "contact_keyword_hits": contact_keyword_hits,
            "social_icon_mentions": int(social_mentions),
            "social_link_count": int(social_links),
            "empty_social_anchor_count": int(empty_social_links),
        }

    request_headers = {
        "User-Agent": "Mozilla/5.0 URL-Phishing-WebApp",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    if httpx is not None:
        error_messages = []
        for verify, tool_name in ((True, "httpx"), (False, "httpx-insecure-fallback")):
            try:
                with httpx.Client(
                    follow_redirects=True,
                    timeout=httpx.Timeout(timeout),
                    headers=request_headers,
                    verify=verify,
                ) as client:
                    response = client.get(url)

                result["tool"] = tool_name
                result["status_code"] = int(response.status_code)
                result["headers"] = {str(k).lower(): str(v) for k, v in response.headers.items()}
                result["redirect_hops"] = len(response.history)
                result.update(_extract_content_signals(response.text[:max_content_bytes]))

                if error_messages:
                    result["error"] = " | ".join(error_messages) + " | fallback=httpx:ok"
                break
            except (httpx.RequestError, httpx.HTTPError, ValueError) as exc:
                error_messages.append(f"{tool_name}:{_format_network_exception(exc)}")

        if result["tool"] is None and error_messages:
            result["error"] = " | ".join(error_messages)

    # Final fallback: urllib to maximize compatibility if httpx is unavailable or failed.
    if not result.get("content_preview"):
        if result["tool"] is None:
            result["tool"] = "urllib"
        request = Request(url, headers=request_headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(max_content_bytes).decode("utf-8", errors="replace")
                if result["status_code"] is None:
                    result["status_code"] = getattr(response, "status", None)
                if not result["headers"]:
                    result["headers"] = {str(k).lower(): str(v) for k, v in dict(response.info().items()).items()}
                result.update(_extract_content_signals(payload))
                if result["error"]:
                    result["error"] = f"{result['error']} | fallback=urllib:ok"
        except (HTTPError, URLError, TimeoutError, OSError, Exception) as exc:
            if result["error"]:
                result["error"] = f"{result['error']} | fallback=urllib:{_format_network_exception(exc)}"
            else:
                result["error"] = _format_network_exception(exc)

    cache.set(cache_key, dict(result))
    return result
