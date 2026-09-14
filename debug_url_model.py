import argparse
import json
import re
import shutil
import socket
import ssl
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import joblib
import pandas as pd

from url_ml.feature_engineering import extract_url_features, normalize_url


SUSPICIOUS_KEYWORDS = {
    "login",
    "verify",
    "password",
    "signin",
    "account",
    "bank",
    "wallet",
    "secure",
    "update",
    "confirm",
    "billing",
    "token",
    "otp",
    "payment",
    "invoice",
}

RISKY_TLDS = {
    "tk",
    "ga",
    "gq",
    "cf",
    "ml",
    "vip",
    "xyz",
    "top",
    "click",
    "work",
    "support",
    "zip",
    "mov",
    "country",
    "stream",
}

TRUSTED_TLDS = {
    "gov",
    "gov.vn",
    "edu",
    "edu.vn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict and debug whether a URL is benign or malicious.",
    )
    parser.add_argument("--model-path", default="artifacts/url_phishing_model.joblib", help="Path to the trained model artifact.")
    parser.add_argument("--url", help="Single URL to inspect.")
    parser.add_argument("--url-file", help="Text file containing one URL per line.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    parser.add_argument("--top-k", type=int, default=6, help="How many important debug features to print.")
    parser.add_argument(
        "--uncertain-threshold",
        type=float,
        default=0.18,
        help="Run external checks when abs(probability - 0.5) is below this threshold.",
    )
    parser.add_argument("--force-enrichment", action="store_true", help="Always run external checks.")
    parser.add_argument("--skip-enrichment", action="store_true", help="Disable DNS/SSL/domain age/content enrichment.")
    parser.add_argument("--timeout", type=int, default=8, help="Timeout in seconds for each external check.")
    parser.add_argument("--cache-hours", type=int, default=24, help="How long to reuse cached enrichment results.")
    parser.add_argument(
        "--cache-file",
        default="artifacts/external_checks_cache.json",
        help="Cache file for external lookup results.",
    )
    parser.add_argument(
        "--max-content-bytes",
        type=int,
        default=4096,
        help="Maximum response content to keep for analysis.",
    )
    parser.add_argument(
        "--always-ssl-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always run DNS + SSL checks (cached), even when model confidence is high.",
    )
    parser.add_argument(
        "--always-domain-age-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Always run RDAP domain-age check (cached).",
    )
    parser.add_argument(
        "--malicious-threshold",
        type=float,
        default=0.5,
        help="Final probability threshold to classify as malicious.",
    )
    parser.add_argument(
        "--suspicious-threshold",
        type=float,
        default=0.25,
        help="Final probability threshold to classify as suspicious (between benign and malicious).",
    )
    return parser.parse_args()


def load_urls(args: argparse.Namespace) -> list[str]:
    urls = []
    if args.url:
        urls.append(args.url.strip())
    if args.url_file:
        file_urls = Path(args.url_file).read_text(encoding="utf-8").splitlines()
        urls.extend(url.strip() for url in file_urls if url.strip())
    if not urls:
        entered_url = input("Nhap URL can kiem tra: ").strip()
        if entered_url:
            urls.append(entered_url)
    if not urls:
        raise ValueError("No URL provided.")
    return urls


def find_debug_signals(features: dict) -> list[str]:
    signals = []
    if features["is_ip"]:
        signals.append("Hostname dang IP thay vi ten mien.")
    if features["entropy"] >= 4.2:
        signals.append("Entropy cao, URL co ve bi xao tron / random.")
    if features["digit_ratio"] >= 0.2:
        signals.append("Ty le chu so cao bat thuong.")
    if features["special_cnt"] >= 10:
        signals.append("So ky tu dac biet cao.")
    if features["query_len"] >= 25:
        signals.append("Chuoi query dai, can xem ky tham so.")
    if features["subdom_cnt"] >= 3:
        signals.append("Nhieu subdomain, co the dang ngu trang.")
    if not signals:
        signals.append("Khong thay dau hieu ky thuat noi bat tu heuristic don gian.")
    return signals


def rank_features(features: dict, artifact: dict, top_k: int) -> list[dict]:
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    feature_stats = artifact["feature_stats"]
    importances = getattr(model, "feature_importances_", [0.0] * len(feature_names))

    ranked = []
    for feature_name, importance in zip(feature_names, importances):
        mean = feature_stats[feature_name]["mean"]
        std = feature_stats[feature_name]["std"] or 1.0
        value = float(features[feature_name])
        z_score = abs((value - mean) / std)
        score = float(importance) * z_score
        if not any(importances):
            score = z_score
        ranked.append(
            {
                "feature": feature_name,
                "value": value,
                "importance": float(importance),
                "debug_score": score,
            }
        )
    ranked.sort(key=lambda item: item["debug_score"], reverse=True)
    return ranked[:top_k]


def load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def get_cached(cache: dict, cache_key: str, ttl_hours: int):
    entry = cache.get(cache_key)
    if not entry:
        return None
    timestamp = entry.get("timestamp")
    if not timestamp:
        return None
    try:
        created_at = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if datetime.now(UTC) - created_at > timedelta(hours=ttl_hours):
        return None
    return entry.get("value")


def set_cached(cache: dict, cache_key: str, value: dict) -> None:
    cache[cache_key] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "value": value,
    }


def run_command(command: list[str], timeout: int) -> tuple[bool, str]:
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


def normalize_text_snippet(text: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:limit]


def flatten_name(name_tuple) -> str:
    parts = []
    for item in name_tuple or []:
        for key, value in item:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def extract_ips_from_nslookup(output: str) -> list[str]:
    matches = re.findall(r"(?:\d{1,3}\.){3}\d{1,3}", output)
    return sorted(set(matches))


def parse_header_block(raw_headers: str) -> dict:
    blocks = [block.strip() for block in re.split(r"\r?\n\r?\n", raw_headers) if block.strip()]
    if not blocks:
        return {"raw": raw_headers, "status_code": None, "headers": {}}

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

    return {
        "raw": raw_headers,
        "status_code": status_code,
        "headers": headers,
        "redirect_hops": max(0, len(blocks) - 1),
    }


def collect_dns_info(hostname: str, timeout: int, cache: dict, cache_hours: int) -> dict:
    cache_key = f"dns::{hostname}"
    cached = get_cached(cache, cache_key, cache_hours)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    result = {
        "check": "dns",
        "hostname": hostname,
        "resolved": False,
        "ips": [],
        "tool": None,
        "raw_output": "",
        "from_cache": False,
    }

    if shutil.which("nslookup"):
        ok, output = run_command(["nslookup", hostname], timeout)
        result["tool"] = "nslookup"
        result["raw_output"] = output
        result["ips"] = extract_ips_from_nslookup(output)
        result["resolved"] = ok and bool(result["ips"])

    if not result["resolved"]:
        previous_output = result["raw_output"]
        result["tool"] = "socket.getaddrinfo" if result["tool"] is None else f"{result['tool']} -> socket.getaddrinfo"
        try:
            addrinfo = socket.getaddrinfo(hostname, None)
            result["ips"] = sorted({item[4][0] for item in addrinfo})
            result["resolved"] = bool(result["ips"])
            result["raw_output"] = (
                (previous_output + "\n\n") if previous_output else ""
            ) + "Resolved via socket.getaddrinfo"
        except OSError as exc:
            result["raw_output"] = ((previous_output + "\n\n") if previous_output else "") + str(exc)

    set_cached(cache, cache_key, dict(result))
    return result


def collect_ssl_info(scheme: str, hostname: str, timeout: int, cache: dict, cache_hours: int) -> dict:
    cache_key = f"ssl::{hostname}"
    cached = get_cached(cache, cache_key, cache_hours)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    result = {
        "check": "ssl",
        "available": False,
        "hostname": hostname,
        "issuer": None,
        "subject": None,
        "not_before": None,
        "not_after": None,
        "days_until_expiration": None,
        "expired": None,
        "error": None,
        "from_cache": False,
    }

    if scheme.lower() != "https":
        result["error"] = "URL does not use HTTPS."
        set_cached(cache, cache_key, dict(result))
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
                "issuer": flatten_name(cert.get("issuer")),
                "subject": flatten_name(cert.get("subject")),
                "not_before": not_before.isoformat(),
                "not_after": not_after.isoformat(),
                "days_until_expiration": int((not_after - datetime.now(UTC)).total_seconds() // 86400),
                "expired": datetime.now(UTC) > not_after,
            }
        )
    except (ssl.SSLError, OSError, KeyError, ValueError) as exc:
        result["error"] = str(exc)

    set_cached(cache, cache_key, dict(result))
    return result


def collect_domain_age(hostname: str, timeout: int, cache: dict) -> dict:
    cache_key = f"rdap::{hostname}"
    cached = get_cached(cache, cache_key, ttl_hours=24 * 7)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    result = {
        "check": "domain_age",
        "hostname": hostname,
        "registered_at": None,
        "expires_at": None,
        "days_since_registration": None,
        "days_until_expiration": None,
        "events": [],
        "error": None,
        "from_cache": False,
        "source": "rdap.org",
    }

    request = Request(
        f"https://rdap.org/domain/{quote(hostname)}",
        headers={"User-Agent": "Mozilla/5.0 URL-Phishing-Debugger"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        events = payload.get("events", [])
        result["events"] = events
        now = datetime.now(UTC)
        for event in events:
            action = (event.get("eventAction") or "").lower()
            date_text = event.get("eventDate")
            if not date_text:
                continue
            try:
                event_time = datetime.fromisoformat(date_text.replace("Z", "+00:00")).astimezone(UTC)
            except ValueError:
                continue
            if action in {"registration", "registered"} and result["registered_at"] is None:
                result["registered_at"] = event_time.isoformat()
                result["days_since_registration"] = int((now - event_time).total_seconds() // 86400)
            if action in {"expiration", "expired", "expiration date"} and result["expires_at"] is None:
                result["expires_at"] = event_time.isoformat()
                result["days_until_expiration"] = int((event_time - now).total_seconds() // 86400)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)

    set_cached(cache, cache_key, dict(result))
    return result


def fetch_http_signals(url: str, timeout: int, cache: dict, cache_hours: int, max_content_bytes: int) -> dict:
    cache_key = f"http::{url}"
    cached = get_cached(cache, cache_key, cache_hours)
    if cached is not None:
        cached["from_cache"] = True
        return cached

    result = {
        "check": "http",
        "tool": None,
        "status_code": None,
        "headers": {},
        "redirect_hops": 0,
        "content_preview": None,
        "keyword_hits": [],
        "password_form_detected": False,
        "error": None,
        "from_cache": False,
    }

    if shutil.which("curl"):
        result["tool"] = "curl"
        ok_headers, raw_headers = run_command(
            ["curl", "-I", "-L", "-sS", "--max-time", str(timeout), url],
            timeout=max(timeout + 2, timeout),
        )
        parsed_headers = parse_header_block(raw_headers)
        result["status_code"] = parsed_headers.get("status_code")
        result["headers"] = parsed_headers.get("headers", {})
        result["redirect_hops"] = parsed_headers.get("redirect_hops", 0)

        ok_body, body = run_command(
            [
                "curl",
                "-L",
                "-sS",
                "--max-time",
                str(timeout),
                "--range",
                f"0-{max_content_bytes - 1}",
                url,
            ],
            timeout=max(timeout + 2, timeout),
        )
        if ok_body:
            lowered = body.lower()
            result["content_preview"] = normalize_text_snippet(body, max_content_bytes)
            result["keyword_hits"] = sorted(keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in lowered)
            result["password_form_detected"] = "password" in lowered and "<input" in lowered
        elif not ok_headers:
            result["error"] = raw_headers or body
    else:
        result["tool"] = "urllib"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 URL-Phishing-Debugger"})
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(max_content_bytes).decode("utf-8", errors="replace")
                headers = dict(response.info().items())
                result["status_code"] = getattr(response, "status", None)
                result["headers"] = {str(key).lower(): str(value) for key, value in headers.items()}
                lowered = payload.lower()
                result["content_preview"] = normalize_text_snippet(payload, max_content_bytes)
                result["keyword_hits"] = sorted(keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in lowered)
                result["password_form_detected"] = "password" in lowered and "<input" in lowered
        except (HTTPError, URLError, TimeoutError) as exc:
            result["error"] = str(exc)

    set_cached(cache, cache_key, dict(result))
    return result


def detect_risk_triggers(url: str, features: dict) -> list[str]:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    hostname = (parsed.hostname or "").lower()
    path_query = ((parsed.path or "") + " " + (parsed.query or "")).lower()

    trigger_reasons = []
    if features.get("is_https") == 0:
        trigger_reasons.append("URL khong dung HTTPS.")
    if "xn--" in hostname:
        trigger_reasons.append("Hostname co dau hieu punycode (co the domain ngu trang).")
    if features.get("dash_cnt", 0) >= 2 and features.get("dom_len", 0) >= 12:
        trigger_reasons.append("Domain dai va nhieu dau gach ngang.")
    if features.get("subdom_cnt", 0) >= 2:
        trigger_reasons.append("Domain co nhieu subdomain.")
    if features.get("entropy", 0) >= 4.0:
        trigger_reasons.append("Entropy URL cao bat thuong.")

    tld = str(features.get("tld", "")).lower().strip(".")
    if tld in RISKY_TLDS:
        trigger_reasons.append(f"TLD co rui ro phishing cao: .{tld}")

    lexical_hits = sorted(keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in hostname or keyword in path_query)
    if lexical_hits:
        trigger_reasons.append(f"URL co tu khoa nhay cam: {', '.join(lexical_hits)}.")

    return trigger_reasons


def is_trusted_tld(tld: str) -> bool:
    return tld.lower().strip(".") in TRUSTED_TLDS


def should_run_enrichment(
    probability: float,
    threshold: float,
    force: bool,
    skip: bool,
    trigger_reasons: list[str],
) -> tuple[bool, str]:
    if skip:
        return False, "External enrichment disabled by --skip-enrichment."
    if force:
        return True, "External enrichment forced by --force-enrichment."
    if trigger_reasons:
        return True, "External enrichment enabled by risk triggers: " + " | ".join(trigger_reasons)
    margin = abs(probability - 0.5)
    if margin < threshold:
        return True, f"Model confidence is low because |{probability:.4f} - 0.5| = {margin:.4f} < {threshold:.4f}."
    return False, f"Model confidence is high enough, so external enrichment is skipped (margin={margin:.4f})."


def compute_enrichment_adjustment(features: dict, external_checks: dict) -> tuple[float, list[str]]:
    adjustment = 0.0
    reasons = []

    tld = str(features.get("tld", "")).lower().strip(".")
    if tld in RISKY_TLDS:
        adjustment += 0.10
        reasons.append(f"TLD .{tld} nam trong nhom rui ro phishing cao.")

    dns_info = external_checks.get("dns", {})
    if dns_info and not dns_info.get("resolved", True):
        adjustment += 0.10
        reasons.append("DNS khong resolve duoc hostname.")

    ssl_info = external_checks.get("ssl", {})
    if ssl_info.get("expired") is True:
        adjustment += 0.20
        reasons.append("SSL certificate da het han.")
    elif isinstance(ssl_info.get("days_until_expiration"), int):
        if ssl_info["days_until_expiration"] < 7:
            adjustment += 0.08
            reasons.append("SSL sap het han trong vong 7 ngay.")
        elif ssl_info["days_until_expiration"] > 180:
            adjustment -= 0.03
            reasons.append("SSL con han dai, giam nhe muc do nghi ngo.")
    elif features.get("is_https") == 0:
        adjustment += 0.04
        reasons.append("URL khong su dung HTTPS nen tang nhe muc do nghi ngo.")

    domain_age = external_checks.get("domain_age", {})
    registered_days = domain_age.get("days_since_registration")
    if isinstance(registered_days, int):
        if registered_days < 30:
            adjustment += 0.25
            reasons.append("Domain moi dang ky duoi 30 ngay.")
        elif registered_days < 180:
            adjustment += 0.12
            reasons.append("Domain con rat moi duoi 180 ngay.")
        elif registered_days > 365 * 3:
            adjustment -= 0.08
            reasons.append("Domain da ton tai lau nam, giam muc do nghi ngo.")

    http_info = external_checks.get("http", {})
    keyword_hits = http_info.get("keyword_hits", [])
    if keyword_hits:
        keyword_adjustment = min(0.20, len(keyword_hits) * 0.04)
        adjustment += keyword_adjustment
        reasons.append(f"Noi dung trang co tu khoa nhay cam: {', '.join(keyword_hits)}.")
    if http_info.get("password_form_detected"):
        adjustment += 0.15
        reasons.append("Trang co input password trong noi dung HTML.")
    if isinstance(http_info.get("redirect_hops"), int) and http_info["redirect_hops"] >= 3:
        adjustment += 0.08
        reasons.append("URL qua nhieu redirect hop.")
    if http_info.get("status_code") in {401, 403, 404, 429, 500, 502, 503}:
        adjustment += 0.03
        reasons.append(f"HTTP status bat thuong: {http_info['status_code']}.")

    if features.get("is_https") == 0 and (keyword_hits or http_info.get("password_form_detected")):
        adjustment += 0.12
        reasons.append("Trang co dau hieu login/nhay cam nhung khong dung HTTPS.")

    return adjustment, reasons


def run_external_enrichment(
    url: str,
    features: dict,
    args: argparse.Namespace,
    include_domain_age: bool,
    include_http: bool,
) -> dict:
    normalized_url = normalize_url(url)
    parsed = urlparse(normalized_url)
    hostname = (parsed.hostname or "").strip().lower()
    cache_path = Path(args.cache_file)
    cache = load_cache(cache_path)

    checks = {
        "dns": collect_dns_info(hostname, args.timeout, cache, args.cache_hours),
        "ssl": collect_ssl_info(parsed.scheme, hostname, args.timeout, cache, args.cache_hours),
        "domain_age": collect_domain_age(hostname, args.timeout, cache)
        if include_domain_age
        else {
            "check": "domain_age",
            "source": "N/A",
            "registered_at": None,
            "expires_at": None,
            "days_since_registration": None,
            "days_until_expiration": None,
            "events": [],
            "error": "Skipped to avoid unnecessary network usage.",
            "from_cache": False,
        },
        "http": fetch_http_signals(normalized_url, args.timeout, cache, args.cache_hours, args.max_content_bytes)
        if include_http
        else {
            "check": "http",
            "tool": "N/A",
            "status_code": None,
            "headers": {},
            "redirect_hops": 0,
            "content_preview": None,
            "keyword_hits": [],
            "password_form_detected": False,
            "error": "Skipped to avoid unnecessary network usage.",
            "from_cache": False,
        },
    }

    save_cache(cache_path, cache)
    adjustment, reasons = compute_enrichment_adjustment(features, checks)
    return {
        "executed": True,
        "hostname": hostname,
        "cache_file": str(cache_path),
        "checks": checks,
        "probability_adjustment": adjustment,
        "adjustment_reasons": reasons,
    }


def classify_probability(probability: float, malicious_threshold: float, suspicious_threshold: float) -> str:
    if probability >= malicious_threshold:
        return "malicious"
    if probability >= suspicious_threshold:
        return "suspicious"
    return "benign"


def predict_url(url: str, artifact: dict, args: argparse.Namespace) -> dict:
    features = extract_url_features(url)
    feature_names = artifact["feature_names"]
    model = artifact["model"]
    model_type = artifact.get("training", {}).get("model_type", "rf")
    feature_frame = pd.DataFrame([{name: features[name] for name in feature_names}])

    if model_type == "dl" and hasattr(model, "predict_with_urls"):
        predicted_class = int(model.predict_with_urls([url], feature_frame)[0])
        base_probability = float(model.predict_proba_with_urls([url], feature_frame)[0][1])
    else:
        predicted_class = int(model.predict(feature_frame)[0])
        if hasattr(model, "predict_proba"):
            base_probability = float(model.predict_proba(feature_frame)[0][1])
        else:
            base_probability = float(predicted_class)

    tld = str(features.get("tld", "")).lower().strip(".")
    trusted_tld = is_trusted_tld(tld)

    trigger_reasons = detect_risk_triggers(url, features)
    if args.skip_enrichment:
        should_enrich = False
        enrich_reason = "External enrichment disabled by --skip-enrichment."
    elif args.force_enrichment:
        should_enrich = True
        enrich_reason = "External enrichment forced by --force-enrichment."
    elif not trusted_tld:
        should_enrich = True
        enrich_reason = f"Non-trusted TLD policy: .{tld or 'unknown'} requires deep checks (domain age/content/ssl)."
    else:
        should_enrich, enrich_reason = should_run_enrichment(
            base_probability,
            args.uncertain_threshold,
            False,
            False,
            trigger_reasons,
        )

    enrichment = {
        "executed": False,
        "reason": enrich_reason,
        "risk_triggers": trigger_reasons,
        "tld_policy": "trusted" if trusted_tld else "non-trusted",
        "tld": tld,
    }
    final_probability = base_probability
    final_prediction = classify_probability(
        final_probability,
        args.malicious_threshold,
        args.suspicious_threshold,
    )
    fusion_reasons = []

    if should_enrich:
        if not trusted_tld and not args.force_enrichment:
            full_mode = True
        else:
            full_mode = args.force_enrichment or bool(trigger_reasons) or abs(base_probability - 0.5) < args.uncertain_threshold
        include_domain_age = full_mode or args.always_domain_age_check
        include_http = full_mode
        enrichment = run_external_enrichment(url, features, args, include_domain_age, include_http)
        enrichment["reason"] = enrich_reason
        enrichment["risk_triggers"] = trigger_reasons
        enrichment["tld_policy"] = "trusted" if trusted_tld else "non-trusted"
        enrichment["tld"] = tld
        final_probability = min(1.0, max(0.0, base_probability + enrichment["probability_adjustment"]))
        final_prediction = classify_probability(
            final_probability,
            args.malicious_threshold,
            args.suspicious_threshold,
        )
        fusion_reasons = enrichment.get("adjustment_reasons", [])
    elif args.always_ssl_check and not args.skip_enrichment:
        enrichment = run_external_enrichment(
            url,
            features,
            args,
            include_domain_age=args.always_domain_age_check,
            include_http=False,
        )
        enrichment["reason"] = "Always-on SSL/DNS mode is active."
        enrichment["risk_triggers"] = trigger_reasons
        enrichment["tld_policy"] = "trusted" if trusted_tld else "non-trusted"
        enrichment["tld"] = tld
        final_probability = min(1.0, max(0.0, base_probability + enrichment["probability_adjustment"]))
        final_prediction = classify_probability(
            final_probability,
            args.malicious_threshold,
            args.suspicious_threshold,
        )
        fusion_reasons = enrichment.get("adjustment_reasons", [])

    return {
        "url": url,
        "prediction": final_prediction,
        "model_prediction": "malicious" if predicted_class == 1 else "benign",
        "malicious_probability": final_probability,
        "base_malicious_probability": base_probability,
        "thresholds": {
            "malicious": args.malicious_threshold,
            "suspicious": args.suspicious_threshold,
        },
        "features": features,
        "top_debug_features": rank_features(features, artifact, args.top_k),
        "heuristic_signals": find_debug_signals(features),
        "enrichment": enrichment,
        "fusion_reasons": fusion_reasons,
    }


def print_human_readable(result: dict) -> None:
    print("=" * 100)
    print(f"URL: {result['url']}")
    print(f"Model prediction: {result['model_prediction']}")
    print(f"Final prediction: {result['prediction']}")
    print(f"Risk tier: {result['prediction'].upper()}")
    print(f"Base malicious probability: {result['base_malicious_probability']:.4f}")
    print(f"Final malicious probability: {result['malicious_probability']:.4f}")

    print("\nTop debug features")
    for item in result["top_debug_features"]:
        print(
            f"- {item['feature']}: value={item['value']:.4f}, "
            f"importance={item['importance']:.4f}, debug_score={item['debug_score']:.4f}"
        )

    print("\nHeuristic signals")
    for signal in result["heuristic_signals"]:
        print(f"- {signal}")

    enrichment = result["enrichment"]
    print("\nExternal enrichment")
    print(f"- executed: {enrichment.get('executed', False)}")
    print(f"- reason: {enrichment.get('reason', 'N/A')}")
    print(f"- tld: .{enrichment.get('tld', 'unknown')}")
    print(f"- tld_policy: {enrichment.get('tld_policy', 'unknown')}")
    risk_triggers = enrichment.get("risk_triggers", [])
    if risk_triggers:
        print(f"- risk_triggers: {' | '.join(risk_triggers)}")

    if enrichment.get("executed"):
        print(f"- probability adjustment: {enrichment['probability_adjustment']:+.4f}")
        print(f"- cache file: {enrichment['cache_file']}")

        print("\nFusion reasons")
        if result["fusion_reasons"]:
            for reason in result["fusion_reasons"]:
                print(f"- {reason}")
        else:
            print("- Khong co ly do bo sung duoc ap vao diem xac suat.")

        checks = enrichment["checks"]
        dns_info = checks["dns"]
        print("\nDNS check")
        print(f"- tool: {dns_info.get('tool')}")
        print(f"- resolved: {dns_info.get('resolved')}")
        print(f"- ips: {', '.join(dns_info.get('ips', [])) or 'N/A'}")
        print(f"- from_cache: {dns_info.get('from_cache')}")

        ssl_info = checks["ssl"]
        print("\nSSL check")
        print(f"- available: {ssl_info.get('available')}")
        print(f"- issuer: {ssl_info.get('issuer') or 'N/A'}")
        print(f"- subject: {ssl_info.get('subject') or 'N/A'}")
        print(f"- not_before: {ssl_info.get('not_before') or 'N/A'}")
        print(f"- not_after: {ssl_info.get('not_after') or 'N/A'}")
        print(f"- days_until_expiration: {ssl_info.get('days_until_expiration')}")
        print(f"- expired: {ssl_info.get('expired')}")
        print(f"- error: {ssl_info.get('error') or 'N/A'}")
        print(f"- from_cache: {ssl_info.get('from_cache')}")

        age_info = checks["domain_age"]
        print("\nDomain age check")
        print(f"- source: {age_info.get('source')}")
        print(f"- registered_at: {age_info.get('registered_at') or 'N/A'}")
        print(f"- expires_at: {age_info.get('expires_at') or 'N/A'}")
        print(f"- days_since_registration: {age_info.get('days_since_registration')}")
        print(f"- days_until_expiration: {age_info.get('days_until_expiration')}")
        print(f"- error: {age_info.get('error') or 'N/A'}")
        print(f"- from_cache: {age_info.get('from_cache')}")

        http_info = checks["http"]
        print("\nHTTP/content check")
        print(f"- tool: {http_info.get('tool')}")
        print(f"- status_code: {http_info.get('status_code')}")
        print(f"- redirect_hops: {http_info.get('redirect_hops')}")
        print(f"- server: {http_info.get('headers', {}).get('server', 'N/A')}")
        print(f"- content_type: {http_info.get('headers', {}).get('content-type', 'N/A')}")
        print(f"- keyword_hits: {', '.join(http_info.get('keyword_hits', [])) or 'N/A'}")
        print(f"- password_form_detected: {http_info.get('password_form_detected')}")
        print(f"- error: {http_info.get('error') or 'N/A'}")
        print(f"- from_cache: {http_info.get('from_cache')}")
        preview = http_info.get("content_preview") or ""
        if preview:
            print("- content_preview:")
            print(f"  {preview}")

    print("\nExtracted features")
    for key, value in result["features"].items():
        if key in {"url", "dom", "tld"}:
            print(f"- {key}: {value}")
        else:
            print(f"- {key}: {float(value):.6f}")


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model_path)
    results = [predict_url(url, artifact, args) for url in load_urls(args)]

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    for result in results:
        print_human_readable(result)


if __name__ == "__main__":
    main()