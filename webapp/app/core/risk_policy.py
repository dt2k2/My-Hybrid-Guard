from url_ml.feature_engineering import normalize_url, safe_urlparse
from webapp.app.config import (
    KNOWN_LEGIT_LOGIN_DOMAINS,
    MALICIOUS_THRESHOLD,
    RISKY_TLDS,
    SUSPICIOUS_KEYWORDS,
    SUSPICIOUS_THRESHOLD,
    TRUSTED_TLDS,
)


def detect_triggers(url: str, features: dict) -> list[str]:
    normalized = normalize_url(url)
    parsed = safe_urlparse(normalized)
    try:
        hostname = (parsed.hostname or "").lower()
    except ValueError:
        hostname = (parsed.netloc or "").split(":")[0].lower()
    path_query = ((parsed.path or "") + " " + (parsed.query or "")).lower()

    reasons = []
    if features.get("is_https") == 0:
        reasons.append("URL không dùng HTTPS.")
    if "xn--" in hostname:
        reasons.append("Hostname có dấu hiệu punycode.")
    if features.get("subdom_cnt", 0) >= 2:
        reasons.append("Domain có nhiều subdomain.")
    if features.get("entropy", 0) >= 4.0:
        reasons.append("Entropy URL cao bất thường.")

    tld = str(features.get("tld", "")).lower().strip(".")
    if tld in RISKY_TLDS:
        reasons.append(f"TLD có rủi ro phishing cao: .{tld}")

    keyword_hits = sorted(keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in hostname or keyword in path_query)
    if keyword_hits:
        reasons.append(f"URL có từ khóa nhạy cảm: {', '.join(keyword_hits)}.")

    return reasons


def is_trusted_tld(tld: str) -> bool:
    return tld.lower().strip(".") in TRUSTED_TLDS


def is_risky_tld(tld: str) -> bool:
    return tld.lower().strip(".") in RISKY_TLDS


def classify_probability(probability: float, malicious_threshold: float = MALICIOUS_THRESHOLD, suspicious_threshold: float = SUSPICIOUS_THRESHOLD) -> str:
    if probability >= malicious_threshold:
        return "malicious"
    if probability >= suspicious_threshold:
        return "suspicious"
    return "benign"


def is_known_legit_domain(hostname: str) -> bool:
    host = (hostname or "").lower().strip(".")
    if not host:
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in KNOWN_LEGIT_LOGIN_DOMAINS)


def apply_adjustment(features: dict, checks: dict, hostname: str = "") -> tuple[float, list[str]]:
    adjustment = 0.0
    reasons = []
    known_legit = is_known_legit_domain(hostname)

    tld = str(features.get("tld", "")).lower().strip(".")
    risky_tld = is_risky_tld(tld)
    if risky_tld:
        adjustment += 0.10
        reasons.append(f"TLD .{tld} nằm trong nhóm rủi ro phishing cao.")

    dns_info = checks.get("dns", {})
    if dns_info and not dns_info.get("resolved", True):
        adjustment += 0.12
        reasons.append("DNS không thể phân giải hostname.")

    ssl_info = checks.get("ssl", {})
    if ssl_info.get("expired") is True:
        adjustment += 0.20
        reasons.append("Chứng chỉ SSL đã hết hạn.")
    elif isinstance(ssl_info.get("days_until_expiration"), int):
        days = ssl_info["days_until_expiration"]
        if days < 7:
            adjustment += 0.08
            reasons.append("SSL sắp hết hạn trong vòng 7 ngày.")
        elif days > 180:
            adjustment -= 0.03
            reasons.append("SSL còn hạn dài, giảm nhẹ mức độ nghi ngờ.")
    elif features.get("is_https") == 0:
        adjustment += 0.03
        reasons.append("URL không dùng HTTPS nên tăng nhẹ mức độ nghi ngờ.")

    age_info = checks.get("domain_age", {})
    registered_days = age_info.get("days_since_registration")
    if isinstance(registered_days, int):
        if registered_days < 30:
            adjustment += 0.25
            reasons.append("Domain mới đăng ký dưới 30 ngày.")
        elif registered_days < 180:
            adjustment += 0.12
            reasons.append("Domain còn rất mới dưới 180 ngày.")
        elif registered_days > 365 * 3:
            adjustment -= 0.08
            reasons.append("Domain đã tồn tại lâu năm, giảm mức độ nghi ngờ.")

    http_info = checks.get("http", {})
    trusted_transport = bool(dns_info.get("resolved")) and bool(ssl_info.get("available")) and not bool(ssl_info.get("expired"))
    parsed_host = (hostname or "").lower().strip(".")
    host_keywords = {"verify", "secure", "billing", "bank", "wallet", "signin", "confirm", "token", "password"}
    host_has_risky_keyword = any(keyword in parsed_host for keyword in host_keywords)
    suspicious_hosting_suffixes = {
        "rf.gd",
        "web.app",
        "pages.dev",
        "vercel.app",
        "github.io",
        "weebly.com",
        "webflow.io",
        "blogspot.com",
    }
    hosted_on_user_content = any(
        parsed_host.endswith(f".{suffix}") or parsed_host == suffix for suffix in suspicious_hosting_suffixes
    )

    if hosted_on_user_content and host_has_risky_keyword and not known_legit:
        adjustment += 0.10
        reasons.append("Hostname thuộc nền tảng hosting mở và chứa từ khóa nhạy cảm.")

    if host_has_risky_keyword and float(features.get("entropy", 0.0)) >= 4.0 and not known_legit:
        adjustment += 0.12
        reasons.append("Hostname chứa từ khóa nhạy cảm và entropy cao bất thường.")

    if known_legit and trusted_transport:
        adjustment -= 0.10
        reasons.append("Domain thuộc nền tảng uy tín và có SSL/DNS ổn định, giảm điểm nghi ngờ.")

    keyword_hits = http_info.get("keyword_hits", [])
    if keyword_hits:
        keyword_score = min(0.10, 0.02 * len(keyword_hits))
        if known_legit and trusted_transport and not host_has_risky_keyword:
            keyword_score = 0.0
            reasons.append("Bỏ qua tín hiệu từ khóa nội dung vì đây là trang đăng nhập hợp lệ trên domain uy tín.")
        elif known_legit and trusted_transport:
            keyword_score = min(0.05, keyword_score * 0.25)
            reasons.append("Giảm trọng số từ khóa nhạy cảm do domain uy tín và kết nối an toàn.")
        elif trusted_transport and not risky_tld and not host_has_risky_keyword:
            keyword_score = min(0.03, keyword_score * 0.30)
            reasons.append("Giảm trọng số từ khóa vì kết nối ổn định và TLD không thuộc nhóm rủi ro cao.")
        if keyword_score > 0:
            adjustment += keyword_score
            reasons.append(f"Nội dung trang có từ khóa nhạy cảm: {', '.join(keyword_hits)}.")

    weak_transport = (not bool(dns_info.get("resolved"))) or (not bool(ssl_info.get("available"))) or bool(ssl_info.get("expired"))
    if risky_tld and features.get("is_https") == 0 and weak_transport and keyword_hits:
        adjustment += 0.20
        reasons.append("Hội tụ tín hiệu mạnh: TLD rủi ro + không HTTPS + transport yếu + có từ khóa nhạy cảm.")

    social_mentions = int(http_info.get("social_icon_mentions") or 0)
    social_links = int(http_info.get("social_link_count") or 0)
    empty_social_anchors = int(http_info.get("empty_social_anchor_count") or 0)
    contact_hits = http_info.get("contact_keyword_hits") or []
    trusted_tld = is_trusted_tld(tld)

    # Heuristic: phishing pages often display trust-looking social/contact icons
    # but omit real destination links. Apply cautiously to avoid raising FP.
    if social_mentions >= 2 and social_links == 0 and (empty_social_anchors > 0 or not contact_hits):
        if risky_tld or weak_transport or features.get("is_https") == 0 or keyword_hits or http_info.get("password_form_detected"):
            adjustment += 0.10
            reasons.append("Trang có biểu tượng social/contact nhưng không thấy liên kết social hợp lệ.")
            if risky_tld and features.get("is_https") == 0:
                adjustment += 0.06
                reasons.append("Tăng mức nghi ngờ do kết hợp social-link bất thường với TLD rủi ro và không HTTPS.")

    # Stronger signal for fake-looking landing pages: many social mentions but no real social links.
    # Apply only when lexical complexity is elevated and domain is not in trusted/legal lists to reduce FP.
    if (
        social_mentions >= 4
        and social_links == 0
        and not known_legit
        and not trusted_tld
        and float(features.get("entropy", 0.0)) >= 3.6
        and float(features.get("spec_ratio", 0.0)) >= 0.22
    ):
        adjustment += 0.24
        reasons.append("Nhiều biểu tượng social/contact nhưng không có liên kết social thực; mẫu landing page đáng ngờ.")

    if http_info.get("password_form_detected"):
        if known_legit and trusted_transport and not host_has_risky_keyword:
            reasons.append("Bỏ qua tín hiệu form mật khẩu vì domain uy tín và kênh kết nối an toàn.")
        elif known_legit and trusted_transport:
            adjustment += 0.03
            reasons.append("Trang có trường mật khẩu nhưng thuộc domain uy tín, chỉ tăng nhẹ điểm nghi ngờ.")
        else:
            adjustment += 0.10
            reasons.append("Trang có trường mật khẩu trong nội dung HTML.")
    if isinstance(http_info.get("redirect_hops"), int) and http_info["redirect_hops"] >= 3:
        adjustment += 0.08
        reasons.append("URL có quá nhiều bước chuyển hướng (redirect).")
    if http_info.get("status_code") in {401, 403, 404, 429, 500, 502, 503}:
        adjustment += 0.03
        reasons.append(f"Mã trạng thái HTTP bất thường: {http_info['status_code']}.")

    if features.get("is_https") == 0 and (keyword_hits or http_info.get("password_form_detected")):
        adjustment += 0.08
        reasons.append("Trang có dấu hiệu đăng nhập/nhạy cảm nhưng không dùng HTTPS.")

    if http_info.get("error") and (host_has_risky_keyword or float(features.get("entropy", 0.0)) >= 4.0) and not known_legit:
        adjustment += 0.08
        reasons.append("Không thể thu thập nội dung HTTP trong khi URL có dấu hiệu đáng ngờ.")

    if age_info.get("error") and host_has_risky_keyword and not known_legit:
        adjustment += 0.05
        reasons.append("Không xác minh được tuổi domain trong khi hostname chứa từ khóa nhạy cảm.")

    return adjustment, reasons


def apply_trust_guardrail(
    base_probability: float,
    adjusted_probability: float,
    features: dict,
    checks: dict,
    hostname: str,
) -> tuple[float, list[str]]:
    reasons = []
    if not is_known_legit_domain(hostname):
        return adjusted_probability, reasons

    dns_info = checks.get("dns", {})
    ssl_info = checks.get("ssl", {})
    http_info = checks.get("http", {})
    age_info = checks.get("domain_age", {})

    strong_transport = (
        bool(dns_info.get("resolved"))
        and bool(ssl_info.get("available"))
        and not bool(ssl_info.get("expired"))
        and features.get("is_https") == 1
    )
    good_http_status = http_info.get("status_code") in {200, 201, 204, 301, 302}
    low_redirect = int(http_info.get("redirect_hops") or 0) < 3
    has_young_age = isinstance(age_info.get("days_since_registration"), int) and age_info["days_since_registration"] < 30

    parsed_host = (hostname or "").lower().strip(".")
    host_keywords = {"verify", "secure", "billing", "bank", "wallet", "signin", "confirm", "token", "password"}
    host_has_risky_keyword = any(keyword in parsed_host for keyword in host_keywords)

    hard_red_flag = (
        not bool(dns_info.get("resolved"))
        or bool(ssl_info.get("expired"))
        or has_young_age
        or host_has_risky_keyword
    )

    if strong_transport and good_http_status and low_redirect and not hard_red_flag:
        cap = 0.22 if base_probability >= 0.80 else 0.20
        if adjusted_probability > cap:
            adjusted_probability = cap
            reasons.append(
                "Áp dụng trust-guardrail: domain uy tín, SSL/DNS/HTTP ổn định nên giới hạn mức nghi ngờ để giảm false positive."
            )

    return adjusted_probability, reasons
