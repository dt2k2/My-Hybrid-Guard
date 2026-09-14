from urllib.parse import urlparse

from url_ml.feature_engineering import normalize_url
from webapp.app.config import (
    DEFAULT_CACHE_HOURS,
    DEFAULT_MAX_CONTENT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    MALICIOUS_THRESHOLD,
    SUSPICIOUS_THRESHOLD,
    UNCERTAIN_THRESHOLD,
)
from webapp.app.core.risk_policy import (
    apply_adjustment,
    apply_trust_guardrail,
    classify_probability,
    detect_triggers,
    is_risky_tld,
    is_trusted_tld,
)
from webapp.app.services.external_checks import collect_dns, collect_domain_age, collect_http, collect_ssl


def _canonicalize_root_url(raw_url: str) -> str:
    normalized = normalize_url(raw_url)
    parsed = urlparse(normalized)
    if parsed.scheme and parsed.netloc and parsed.path == "":
        return f"{parsed.scheme}://{parsed.netloc}/"
    return normalized


class UrlAnalyzerService:
    def __init__(self, model_service, cache_store) -> None:
        self.model_service = model_service
        self.cache = cache_store

    def analyze(self, raw_url: str, recheck: bool = False) -> dict:
        scheme_missing = "://" not in (raw_url or "")
        canonical_input = _canonicalize_root_url(raw_url)
        features, predicted_class, base_probability = self.model_service.predict(canonical_input)
        normalized_url = normalize_url(canonical_input)
        parsed = urlparse(normalized_url)
        hostname = (parsed.hostname or "").lower()
        tld = str(features.get("tld", "")).lower().strip(".")

        triggers = detect_triggers(raw_url, features)
        trusted_tld = is_trusted_tld(tld)
        risky_tld = is_risky_tld(tld)

        low_confidence = abs(base_probability - 0.5) < UNCERTAIN_THRESHOLD
        medium_risk_non_trusted = (not trusted_tld) and base_probability >= 0.10
        full_checks = risky_tld or bool(triggers) or low_confidence or medium_risk_non_trusted

        checks = {
            "dns": collect_dns(hostname, DEFAULT_TIMEOUT_SECONDS, self.cache, DEFAULT_CACHE_HOURS, no_cache=recheck),
            "ssl": collect_ssl(
                parsed.scheme,
                hostname,
                DEFAULT_TIMEOUT_SECONDS,
                self.cache,
                DEFAULT_CACHE_HOURS,
                no_cache=recheck,
            ),
            "domain_age": collect_domain_age(hostname, DEFAULT_TIMEOUT_SECONDS, self.cache, no_cache=recheck)
            if full_checks
            else {
                "error": "Bỏ qua do không thuộc TLD rủi ro và không có tín hiệu rủi ro mạnh.",
                "from_cache": False,
            },
            "http": collect_http(
                normalized_url,
                DEFAULT_TIMEOUT_SECONDS,
                DEFAULT_MAX_CONTENT_BYTES,
                self.cache,
                DEFAULT_CACHE_HOURS,
                no_cache=recheck,
            )
            if full_checks
            else {
                "error": "Bỏ qua do không thuộc TLD rủi ro và không có tín hiệu rủi ro mạnh.",
                "from_cache": False,
            },
        }

        # If user inputs domain without scheme, probe HTTPS directly to avoid false penalties
        # caused by the temporary http:// normalization.
        inferred_https = False
        if scheme_missing:
            https_ssl = collect_ssl(
                "https",
                hostname,
                DEFAULT_TIMEOUT_SECONDS,
                self.cache,
                DEFAULT_CACHE_HOURS,
                no_cache=recheck,
            )
            if https_ssl.get("available"):
                checks["ssl"] = https_ssl
                inferred_https = True

        features_for_scoring = dict(features)
        if inferred_https:
            features_for_scoring["is_https"] = 1

        adjustment, fusion_reasons = apply_adjustment(features_for_scoring, checks, hostname=hostname)
        final_probability = max(0.0, min(1.0, base_probability + adjustment))
        final_probability, guardrail_reasons = apply_trust_guardrail(
            base_probability=base_probability,
            adjusted_probability=final_probability,
            features=features_for_scoring,
            checks=checks,
            hostname=hostname,
        )
        if inferred_https:
            guardrail_reasons = [
                "Không thấy scheme trong URL đầu vào; hệ thống đã tự kiểm tra HTTPS và phát hiện kết nối an toàn.",
            ] + guardrail_reasons
        fusion_reasons = guardrail_reasons + fusion_reasons
        final_prediction = classify_probability(final_probability, MALICIOUS_THRESHOLD, SUSPICIOUS_THRESHOLD)

        self.cache.save()

        return {
            "url": raw_url,
            "normalized_url": normalized_url,
            "model_prediction": "malicious" if predicted_class == 1 else "benign",
            "final_prediction": final_prediction,
            "base_malicious_probability": base_probability,
            "final_malicious_probability": final_probability,
            "risk_tier": final_prediction.upper(),
            "thresholds": {
                "malicious": MALICIOUS_THRESHOLD,
                "suspicious": SUSPICIOUS_THRESHOLD,
            },
            "risk_triggers": triggers,
            "top_debug_features": self.model_service.rank_features(features, top_k=8),
            "features": features,
            "fusion_reasons": fusion_reasons,
            "checks": checks,
            "policy": {
                "tld": tld,
                "tld_policy": "trusted" if trusted_tld else ("risky" if risky_tld else "standard"),
                "risky_tld": risky_tld,
                "full_checks": full_checks,
                "low_confidence": low_confidence,
                "medium_risk_non_trusted": medium_risk_non_trusted,
                "scheme_missing_input": scheme_missing,
                "inferred_https": inferred_https,
                "recheck": recheck,
            },
        }
