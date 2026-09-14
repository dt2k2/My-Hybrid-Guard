from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "url_phishing_model.joblib"
CACHE_PATH = ARTIFACTS_DIR / "web_external_cache.json"

DL_MODEL_PATH = ARTIFACTS_DIR / "dl_smoke" / "url_phishing_model.joblib"
DT_MODEL_PATH = ARTIFACTS_DIR / "dt_model" / "url_phishing_model.joblib"

MODEL_PATHS = {
    "rf": MODEL_PATH,
    "dl": DL_MODEL_PATH,
    "dt": DT_MODEL_PATH,
}

DEFAULT_MODEL_MODE = "rf"

DEFAULT_TIMEOUT_SECONDS = 8
DEFAULT_CACHE_HOURS = 24
DEFAULT_MAX_CONTENT_BYTES = 4096

MALICIOUS_THRESHOLD = 0.5
SUSPICIOUS_THRESHOLD = 0.35
UNCERTAIN_THRESHOLD = 0.18

RISKY_TLDS = {
    "rf.gd",
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

KNOWN_LEGIT_LOGIN_DOMAINS = {
    "facebook.com",
    "fb.com",
    "instagram.com",
    "messenger.com",
    "whatsapp.com",
    "google.com",
    "youtube.com",
    "microsoft.com",
    "live.com",
    "outlook.com",
    "apple.com",
    "icloud.com",
    "amazon.com",
    "aws.amazon.com",
    "github.com",
    "gitlab.com",
    "paypal.com",
    "stripe.com",
}
