"""Common domain types, enumerations, and constants for FaceTrace."""

from enum import Enum


class PlatformType(str, Enum):
    """Known public platforms where evidence can be discovered."""

    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    REDDIT = "reddit"
    LINKEDIN = "linkedin"
    TIKTOK = "tiktok"
    FACEBOOK = "facebook"
    YOUTUBE = "youtube"
    PINTEREST = "pinterest"
    WIKIPEDIA = "wikipedia"
    NEWS = "news"
    WEB = "web"
    UNKNOWN = "unknown"

    @classmethod
    def from_url(cls, url: str) -> "PlatformType":
        """Infer platform type from domain name."""
        lower_url = url.lower()
        if "instagram.com" in lower_url:
            return cls.INSTAGRAM
        if "facebook.com" in lower_url or "fb.com" in lower_url:
            return cls.FACEBOOK
        if "twitter.com" in lower_url or "x.com" in lower_url:
            return cls.TWITTER
        if "reddit.com" in lower_url:
            return cls.REDDIT
        if "linkedin.com" in lower_url:
            return cls.LINKEDIN
        if "tiktok.com" in lower_url:
            return cls.TIKTOK
        if "youtube.com" in lower_url:
            return cls.YOUTUBE
        if "pinterest.com" in lower_url:
            return cls.PINTEREST
        if "wikipedia.org" in lower_url or "wikimedia.org" in lower_url:
            return cls.WIKIPEDIA
        return cls.WEB


class VerificationStatus(str, Enum):
    """Integrity verification outcome status."""

    VERIFIED = "VERIFIED"
    TAMPERED = "TAMPERED"
    NOT_FOUND = "NOT_FOUND"
    ERROR = "ERROR"
    BLOCKCHAIN_ERROR = "BLOCKCHAIN_ERROR"


class BlockchainNetwork(str, Enum):
    """Target blockchain networks."""

    SEPOLIA = "sepolia"
    LOCAL = "local"
    MOCK = "mock"
