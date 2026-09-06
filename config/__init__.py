"""Shared application configuration values and environment loading."""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_project_environment() -> None:
    """Load local credentials and tracing settings without overriding the shell."""

    load_dotenv(PROJECT_ENV_FILE, override=False)

# Deep Research sends these as You.com domain preferences. They improve the
# likelihood of authoritative medical results without excluding other domains.
PREFERRED_MEDICAL_DOMAINS = (
    "nih.gov",
    "cdc.gov",
    "fda.gov",
    "medlineplus.gov",
    "mayoclinic.org",
    "clevelandclinic.org",
    "hopkinsmedicine.org",
    "jamanetwork.com",
    "nejm.org",
)
