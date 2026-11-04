"""The app with allin1 simulated (allin1_fake): ``uvicorn
scripts.ux_sim.allin1_server:app`` with UX_SIM_META pointing at meta.json."""
from scripts.ux_sim import allin1_fake

allin1_fake.install()

from api.server import app  # noqa: E402,F401
