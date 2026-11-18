from fastapi.testclient import TestClient

from adsci.app.main import create_app
from adsci.config import Settings


def test_serves_built_frontend_with_spa_fallback(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>app</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    settings = Settings(data_dir=tmp_path / "data", secret_key_file=tmp_path / "k", static_dir=dist)
    with TestClient(create_app(settings)) as c:
        assert c.get("/").text == "<html>app</html>"
        assert c.get("/some/client/route").text == "<html>app</html>"
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/api/health").json() == {"status": "ok"}
        assert c.get("/api/nope").status_code == 404  # API 404s are not swallowed by the SPA


def test_no_static_dir_means_api_only(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", secret_key_file=tmp_path / "k", static_dir=None)
    with TestClient(create_app(settings)) as c:
        assert c.get("/").status_code == 404
