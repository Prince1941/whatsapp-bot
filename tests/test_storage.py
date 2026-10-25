from app.models import StyleProfile, OllamaConfig, PrivacyFilterConfig


def test_storage_save_and_retrieve_profile(temp_db):
    profile = StyleProfile(
        id="default",
        profile_name="My Coding Style",
        sender_name="Prince",
        custom_notes="Casual and friendly",
    )
    temp_db.save_profile(profile)

    loaded = temp_db.get_profile("default")
    assert loaded is not None
    assert loaded.profile_name == "My Coding Style"
    assert loaded.sender_name == "Prince"
    assert loaded.custom_notes == "Casual and friendly"


def test_storage_delete_profile(temp_db):
    profile = StyleProfile(profile_name="Ephemeral", sender_name="Tester")
    temp_db.save_profile(profile)
    assert temp_db.get_profile() is not None

    temp_db.delete_profile("default")
    assert temp_db.get_profile() is None


def test_storage_ollama_and_privacy_configs(temp_db):
    ollama_cfg = OllamaConfig(base_url="http://127.0.0.1:11434", model="mistral", timeout_seconds=45.0)
    temp_db.save_ollama_config(ollama_cfg)

    loaded_ollama = temp_db.get_ollama_config()
    assert loaded_ollama.base_url == "http://127.0.0.1:11434"
    assert loaded_ollama.model == "mistral"
    assert loaded_ollama.timeout_seconds == 45.0

    priv_cfg = PrivacyFilterConfig(custom_banned_terms=["topsecret"])
    temp_db.save_privacy_config(priv_cfg)

    loaded_priv = temp_db.get_privacy_config()
    assert "topsecret" in loaded_priv.custom_banned_terms
