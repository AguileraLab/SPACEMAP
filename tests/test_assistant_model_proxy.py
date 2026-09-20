from spacemap_assistant import model_client


def test_local_model_opener_has_no_proxy_handler() -> None:
    proxy_handlers = [
        handler
        for handler in model_client.LOCAL_ONLY_OPENER.handlers
        if handler.__class__.__name__ == "ProxyHandler"
    ]
    assert proxy_handlers == []
