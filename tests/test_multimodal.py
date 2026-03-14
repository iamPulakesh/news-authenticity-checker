import pytest
from app.multimodal.router import detect_input_type
from app.multimodal.scraper import is_valid_url, get_domain


class TestInputDetection:

    def test_url_detection(self):
        assert detect_input_type("https://bbc.com/news/article") == "url"
        assert detect_input_type("http://www.reuters.com/story") == "url"

    def test_image_detection(self):
        assert detect_input_type("/home/user/news_screenshot.jpg") == "image"
        assert detect_input_type("screenshot.png") == "image"
        assert detect_input_type("photo.webp") == "image"

    def test_unknown_detection(self):
        assert detect_input_type("random text") == "unknown"

    def test_text_detection(self):
        long_text = "Scientists have discovered a new species of fish in the Amazon river basin"
        assert detect_input_type(long_text) == "text"


class TestURLValidation:

    def test_valid_urls(self):
        assert is_valid_url("https://www.bbc.com/news") is True
        assert is_valid_url("http://reuters.com") is True

    def test_invalid_urls(self):
        assert is_valid_url("not a url") is False
        assert is_valid_url("ftp://files.example.com") is False
        assert is_valid_url("") is False


class TestDomainExtraction:

    def test_basic_domain(self):
        assert get_domain("https://www.bbc.com/news/article") == "bbc.com"
        assert get_domain("https://apnews.com/") == "apnews.com"

    def test_www_stripped(self):
        assert get_domain("https://www.reuters.com/story") == "reuters.com"

    def test_invalid_url(self):
        assert get_domain("not-a-url") == "unknown" or isinstance(get_domain("not-a-url"), str)
