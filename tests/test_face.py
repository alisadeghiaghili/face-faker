"""
Tests for Face Faker face pipeline
"""

import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from PIL import Image

from face_faker.face.pipeline import get_person_image


def create_test_image(width=100, height=100):
    """Create a test PIL Image."""
    return Image.new('RGB', (width, height), color='red')


def create_test_image_bytes():
    """Create test image as bytes."""
    img = create_test_image()
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()


class TestGetPersonImage:
    """Test cases for get_person_image."""

    @patch('face_faker.face.pipeline.requests.get')
    def test_successful_fetch(self, mock_get):
        """Test successful image fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = create_test_image_bytes()
        mock_get.return_value = mock_response

        result = get_person_image()

        assert result is not None
        assert isinstance(result, Image.Image)
        mock_get.assert_called_once()

    @patch('face_faker.face.pipeline.requests.get')
    def test_failed_fetch_returns_none(self, mock_get):
        """Test failed fetch returns None."""
        mock_get.side_effect = Exception("Network error")

        result = get_person_image()

        assert result is None

    @patch('face_faker.face.pipeline.requests.get')
    def test_non_200_status_returns_none(self, mock_get):
        """Test non-200 status returns None."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = get_person_image()

        assert result is None

    @patch('face_faker.face.pipeline.requests.get')
    def test_timeout_returns_none(self, mock_get):
        """Test timeout returns None."""
        import requests
        mock_get.side_effect = requests.Timeout("Connection timed out")

        result = get_person_image()

        assert result is None


class TestModuleExports:
    """Test module exports."""

    def test_import_generate_id_faces(self):
        """Test that generate_id_faces can be imported."""
        from face_faker import generate_id_faces
        assert callable(generate_id_faces)

    def test_import_version(self):
        """Test that version can be imported."""
        from face_faker import __version__
        assert __version__ == "2.0.0"

    def test_main_package_exports(self):
        """Test main package exports only face-related items."""
        import face_faker
        assert hasattr(face_faker, 'generate_id_faces')
        assert hasattr(face_faker, '__version__')
        # Should NOT have these (Farsi Faker)
        assert not hasattr(face_faker, 'PersianNamesGenerator')
        assert not hasattr(face_faker, 'IdentityGenerator')
        assert not hasattr(face_faker, 'NationalIDGenerator')
