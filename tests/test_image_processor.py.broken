"""
å¤æ¨¡æç¥è¯çè§?- å¾åå¤çå¨æµè¯?
Multimodal Knowledge Understanding - Image Processor Tests

æµè¯ImageProcessorçå®æ´åè?
"""
# pylint: disable=redefined-outer-name

import pytest
import tempfile
from PIL import Image
import numpy as np

from image_processor import (
    ImageProcessor,
    ImageProcessorError,
    OCREngine,
    ImageFormat,
    ImageMetadata,
    OCRResult,
    ObjectDetectionResult,
    ImageFeatures,
    ImageAnalysisResult
)
from entity_extractor import Entity, EntityType


@pytest.fixture
def image_processor():
    """åå»ºImageProcessorå®ä¾"""
    return ImageProcessor(
        ocr_engine=OCREngine.TESSERACT,
        enable_object_detection=False,
        enable_scene_classification=False
    )


@pytest.fixture
def test_image_path():
    """åå»ºæµè¯å¾å"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # åå»ºç®åçæµè¯å¾å
        img = Image.new('RGB', (100, 100), color='red')
        img.save(f.name)
        yield f.name

    # æ¸
ç
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def test_image_with_text():
    """åå»ºå
å«ææ¬çæµè¯å¾å?""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # åå»ºå
å«ææ¬çå¾å?
        img = Image.new('RGB', (200, 100), color='white')
        # æ³? å®é
åºæ·»å ææ¬ç»å?è¿éç®å?
        img.save(f.name)
        yield f.name

    Path(f.name).unlink(missing_ok=True)


class TestImageProcessorInit:
    """æµè¯ImageProcessoråå§å?""

    def test_default_initialization(self):
        """æµè¯é»è®¤åå§å?""
        processor = ImageProcessor()
        assert processor.ocr_engine == OCREngine.TESSERACT
        assert not processor.enable_object_detection
        assert not processor.enable_scene_classification
        assert processor.max_image_size == (2048, 2048)

    def test_custom_initialization(self):
        """æµè¯èªå®ä¹åå§å"""
        processor = ImageProcessor(
            ocr_engine=OCREngine.PADDLE,
            enable_object_detection=True,
            enable_scene_classification=True,
            max_image_size=(1024, 1024)
        )
        assert processor.ocr_engine == OCREngine.PADDLE
        assert processor.enable_object_detection
        assert processor.enable_scene_classification
        assert processor.max_image_size == (1024, 1024)

    def test_statistics_initialized(self, image_processor):
        """æµè¯ç»è®¡åå§å?""
        stats = image_processor.get_statistics()
        assert stats["total_processed"] == 0
        assert stats["ocr_count"] == 0
        assert stats["detection_count"] == 0
        assert stats["classification_count"] == 0


class TestImageLoading:
    """æµè¯å¾åå è½½"""

    def test_load_valid_image(self, image_processor, test_image_path):
        """æµè¯å è½½ææå¾å"""
        img = image_processor.load_image(test_image_path)
        assert img is not None
        assert img.width == 100
        assert img.height == 100

    def test_load_invalid_image(self, image_processor):
        """æµè¯å è½½æ æå¾å"""
        with pytest.raises(ImageProcessorError):
            image_processor.load_image("nonexistent.png")

    def test_image_resize_when_too_large(self):
        """æµè¯è¿å¤§å¾åèªå¨è°æ´"""
        processor = ImageProcessor(max_image_size=(50, 50))

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            img = Image.new('RGB', (200, 200), color='blue')
            img.save(temp_path)

        try:
            loaded_img = processor.load_image(temp_path)
            assert loaded_img.width <= 50
            assert loaded_img.height <= 50
        finally:
            try:
                Path(temp_path).unlink()
            except PermissionError:
                pass  # Windowsæä»¶éé®é¢?å¿½ç¥


class TestImageMetadata:
    """æµè¯å¾åå
æ°æ®æå?""

    def test_get_metadata(self, image_processor, test_image_path):
        """æµè¯è·åå
æ°æ?""
        img = image_processor.load_image(test_image_path)
        metadata = image_processor.get_metadata(img, test_image_path)

        assert isinstance(metadata, ImageMetadata)
        assert metadata.width == 100
        assert metadata.height == 100
        assert metadata.mode == 'RGB'
        assert metadata.size_bytes > 0
        assert len(metadata.hash_md5) == 32  # MD5 hashé¿åº¦

    def test_metadata_hash_consistency(self, image_processor, test_image_path):
        """æµè¯å
æ°æ®hashä¸è´æ?""
        img = image_processor.load_image(test_image_path)
        metadata1 = image_processor.get_metadata(img)
        metadata2 = image_processor.get_metadata(img)

        assert metadata1.hash_md5 == metadata2.hash_md5


class TestOCR:
    """æµè¯OCRææ¬æå"""

    def test_ocr_tesseract_empty_image(self, image_processor, test_image_path):
        """æµè¯Tesseract OCR(ç©ºå¾å?"""
        img = image_processor.load_image(test_image_path)
        result = image_processor.extract_text_ocr(img)

        assert isinstance(result, OCRResult)
        assert isinstance(result.text, str)
        assert 0.0 <= result.confidence <= 1.0

    def test_ocr_paddle_not_installed(self):
        """æµè¯PaddleOCRæªå®è£
æ¶çéçº?""
        processor = ImageProcessor(ocr_engine=OCREngine.PADDLE)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            img = Image.new('RGB', (100, 100), color='white')
            img.save(temp_path)

        try:
            loaded_img = processor.load_image(temp_path)
            result = processor.extract_text_ocr(loaded_img)

            # å¦æPaddleOCRæªå®è£?åºè¿åç©ºç»æ
            assert isinstance(result, OCRResult)
        finally:
            try:
                Path(temp_path).unlink()
            except PermissionError:
                pass  # Windowsæä»¶éé®é¢?å¿½ç¥

    def test_ocr_result_to_dict(self):
        """æµè¯OCRç»æè½¬å­å
?""
        result = OCRResult(
            text="Hello World",
            confidence=0.95,
            bounding_boxes=[(10, 10, 100, 50)],
            language="en"
        )

        result_dict = result.to_dict()
        assert result_dict["text"] == "Hello World"
        assert result_dict["confidence"] == 0.95
        assert len(result_dict["bounding_boxes"]) == 1


class TestObjectDetection:
    """æµè¯å¯¹è±¡æ£æµ?""

    def test_object_detection_disabled(self, image_processor, test_image_path):
        """æµè¯ç¦ç¨å¯¹è±¡æ£æµ?""
        img = image_processor.load_image(test_image_path)
        objects = image_processor.detect_objects(img)

        assert isinstance(objects, list)
        assert len(objects) == 0

    def test_object_detection_enabled(self, test_image_path):
        """æµè¯å¯ç¨å¯¹è±¡æ£æµ?""
        processor = ImageProcessor(enable_object_detection=True)
        img = processor.load_image(test_image_path)
        objects = processor.detect_objects(img)

        assert isinstance(objects, list)
        # ç®åå®ç°è¿åç©ºåè¡¨

    def test_object_detection_result_to_dict(self):
        """æµè¯å¯¹è±¡æ£æµç»æè½¬å­å
¸"""
        result = ObjectDetectionResult(
            label="person",
            confidence=0.92,
            bbox=(50, 50, 200, 300)
        )

        result_dict = result.to_dict()
        assert result_dict["label"] == "person"
        assert result_dict["confidence"] == 0.92
        assert result_dict["bbox"] == (50, 50, 200, 300)


class TestSceneClassification:
    """æµè¯åºæ¯åç±»"""

    def test_scene_classification_disabled(self, image_processor, test_image_path):
        """æµè¯ç¦ç¨åºæ¯åç±»"""
        img = image_processor.load_image(test_image_path)
        label, confidence = image_processor.classify_scene(img)

        assert label == "unknown"
        assert confidence == 0.0

    def test_scene_classification_enabled(self, test_image_path):
        """æµè¯å¯ç¨åºæ¯åç±»"""
        processor = ImageProcessor(enable_scene_classification=True)
        img = processor.load_image(test_image_path)
        label, confidence = processor.classify_scene(img)

        assert isinstance(label, str)
        assert 0.0 <= confidence <= 1.0


class TestFeatureExtraction:
    """æµè¯ç¹å¾æå"""

    def test_extract_features(self, image_processor, test_image_path):
        """æµè¯ç¹å¾æå"""
        img = image_processor.load_image(test_image_path)
        features = image_processor.extract_features(img)

        assert isinstance(features, ImageFeatures)
        assert len(features.dominant_colors) > 0
        assert features.brightness >= 0.0
        assert features.contrast >= 0.0
        assert features.sharpness >= 0.0

    def test_dominant_colors_extraction(self, image_processor):
        """æµè¯ä¸»è²è°æå?""
        # åå»ºçº¯è²å¾å
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            img = Image.new('RGB', (100, 100), color=(255, 0, 0))  # çº¯çº¢è?
            img.save(temp_path)

        try:
            loaded_img = image_processor.load_image(temp_path)
            features = image_processor.extract_features(loaded_img)

            # ä¸»è²è°åºè¯¥æ¯çº¢è²
            dominant = features.dominant_colors[0]
            assert dominant[0] == 255  # R
            assert dominant[1] == 0    # G
            assert dominant[2] == 0    # B
        finally:
            try:
                Path(temp_path).unlink()
            except PermissionError:
                pass  # Windowsæä»¶éé®é¢?å¿½ç¥

    def test_features_to_dict(self):
        """æµè¯ç¹å¾è½¬å­å
?""
        features = ImageFeatures(
            embedding=np.array([0.1, 0.2, 0.3]),
            dominant_colors=[(255, 0, 0)],
            brightness=128.5,
            contrast=45.2,
            sharpness=12.8
        )

        features_dict = features.to_dict()
        assert features_dict["embedding"] == [0.1, 0.2, 0.3]
        assert features_dict["dominant_colors"] == [(255, 0, 0)]
        assert features_dict["brightness"] == 128.5


class TestEntityExtraction:
    """æµè¯å®ä½æå"""

    def test_extract_entities_from_empty_text(self, image_processor):
        """æµè¯ä»ç©ºææ¬æåå®ä½"""
        entities = image_processor.extract_entities_from_ocr("")
        assert len(entities) == 0

    def test_extract_entities_from_text(self, image_processor):
        """æµè¯ä»ææ¬æåå®ä½?""
        text = "Python is a Programming language created by Guido"
        entities = image_processor.extract_entities_from_ocr(text)

        assert len(entities) > 0
        assert all(isinstance(e, Entity) for e in entities)
        assert all(e.entity_type == EntityType.CONCEPT for e in entities)

    def test_entity_confidence(self, image_processor):
        """æµè¯å®ä½ç½®ä¿¡åº?""
        text = "Django Framework"
        entities = image_processor.extract_entities_from_ocr(text)

        for entity in entities:
            assert 0.0 <= entity.confidence <= 1.0


class TestFullProcessing:
    """æµè¯å®æ´å¤çæµç¨"""

    def test_process_image_with_ocr(self, image_processor, test_image_path):
        """æµè¯å¸¦OCRçå®æ´å¤ç?""
        result = image_processor.process(test_image_path, enable_ocr=True)

        assert isinstance(result, ImageAnalysisResult)
        assert result.image_path == test_image_path
        assert isinstance(result.metadata, ImageMetadata)
        assert result.ocr_results is not None
        assert isinstance(result.features, ImageFeatures)

    def test_process_image_without_ocr(self, image_processor, test_image_path):
        """æµè¯ä¸å¸¦OCRçå¤ç?""
        result = image_processor.process(test_image_path, enable_ocr=False)

        assert result.ocr_results is None
        assert len(result.entities) == 0

    def test_process_updates_statistics(self, image_processor, test_image_path):
        """æµè¯å¤çæ´æ°ç»è®¡"""
        initial_count = image_processor.get_statistics()["total_processed"]

        image_processor.process(test_image_path)

        updated_count = image_processor.get_statistics()["total_processed"]
        assert updated_count == initial_count + 1

    def test_result_to_dict(self, image_processor, test_image_path):
        """æµè¯ç»æè½¬å­å
?""
        result = image_processor.process(test_image_path)
        result_dict = result.to_dict()

        assert "image_path" in result_dict
        assert "metadata" in result_dict
        assert "ocr_results" in result_dict
        assert "objects" in result_dict
        assert "features" in result_dict
        assert "timestamp" in result_dict


class TestStatistics:
    """æµè¯ç»è®¡åè½"""

    def test_initial_statistics(self, image_processor):
        """æµè¯åå§ç»è®¡"""
        stats = image_processor.get_statistics()

        assert stats["total_processed"] == 0
        assert stats["ocr_count"] == 0
        assert stats["avg_processing_time_ms"] == 0.0

    def test_statistics_after_processing(self, image_processor, test_image_path):
        """æµè¯å¤çåç»è®?""
        image_processor.process(test_image_path, enable_ocr=True)

        stats = image_processor.get_statistics()
        assert stats["total_processed"] == 1
        # OCRè®¡æ°å¯è½ä¸?(Tesseractæªå®è£?æ?
        assert stats["ocr_count"] in [0, 1]
        assert stats["avg_processing_time_ms"] > 0.0

    def test_statistics_immutability(self, image_processor):
        """æµè¯ç»è®¡ä¸å¯åæ?""
        stats1 = image_processor.get_statistics()
        stats1["total_processed"] = 999

        stats2 = image_processor.get_statistics()
        assert stats2["total_processed"] == 0  # åå§å¼æªè¢«ä¿®æ?


class TestErrorHandling:
    """æµè¯éè¯¯å¤ç"""

    def test_invalid_image_path(self, image_processor):
        """æµè¯æ æå¾åè·¯å¾"""
        with pytest.raises(ImageProcessorError):
            image_processor.process("invalid_path.jpg")

    def test_corrupted_image_handling(self, image_processor):
        """æµè¯æåå¾åå¤ç"""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            f.write(b"not an image")

        try:
            with pytest.raises(ImageProcessorError):
                image_processor.process(temp_path)
        finally:
            try:
                Path(temp_path).unlink()
            except PermissionError:
                pass  # Windowsæä»¶éé®é¢?å¿½ç¥
